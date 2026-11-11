import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,  # int (square) or [H, W]
            in_channels=3,
            cond_channels=args.cond_channels,
            weather_channels=args.weather_channels,
            weather_times=args.weather_times,
            weather_size=args.weather_size,
            weather_patch_size=args.weather_patch_size,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            circular_rope=getattr(args, 'circular_rope', False),
            cylindrical=getattr(args, 'cylindrical', False),
        )
        self.img_size = self.net.input_size  # (H, W)

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        # conditioning dropout during training (also enables CFG):
        # each conditioning frame (I0, It_prev) is blacked out independently with cond_drop_prob, both together
        # with cond_drop_both_prob, and the weather is replaced by the null weather tokens with weather_drop_prob
        self.cond_drop_prob = getattr(args, 'cond_drop_prob', 0.0)
        self.cond_drop_both_prob = getattr(args, 'cond_drop_both_prob', 0.0)
        self.weather_drop_prob = getattr(args, 'weather_drop_prob', 0.0)

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        # weather CFG: v(weather) + (cfg_scale - 1) * (v(weather) - v(null weather)), frames always kept
        self.cfg_scale = getattr(args, 'cfg', 1.0)

    # channels of cond (see make_conditioning): I0 (3), It_prev (3), angle_mask (cos, sin, 0), view_mask (1)
    ANGLE_COS, ANGLE_SIN = 6, 7

    def column_angles(self, cond):
        """
        (N, W) azimuth of every column in radians (0 = North) for cylindrical models, None otherwise. Read back from
        the angle_mask channels (cos, sin, normalized to [-1, 1]), so it follows any crop or resize of the frames.
        """
        if not self.net.cylindrical:
            return None
        return torch.atan2(cond[:, self.ANGLE_SIN].float().mean(1), cond[:, self.ANGLE_COS].float().mean(1))

    @staticmethod
    def make_conditioning(I0, It_prev, w0, wt, wt_prev, angle_mask, view_mask):
        """
        Channel-concatenates the conditioning frames, and stacks the weather fields along a time axis
        in the same order as the frames ([It,] I0, It_prev): (N, 3, C_w, H_w, W_w)
        """
        cond = torch.cat([I0, It_prev, angle_mask, view_mask], dim=1)
        w = torch.stack([wt, w0, wt_prev], dim=1)
        return cond, w

    def drop_frames(self, I0, It_prev):
        """
        Blacks out (-1 in [-1, 1]) I0 and It_prev independently with probability cond_drop_prob each,
        and both together with probability cond_drop_both_prob
        """
        both = torch.rand(I0.shape[0], 1, 1, 1, device=I0.device) < self.cond_drop_both_prob
        def drop(frame):
            dropped = both | (torch.rand(frame.shape[0], 1, 1, 1, device=frame.device) < self.cond_drop_prob)
            return torch.where(dropped, torch.full_like(frame, -1.0), frame)
        return drop(I0), drop(It_prev)

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, I0, It, It_prev, w0, wt, wt_prev, angle_mask, view_mask):
        x = It
        w_drop = None
        if self.training:
            I0, It_prev = self.drop_frames(I0, It_prev)
            w_drop = torch.rand(x.shape[0], device=x.device) < self.weather_drop_prob
        cond, w = self.make_conditioning(I0, It_prev, w0, wt, wt_prev, angle_mask, view_mask)

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), cond, w, w_drop, angles=self.column_angles(cond))
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss over the valid field of view only: view_mask is normalized to [-1, 1] with the image
        # transforms (1 inside the original FOV, -1 in the padding added by PanoramaExpand)
        valid = (view_mask > 0).to(v.dtype)
        loss = ((v - v_pred) ** 2).mean(dim=1, keepdim=True) * valid
        # average over all valid pixels in the batch, so fully padded crops contribute nothing
        loss = loss.sum() / valid.sum().clamp_min(1)

        return loss

    @torch.no_grad()
    def generate(self, I0, It_prev, w0, wt, wt_prev, angle_mask, view_mask):
        """Samples It given the conditioning frames and weather"""
        cond, w = self.make_conditioning(I0, It_prev, w0, wt, wt_prev, angle_mask, view_mask)
        device = I0.device
        bsz = I0.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, *I0.shape[-2:], device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, cond, w)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], cond, w)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, cond, w):
        """Velocity at (z, t), with weather CFG when cfg_scale != 1. t: (N, 1, 1, 1)"""
        if self.cfg_scale == 1.0:
            x_pred = self.net(z, t.flatten(), cond, w, angles=self.column_angles(cond))
            return (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

        # conditional and null-weather predictions in one batch
        n = z.shape[0]
        w_drop = torch.arange(2 * n, device=z.device) >= n
        angles = self.column_angles(cond)
        x_pred = self.net(torch.cat([z, z]), t.flatten().repeat(2), torch.cat([cond, cond]), torch.cat([w, w]), w_drop,
                          angles=None if angles is None else torch.cat([angles, angles]))
        v_cond, v_uncond = ((x_pred - torch.cat([z, z])) / (1.0 - t.repeat(2, 1, 1, 1)).clamp_min(self.t_eps)).chunk(2)
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, cond, w):
        v_pred = self._forward_sample(z, t, cond, w)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, cond, w):
        v_pred_t = self._forward_sample(z, t, cond, w)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, cond, w)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
