# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm, rotate_half, \
    CylindricalPosEmbed, patch_azimuths, cylindrical_rope


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True, strict_width=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patch_size, patch_size)
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"image size {img_size} must be a multiple of the patch size {patch_size}"
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # strict_width=False: any width that is a multiple of the patch size (panoramas of any field of view)
        self.strict_width = strict_width

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.strict_width:
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        else:
            assert H == self.img_size[0] and W % self.patch_size[1] == 0, \
                f"Input image size ({H}*{W}): height must be {self.img_size[0]}, width a multiple of {self.patch_size[1]}."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0., fused=False):
        super().__init__()
        self.num_heads = num_heads
        # fused: torch's memory-efficient scaled_dot_product_attention (FlashAttention on recent GPUs) in the autocast
        # dtype, O(N) memory instead of the full N x N fp32 attention matrix; needed for long token sequences
        self.fused = fused
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if isinstance(rope, tuple):
            cos, sin = (r.unsqueeze(1).to(q.dtype) for r in rope)  # (B, 1, N, D), broadcast over heads
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        else:
            q = rope(q)
            k = rope(k)

        dropout_p = self.attn_drop.p if self.training else 0.
        if self.fused:
            q, k = q.to(v.dtype), k.to(v.dtype)
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            x = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, fused_attn=False):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop, fused=fused_attn)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        cond_channels=0,
        weather_channels=0,
        weather_times=3,
        weather_size=16,
        weather_patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        in_context_start=8,
        circular_rope=False,
        cylindrical=False,
        turn_tokens=48,
        fused_attn=False,
        grad_ckpt=False
    ):
        super().__init__()
        # recompute each block's activations in the backward pass instead of storing them (saves memory, ~30% slower)
        self.grad_ckpt = grad_ckpt
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        # (H, W): 256 for square crops, (256, 2048) for full panoramas
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
        # column position (RoPE and sincos pos_embed) periodic over the width, for 360 deg panoramas
        self.circular_rope = circular_rope
        # panoramas of any width: position embedding and column RoPE from each column's azimuth (0 = North),
        # patch grid from the input; input_size only sets the height
        self.cylindrical = cylindrical
        self.turn_tokens = turn_tokens
        assert not (circular_rope and cylindrical), "circular_rope and cylindrical are exclusive"
        self.in_context_start = in_context_start

        # time embed
        self.t_embedder = TimestepEmbedder(hidden_size)

        # linear embed of the noisy image channel-concatenated with the conditioning frames
        self.x_embedder = BottleneckPatchEmbed(self.input_size, patch_size, in_channels + cond_channels, bottleneck_dim, hidden_size, bias=True,
                                               strict_width=not cylindrical)

        # weather fields are patchified into in-context tokens by an embedder shared across time steps;
        # a learned time-slot embedding (one per weather time) tells the time steps apart
        assert weather_size % weather_patch_size == 0, "weather_size must be a multiple of weather_patch_size"
        self.weather_times = weather_times
        self.w_embedder = BottleneckPatchEmbed(weather_size, weather_patch_size, weather_channels, bottleneck_dim, hidden_size, bias=True)
        self.w_pos_embed = nn.Parameter(torch.zeros(1, 1, self.w_embedder.num_patches, hidden_size), requires_grad=False)
        self.w_time_embed = nn.Parameter(torch.zeros(1, weather_times, 1, hidden_size), requires_grad=True)
        # learned null weather tokens, used in place of the weather when it is dropped (training) or for CFG
        self.w_null = nn.Parameter(torch.zeros(1, weather_times * self.w_embedder.num_patches, hidden_size), requires_grad=True)
        self.in_context_len = weather_times * self.w_embedder.num_patches

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        if cylindrical:
            self.pos_embed = None
            self.cyl_pos_embed = CylindricalPosEmbed(hidden_size, self.x_embedder.grid_size[0], patch_size)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)


        # rope
        half_head_dim = hidden_size // num_heads // 2
        grid_size = self.x_embedder.grid_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=grid_size,
            num_cls_token=0,
            circular_w=circular_rope
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=grid_size,
            num_cls_token=self.in_context_len,
            circular_w=circular_rope
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     fused_attn=fused_attn)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size, circular_w=self.circular_rope)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w_pos_embed = get_2d_sincos_pos_embed(self.w_pos_embed.shape[-1], self.w_embedder.grid_size)
        self.w_pos_embed.data.copy_(torch.from_numpy(w_pos_embed).float()[None, None])
        nn.init.normal_(self.w_time_embed, std=.02)
        nn.init.normal_(self.w_null, std=.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        for embedder in (self.x_embedder, self.w_embedder):
            w1 = embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(embedder.proj2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p, grid_size=None):
        """
        x: (N, T, patch_size**2 * C)
        grid_size: (h, w) patch grid of the input (default: the model's fixed grid)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        h, w = grid_size or self.x_embedder.grid_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, cond, w, w_drop=None, angles=None):
        """
        x: (N, C, H, W) noisy image
        t: (N,)
        cond: (N, C_cond, H, W) conditioning frames
        w: (N, T, C_w, H_w, W_w) weather fields at T times
        w_drop: (N,) bool, samples whose weather is replaced by the null weather tokens
        angles: (N, W) azimuth in radians of every column (cylindrical models only)
        """
        grid_size = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        # weather tokens (shared embedder per time + spatial and time-slot embeddings), pooled into the time embedding for adaLN
        N, T = w.shape[:2]
        w_tokens = self.w_embedder(w.flatten(0, 1)).unflatten(0, (N, T))
        w_tokens = (w_tokens + self.w_pos_embed + self.w_time_embed).flatten(1, 2)
        if w_drop is not None:
            w_tokens = torch.where(w_drop.view(-1, 1, 1), self.w_null.to(w_tokens.dtype), w_tokens)
        t_emb = self.t_embedder(t)
        c = t_emb + w_tokens.mean(dim=1)

        # forward JiT
        x = self.x_embedder(torch.cat([x, cond], dim=1))
        if self.cylindrical:
            assert angles is not None, "cylindrical models need the column azimuths"
            az = patch_azimuths(angles, self.patch_size)  # (N, gw)
            x = x + self.cyl_pos_embed(az).to(x.dtype)
            half_head_dim = self.hidden_size // self.num_heads // 2
            rope = cylindrical_rope(az, grid_size[0], half_head_dim, turn_tokens=self.turn_tokens)
            rope_incontext = cylindrical_rope(az, grid_size[0], half_head_dim, num_cls_token=self.in_context_len,
                                              turn_tokens=self.turn_tokens)
        else:
            x += self.pos_embed
            rope, rope_incontext = self.feat_rope, self.feat_rope_incontext

        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                x = torch.cat([w_tokens, x], dim=1)
            r = rope if i < self.in_context_start else rope_incontext
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, c, r, use_reentrant=False)
            else:
                x = block(x, c, r)

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size, grid_size)

        return output


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_start=4, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_start=8, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_start=10, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_start=10, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}
