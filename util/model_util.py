# --------------------------------------------------------
# References:
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

from math import pi

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)


def circular_freqs(freqs, period):
    """
    Rounds angular frequencies (rad per position) to the nearest ones that are periodic over `period` positions,
    2 pi k / period with integer k, i.e. a whole number of cycles around a 360 deg panorama. The spectrum of the
    standard frequencies is kept: frequencies below half a cycle per panorama round to k = 0 (no rotation,
    position-free channels, as the standard low frequencies nearly are over the grid).
    """
    k = np.round(np.asarray(freqs, dtype=np.float64) * period / (2 * np.pi))
    return 2 * np.pi * k / period


class VisionRotaryEmbeddingFast(nn.Module):
    """
    2D axial RoPE over a token grid of pt_seq_len = (h, w) (or an int for a square grid): the first half of the
    rotated dimensions encodes the row, the second half the column. With circular_w, the column frequencies are
    integer multiples of 2 pi / w, so the rotation is periodic in the column: token w-1 and token 0 are neighbours
    (360 deg panoramas). num_cls_token in-context tokens are prepended with the identity rotation.
    """
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        num_cls_token = 0,
        circular_w = False
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        h, w = (pt_seq_len, pt_seq_len) if isinstance(pt_seq_len, int) else pt_seq_len
        if ft_seq_len is None: ft_seq_len = (h, w)
        ft_h, ft_w = (ft_seq_len, ft_seq_len) if isinstance(ft_seq_len, int) else ft_seq_len

        freqs_w = torch.from_numpy(circular_freqs(freqs.numpy(), ft_w)).float() if circular_w else freqs
        t_h = torch.arange(ft_h) / ft_h * h
        t_w = torch.arange(ft_w) if circular_w else torch.arange(ft_w) / ft_w * w

        freqs_h = repeat(torch.einsum('..., f -> ... f', t_h, freqs), '... n -> ... (n r)', r = 2)
        freqs_w = repeat(torch.einsum('..., f -> ... f', t_w, freqs_w), '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)  # [h, w, D]

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()  # [N_cls+N_img, D]
            self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
        else:
            self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
            self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

    def forward(self, t): return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, circular_w=False):
    """
    grid_size: int of the grid height and width, or (h, w)
    circular_w: the column embedding is periodic over w (360 deg panoramas)
    return:
    pos_embed: [h*w, embed_dim] or [extra_tokens+h*w, embed_dim] (w/ or w/o cls_token)
    """
    h, w = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, w_period=w if circular_w else None)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, w_period=None):
    assert embed_dim % 2 == 0

    # half of the dimensions encode grid[0] (the column), the other half grid[1] (the row)
    omega_w = circular_freqs(1. / 10000 ** (np.arange(embed_dim // 4) / (embed_dim / 4.)), w_period) if w_period else None
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], omega=omega_w)  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, omega=None):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    omega: (D/2,) frequencies, default 1 / 10000^(2i/D)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    if omega is None:
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class CylindricalPosEmbed(nn.Module):
    """
    Absolute position embedding of panoramas of any width, built from each column's direction:
    the first half of the dimensions encodes the patch row (1D sin-cos), the second half the patch azimuth with
    integer harmonics k = 1..max_az_freq (cycles per 360 deg), so the embedding is periodic across North/the seam
    and the same heading gets the same embedding in every camera, whatever its field of view or resolution.
    The azimuth of a patch is the circular mean of its columns' azimuths.
    """
    def __init__(self, embed_dim, grid_h, patch_size, max_az_freq=32):
        super().__init__()
        assert embed_dim % 4 == 0
        self.patch_size = patch_size
        row = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, np.arange(grid_h, dtype=np.float32))  # (gh, D/2)
        self.register_buffer("row_embed", torch.from_numpy(row).float(), persistent=False)
        k = torch.round(torch.logspace(0, np.log10(max_az_freq), embed_dim // 4))
        self.register_buffer("az_freqs", k, persistent=False)

    def forward(self, az):
        """az: (B, gw) patch azimuths in radians -> (B, gh * gw, D), row-major like the patch embedding"""
        B, gw = az.shape
        gh = self.row_embed.shape[0]
        ang = az.float()[..., None] * self.az_freqs                       # (B, gw, D/4)
        az_emb = torch.cat([ang.sin(), ang.cos()], dim=-1)                # (B, gw, D/2)
        row = self.row_embed[None, :, None].expand(B, gh, gw, -1)
        return torch.cat([row, az_emb[:, None].expand(B, gh, gw, -1)], dim=-1).reshape(B, gh * gw, -1)


def patch_azimuths(angles, patch_size):
    """(B, W) column azimuths in radians -> (B, W / patch_size) patch azimuths (circular mean, safe across the wrap)"""
    B, W = angles.shape
    a = angles.float().reshape(B, W // patch_size, patch_size)
    return torch.atan2(a.sin().mean(-1), a.cos().mean(-1))


def cylindrical_rope(az, grid_h, dim, num_cls_token=0, turn_tokens=48, theta=10000):
    """
    2D axial RoPE tables for panoramas of any width, the column rotation driven by the patch azimuth.
    Rows use the standard frequencies; for the columns, the standard frequencies (rad per token) are rounded to
    whole cycles per 360 deg for a nominal turn_tokens patches per 360 deg, so the rotation of a column depends
    on its heading only: attention between two patches depends on their azimuth difference, wraps around
    North for 360 deg cameras, and matches across cameras of different resolutions.
    az: (B, gw) patch azimuths; dim: rotated dims per axis (half the head dim).
    Returns (cos, sin), each (B, num_cls_token + grid_h * gw, 2 * dim); in-context tokens get the identity.
    """
    B, gw = az.shape
    freqs = 1. / (theta ** (torch.arange(0, dim, 2, device=az.device)[:(dim // 2)].float() / dim))
    k = torch.round(freqs * turn_tokens / (2 * pi))                          # cycles per 360 deg
    f_h = repeat(torch.arange(grid_h, device=az.device).float()[:, None] * freqs, '... n -> ... (n r)', r=2)  # (gh, dim)
    f_w = repeat(az.float()[..., None] * k, '... n -> ... (n r)', r=2)       # (B, gw, dim)
    f = torch.cat([f_h[None, :, None].expand(B, grid_h, gw, -1), f_w[:, None].expand(B, grid_h, gw, -1)], dim=-1)
    f = f.reshape(B, grid_h * gw, 2 * dim)
    cos, sin = f.cos(), f.sin()
    if num_cls_token > 0:
        cos = torch.cat([cos.new_ones(B, num_cls_token, 2 * dim), cos], dim=1)
        sin = torch.cat([sin.new_zeros(B, num_cls_token, 2 * dim), sin], dim=1)
    return cos, sin
