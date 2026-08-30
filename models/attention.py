"""Transformer 瓶颈与空间注意力（2D / 3D 共用实现）。

修复：旧版把 (B,C,H,W) 直接 flatten 成 token 序列送进 MultiheadAttention，
**没有任何位置编码**。自注意力对 token 排列是等变的，因此该模块贡献的全局信息
里完全不含空间位置 —— 它能告诉网络"某处有高速区"，却说不清在哪。对流场这种强
空间结构的任务是实质性缺陷。这里补上 2D / 3D 的 sin-cos 位置编码（无参数、
对分辨率自适应，换网格尺寸不需要重训）。
"""

import math

import torch
import torch.nn as nn


def _sincos_1d(positions, dim, device, dtype):
    """给定位置向量，生成 (len(positions), dim) 的 sin-cos 编码。"""
    if dim % 2 == 1:                       # 奇数维补一列后再裁掉
        enc = _sincos_1d(positions, dim + 1, device, dtype)
        return enc[:, :dim]
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                     * (-math.log(10000.0) / max(half, 1)))
    ang = positions.to(torch.float32).unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1).to(dtype)


def sincos_positional_encoding(channels, spatial, device, dtype):
    """任意维度网格的可分离 sin-cos 位置编码。

    Args:
        channels: 通道数 C
        spatial:  空间尺寸元组，如 (H, W) 或 (D, H, W)
    Returns:
        (1, C, *spatial)
    """
    n_axes = len(spatial)
    per_axis = channels // n_axes
    remainder = channels - per_axis * n_axes

    encodings = []
    for axis, size in enumerate(spatial):
        dim = per_axis + (remainder if axis == n_axes - 1 else 0)
        if dim <= 0:
            encodings.append(torch.zeros((size, 0), device=device, dtype=dtype))
            continue
        pos = torch.arange(size, device=device, dtype=torch.float32)
        encodings.append(_sincos_1d(pos, dim, device, dtype))

    # 逐轴广播拼接成 (C, *spatial)
    parts = []
    for axis, enc in enumerate(encodings):
        if enc.shape[1] == 0:
            continue
        view = [1] * (n_axes + 1)
        view[0] = enc.shape[1]
        view[axis + 1] = spatial[axis]
        expand = [enc.shape[1]] + list(spatial)
        moved = enc.transpose(0, 1).reshape(view).expand(expand)
        parts.append(moved)
    out = torch.cat(parts, dim=0) if parts else \
        torch.zeros((channels,) + tuple(spatial), device=device, dtype=dtype)
    return out.unsqueeze(0)


class TransformerBottleneck(nn.Module):
    """编解码器最深处的全局自注意力块，2D / 3D 通用。"""

    def __init__(self, channels, num_heads=4, mlp_ratio=4.0, dropout=0.0,
                 use_positional_encoding=True, max_tokens=8192):
        super(TransformerBottleneck, self).__init__()
        # 保证 embed_dim 能被 num_heads 整除
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        self.max_tokens = max_tokens

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )
        self._warned_tokens = False

    def forward(self, x):
        shape = x.shape
        spatial = shape[2:]
        n_tokens = 1
        for s in spatial:
            n_tokens *= int(s)

        if n_tokens > self.max_tokens:
            # O(N^2) 注意力在大网格上会爆显存；跳过并提示，而不是让它崩掉
            if not self._warned_tokens:
                print('[TransformerBottleneck] token 数 {} 超过上限 {}，'
                      '本层跳过（可调 --attn-max-tokens）'.format(n_tokens, self.max_tokens))
                self._warned_tokens = True
            return x

        if self.use_positional_encoding:
            pos = sincos_positional_encoding(shape[1], tuple(int(s) for s in spatial),
                                             x.device, x.dtype)
            x = x + pos

        x_flat = x.flatten(2).transpose(1, 2)          # (B, N, C)
        h = self.norm1(x_flat)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.transpose(1, 2).contiguous().view(*shape)


class SpatialAttention(nn.Module):
    """通道维 avg/max 池化 -> 卷积 -> sigmoid 权重图。ndim=2 或 3。"""

    def __init__(self, kernel_size=7, ndim=2):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        self.conv = conv(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn
