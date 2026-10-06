"""EDM-style convolutional backbone predicting pixel-space flow velocities.

Uses raw flow time, without diffusion noise sampling or EDM preconditioning.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.attention import scaled_attention
from src.models.hybrid import TimeEmbedding


@dataclass(frozen=True)
class EDMUNetConfig:
    base_channels: int = 128
    width: int = 512
    depth: int = 2
    heads: int = 8
    dropout: float = 0.0
    backbone: str = "edm_unet"

    def __post_init__(self):
        if self.backbone != "edm_unet":
            raise ValueError("EDMUNetConfig requires backbone='edm_unet'")
        if self.base_channels < 32 or self.base_channels % 32:
            raise ValueError("base_channels must be a positive multiple of 32")
        if self.width != 4 * self.base_channels:
            raise ValueError("Require width == 4 * base_channels")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if self.heads < 1 or any(c % self.heads for c in self.channels[2:]):
            raise ValueError("heads must divide the 16x16 and 4x4 attention channel counts")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def channels(self):
        return tuple(self.base_channels * scale for scale in (1, 2, 3, 4, 4))


class ResBlock(nn.Module):
    """GroupNorm/FiLM residual convolution with variance-scaled addition."""

    def __init__(self, in_channels, out_channels, condition_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * out_channels))
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return (self.skip(x) + h) * (2 ** -0.5)


class SpatialAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(
            batch, 3, self.heads, channels // self.heads, height * width
        ).permute(1, 0, 2, 4, 3).unbind(0)
        h = scaled_attention(q, k, v).transpose(-2, -1).reshape(x.shape)
        return (x + self.proj(h)) * (2 ** -0.5)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, config, attention=False, depth=None):
        super().__init__()
        depth = config.depth if depth is None else depth
        self.blocks = nn.ModuleList([
            ResBlock(in_channels if i == 0 else out_channels, out_channels,
                     config.width, config.dropout) for i in range(depth)
        ])
        self.attention = SpatialAttention(out_channels, config.heads) if attention else nn.Identity()

    def forward(self, x, condition):
        for i, block in enumerate(self.blocks):
            x = block(x, condition)
            if i == 0:
                x = self.attention(x)
        return x


class EDMUNetFlowNet(nn.Module):
    """64 -> 32 -> 16 -> 8 -> 4 U-Net; attention at 16 and the bottleneck."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, EDMUNetConfig) else EDMUNetConfig(**(config or {}))
        c = self.config
        channels = c.channels
        self.category = nn.Embedding(151, c.width)
        self.time = TimeEmbedding(c.width)
        # Optional t-r conditioning preserves the project's MeanFlow interface.
        self.interval = TimeEmbedding(c.width)
        self.stem = nn.Conv2d(3, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList([
            Stage(ch, ch, c, attention=i == 2) for i, ch in enumerate(channels)
        ])
        self.downsample = nn.ModuleList([
            nn.Conv2d(a, b, 3, stride=2, padding=1)
            for a, b in zip(channels[:-1], channels[1:])
        ])
        self.bottleneck = Stage(channels[-1], channels[-1], c, attention=True, depth=2)
        self.decoder = nn.ModuleList([
            Stage(2 * ch, ch, c, attention=i == 2) for i, ch in enumerate(channels)
        ])
        self.upsample = nn.ModuleList([
            nn.Conv2d(b, a, 3, padding=1)
            for a, b in zip(channels[:-1], channels[1:])
        ])
        self.output_norm = nn.GroupNorm(32, channels[0], eps=1e-6)
        self.output = nn.Conv2d(channels[0], 3, 3, padding=1)
        self.apply(self._init)
        nn.init.normal_(self.category.weight, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight.flatten(1))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, timestep, *, category, interval=None):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 64, 64):
            raise ValueError("Expected [B, 3, 64, 64]")
        if timestep.shape != (x.shape[0],) or category.shape != (x.shape[0],):
            raise ValueError("timestep and category must have shape [B]")
        interval = torch.zeros_like(timestep) if interval is None else interval
        if interval.shape != timestep.shape:
            raise ValueError("interval must have shape [B]")
        condition = self.time(timestep) + self.category(category) + self.interval(interval)
        h = self.stem(x)
        skips = []
        for i, stage in enumerate(self.encoder):
            h = stage(h, condition)
            skips.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
        h = self.bottleneck(h, condition)
        for i in reversed(range(len(self.decoder))):
            h = self.decoder[i](torch.cat((h, skips[i]), dim=1), condition)
            if i > 0:
                h = self.upsample[i - 1](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(F.silu(self.output_norm(h)))

    def architecture_config(self):
        return asdict(self.config)
