"""Pixel-space adaLN-Zero transformer. Original implementation; see EXPERIMENTS.md."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiTConfig:
    width: int = 512
    depth: int = 12
    heads: int = 8
    patch_size: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    backbone: str = "dit"

    def __post_init__(self):
        if self.backbone != "dit":
            raise ValueError("Only dit is implemented; U-Net is a future extension.")
        if (
            self.width < 4
            or self.depth < 1
            or self.heads < 1
            or self.width % self.heads
        ):
            raise ValueError(
                "Require positive depth/heads and width >= 4 divisible by heads."
            )
        if self.patch_size not in (2, 4, 8):
            raise ValueError("patch_size must be 2, 4 or 8")
        if (
            not math.isfinite(self.mlp_ratio)
            or self.mlp_ratio < 1
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("Require finite mlp_ratio >= 1 and dropout in [0, 1).")


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer(
            "frequencies", torch.exp(-math.log(10000) * torch.arange(128) / 128)
        )
        self.net = nn.Sequential(
            nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(self, t):
        angles = t.float()[:, None] * self.frequencies[None] * 1000
        return self.net(torch.cat((angles.cos(), angles.sin()), dim=-1))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.dropout = c.dropout
        self.norm1 = nn.LayerNorm(c.width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(c.width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.proj = nn.Linear(c.width, c.width)
        hidden = int(c.width * c.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(c.width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(c.dropout),
            nn.Linear(hidden, c.width),
            nn.Dropout(c.dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(c.width, 6 * c.width))

    def forward(self, x, condition):
        s1, a1, g1, s2, a2, g2 = self.modulation(condition).chunk(6, dim=-1)
        y = modulate(self.norm1(x), s1, a1)
        b, n, d = y.shape
        q, k, v = (
            self.qkv(y)
            .reshape(b, n, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        y = self.proj(y.transpose(1, 2).reshape(b, n, d))
        x = x + g1[:, None] * F.dropout(y, self.dropout, self.training)
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, a2))


class PixelDiT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = (
            config if isinstance(config, DiTConfig) else DiTConfig(**(config or {}))
        )
        c = self.config
        p = c.patch_size
        self.patch = nn.Conv2d(3, c.width, p, stride=p)
        self.position = nn.Parameter(torch.empty(1, (64 // p) ** 2, c.width))
        self.category = nn.Embedding(151, c.width)
        self.time = TimeEmbedding(c.width)
        # Reserved for interval objectives; standard FM always uses interval=0.
        self.interval = TimeEmbedding(c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.depth)])
        self.final_norm = nn.LayerNorm(c.width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(c.width, 2 * c.width)
        )
        self.output = nn.Linear(c.width, p * p * 3)
        self.apply(self._init)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.category.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        for layer in (self.interval.net[-1], self.final_modulation[-1], self.output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
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
        condition = (
            self.category(category) + self.time(timestep) + self.interval(interval)
        )
        tokens = self.patch(x).flatten(2).transpose(1, 2) + self.position
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        patches = self.output(modulate(self.final_norm(tokens), shift, scale))
        p = self.config.patch_size
        side = 64 // p
        return (
            patches.reshape(x.shape[0], side, side, p, p, 3)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(x.shape[0], 3, 64, 64)
        )

    def architecture_config(self):
        return asdict(self.config)
