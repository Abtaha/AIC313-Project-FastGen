"""InceptFlow: local multi-scale convolutions around global rotary attention."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.models.hybrid import HybridConfig, HybridFlowNet, TransformerBlock
from src.models.hybrid_v2 import Rotary2D
from src.utils import count_parameters


@dataclass(frozen=True)
class InceptFlowConfig(HybridConfig):
    depth: int = 6
    backbone: str = "inceptflow"

    def __post_init__(self):
        if self.backbone != "inceptflow":
            raise ValueError("InceptFlowConfig requires backbone='inceptflow'")
        self._validate_dimensions()
        if (self.width // self.heads) % 4:
            raise ValueError("2D RoPE requires head_dim divisible by four")


class InceptionResBlock(nn.Module):
    """Conditioned 1x1, 3x3, stacked 3x3, and dilated 3x3 branches.

    Equal-width blocks start as identity. A channel-changing block starts as
    its learned skip projection, allowing decoder skip concatenation.
    """

    def __init__(self, channels, condition_dim, *, in_channels=None):
        super().__init__()
        in_channels = channels if in_channels is None else in_channels
        if channels < 4 or channels % 4:
            raise ValueError("channels must be a positive multiple of four")
        if in_channels < 32 or in_channels % 32:
            raise ValueError("in_channels must be a positive multiple of 32")
        branch = channels // 4
        self.norm = nn.GroupNorm(32, in_channels)
        self.branch1 = nn.Conv2d(in_channels, branch, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch, 1), nn.SiLU(),
            nn.Conv2d(branch, branch, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch, 1), nn.SiLU(),
            nn.Conv2d(branch, branch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(branch, branch, 3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch, 1), nn.SiLU(),
            nn.Conv2d(branch, branch, 3, padding=2, dilation=2),
        )
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * channels))
        self.proj = nn.Conv2d(channels, channels, 1)
        self.skip = (nn.Identity() if in_channels == channels
                     else nn.Conv2d(in_channels, channels, 1))
        self.apply(HybridFlowNet._init)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, condition):
        h = F.silu(self.norm(x))
        h = torch.cat([branch(h) for branch in
                       (self.branch1, self.branch2, self.branch3, self.branch4)], dim=1)
        shift, scale = self.condition(condition).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.proj(F.silu(h))


class _ConditionedSequence(nn.ModuleList):
    def forward(self, x, condition):
        for block in self:
            x = block(x, condition)
        return x


class InceptFlowNet(HybridFlowNet):
    """64 -> 32 -> 16 U-Net with six full-attention blocks by default.

    Encoder: two ResBlocks at 64, ResBlock + Inception at 32, ResBlock at 16.
    Decoder: two Inception blocks at 16, Inception at 32, ResBlock at 64.
    Category, timestep, and MeanFlow interval conditioning are shared throughout.
    """

    config_type = InceptFlowConfig
    learned_position = False
    add_post_bottleneck = True

    def make_transformer_block(self, config):
        return TransformerBlock(config.width, config.heads, config.mlp_ratio,
                                config.dropout, rotary=Rotary2D(config.width // config.heads))

    def __init__(self, config=None):
        super().__init__(config)
        c = self.config
        self.enc32[1] = InceptionResBlock(2 * c.base_channels, c.width)
        self.post_bottleneck = _ConditionedSequence([
            InceptionResBlock(c.width, c.width) for _ in range(2)
        ])
        self.dec32 = InceptionResBlock(2 * c.base_channels, c.width,
                                       in_channels=4 * c.base_channels)
        if count_parameters(self, trainable_only=False) > 100_000_000:
            raise ValueError("InceptFlow must have no more than 100M total parameters")

    def parameter_breakdown(self):
        conditioning = sum(count_parameters(module, False)
                           for module in (self.category, self.time, self.interval))
        transformer = count_parameters(self.transformer, False)
        normalization = count_parameters(self.transformer_norm, False)
        total = count_parameters(self, False)
        return dict(transformer=transformer, transformer_norm=normalization,
                    spatial_with_film=total - conditioning - transformer - normalization,
                    global_conditioning=conditioning, total=total)
