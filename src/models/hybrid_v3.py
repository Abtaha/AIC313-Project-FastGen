"""Deep multiscale U-Net with two adaLN-Zero DiT blocks at 8x8."""

from dataclasses import dataclass
import math

from torch import nn

from src.models.edm_unet import EDMUNetConfig, EDMUNetFlowNet
from src.models.hybrid import TransformerBlock
from src.models.hybrid_v2 import Rotary2D
from src.utils import count_parameters


@dataclass(frozen=True)
class HybridV3Config(EDMUNetConfig):
    # depth counts DiT blocks; convolutional stages always have two blocks.
    depth: int = 2
    mlp_ratio: float = 4.0
    backbone: str = "hybrid_v3"

    def __post_init__(self):
        if self.backbone != "hybrid_v3":
            raise ValueError("HybridV3Config requires backbone='hybrid_v3'")
        # Reuse the multiscale channel, head, depth, and dropout constraints.
        EDMUNetConfig(base_channels=self.base_channels, width=self.width,
                      depth=self.depth, heads=self.heads, dropout=self.dropout)
        if (self.width // self.heads) % 4:
            raise ValueError("2D RoPE requires head_dim divisible by four")
        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio < 1:
            raise ValueError("mlp_ratio must be finite and >= 1")


class HybridV3FlowNet(EDMUNetFlowNet):
    """64 -> 32 -> 16 -> 8 (DiT) -> 4, then a symmetric conv decoder.

    Keep two residual blocks at every encoder/decoder stage and spatial
    attention between the 16x16 blocks. One extra residual block plus
    attention at the 4x4 bottleneck leaves room for DiT under 100M.
    The 8x8 skip is saved after global mixing; DiT runs once per forward.
    """

    config_type = HybridV3Config
    residual_depth = 2
    bottleneck_depth = 1

    def __init__(self, config=None):
        super().__init__(config)
        c = self.config
        self.transformer = nn.ModuleList([
            TransformerBlock(c.width, c.heads, c.mlp_ratio, c.dropout,
                             rotary=Rotary2D(c.width // c.heads, height=8, width=8))
            for _ in range(c.depth)
        ])
        self.transformer_norm = nn.LayerNorm(c.width, eps=1e-6)
        self.transformer.apply(self._init)
        for block in self.transformer:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        if count_parameters(self, trainable_only=False) > 100_000_000:
            raise ValueError("Hybrid-v3 must have no more than 100M total parameters")

    def mix_encoder_stage(self, index, x, condition):
        if index != 3:
            return x
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.transformer:
            tokens = block(tokens, condition)
        return self.transformer_norm(tokens).transpose(1, 2).reshape_as(x)

    def parameter_breakdown(self):
        conditioning = sum(count_parameters(module, False)
                           for module in (self.category, self.time, self.interval))
        transformer = count_parameters(self.transformer, False)
        normalization = count_parameters(self.transformer_norm, False)
        total = count_parameters(self, False)
        return dict(transformer=transformer, transformer_norm=normalization,
                    spatial_with_film=total-conditioning-transformer-normalization,
                    global_conditioning=conditioning, total=total)
