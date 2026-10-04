"""Hybrid-v2: eight rotary Transformer blocks and a stronger convolutional path."""

from dataclasses import dataclass

import torch
from torch import nn

from src.models.hybrid import HybridConfig, HybridFlowNet, TransformerBlock
from src.utils import count_parameters


@dataclass(frozen=True)
class HybridV2Config(HybridConfig):
    depth: int = 8
    backbone: str = "hybrid_v2"

    def __post_init__(self):
        if self.backbone != "hybrid_v2":
            raise ValueError("HybridV2Config requires backbone='hybrid_v2'")
        self._validate_dimensions()
        if (self.width // self.heads) % 4:
            raise ValueError("2D RoPE requires head_dim divisible by four")


class Rotary2D(nn.Module):
    """Rotate adjacent Q/K pairs: half the head dimensions per spatial axis.

    Row-major tokens match flatten(2): x varies fastest, y varies by row.
    At head_dim=80, each axis rotates 20 pairs (40 dimensions).
    The fixed tables are buffers, not parameters, and are rebuilt on loading.
    """

    def __init__(self, head_dim, height=16, width=16):
        super().__init__()
        if head_dim < 4 or head_dim % 4:
            raise ValueError("2D RoPE requires head_dim divisible by four")
        self.head_dim = head_dim
        self.tokens = height * width
        axis_dim = head_dim // 2
        frequency = 10000.0 ** (-torch.arange(0, axis_dim, 2).float() / axis_dim)
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        angles = torch.cat((x.flatten()[:, None] * frequency,
                            y.flatten()[:, None] * frequency), dim=-1)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def rotate(self, value):
        if value.shape[-2:] != (self.tokens, self.head_dim):
            raise ValueError("RoPE input must match the spatial grid and head dimension")
        pairs = value.reshape(*value.shape[:-1], -1, 2)
        even, odd = pairs.unbind(-1)
        cosine, sine = self.cos.to(value.dtype), self.sin.to(value.dtype)
        return torch.stack((even * cosine - odd * sine,
                            even * sine + odd * cosine), dim=-1).flatten(-2)

    def forward(self, q, k):
        return self.rotate(q), self.rotate(k)


class HybridV2FlowNet(HybridFlowNet):
    """Default: 99,595,363 parameters, including all conditioning modules.

    Encoder blocks: 2 at 64x64, 3 at 32x32, 1 before attention at 16x16.
    Decoder blocks: 1 after attention at 16x16, 3 at 32x32, 3 at 64x64.
    The fully preferred 3/3 encoder layout exceeds 100M by 262,243 parameters;
    removing one 160-channel encoder block preserves every decoder block.
    """

    config_type = HybridV2Config
    encoder_blocks = (2, 3)
    decoder_blocks = 3
    learned_position = False
    add_post_bottleneck = True

    def make_transformer_block(self, config):
        return TransformerBlock(config.width, config.heads, config.mlp_ratio,
                                config.dropout, rotary=Rotary2D(config.width // config.heads))

    def __init__(self, config=None):
        super().__init__(config)
        if count_parameters(self, trainable_only=False) > 100_000_000:
            raise ValueError("Hybrid-v2 must have no more than 100M total parameters")

    def parameter_breakdown(self):
        """Disjoint counts that include FiLM and all timestep/interval embeddings."""
        conditioning = sum(count_parameters(module, trainable_only=False)
                           for module in (self.category, self.time, self.interval))
        transformer = count_parameters(self.transformer, trainable_only=False)
        normalization = count_parameters(self.transformer_norm, trainable_only=False)
        spatial = sum(count_parameters(module, trainable_only=False) for module in (
            self.stem, self.enc64, self.down64, self.enc32, self.down32,
            self.bottleneck, self.post_bottleneck, self.up32_conv, self.dec32,
            self.up64_conv, self.dec64, self.output_norm, self.output,
        ))
        total = count_parameters(self, trainable_only=False)
        return dict(transformer=transformer, transformer_norm=normalization,
                    spatial_with_film=spatial, global_conditioning=conditioning, total=total)
