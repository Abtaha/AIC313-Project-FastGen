"""Convolutional U-Net with an adaLN-Zero Transformer bottleneck."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.attention import scaled_attention


@dataclass(frozen=True)
class HybridConfig:
    # Transformer bottleneck.
    width: int = 640
    depth: int = 10
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # U-Net stem.
    base_channels: int = 160

    backbone: str = "hybrid"

    def __post_init__(self):
        if self.backbone != "hybrid":
            raise ValueError("HybridConfig requires backbone='hybrid'")
        self._validate_dimensions()

    def _validate_dimensions(self):
        if self.width < 4 or self.heads < 1 or self.width % self.heads != 0:
            raise ValueError("width must be positive and divisible by positive heads")

        if self.base_channels < 32 or self.base_channels % 32:
            raise ValueError("base_channels must be a positive multiple of 32 for GroupNorm")

        if self.width != 4 * self.base_channels:
            raise ValueError("This implementation expects width == 4 * base_channels")

        if self.depth < 1:
            raise ValueError("depth must be positive")

        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio < 1:
            raise ValueError("mlp_ratio must be finite and >= 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(10000) * torch.arange(128) / 128),
        )

        self.net = nn.Sequential(
            nn.Linear(256, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, t):
        angles = t.float()[:, None] * self.frequencies[None] * 1000

        embedding = torch.cat(
            (angles.cos(), angles.sin()),
            dim=-1,
        )

        return self.net(embedding)


class ResBlock(nn.Module):
    """
    Spatial residual block with timestep/category FiLM conditioning.

    Convolution handles local image structure.
    Conditioning modulates each feature channel.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        condition_dim,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(32, out_channels)

        self.condition = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                condition_dim,
                2 * out_channels,
            ),
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        )

    def forward(self, x, condition):
        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)

        shift, scale = self.condition(condition).chunk(
            2,
            dim=-1,
        )

        shift = shift[:, :, None, None]
        scale = scale[:, :, None, None]

        h = h * (1 + scale) + shift

        h = F.silu(h)
        h = self.conv2(h)

        return residual + h


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class TransformerBlock(nn.Module):
    """
    DiT-style adaLN-Zero block used only at 16×16.

    64×64 image
        -> conv downsample
        -> 16×16
        -> 256 tokens

    This gives global attention without making raw pixels the
    transformer's responsibility.
    """

    def __init__(
        self,
        width,
        heads,
        mlp_ratio=4.0,
        dropout=0.0,
        rotary=None,
    ):
        super().__init__()

        self.heads = heads
        self.dropout = dropout
        self.rotary = rotary

        self.norm1 = nn.LayerNorm(
            width,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(
            width,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.qkv = nn.Linear(
            width,
            3 * width,
        )

        self.proj = nn.Linear(
            width,
            width,
        )

        hidden = int(width * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                width,
                6 * width,
            ),
        )

    def forward(self, x, condition):
        (
            shift1,
            scale1,
            gate1,
            shift2,
            scale2,
            gate2,
        ) = self.modulation(condition).chunk(
            6,
            dim=-1,
        )

        y = modulate(
            self.norm1(x),
            shift1,
            scale1,
        )

        batch, tokens, width = y.shape

        qkv = self.qkv(y)

        q, k, v = (
            qkv.reshape(
                batch,
                tokens,
                3,
                self.heads,
                width // self.heads,
            )
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        if self.rotary is not None:
            q, k = self.rotary(q, k)

        y = scaled_attention(
            q,
            k,
            v,
            dropout_p=(self.dropout if self.training else 0.0),
        )

        y = y.transpose(1, 2).reshape(
            batch,
            tokens,
            width,
        )

        y = self.proj(y)

        x = x + gate1[:, None] * F.dropout(
            y,
            self.dropout,
            self.training,
        )

        y = self.mlp(
            modulate(
                self.norm2(x),
                shift2,
                scale2,
            )
        )

        return x + gate2[:, None] * y


class HybridFlowNet(nn.Module):
    """
    64×64 category-conditioned velocity network.

    Conv U-Net:
        64 -> 32 -> 16 -> 32 -> 64

    Transformer:
        only at 16×16.
    """

    config_type = HybridConfig
    encoder_blocks = (2, 2)
    decoder_blocks = 1
    learned_position = True
    add_post_bottleneck = False

    def make_transformer_block(self, config):
        return TransformerBlock(config.width, config.heads, config.mlp_ratio, config.dropout)

    def __init__(self, config=None):
        super().__init__()

        self.config = (
            config
            if isinstance(config, self.config_type)
            else self.config_type(**(config or {}))
        )

        c = self.config

        c1 = c.base_channels
        c2 = c1 * 2
        c3 = c.width

        # -------------------------
        # CONDITIONING
        # -------------------------

        self.category = nn.Embedding(
            151,
            c3,
        )

        self.time = TimeEmbedding(c3)

        # MeanFlow conditions on t-r; standard FM uses interval=0.
        self.interval = TimeEmbedding(c3)

        # -------------------------
        # ENCODER: 64×64
        # -------------------------

        self.stem = nn.Conv2d(
            3,
            c1,
            kernel_size=3,
            padding=1,
        )

        self.enc64 = nn.ModuleList(
            [
                ResBlock(c1, c1, c3) for _ in range(self.encoder_blocks[0])
            ]
        )

        # 64 -> 32
        self.down64 = nn.Conv2d(
            c1,
            c2,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # -------------------------
        # ENCODER: 32×32
        # -------------------------

        self.enc32 = nn.ModuleList(
            [
                ResBlock(c2, c2, c3) for _ in range(self.encoder_blocks[1])
            ]
        )

        # 32 -> 16
        self.down32 = nn.Conv2d(
            c2,
            c3,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # -------------------------
        # BOTTLENECK: 16×16
        # -------------------------

        self.bottleneck = ResBlock(
            c3,
            c3,
            c3,
        )

        # 16×16 = exactly 256 transformer tokens.
        if self.learned_position:
            self.position = nn.Parameter(
                torch.empty(1, 16 * 16, c3)
            )

        self.transformer = nn.ModuleList(
            [
                self.make_transformer_block(c)
                for _ in range(c.depth)
            ]
        )

        self.transformer_norm = nn.LayerNorm(
            c3,
            eps=1e-6,
        )
        if self.add_post_bottleneck:
            self.post_bottleneck = ResBlock(c3, c3, c3)

        # -------------------------
        # DECODER: 16 -> 32
        # -------------------------

        self.up32_conv = nn.Conv2d(
            c3,
            c2,
            kernel_size=3,
            padding=1,
        )

        # c2 decoder + c2 encoder skip
        self.dec32 = ResBlock(
            c2 + c2,
            c2,
            c3,
        )
        if self.decoder_blocks > 1:
            self.dec32 = nn.ModuleList([self.dec32] + [
                ResBlock(c2, c2, c3) for _ in range(self.decoder_blocks - 1)
            ])

        # -------------------------
        # DECODER: 32 -> 64
        # -------------------------

        self.up64_conv = nn.Conv2d(
            c2,
            c1,
            kernel_size=3,
            padding=1,
        )

        # c1 decoder + c1 encoder skip
        self.dec64 = ResBlock(
            c1 + c1,
            c1,
            c3,
        )
        if self.decoder_blocks > 1:
            self.dec64 = nn.ModuleList([self.dec64] + [
                ResBlock(c1, c1, c3) for _ in range(self.decoder_blocks - 1)
            ])

        self.output_norm = nn.GroupNorm(
            32,
            c1,
        )

        self.output = nn.Conv2d(
            c1,
            3,
            kernel_size=3,
            padding=1,
        )

        self.apply(self._init)

        if self.learned_position:
            nn.init.normal_(self.position, std=0.02)

        nn.init.normal_(
            self.category.weight,
            std=0.02,
        )

        # Start residual image blocks near identity.
        for module in self.modules():
            if isinstance(module, ResBlock):
                nn.init.zeros_(module.conv2.weight)

                if module.conv2.bias is not None:
                    nn.init.zeros_(module.conv2.bias)

        # adaLN-Zero transformer.
        for block in self.transformer:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)

        # Initial predicted velocity = zero.
        nn.init.zeros_(self.output.weight)

        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    @staticmethod
    def _init(module):
        if isinstance(
            module,
            (nn.Linear, nn.Conv2d),
        ):
            nn.init.xavier_uniform_(module.weight.flatten(1))

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x,
        timestep,
        *,
        category,
        interval=None,
    ):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 64, 64):
            raise ValueError("Expected [B, 3, 64, 64]")

        batch = x.shape[0]

        if timestep.shape != (batch,):
            raise ValueError("timestep must have shape [B]")

        if category.shape != (batch,):
            raise ValueError("category must have shape [B]")

        if interval is None:
            interval = torch.zeros_like(timestep)

        if interval.shape != (batch,):
            raise ValueError("interval must have shape [B]")

        # Global conditioning vector.
        condition = (
            self.category(category) + self.time(timestep) + self.interval(interval)
        )

        # =========================
        # 64 × 64
        # =========================

        h64 = self.stem(x)

        for block in self.enc64:
            h64 = block(
                h64,
                condition,
            )

        # =========================
        # 32 × 32
        # =========================

        h32 = self.down64(h64)

        for block in self.enc32:
            h32 = block(
                h32,
                condition,
            )

        # =========================
        # 16 × 16
        # =========================

        h16 = self.down32(h32)

        h16 = self.bottleneck(
            h16,
            condition,
        )

        # Convert feature map to tokens.
        tokens = h16.flatten(2).transpose(1, 2)

        if self.learned_position:
            tokens = tokens + self.position

        for block in self.transformer:
            tokens = block(
                tokens,
                condition,
            )

        tokens = self.transformer_norm(tokens)

        # Tokens -> feature map.
        h16 = tokens.transpose(1, 2).reshape(
            batch,
            self.config.width,
            16,
            16,
        )
        if self.add_post_bottleneck:
            h16 = self.post_bottleneck(h16, condition)

        # =========================
        # 32 × 32 decoder
        # =========================

        h = F.interpolate(
            h16,
            scale_factor=2,
            mode="nearest",
        )

        h = self.up32_conv(h)

        h = torch.cat(
            (h, h32),
            dim=1,
        )

        for block in self.dec32 if isinstance(self.dec32, nn.ModuleList) else (self.dec32,):
            h = block(h, condition)

        # =========================
        # 64 × 64 decoder
        # =========================

        h = F.interpolate(
            h,
            scale_factor=2,
            mode="nearest",
        )

        h = self.up64_conv(h)

        h = torch.cat(
            (h, h64),
            dim=1,
        )

        for block in self.dec64 if isinstance(self.dec64, nn.ModuleList) else (self.dec64,):
            h = block(h, condition)

        h = self.output_norm(h)
        h = F.silu(h)

        return self.output(h)

    def architecture_config(self):
        return asdict(self.config)
