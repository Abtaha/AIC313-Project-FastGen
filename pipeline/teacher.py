"""Class-conditional EDM teacher trained from scratch on the FastGen train split.

EDM equations and sampler follow NVlabs/edm (see src/edm/NOTICE.md).
Noise arguments are standard deviations, not discrete DDPM timesteps.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.edm.networks import SongUNet
from src.utils import count_parameters


@dataclass(frozen=True)
class TeacherConfig:
    resolution: int = 64
    channels: int = 3
    num_classes: int = 151
    model_channels: int = 96
    channel_mult: tuple[int, ...] = (1, 2, 2, 2)
    num_blocks: int = 4
    attn_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.10
    label_dropout: float = 0.10
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0


class EDMTeacher(nn.Module):
    """DDPM++ with EDM preconditioning; forward returns a clean-image estimate."""

    def __init__(self, config: TeacherConfig | None = None):
        super().__init__()
        self.config = config or TeacherConfig()
        c = self.config
        if c.num_classes < 1 or c.sigma_data <= 0:
            raise ValueError("num_classes and sigma_data must be positive")
        if c.resolution < 4 or c.resolution & (c.resolution - 1):
            raise ValueError("resolution must be a power of two >= 4")
        if not c.channel_mult or c.resolution // 2 ** (len(c.channel_mult) - 1) < 4:
            raise ValueError("channel_mult must leave a bottleneck of at least 4x4")
        if (c.model_channels not in (4, 8, 16) and c.model_channels % 32) or c.model_channels < 4 or c.num_blocks < 1:
            raise ValueError("model_channels must be 4/8/16 or a multiple of 32; num_blocks >= 1")
        if not 0 <= c.dropout < 1 or not 0 <= c.label_dropout < 1:
            raise ValueError("dropout and label_dropout must be in [0, 1)")
        self.backbone = SongUNet(
            img_resolution=c.resolution, in_channels=c.channels,
            out_channels=c.channels, label_dim=c.num_classes,
            model_channels=c.model_channels, channel_mult=list(c.channel_mult),
            num_blocks=c.num_blocks, attn_resolutions=list(c.attn_resolutions),
            dropout=c.dropout, embedding_type="positional",
            encoder_type="standard", decoder_type="standard",
            resample_filter=[1, 1], label_dropout=c.label_dropout,
        )
        self.num_parameters = count_parameters(self, trainable_only=False)
        if self.num_parameters > 100_000_000:
            raise ValueError(f"Teacher has {self.num_parameters:,} parameters; limit is 100M")

    def forward(self, x, sigma, category=None):
        if x.ndim != 4 or tuple(x.shape[1:]) != (
            self.config.channels, self.config.resolution, self.config.resolution
        ):
            raise ValueError("x must have shape [batch, channels, resolution, resolution]")
        if category is None:
            labels = x.new_zeros(x.shape[0], self.config.num_classes).float()
        else:
            category = torch.as_tensor(category, device=x.device)
            if category.shape != (x.shape[0],) or category.dtype != torch.long:
                raise ValueError("category must be a long tensor of shape [batch]")
            labels = F.one_hot(category, self.config.num_classes).float()
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1)
        if sigma.numel() not in (1, x.shape[0]) or not torch.all(torch.isfinite(sigma) & (sigma > 0)):
            raise ValueError("sigma must be positive, finite, and scalar or per image")
        sigma = sigma.expand(x.shape[0]).reshape(-1, 1, 1, 1)
        x = x.float()
        sd = self.config.sigma_data
        c_skip = sd ** 2 / (sigma.square() + sd ** 2)
        c_out = sigma * sd / (sigma.square() + sd ** 2).sqrt()
        c_in = (sigma.square() + sd ** 2).rsqrt()
        residual = self.backbone(c_in * x, sigma.log().flatten() / 4, labels)
        return c_skip * x + c_out * residual.float()

    def guided(self, x, sigma, category, guidance=1.0):
        """CFG denoising: D_null + guidance * (D_class - D_null).

        Costs one backbone evaluation at guidance=1 or category=None,
        otherwise two. Guidance is for eval-mode teacher targets only.
        """
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")
        if guidance != 1 and category is not None and self.training:
            raise RuntimeError("Call teacher.eval() before guided denoising")
        if guidance != 1 and category is not None and self.config.label_dropout == 0:
            raise ValueError("CFG requires a teacher trained with label_dropout > 0")
        conditional = self(x, sigma, category)
        if guidance == 1 or category is None:
            return conditional
        unconditional = self(x, sigma, None)
        return unconditional + guidance * (conditional - unconditional)

    def loss(self, images, category, *, p_mean=-1.2, p_std=1.2, generator=None):
        """EDM log-normal noise sampling and weighted denoising MSE."""
        sigma = (torch.randn(
            (len(images), 1, 1, 1), device=images.device, generator=generator
        ) * p_std + p_mean).exp()
        noise = torch.randn(images.shape, device=images.device, generator=generator)
        prediction = self(images.float() + sigma * noise, sigma, category)
        weight = (sigma.square() + self.config.sigma_data ** 2) / (
            sigma * self.config.sigma_data
        ).square()
        return (weight * (prediction - images.float()).square()).mean()

    def sigma_schedule(self, num_steps=40, *, device=None):
        """Karras schedule with a terminal zero; Heun costs 2*num_steps-1 NFE."""
        c = self.config
        if num_steps < 2 or not 0 < c.sigma_min < c.sigma_max or c.rho <= 0:
            raise ValueError("need num_steps >= 2, 0 < sigma_min < sigma_max, rho > 0")
        ramp = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float64)
        sigmas = (c.sigma_max ** (1 / c.rho) + ramp * (
            c.sigma_min ** (1 / c.rho) - c.sigma_max ** (1 / c.rho)
        )) ** c.rho
        return torch.cat([sigmas, sigmas.new_zeros(1)])

    @torch.no_grad()
    def ode_step(self, x, sigma, sigma_next, category, *, solver="heun", guidance=1.0):
        """Deterministic teacher transition for future distillation targets.

        No clipping is applied. Heun uses two denoiser calls except at terminal
        zero, where one Euler call is used. CFG doubles the backbone NFE count.
        The input is already sigma-scaled.
        """
        if self.training:
            raise RuntimeError("Call teacher.eval() before creating ODE targets")
        if solver not in ("heun", "euler"):
            raise ValueError("solver must be heun or euler")
        s = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        sn = torch.as_tensor(sigma_next, device=x.device, dtype=x.dtype)
        if s.numel() != 1 or sn.numel() != 1 or not 0 <= sn < s:
            raise ValueError("ODE step requires scalar 0 <= sigma_next < sigma")
        d = (x - self.guided(x, s, category, guidance).to(x.dtype)) / s
        result = x + (sn - s) * d
        if solver == "heun" and sn > 0:
            d_next = (result - self.guided(result, sn, category, guidance).to(x.dtype)) / sn
            result = x + (sn - s) * (d + d_next) / 2
        return result

    @torch.no_grad()
    def sample(self, shape, *, device=None, category=None, num_steps=40,
               solver="heun", generator=None, latents=None, return_trajectory=False,
               churn=0.0, churn_min=0.0, churn_max=math.inf, noise_scale=1.0,
               guidance=1.0):
        """Teacher sampling, optionally returning unclipped states and NFE count.

        ``latents`` is unit Gaussian noise. For paired distillation targets use
        explicit latents, churn=0, and return_trajectory=True. Trajectory storage
        is opt-in because it scales with batch size and number of steps.
        """
        if self.training:
            raise RuntimeError("Call teacher.eval() before sampling")
        if len(shape) != 4 or tuple(shape[1:]) != (
            self.config.channels, self.config.resolution, self.config.resolution
        ) or shape[0] < 1:
            raise ValueError("shape must match the configured teacher image dimensions")
        device = torch.device(device or next(self.parameters()).device)
        if category is None:
            category = torch.randint(self.config.num_classes, (shape[0],),
                                     device=device, generator=generator)
        else:
            category = torch.as_tensor(category, device=device, dtype=torch.long)
        if category.shape != (shape[0],):
            raise ValueError("category must have one ID per image")
        if churn < 0 or noise_scale <= 0:
            raise ValueError("churn must be nonnegative and noise_scale positive")
        sigmas = self.sigma_schedule(num_steps, device=device)
        if latents is None:
            latents = torch.randn(shape, device=device, generator=generator)
        elif tuple(latents.shape) != tuple(shape):
            raise ValueError("latents must match shape")
        # Float64 solver state follows NVIDIA; denoiser itself runs in float32.
        x = latents.to(device=device, dtype=torch.float64) * sigmas[0]
        states = [x.clone()] if return_trajectory else None
        nfe = 0
        for s, sn in zip(sigmas[:-1], sigmas[1:]):
            gamma = min(churn / num_steps, math.sqrt(2) - 1) if churn_min <= s <= churn_max else 0
            sh = s * (1 + gamma)
            if gamma:
                noise = torch.randn(shape, device=device, dtype=x.dtype, generator=generator)
                x = x + (sh.square() - s.square()).sqrt() * noise_scale * noise
            x = self.ode_step(x, sh, sn, category, solver=solver, guidance=guidance)
            nfe += (2 if solver == "heun" and sn > 0 else 1) * (2 if guidance != 1 else 1)
            if states is not None:
                states.append(x.clone())
        if return_trajectory:
            return {"samples": x.float(), "states": torch.stack(states),
                    "sigmas": sigmas, "category": category, "nfe": nfe}
        return x.float().clamp(-1, 1)

    @classmethod
    def load_checkpoint(cls, path, *, device="cpu", freeze=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "fastgen-edm-teacher-v1":
            raise ValueError("Expected an EDM teacher checkpoint")
        teacher = cls(TeacherConfig(**checkpoint["config"]))
        teacher.load_state_dict(checkpoint["state_dict"])
        teacher.to(device).eval()
        if freeze:
            teacher.requires_grad_(False)
        return teacher

    def checkpoint(self):
        return {"format": "fastgen-edm-teacher-v1", "config": asdict(self.config),
                "state_dict": self.state_dict()}
