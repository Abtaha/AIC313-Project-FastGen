"""Consistency boundary parameterization and exactly counted student sampling.

Uses the existing attributed SongUNet implementation, without a teacher or EMA
registered inside the inference model. See README.md for the objective/reference.
"""

from dataclasses import asdict, replace

import torch
from torch import nn
from torch.nn import functional as F

from teacher import EDMTeacher, TeacherConfig


class ConsistencyStudent(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = replace(config or TeacherConfig(), dropout=0.0, label_dropout=0.0)
        # Reuse architecture construction/validation; retain only its backbone.
        self.backbone = EDMTeacher(self.config).backbone
        c = self.config
        ramp = torch.arange(4, dtype=torch.float32) / 4
        self.register_buffer("four_sigmas", (c.sigma_max**(1/c.rho) + ramp *
            (c.sigma_min**(1/c.rho)-c.sigma_max**(1/c.rho))) ** c.rho)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Original consistency checkpoints predate the persistent sampling grid.
        if prefix + "four_sigmas" not in state_dict:
            state_dict[prefix + "four_sigmas"] = self.four_sigmas.clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def set_four_sigmas(self, values):
        values = torch.as_tensor(values, dtype=torch.float32, device=self.four_sigmas.device)
        c = self.config
        if values.shape != (4,) or not torch.isfinite(values).all() or not torch.all(values[:-1] > values[1:]):
            raise ValueError("Four finite strictly decreasing noise levels required")
        if not torch.isclose(values[0], values.new_tensor(c.sigma_max)) or values[-1] < c.sigma_min:
            raise ValueError("Grid must start at sigma_max and stay >= sigma_min")
        self.four_sigmas.copy_(values)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != "fastgen-consistency-v1":
            raise ValueError("Expected an exported consistency/DMD student checkpoint")
        model = cls(TeacherConfig(**data["config"]))
        model.load_state_dict({k.removeprefix("backbone."): v for k, v in data["state_dict"].items()})
        return model.to(device).eval()

    @classmethod
    def from_teacher(cls, teacher):
        model = cls(teacher.config).to(next(teacher.parameters()).device)
        model.backbone.load_state_dict(teacher.backbone.state_dict())
        return model

    def forward(self, x, sigma, category):
        c = self.config
        if x.ndim != 4 or tuple(x.shape[1:]) != (c.channels, c.resolution, c.resolution):
            raise ValueError("Input shape differs from student configuration")
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1)
        if sigma.numel() not in (1, len(x)) or not torch.all(torch.isfinite(sigma) & (sigma >= c.sigma_min)):
            raise ValueError("sigma must be finite, >= sigma_min, and scalar or per image")
        category = torch.as_tensor(category, device=x.device)
        if category.dtype != torch.long or category.shape != (len(x),):
            raise ValueError("category must be a long tensor of shape [batch]")
        sigma = sigma.expand(len(x)).reshape(-1, 1, 1, 1)
        delta = sigma - c.sigma_min
        skip = c.sigma_data**2 / (delta.square() + c.sigma_data**2)
        out = delta * c.sigma_data / (sigma.square() + c.sigma_data**2).sqrt()
        scaled = x.float() / (sigma.square() + c.sigma_data**2).sqrt()
        labels = F.one_hot(category, c.num_classes).float()
        residual = self.backbone(scaled, sigma.log().flatten() / 4, labels)
        # f(x, sigma_min) = x exactly, which prevents the constant-solution collapse.
        return skip * x.float() + out * residual.float()

    def sampling_sigmas(self, nfe, *, device):
        if nfe not in (1, 4):
            raise ValueError("Student sampling supports exactly 1 or 4 NFE")
        if nfe == 4:
            return self.four_sigmas.to(device)
        c = self.config
        # Four positive levels; zero is never passed to the log-noise embedding.
        ramp = torch.arange(nfe, device=device, dtype=torch.float32) / nfe
        return (c.sigma_max**(1/c.rho) + ramp *
                (c.sigma_min**(1/c.rho) - c.sigma_max**(1/c.rho))) ** c.rho

    @torch.no_grad()
    def sample(self, shape, *, device="cuda", category=None, nfe=1, generator=None, latents=None):
        if self.training:
            raise RuntimeError("Call eval() before student sampling")
        c = self.config
        if len(shape) != 4 or shape[0] < 1 or tuple(shape[1:]) != (c.channels, c.resolution, c.resolution):
            raise ValueError("Sample shape differs from student configuration")
        device = torch.device(device)
        if category is None:
            category = torch.randint(c.num_classes, (shape[0],), device=device, generator=generator)
        if latents is None:
            latents = torch.randn(shape, device=device, generator=generator)
        if tuple(latents.shape) != tuple(shape):
            raise ValueError("latents must match shape")
        sigmas = self.sampling_sigmas(nfe, device=device)
        x = latents.to(device=device, dtype=torch.float32) * sigmas[0]
        for index, sigma in enumerate(sigmas):
            if index:
                noise = torch.randn(shape, device=device, generator=generator)
                x = x + (sigma.square() - c.sigma_min**2).clamp_min(0).sqrt() * noise
            x = self(x, sigma, category)
        return x.clamp(-1, 1)

    def export(self, *, category_to_id, step, guidance):
        """State keys match the provided Model wrappers, in either NFE mode."""
        return {"format": "fastgen-consistency-v1", "config": asdict(self.config),
                "state_dict": {"backbone." + k: v.detach().cpu() for k, v in self.state_dict().items()},
                "category_to_id": category_to_id, "step": step, "teacher_guidance": guidance}
