"""Frozen EDM with learned LD3 times or Bézier stochastic interpolants.

Independent equation implementation; paper links and experimental deviations
are documented in README.md. No upstream sampler code is vendored here.
"""

from dataclasses import asdict
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from teacher import EDMTeacher, TeacherConfig


def bernstein(control, time):
    degree = len(control) - 1
    return sum(math.comb(degree, k) * time**k * (1-time)**(degree-k) * value
               for k, value in enumerate(control))


def ordered_points(logits):
    # Explicit fixed endpoints avoid roundoff moving the boundary conditions.
    return torch.cat([logits.new_zeros(1), logits.softmax(0).cumsum(0)[:-1], logits.new_ones(1)])


class LD3Schedule(nn.Module):
    def __init__(self, steps, config):
        super().__init__()
        if steps < 2:
            raise ValueError("LD3 needs at least two solver calls; one call has no interior time")
        self.config, self.steps = config, steps
        ramp = torch.arange(steps, dtype=torch.float32) / steps
        sigma = (config.sigma_max**(1/config.rho) + ramp *
                 (config.sigma_min**(1/config.rho) - config.sigma_max**(1/config.rho))) ** config.rho
        normalized = (sigma - config.sigma_min) / (config.sigma_max - config.sigma_min)
        gaps = torch.cat([normalized, normalized.new_zeros(1)])
        self.logits = nn.Parameter((gaps[:-1] - gaps[1:]).log())
        # Decouple model conditioning from solver times, as in LD3 section 4.3.
        self.condition_offsets = nn.Parameter(torch.zeros(steps))

    def times(self):
        c = self.config
        points = ordered_points(self.logits)
        positive = c.sigma_min + (c.sigma_max-c.sigma_min) * (1-points[:-1])
        return torch.cat([positive, positive.new_zeros(1)])

    def condition(self, sigma, index):
        return (sigma * (0.5 * self.condition_offsets[index].tanh()).exp()).clamp(
            self.config.sigma_min, self.config.sigma_max)


class BezierSchedule(nn.Module):
    def __init__(self, config, degree=8):
        super().__init__()
        if degree < 2:
            raise ValueError("Bezier degree must be >= 2")
        self.config, self.degree = config, degree
        self.alpha_logits = nn.Parameter(torch.zeros(degree))
        self.beta_logits = nn.Parameter(torch.zeros(degree))
        # Normalize the finite EDM prior: z = a0 * (sigma_max * noise).
        self.alpha_start = 1 / math.sqrt(1 + config.sigma_max**2)
        self.beta_start = config.sigma_max * self.alpha_start

    def coefficients(self, time):
        a = self.alpha_start + (1-self.alpha_start) * ordered_points(self.alpha_logits)
        b = self.beta_start * (1-ordered_points(self.beta_logits))
        alpha, beta = bernstein(a, time), bernstein(b, time)
        da = self.degree * bernstein(a[1:]-a[:-1], time)
        db = self.degree * bernstein(b[1:]-b[:-1], time)
        return alpha, beta, da, db


class LearnedSampler(nn.Module):
    def __init__(self, teacher, *, method, nfe=4, guidance=2.0, degree=8):
        super().__init__()
        if method not in ("ld3", "bezier") or nfe not in (1, 4):
            raise ValueError("Choose ld3/bezier and NFE 1/4")
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")
        cost = 1 if guidance == 1 else 2
        if nfe % cost:
            raise ValueError("CFG requires two backbone calls per solver evaluation")
        self.teacher = teacher.eval().requires_grad_(False)
        self.method, self.nfe, self.guidance, self.degree = method, nfe, guidance, degree
        self.steps = nfe // cost
        self.schedule = (LD3Schedule(self.steps, teacher.config) if method == "ld3" else
                         BezierSchedule(teacher.config, degree))
        self.schedule.to(next(teacher.parameters()).device)
        if self.count_parameters() > 100_000_000:
            raise ValueError("Entire sampler including frozen teacher exceeds 100M")

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def denoise(self, x, sigma, category, rematerialize):
        def call(state, level):
            return self.teacher.guided(state, level, category, self.guidance)
        if rematerialize and torch.is_grad_enabled():
            return checkpoint(call, x, sigma, use_reentrant=False)
        return call(x, sigma)

    def forward(self, latents, category, *, rematerialize=True):
        """Differentiable sampling; no clipping or extra terminal denoiser call."""
        c = self.teacher.config
        if self.method == "ld3":
            times = self.schedule.times()
            x = latents.float() * c.sigma_max
            for index in range(self.steps):
                sigma = times[index]
                prediction = self.denoise(x, self.schedule.condition(sigma, index), category, rematerialize)
                x = x + (times[index+1]-sigma) * (x-prediction) / sigma
            return x
        x = latents.float() * self.schedule.beta_start
        for index in range(self.steps):
            time = x.new_tensor(index / self.steps)
            a, b, da, db = self.schedule.coefficients(time)
            # EDM source coordinates: y=x/a, sigma=b/a. All evaluation
            # times precede t=1, where b=0, so no singular endpoint is queried.
            sigma = (b/a).clamp(c.sigma_min, c.sigma_max)
            prediction = self.denoise(x/a, sigma, category, rematerialize)
            velocity = (db/b) * x + (da - a*db/b) * prediction
            x = x + velocity / self.steps
        return x

    @torch.no_grad()
    def sample(self, shape, *, device="cuda", category=None, generator=None, latents=None):
        c = self.teacher.config
        if len(shape) != 4 or shape[0] < 1 or tuple(shape[1:]) != (c.channels, c.resolution, c.resolution):
            raise ValueError("Sample shape differs from teacher configuration")
        if category is None:
            category = torch.randint(c.num_classes, (shape[0],), device=device, generator=generator)
        if latents is None:
            latents = torch.randn(shape, device=device, generator=generator)
        if tuple(latents.shape) != tuple(shape):
            raise ValueError("latents must match shape")
        return self(latents.to(device), category, rematerialize=False).clamp(-1, 1)

    def export(self):
        return {"format": "fastgen-learned-scheduler-v1", "config": asdict(self.teacher.config),
                "method": self.method, "nfe": self.nfe, "guidance": self.guidance, "degree": self.degree,
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != "fastgen-learned-scheduler-v1":
            raise ValueError("Not a learned scheduler checkpoint")
        model = cls(EDMTeacher(TeacherConfig(**data["config"])),
                    **{k: data[k] for k in ("method", "nfe", "guidance", "degree")})
        model.load_state_dict(data["state_dict"])
        return model.to(device).eval()


def matching_loss(prediction, target):
    """Dependency-free multiscale L2; explicitly not LPIPS."""
    loss = prediction.new_zeros(())
    for factor in (1, 2, 4):
        if factor == 1:
            left, right = prediction, target
        else:
            left = torch.nn.functional.avg_pool2d(prediction, factor)
            right = torch.nn.functional.avg_pool2d(target, factor)
        loss = loss + (left-right).square().mean() / 3
    return loss


@torch.no_grad()
def project_latents(proposal, original, radius):
    delta = proposal-original
    norms = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
    return original + delta * (radius/norms).clamp(max=1)[:, None, None, None]
