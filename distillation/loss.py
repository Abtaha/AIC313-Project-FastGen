"""Adjacent-noise consistency distillation with a frozen guided Heun teacher."""

import torch


def noise_grid(config, intervals, device):
    if intervals < 2:
        raise ValueError("At least two noise intervals are required")
    ramp = torch.linspace(0, 1, intervals + 1, device=device)
    grid = (config.sigma_min**(1/config.rho) + ramp *
            (config.sigma_max**(1/config.rho) - config.sigma_min**(1/config.rho))) ** config.rho
    grid[0], grid[-1] = config.sigma_min, config.sigma_max
    return grid


@torch.no_grad()
def heun_target(teacher, x, high, low, category, guidance):
    """Per-example sigma pairs; no zero endpoint, clipping, or teacher gradients."""
    hi, lo = high[:, None, None, None], low[:, None, None, None]
    derivative = (x - teacher.guided(x, high, category, guidance)) / hi
    proposal = x + (lo - hi) * derivative
    corrected = (proposal - teacher.guided(proposal, low, category, guidance)) / lo
    return x + (lo - hi) * (derivative + corrected) / 2


def consistency_loss(student, target, teacher, images, category, *, grid, guidance,
                     loss_type="pseudo-huber", trajectory=False, trajectory_source="student"):
    if trajectory:
        noisy, low, high = trajectory_input(student, images, category, grid,
                                           source=trajectory_source)
    else:
        index = torch.randint(len(grid) - 1, (len(images),), device=images.device)
        low, high = grid[index], grid[index + 1]
        noisy = images.float() + high[:, None, None, None] * torch.randn_like(images)
    with torch.no_grad():
        # Teacher integration stays FP32 even when the student uses AMP.
        with torch.autocast(device_type=images.device.type, enabled=False):
            previous = heun_target(teacher, noisy, high, low, category, guidance)
        expected = target(previous, low, category)
    difference = student(noisy, high, category).float() - expected.float()
    if loss_type == "mse":
        distance = difference.square().flatten(1).mean(1)
    elif loss_type == "pseudo-huber":
        constant = 0.00054 * (images[0].numel() ** 0.5)
        distance = (difference.square() + constant**2).sqrt().sub(constant).flatten(1).mean(1)
    else:
        raise ValueError("Unknown consistency loss")
    return (distance / (high - low)).mean()


@torch.no_grad()
def trajectory_input(student, images, category, grid, *, source="student", stage=None):
    """Stopped-gradient sampler prefix, including the exact inference re-noising.

    Adaptation of backward simulation (DMD2 sec. 4.5) to our local consistency
    objective, not the paper's full multi-step DMD/GAN objective. No intermediate
    clipping: ConsistencyStudent.sample clips only after its final call.
    """
    if source not in ("student", "real"):
        raise ValueError("Trajectory source must be student or real")
    sigmas = student.sampling_sigmas(4, device=images.device)
    stage = int(torch.randint(4, (), device=images.device)) if stage is None else stage
    if stage not in range(4):
        raise ValueError("Trajectory stage must be in 0..3")
    high = sigmas[stage]
    lower = grid[grid < high]
    if not len(lower):
        raise ValueError("Inference sigma requires a strictly lower consistency target")
    low = lower[-1]
    if source == "real":
        x = images.float() + high * torch.randn_like(images)
    else:
        x = torch.randn_like(images) * sigmas[0]
        mode = student.training
        try:
            student.eval()
            with torch.autocast(device_type=images.device.type, enabled=False):
                for index in range(stage):
                    x = student(x, sigmas[index], category)
                    scale = (sigmas[index+1].square()-student.config.sigma_min**2).clamp_min(0).sqrt()
                    x = x + scale * torch.randn_like(x)
        finally:
            student.train(mode)
    return x.detach(), low.expand(len(images)), high.expand(len(images))
