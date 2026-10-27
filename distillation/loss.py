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
                     loss_type="pseudo-huber"):
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
