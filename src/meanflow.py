"""Category-conditioned MeanFlow, without a teacher or classifier-free guidance.

Based on Geng et al., Mean Flows for One-step Generative Modeling (2025),
https://arxiv.org/abs/2505.13447, Algorithms 1/2 and Eq. (22).
See also the authors' implementation: https://github.com/Gsunshine/meanflow.
The separate no-grad JVP follows https://github.com/CaptainAmu/py-MeanFlow.
"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


MEANFLOW_DEFAULTS = {
    "mf_ratio": 0.25,
    "mf_time_mean": -0.4,
    "mf_time_std": 1.0,
    "mf_power": 1.0,
    "mf_eps": 1e-3,
}


def sample_times(batch_size, device, *, ratio=0.25, mean=-0.4, std=1.0):
    """Ordered logit-normal times, mixing finite intervals with r=t samples."""
    pair = (torch.randn(2, batch_size, device=device) * std + mean).sigmoid()
    t = pair.max(dim=0).values
    r = pair.min(dim=0).values
    # Match the reference's fixed batch proportion (rounding diagonal count down).
    diagonal_count = int(batch_size * (1 - ratio))
    r[:diagonal_count] = t[:diagonal_count]
    return r, t


def _random_state(device):
    state = [torch.get_rng_state(), None]
    if device.type == "cuda":
        state[1] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state[1] = torch.mps.get_rng_state()
    return state


def _restore_random_state(state, device):
    torch.set_rng_state(state[0])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state[1], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state[1])


def meanflow_loss(model, images, categories, noise, r, t, *, power=1.0, eps=1e-3):
    """Regress to sg(v - (t-r)*du/dt), with an exact forward-mode JVP.

    Paper convention: data at t=0, Gaussian noise at t=1. Parameterize
    u(z,r,t) as net(z,t,interval=t-r), so the interval derivative is included.
    The detached adaptive weights use per-image mean squared pixel error.
    """
    mix = t[:, None, None, None]
    z = (1 - mix) * images + mix * noise
    velocity = noise - images

    def average_velocity(z, r, t):
        return model(z, t, category=categories, interval=t - r)

    # Fused attention kernels do not support forward AD in the course's PyTorch
    # version. Math SDPA works for both DiT and the hybrid bottleneck.
    # Compute the JVP without a reverse-mode graph, then do one normal forward
    # for parameter gradients. Replay the RNG so any dropout masks agree.
    state = _random_state(images.device)
    with sdpa_kernel(SDPBackend.MATH):
        try:
            # Forward AD through autocast convolutions can mix primal/tangent
            # dtypes after normalization. Keep this target-only pass in FP32;
            # the gradient-enabled prediction still uses the caller's AMP mode.
            with torch.no_grad(), torch.autocast(images.device.type, enabled=False):
                _, derivative = torch.func.jvp(
                    average_velocity, (z.float(), r.float(), t.float()),
                    (velocity.float(), torch.zeros_like(r).float(), torch.ones_like(t).float()),
                )
        finally:
            _restore_random_state(state, images.device)
        prediction = average_velocity(z, r, t)

    interval = (t - r)[:, None, None, None]
    target = (velocity.float() - interval * derivative.float()).detach()
    error = (prediction.float() - target).square().flatten(1).mean(1)
    weight = (error.detach() + eps).pow(-power)
    return (weight * error).mean(), error.detach().mean()
