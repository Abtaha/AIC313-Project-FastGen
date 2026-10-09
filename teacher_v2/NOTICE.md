# EDM2 attribution

`networks.py` is adapted from NVIDIA's `training/networks_edm2.py` at revision
`4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc` of https://github.com/NVlabs/edm2.
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. The upstream CC BY-NC-SA 4.0
license is preserved in `LICENSE.txt`.

Adaptations: removed pickle persistence and the unused preconditioning/
uncertainty-estimation class; replaced scalar NumPy functions with `math` and
the resampling filter construction with Torch; replaced explicit attention
matrices with PyTorch SDPA, with a CPU forward/gradient equivalence test. CUDA
fused implementations can have numerical differences. No weights are fetched.

The independent `teacher_v2/model.py` includes EDM preconditioning, CFG dropout,
the Fourier log-variance head, and the uncertainty-weighted EDM2 loss.
`teacher_v2/train.py` adapts the inverse-square-root LR schedule and Adam betas
from upstream `training/training_loop.py`, scaled to the small-data experiment.
Exponential EMA is retained; post-hoc power-function EMA is not implemented.
Raw and EMA snapshots are retained for subsequent profile comparisons.

Cite:

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli
Laine. **Analyzing and Improving the Training Dynamics of Diffusion Models.**
CVPR 2024. https://arxiv.org/abs/2312.02696
