# EDM source attribution

`networks.py` contains the DDPM++/NCSN++ backbone and supporting layers adapted
from NVIDIA's official EDM implementation:

- Repository: https://github.com/NVlabs/edm
- Revision: `008a4e5316c8e3bfe61a62f874bddba254295afb`
- Source file: `training/networks.py`
- Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International; the full upstream license is in `LICENSE.txt`.

Adaptations: omitted unused architectures and preconditioners; removed the
pickle persistence decorators/import because local checkpoints use state
dictionaries; replaced scalar NumPy math with Python `math`. The backbone,
initialization, attention, resampling, and residual computations are retained.
No pretrained weights are included or downloaded.

`teacher.py` implements EDM preconditioning, weighted denoising loss, Karras
noise discretization, and Euler/Heun sampling following `training/networks.py`,
`training/loss.py`, and `generate.py` in the same repository. These adapted
portions are also covered by the upstream license.

Paper to cite in the project write-up:

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. **Elucidating the Design
Space of Diffusion-Based Generative Models.** NeurIPS 2022.
https://arxiv.org/abs/2206.00364

The DDPM++ architecture also draws on Song et al., **Score-Based Generative
Modeling through Stochastic Differential Equations**, ICLR 2021, as credited
in the upstream source.
