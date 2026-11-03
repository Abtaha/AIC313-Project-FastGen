# Attribution and methods

All submitted generator and teacher weights were trained locally on the supplied
Pokémon training split. No externally pretrained generator, perceptual network,
classifier, or image decoder is used. Inception is used only by the provided FID
evaluation implementation.

## Source code

- Course base code: KAIST Visual AI Group,
  <https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen>, official
  revision `3ccf3261495bd144bf844b7f298d10406b53117a`. The supplied data interface,
  downloader, evaluator, parameter utilities, fixed manifests/mapping, and base
  `Model` class are unchanged. Only the NFE-specific subclasses are implemented.
- NVIDIA EDM: <https://github.com/NVlabs/edm>, revision
  `008a4e5316c8e3bfe61a62f874bddba254295afb`, `training/networks.py`. The adapted
  SongUNet backbone is under `src/edm/`. Its copyright headers, full CC BY-NC-SA
  4.0 license, and detailed adaptation notice are retained in that folder.

## Papers and our adaptations

- Karras et al. **Elucidating the Design Space of Diffusion-Based Generative
  Models.** NeurIPS 2022. <https://arxiv.org/abs/2206.00364>.
  EDM preconditioning/loss, log-normal sigma sampling, Karras schedule, Heun
  teacher, and image-based EMA.
- Song et al. **Score-Based Generative Modeling through Stochastic Differential
  Equations.** ICLR 2021. <https://arxiv.org/abs/2011.13456>.
  DDPM++/SongUNet ancestry, as credited by NVIDIA's source.
- Song et al. **Consistency Models.** ICML 2023.
  <https://arxiv.org/abs/2303.01469>.
  Adjacent-noise teacher distillation, consistency boundary parameterization,
  EMA targets, and repeated denoising/re-noising. Our objective uses inverse-gap
  weighted pseudo-Huber distance and a guided local Heun teacher.
- Yin et al. **One-step Diffusion with Distribution Matching Distillation.**
  CVPR 2024. <https://arxiv.org/abs/2311.18828>.
  Real/fake score gradient and teacher-pair regression. We use an EDM-space
  implementation, consistency initialization, and multiscale pixel L2 rather
  than LPIPS; no pretrained perceptual network is introduced.
- Yin et al. **Improved Distribution Matching Distillation for Fast Image
  Synthesis.** NeurIPS 2024. <https://arxiv.org/abs/2405.14867>.
  Multiple fake-score updates and a classifier on the fake denoiser's middle
  features. Our DMD2-inspired adaptation uses 3:1 updates, a small retained paired
  regression term, an EDM noise distribution, and real-image supervision from
  the official training manifest. It is not an exact DMD2 reproduction.

The four-pass noise grid is a fixed synthetic-training-pair-selected grid. The
one-pass sampler is a single conditional generator call. Neither submitted
sampler evaluates its training teacher, fake-score estimator, or GAN head.
