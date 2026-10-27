# Guided consistency distillation

This package distills the frozen V1 EDM checkpoint into a conditional consistency
student initialized from its SongUNet weights. V2 is not imported. The teacher's
guidance (default 2.0) is baked into training targets; students do not run CFG.
This is a first experiment, not a claim that student FID matches teacher FID.

The objective follows [Consistency Models](https://arxiv.org/abs/2303.01469):
noise a training image at a random upper grid level, integrate the guided teacher
one Heun interval downward, and match a stop-gradient EMA student at the lower
level. The online student predicts at the upper level. Boundary preconditioning
enforces `f(x, sigma_min) = x`. The default loss is pseudo-Huber with inverse
interval weighting; `--loss mse` provides an L2 alternative. Grid endpoints are
0.002 and 80, with 40 Karras intervals. Teacher integration is FP32; student and
target can use BF16. Student dropout and label dropout are disabled.

All 17,079 official training images are available through uniform-class sampling
with replacement and optional horizontal flips. Validation is never loaded by
training. Teacher weights remain frozen. The default inference model has
34,798,755 parameters, student plus its separate EMA has 69,597,510, and the
separate teacher has 34,798,755. No teacher or EMA is exported inside the student.
Peak **allocated** CUDA memory is checked against 20 GB, including optimizer state.

From the repository root, begin with a short GPU sanity run:

```bash
uv run python -m distillation.train \
  --teacher /root/teacher.ckpt \
  --outdir checkpoints/distillation-smoke \
  --steps 100 --batch-size 128 --microbatch 8 \
  --guidance 2.0 --amp bf16 --save-every 100 --sample-every 100
```

Then use a fresh output directory for a 10k-step pilot:

```bash
uv run python -m distillation.train \
  --teacher /root/teacher.ckpt \
  --outdir checkpoints/distillation-cd-g2 \
  --steps 10000 --batch-size 128 --microbatch 8 \
  --guidance 2.0 --num-intervals 40 --lr 0.0001 --amp bf16
```

Outputs are `stats.jsonl`, configuration/provenance, sample grids at both NFE
budgets, an inference-only `student.ckpt`, and resumable `training-state.pt`.
Only the latest checkpoints are retained to limit disk use. To extend a run,
repeat its command with the same outdir and recipe, a larger `--steps`, and
`--resume <outdir>/training-state.pt`. RNG and optimizer state are restored;
prefetching and restarting the data iterator mean sample order is not exactly
reproduced. A different teacher hash, manifest, category mapping, or recipe is
rejected on resume. Preserve a promising `student.ckpt` before extending a run.

`model.py` contains thin `ModelOneNFE` / `ModelFewNFE` adapters. The provided base
`Model` and evaluator are unchanged. Both modes load the same inference checkpoint;
one calls the backbone exactly once, four calls it exactly four times using
consistency re-noising between positive noise levels. These are not Heun steps.
The initial four-level schedule is a baseline and still needs measurement.
Default teacher architecture/configuration is required by the unchanged evaluator;
custom architectures can be loaded programmatically with `config=TeacherConfig(...)`.

Use the **provided** competition evaluator (20 generated images/class, full
validation reference, one pooled CleanFID). Choose new output/reference paths so
stale PNGs cannot contaminate a run. No alternate FID implementation is added:

```bash
uv run evaluate.py --model_checkpoint checkpoints/distillation-cd-g2/student.ckpt \
  --evaluate_mode one_nfe --batch_size 8 \
  --reference_dir results/cd-step10000/reference \
  --output_dir results/cd-step10000/generated
uv run evaluate.py --model_checkpoint checkpoints/distillation-cd-g2/student.ckpt \
  --evaluate_mode few_nfe --batch_size 8 \
  --reference_dir results/cd-step10000/reference \
  --output_dir results/cd-step10000/generated
```

Tests (no dataset downloads):

```bash
uv run python -m unittest distillation.test_distillation
```

SongUNet is reused from `src/edm/networks.py`; its NVIDIA attribution and license
remain in `src/edm/NOTICE.md`. This package implements the consistency equations
directly and introduces no dependencies beyond `requirements.txt`.

## Frozen-teacher scheduler experiments

`schedulers.py` implements two additional approaches from equations, with no
upstream sampler code copied into the repository:

- [LD3: Learning to Discretize Denoising Diffusion ODEs](https://arxiv.org/abs/2405.15506),
  Tong et al., ICLR 2025. [Reference implementation](https://github.com/vinhsuhi/LD3).
  Learn decreasing solver times using cumulative softmax, bounded decoupled model
  conditioning, and per-example soft teacher-forcing inputs projected into an L2
  ball. These input perturbations are used only for fitting; inference and
  monitoring start from independent unmodified Gaussian noise. The radius is in
  unit-latent L2 distance (equivalent to radius times sigma_max in EDM coordinates).
- [BézierFlow](https://arxiv.org/abs/2512.13255), Min et al., ICLR 2026.
  [Reference implementation](https://github.com/KAIST-Visual-AI-Group/BezierFlow).
  Learn monotonic Bernstein-polynomial alpha/beta curves with fixed endpoints
  and analytic derivatives, then integrate the transformed velocity field.
  With `y=x/alpha` and `sigma=beta/alpha`, the EDM denoiser gives
  `v=(beta'/beta)*x + (alpha'-alpha*beta'/beta)*D(y,sigma)`.

These are **adapted pilots**, not exact reproductions of the papers' reported
FID. Both use Euler integration and dependency-free multiscale L2 instead of
LPIPS. LD3 uses bounded multiplicative conditioning offsets rather than additive
offsets. BézierFlow uses a finite normalized EDM prior
`alpha(0)=1/sqrt(1+sigma_max^2)`, `beta(0)=sigma_max*alpha(0)`, and endpoints
`alpha(1)=1, beta(1)=0`; this avoids querying an infinite sigma. All stages are
before time 1, so no singular beta=0 denoiser evaluation occurs. Degree defaults
to 8. One-pass BézierFlow only learns combinations of the initial denoiser
prediction and noise; its capacity is limited, and matching distilled-student
quality is not guaranteed. LD3 rejects one-pass runs with no interior timestep.

The pretrained **local V1 teacher** remains frozen but input/noise gradients pass
through it. Activation rematerialization reduces training memory. The entire
inference sampler, frozen network included, has about 34.8M parameters. CFG costs
two backbone calls, so `--nfe 4 --guidance 2` performs exactly two Euler updates.
`--nfe 1` requires `--guidance 1`. Hooks test actual calls, including roundtrips.

`train_scheduler.py` creates reusable teacher-generated noise/image pairs without
reading real validation images. Default cache: two fitting pairs plus one
monitoring pair per class, 453 pairs total. Guidance, seed, teacher hash, and
category mapping are verified before reusing a cache. Both four-call methods use
the same guided targets; the one-call probe uses a separate conditional cache.
Keep caches/checkpoints in ignored `checkpoints/`, and logs in ignored `results/`.

Example four-call LD3 pilot, followed by BézierFlow using the same cache:

```bash
uv run python -m distillation.train_scheduler \
  --teacher /root/teacher.ckpt --method ld3 --nfe 4 --guidance 2 \
  --outdir checkpoints/scheduler-ld3-4nfe \
  --cache checkpoints/scheduler-targets-g2.pt --steps 500 --report-fid
uv run python -m distillation.train_scheduler \
  --teacher /root/teacher.ckpt --method bezier --nfe 4 --guidance 2 \
  --outdir checkpoints/scheduler-bezier-4nfe \
  --cache checkpoints/scheduler-targets-g2.pt --steps 500 --report-fid
```

For the one-call probe, use `--method bezier --nfe 1 --guidance 1`, a fresh outdir,
and `--cache checkpoints/scheduler-targets-g1.pt`. `--memory-fraction 0.14` can
bound the process allocator when sharing the GPU. Models/solver gradients are
FP32. These pilots run sequentially to limit contention with existing jobs.

Each run writes initial, latest, and best standalone sampler checkpoints plus
resumable optimizer/RNG/input-perturbation state. `best.ckpt` is selected by
matching error on held-out **synthetic noise**, never official validation FID.
Initial and latest grids use those same fixed probes. `--resume` requires the
original outdir, teacher, and recipe; `--steps` can be extended.

`--report-fid` loads that selected checkpoint and reuses the provided evaluator's
PNG conversion, full validation reference, 20 samples/class, and pooled CleanFID.
It writes `fid.json` after fitting; validation FID does not feed the optimizer or
checkpoint selector. Temporary FID PNGs are cleaned after scoring. The standalone
checkpoint loader is `LearnedSampler.load_checkpoint`; these experiment artifacts
are separate from the consistency student checkpoints expected by the current
`ModelOneNFE`/`ModelFewNFE` adapters. Submission integration follows if a pilot wins.

```bash
uv run python -m unittest distillation.test_schedulers
```
