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
