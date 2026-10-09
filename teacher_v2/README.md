# EDM2 teacher

Independent V2 implementation; no imports from the V1 teacher or trainer.

| File | Purpose |
| --- | --- |
| `model.py` | EDM preconditioning, uncertainty loss, CFG, and sampling |
| `networks.py` | NVIDIA magnitude-preserving U-Net |
| `train.py` | Balanced training, LR schedule, EMA, checkpoints |
| `data.py` | Class sampling, training holdout, loss diagnostics, pixel statistics |
| `evaluate.py` | Competition-protocol FID |
| `sample.py` | Export sample PNGs |

Defaults: width 112, multipliers `[1,2,3]`, three residual blocks, attention at
16x16, batch 64, microbatch 2, BF16, Adam `(0.9,0.99)`, LR 2e-4, noise
`P_mean=-0.4/P_std=1.0`, sigma_data 0.5. LR warms up over 100 kimg and decays
as inverse square root after 128 kimg. **43,648,348 parameters per teacher;
87,296,696 including the full EMA and uncertainty heads**, checked against 100M.

Run commands from the repository root.

## Train

GPU smoke test:

```bash
uv run python -m teacher_v2.train --outdir checkpoints/edm2-smoke \
  --steps 5 --batch-size 8 --microbatch 1 \
  --fid-every 0 --loss-every 0 --save-every 5 --log-every 1
```

Default 5k pilot:

```bash
uv run python -m teacher_v2.train
```

All 17,079 permitted training images, with training-holdout diagnostics disabled:

```bash
uv run python -m teacher_v2.train --outdir checkpoints/edm2-112-full \
  --holdout-fraction 0 --loss-every 0
```

By default, 10% of the training split is reserved for fixed-sigma loss monitoring.
Class-balanced replacement sampling uses only the fitting subset. Validation
images are used only for FID, never for gradients. The uncertainty-weighted
training loss may become negative; diagnostic losses remain ordinary weighted MSE.

Resume and extend with the same configuration:

```bash
uv run python -m teacher_v2.train --steps 20000 \
  --resume checkpoints/edm2-112/training-state.pt
```

Use a fresh directory for this recipe. Check actual GPU memory before increasing
microbatch. CPU tests do not establish A100 memory use, speed, or convergence.

## Evaluate and sample

FID matches `evaluate.py`: full validation set, exactly 20 generated images per
class (3,020 total), identical PNG conversion, and one pooled CleanFID score.
Each evaluation uses fresh directories. Default 18-step Heun uses 35 NFE;
CFG other than guidance 1 doubles that count. Match seed, guidance, steps, and
batch size when comparing with the V1 baseline.

```bash
uv run python -m teacher_v2.evaluate --checkpoint checkpoints/edm2-112/best-fid.ckpt
uv run python -m teacher_v2.sample --checkpoint checkpoints/edm2-112/best-fid.ckpt
uv run python -m teacher_v2.data --output results/teacher-data-stats.json
```

## Outputs

`teacher.ckpt` is latest EMA; `training-state.pt` restores optimizer and progress.
`best-fid.ckpt` records `competition_fid`; `best-loss.ckpt` tracks holdout loss.
`stats.jsonl` contains loss, LR, timing, and peak memory; `monitor.jsonl` contains
diagnostics/FID. Sample grids are in `samples/`. Raw and EMA models are retained
in `snapshots/` every 1k steps, about 350 MB per pair; disable with
`--snapshot-every 0`. Snapshot retention does not implement post-hoc power-function EMA.

Attribution and CC BY-NC-SA 4.0 license: [NOTICE.md](NOTICE.md), [LICENSE.txt](LICENSE.txt).
Tests: `uv run python -m unittest discover -s tests -v`.
