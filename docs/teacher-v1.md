# EDM teacher for FastGen

The teacher uses the official NVlabs DDPM++ (`SongUNet`) backbone with EDM
preconditioning, log-normal noise sampling, weighted denoising MSE, and the
Karras Euler/Heun sampler. It is randomly initialized; no pretrained generation
model is used. Source attribution, adaptation details, and license are in
[`src/edm/NOTICE.md`](../src/edm/NOTICE.md).

The default is a 64x64 RGB, 151-category teacher with base width 96, multipliers
`[1,2,2,2]`, four residual blocks per resolution, attention at 16x16, dropout
0.10, and class-label dropout 0.10. It has **34,798,755 parameters**, or
**69,597,510 including the separate training EMA copy**. Both are checked against
the user-confirmed **100M** limit. Arbitrary widths are restricted to compatible
GroupNorm channel counts. The frozen teacher checkpoint contains only one model.

`model.py`, the base `Model`, dataset/evaluation code, and fixed manifests are
unchanged. The student classes remain placeholders for the later distillation
work. Teacher checkpoints are intentionally separate from `one_nfe.ckpt` and
`few_nfe.ckpt`; they cannot be evaluated as students.

## Train

Use the provided environment and requirements. No added runtime dependencies
are needed. Run on the course CUDA GPU:

For `uv`, this repository uses `requirements.txt` rather than `pyproject.toml`:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt torch==2.6.0 torchvision==0.21.0
uv run python -m unittest discover -s tests -v
```

On the course Linux CUDA 12.4 host, install the CUDA builds first, then the
remaining provided dependencies:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt
uv run train_teacher.py --outdir checkpoints/teacher
```

macOS has no CUDA; use the local environment for tests. Full training with the
default BF16/CUDA settings requires the NVIDIA GPU. `uv add` is not applicable
without a `pyproject.toml`.

```bash
python train_teacher.py --outdir checkpoints/teacher
```

Defaults: effective batch 128, microbatch 16 (8 accumulation passes), BF16,
horizontal flips, Adam at 2e-4 with a 100-kimg warmup, and image-based EMA with
500-kimg half-life and the 5% ramp-up. The 100k-step duration is a starting budget,
not a validated optimum: on the complete 17,079-image split it corresponds to
about 749 passes; use monitoring to select an appropriate duration.

Start with a short run to measure memory and speed on the actual GPU:

```bash
python train_teacher.py --steps 20 --log-every 1 --outdir checkpoints/teacher-smoke
```

Peak CUDA allocated and reserved memory are logged in decimal GB. Only allocated
memory triggers the 20-GB guard, per the project measurement convention. If it
approaches the limit, lower `--microbatch`; if there is headroom, raise it to a
divisor of `--batch-size`. VRAM and BF16 execution have not been measured on a
CUDA GPU in this development environment. An allocation can fail before the
guard executes, so begin with the short run.

Resume training:

```bash
python train_teacher.py --resume checkpoints/teacher/training-state.pt
```

Use the same architecture, seed, batch size, holdout settings, and selection
configuration when resuming. Model/EMA, optimizer, scaler, steps, RNG states,
best metrics, and elapsed time are restored. The shuffled data iterator starts
again, so resuming is **not bitwise equivalent** to an uninterrupted run.

Outputs:

- `teacher.ckpt`: latest EMA teacher and reconstruction configuration.
- `training-state.pt`: full resumable training state.
- `best-fid.ckpt`: best training-holdout FID EMA snapshot.
- `best-loss.ckpt`: lowest mean fixed-sigma training-holdout loss snapshot.
- `stats.jsonl`: training loss, gradient norm, learning rate, timing, peak memory.
- `monitor.jsonl`: fit/holdout losses at sigma 0.1, 0.5, 2, 10, and holdout FID.
- `samples/step-*.png`: eight fixed-seed class-conditional samples.
- `config.json`, `monitor-split.json`: settings and reproducible holdout indices.

## Feedback without using the official validation split

The README prohibits validation images for training, distillation, or
hyperparameter selection. Monitoring therefore reserves a fixed, stratified
10% **from train_split.txt**. It never opens `val_split.txt`. The teacher is fit
on the remaining training images during these runs.

Fixed-noise loss monitoring runs every 2k updates on up to 512 fit and holdout
images. FID runs every 5k updates with 5k generated images, 18 Heun steps
(35 NFEs without guidance), and labels drawn from the empirical fit-category
distribution. FID uses the clean-fid evaluation Inception, already part of the
provided evaluation dependencies; it may download its metric weights on first
use. Generated/reference PNGs are temporary and removed after scoring.

These small-holdout FIDs are for comparisons within a run, not official FastGen
scores. They have sampling bias and noise; compare identical counts, steps, and
seeds. Neither their scale nor class proportions match the official balanced
3,020-sample evaluation exactly. Generation and FID are costly and run in FP32;
adjust `--fid-every`, or set it to 0 to disable. Increasing fit loss versus holdout
loss can inform dropout/duration choices. `--early-stop-patience N` optionally
stops after N loss checks without improvement; it is off by default.

After selecting settings/duration, train a fresh teacher on the full permitted
training split if desired:

```bash
python train_teacher.py --holdout-fraction 0 --loss-every 0 --fid-every 0 \
  --steps CHOSEN_STEPS --outdir checkpoints/teacher-full
```

## Guidance and sampling

Class dropout zeros the one-hot vector during training. At inference,
`teacher(x, sigma, None)` is the null-condition denoiser. CFG is
`D_null + guidance * (D_class - D_null)`; `guidance=1` costs one backbone call,
other weights cost two. CFG requires a teacher trained with label dropout.

Compare guidance values on the best checkpoint using paired seeds and the
empirical training-category distribution:

```bash
python sweep_teacher_guidance.py --run checkpoints/teacher \
  --guidance-values 1 1.5 2 3
```

It writes `guidance-sweep.json`; use its selected value when producing teacher
targets. Use `--checkpoint .../best-loss.ckpt` if FID was disabled during training.
The sweep needs the original run's monitoring split and dataset.

Generate PNGs for a chosen category (ID mapping is the provided JSON):

```bash
python sample_teacher.py --checkpoint checkpoints/teacher/best-fid.ckpt \
  --category 0 --guidance 2 --steps 40 --outdir results/teacher
```

40 Heun steps cost 79 NFEs at guidance 1 and 158 NFEs at guidance 2. The teacher
has no student NFE restriction. Guidance is not a second pass students may add
for free: later distillation must absorb it into the student's single conditional
prediction or count the extra backbone calls against the student budget.

## Distillation interface

```python
import torch
from teacher import EDMTeacher

teacher = EDMTeacher.load_checkpoint("checkpoints/teacher/best-fid.ckpt", device="cuda")
labels = torch.tensor([0, 1, 2, 3], device="cuda")
z = torch.randn(4, 3, 64, 64, device="cuda")
targets = teacher.sample(
    z.shape, device="cuda", category=labels, latents=z,
    num_steps=40, guidance=2.0, churn=0, return_trajectory=True,
)
# targets['samples']: unclipped terminal clean-image target.
# targets['states']: [steps+1, batch, 3, 64, 64], float64 ODE states.
# targets['sigmas']: matching descending sigma levels, including terminal zero.
# targets['nfe']: actual per-image backbone evaluation count, including CFG.

# A local deterministic transition for consistency/progressive distillation:
x_next = teacher.ode_step(
    targets['states'][0], targets['sigmas'][0], targets['sigmas'][1],
    labels, guidance=2.0,
)
```

`latents` means unit Gaussian noise; the sampler scales it by sigma_max. ODE
inputs are already sigma-scaled. All target generation should use the frozen
EMA teacher in eval mode. Churn defaults to zero for deterministic targets.
Intermediate states are unclipped; ordinary display sampling clamps to [-1,1].
Trajectory storage scales with batch size and solver steps, so prefer
`ode_step` when only adjacent targets are needed. A trained teacher target
does not by itself make an undistilled 1-step or 4-step student work; their
objectives, training, and evaluation integration are the next implementation.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover preconditioning, gradients, EMA updates, null labels/CFG arithmetic,
actual backbone NFE counts, deterministic trajectories, checkpoint reconstruction,
parameter limits, disjoint train holdout, and training/resume using synthetic
RGB files with no validation manifest. FID selection plumbing is tested with
a mocked score; a real dataset FID and CUDA memory measurement require a
training run. Existing course files remain unchanged.
