# Training pipeline for the best completed models

This folder is self-contained: copy it to another machine and run the commands
here. It does not import Python code from its parent repository. It contains
the best **completed, measured** teacher and student recipes, not every
experiment. No checkpoint binaries or training images are bundled.

| Model | Measured competition FID | Inference calls | Fixed endpoint |
|---|---:|---:|---|
| V1 EDM teacher | 20.1758 | 158 | 100k updates; 40 Heun steps, CFG 2 |
| One-pass DMD + adversarial refinement | 38.6546 | 1 | Refined CD → DMD 2k → adversarial refinement to 5k |
| Refined consistency | 26.3088 | 4 | CD 9k → 2k lower-LR refinement; fixed noise grid |

These scores describe the preserved historical checkpoints, not guaranteed
scores for a retraining. `REFERENCE_RESULTS.json` records hashes, seeds, and
known provenance limits. The completed adversarial control is included; the
ongoing 5:1/LR sweeps are omitted because their results are not established.

## Environment and data

Use the provided course environment when available. The observed training
environment was Python 3.10.22, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124,
and a single 20-GB A100 vGPU. Other observed package versions are pinned in
`requirements.txt`. Setup for a fresh CUDA 12.4 environment:

```bash
cd pipeline
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python verify.py
```

The fixed train/validation manifests and class mapping are included under
`data/pokemon-generation-one-22k/`. Training downloads the Kaggle image data
when missing. Kaggle access may require your own credentials; none are bundled.
Alternatively point `--data-root` at an existing image tree containing
`PokemonData/`. Transparency is composited onto white by the supplied download
utility, using the same convention as the experiments. An existing image tree
can be modified by that normalization; use an appropriate working copy.

All entry points accept `--data-root`, `--split-dir`, and `--workdir`. Paths are
resolved from your calling directory. Defaults are local to this folder. Do
not change the supplied split manifests. Training and synthetic cache generation
never load validation images; only the explicit evaluation command does.

## 1. Train the teacher

```bash
python train_teacher.py
```

This trains the 34,798,755-parameter class-conditional EDM/SongUNet teacher from
random initialization for 100,000 updates. Batch 128, microbatch 64, BF16,
horizontal flips, dropout/label dropout .10, Adam LR `2e-4`, 100-kimg warmup,
and 500-kimg EMA half-life with the original 5% ramp-up are configured in
`recipes.json`. The exported teacher is the EMA network:

```text
runs/teacher/teacher.ckpt
runs/teacher/training-state.pt
```

**Historical limitation:** the original 100k inference checkpoint verifies the
architecture, step, 12.8M seen images, and full training-manifest fingerprint.
It does not retain its optimizer, fitting indices, or complete resume history.
The script therefore uses the historical trainer's stratified 10% internal
training holdout, consistent with the early logs. This subset choice is a
reconstruction, not verified metadata for the final teacher. Internal loss/FID
selection is disabled: the exported endpoint is fixed at 100k. Matching the
original teacher weights or exact FID cannot be promised. You may supply the
preserved original teacher to the student scripts to reproduce that lineage
without retraining this uncertain historical stage.

## 2. Train the four-pass student

```bash
python train_few_nfe.py
```

The script performs three stages and saves all required intermediate artifacts:

1. Guided consistency distillation to 2k, batch 128/microbatch 8.
2. Resume the same optimizer/student/EMA to 9k with microbatch 32. The teacher
   uses CFG 2; the objective uses 40 noise intervals, pseudo-Huber distance,
   LR `1e-4`, 500-step warmup, EMA decay .95, and balanced class sampling.
3. Initialize student and EMA from the exported 9k EMA weights, start a fresh
   optimizer, and train **2,000 additional updates** at LR `2e-5`, no warmup,
   microbatch 16. Other objective settings stay fixed.

All 17,079 official training images are available to both student stages.
The last stage's counter starts at zero; its step 2k represents 11k total
updates along the model lineage. The 9k branch point is intentional; the
historical default run continued to 10k, but those later weights are not used.

```text
runs/consistency/snapshots/student-001000.ckpt
runs/consistency/snapshots/student-009000.ckpt
runs/few-refine/snapshots/student-002000.ckpt
```

The fixed four-call levels `[80, 5, .5, .02]` are applied at export, without a
validation search. They were selected by matching synthetic training teacher
pairs in the recorded experiment. Each extra pass denoises a re-noised image;
these are four actual backbone calls, not four Heun steps.

## 3. Train the one-pass student

```bash
python train_one_nfe.py
```

This initializes from the **refined 2k EMA snapshot** saved above, before the
four-pass sampling-grid adjustment at export. The generator starts with the
stronger consistency weights; it is trained to generate in one direct pass.

The script creates/verifies a deterministic guided-teacher cache: 40 Heun
steps, CFG 2, seed 123, two fitting and one independent synthetic monitoring
pair per class (453 pairs total). These are generated images, not validation
images. It then trains:

1. DMD for 2k updates with **3:1** fake-score updates throughout. Batch
   64/microbatch 8, BF16, generator LR `5e-6`, fake LR `1e-4`, seed 789.
   The first 200 updates train only the fake-score network. Regression weight
   is .25. The GAN head is initialized and checkpointed but remains inactive.
   The historical endpoint measured **47.6906 FID**.
2. Preserve the full 2k state, then resume the same run to absolute step **5k**,
   adding 3k updates. Enable GAN loss with coefficient `.003` and reduce
   regression to `.025` after step 2000. The endpoint measured **38.6546 FID**.
   Stage lengths are fixed; validation FID does not gate the transition.

```text
runs/teacher-pairs.pt
runs/one-adversarial/stage1-step2000/training-state.pt
runs/one-adversarial/snapshots/one_nfe-005000.ckpt
```

The loss uses a frozen guided teacher, a trainable conditional fake-score
denoiser, distribution-matching gradients, and multiscale L2 paired regression.
The small conditional classifier shares the fake-score encoder/middle blocks.
It learns from all 17,079 official training images, with uniform class/image
sampling, noisy inputs, and horizontal flips. The generator receives a
non-saturating GAN gradient through frozen classifier weights. Each adversarial
critic update uses fresh detached generated images. This is a DMD2-inspired
adaptation retaining a small regression anchor, not an exact DMD2 reproduction.

There is no generator EMA: generator + fake score + another full EMA would
exceed 100M parameters. The training framework includes the GAN head and totals
**69,991,239 parameters**. The one-pass export contains only the generator and
does not call the fake-score model or GAN head at inference. The new run uses a
distinct output directory so older one-pass artifacts are not overwritten.

## Resume, inspect, and export

Use the same arguments/workdir and add `--resume` after interruption. Completed
stages are skipped; incomplete stages restore their own full state. Consistency
data iterator ordering is not guaranteed to be bitwise identical across resumes.
Fresh commands refuse to overwrite an existing run. Preview commands without
training using `--dry-run`:

```bash
python train_few_nfe.py --dry-run
python train_one_nfe.py --dry-run
python train_few_nfe.py --resume
python train_one_nfe.py --resume
python export_models.py
```

To use an existing teacher, pass `--teacher /absolute/path/teacher.ckpt` to both
student scripts and the export command. The export directory must be empty.

```text
submission/teacher.ckpt
submission/one_nfe.ckpt
submission/few_nfe.ckpt
submission/manifest.json
```

The manifest records source/export SHA-256 hashes and parameter counts. The
four-pass export changes only the sampling-grid buffer; learned weights are
unchanged. `model.py`, `teacher.py`, `distillation/model.py`, and `src/` contain
the inference definitions. No optimizer or fake-score network is needed for
student inference. See the original course submission format for required paths;
this training folder is not an automatically assembled competition submission.

## Official evaluation, after training

```bash
python evaluate_models.py --kind teacher --checkpoint submission/teacher.ckpt \
  --outdir results/teacher --seed 0 --batch-size 8
python evaluate_models.py --kind one_nfe --checkpoint submission/one_nfe.ckpt \
  --outdir results/one --seed 10789 --batch-size 8
python evaluate_models.py --kind few_nfe --checkpoint submission/few_nfe.ckpt \
  --outdir results/four --seed 0 --batch-size 32
```

This reports one pooled CleanFID: 20 generated images per class, 3,020 total,
against all 3,020 official validation images. It uses the supplied torchvision
PNG conversion and fresh temporary directories. FID metric weights may download
on first evaluation. Results go to each new output directory's `fid.json`.
Validation is for reporting, not training or hyperparameter selection.

All inference models have 34,798,755 parameters. Teacher + EMA and consistency +
EMA each have 69,597,510; one-pass generator + fake score + GAN head has
69,991,239. Training guards
enforce 100M and 20 GB allocated VRAM. Changing microbatch can change random
streams and numerical results; check memory on the actual course GPU.

Source provenance is in `SOURCE_MANIFEST.json`. NVIDIA's backbone attribution
and CC BY-NC-SA 4.0 license are retained under `src/edm/`. Methods follow
[EDM](https://arxiv.org/abs/2206.00364),
[Consistency Models](https://arxiv.org/abs/2303.01469), and
[DMD](https://arxiv.org/abs/2311.18828), and
[DMD2](https://arxiv.org/abs/2405.14867), with the adaptations documented above.
