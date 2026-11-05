# Reproducing the submitted models

This submission is self-contained: run the commands from its root directory. It does not import Python code from its parent repository. It contains
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
and a single 20-GB A100 vGPU. `requirements.txt` is the unmodified TA-provided
dependency list; exact observed versions are recorded in
`SUBMISSION_MANIFEST.json`. Setup for a fresh CUDA 12.4 environment:

```bash
cd "midterm submission"  # or the extracted team directory
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
the packaged checkpoints are already installed at the required root-level paths.

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

## Teacher checkpoint retention

Teacher training retains EMA exports every 5,000 steps and complete resumable
states every 20,000 steps in the run's `snapshots/` directory. The latest
`teacher.ckpt` and `training-state.pt` still update every `--save-every` steps.
Use `--snapshot-every` and `--state-snapshot-every` to change retention intervals
(multiples of `--save-every`; zero disables that type).

For the existing 200k run, started before retention was added, a separate CPU
watcher preserves the first available save and subsequent scheduled saves without
restarting the GPU job:

```bash
.venv/bin/python -m training.retain_checkpoints \
  --outdir checkpoints/teacher-200k \
  --exit-status results/teacher-200k.exit-status
```

The watcher must remain running (the server uses tmux session `teacher_retention`).
Retention uses hard links to completed atomic saves; later checkpoint replacements
do not alter saved snapshots. The 200k run needs approximately 11 GB for scheduled
snapshots, plus its initial preserved save and latest files.

## November refinement experiments

`experiments/refinement.py` runs two independent GPU jobs from immutable baseline
copies. First run `--task prepare`, then run `--task teacher` and `--task student`
in separate tmux sessions. Every invocation needs `--baseline-root` (the previous
experiment repository), `--teacher` (the original teacher export), and `--workdir`
(an experiment output directory).

* Teacher: initialize from the original 100k-step EMA, fresh Adam at 2e-5, and
  train to absolute step 110k. Batch 128, microbatch 32, BF16, unchanged natural
  image sampling and 10% internal holdout. Save EMA every 1k and full state every
  5k; measure denoising loss every 1k. Compare the original and 101k/105k/110k
  exports at guidance 1.5, 2.0, 2.5 on the training-only reference.
* Student: fork the exact 5k generator, fake-score, GAN-head and optimizer state.
  Use 5 fake updates per loop and freeze the generator through absolute step
  5100. Both branches then receive 1,000 generator updates. `fixed5` keeps the
  GAN-phase regression weight at .025; `decay5` linearly reduces it from .025 at
  5100 to zero at 5600. GAN weight, learning rates, batches and initial RNG state
  are matched. Fresh-sample fake-score diagnostics are logged every 250 steps.
* Monitoring: a deterministic 10 images/class (1,510 total) from train_split.txt,
  excluded from the new real-image GAN updates. Generate 20/class (3,020) using
  the official image conversion and CleanFID functions. Parents may have seen
  these reference images during earlier training: this is a training-only
  monitoring metric, not an untouched holdout or the official validation FID.
  No official validation images are loaded by this experiment runner.

The student runner executes `decay5` first, then its matched `fixed5` control,
retaining samples, diagnostics, snapshot scores and wall-clock timings. It does
not replace the submitted model checkpoints. Student branches deliberately use
the existing frozen teacher rather than the simultaneously refined teacher.

For the next four-pass refinement, run the same experiment entry point with
`--task few`. It initializes live and EMA student weights from the preserved
best four-pass export with a fresh Adam optimizer, trains 2,000 updates at 5e-6,
and saves every 250 steps. Teacher, loss, EMA decay, balanced sampling and the
fixed `[80, 5, .5, .02]` sampling schedule remain the same. The baseline and all
eight snapshots are evaluated against the same training-only monitoring set;
results are written to `few-comparison.json`. The submitted checkpoints are
preserved separately.

`--task one` runs two further one-pass experiments using the completed `fixed5`
branch as a matched control. `gan-g01` changes only the generator adversarial
coefficient from .003 to .01; the critic coefficient stays .003. `cache16`
changes only the paired target cache from two to 16 fitting examples per class
(plus one synthetic monitoring example). Cache generation uses the original
40-step, guidance-2 teacher. Both retain regression .025 and restore the same
5k generator/fake-score/GAN/optimizer parent state, with the same 100 score-only
warmup steps and up to 1,000 generator updates. Evaluate at 5250, 5500, 5750,
6000 and 6100 using the existing training-only reference. From step 5500 onward,
a branch stops if its FID is over three points worse than its matched control.
Outputs, plan, per-arm status and comparisons are under `one-followups/`.

## Fresh teacher with an untouched holdout

`data/holdout-v1/` is a separate experimental split of the permitted training
images: 15,569 fitting images and 1,510 holdout images (10 per category), selected
with seed 20261104. The provided split files are unchanged. `partition.json`
records hashes, counts, source manifest and the sampling seed. The split can be
recreated with `python -m training.holdout --outdir NEW_DIRECTORY`.

Start the fresh V1 teacher with:

```bash
python -m experiments.train_holdout_teacher
```

Use `--resume` after interruption, or `--dry-run` to print the command. The fixed
budget is 100k updates, batch 128, microbatch 32, BF16, LR 2e-4, and the original
V1 architecture/objective/EMA. It initializes randomly. Denoising losses use
fixed noise every 2k steps. EMA FID on the unseen, class-balanced holdout uses
1,510 generated images, 18 Heun steps and guidance 2 every 10k updates. These
holdout FIDs are distinct from official-reference FID. Snapshots are retained
and `best-fid.ckpt` is selected using this holdout, not the official validation
set. Defaults write to `runs/holdout-v1/teacher/`.

For subsequent consistency/student/DMD/GAN training, pass the same
`--split-dir data/holdout-v1` and use teacher/student checkpoints from this fresh
chain. Checkpoints carry the partition metadata. Training rejects mixing a
clean checkpoint with the original full split, altered manifests, or old
initialization weights without matching clean-partition provenance. The
consistency and GAN loaders read only the fitting manifest. Synthetic teacher
pairs inherit the frozen clean teacher's provenance through its checkpoint hash.
Legacy experiments and submitted checkpoints remain separate.

## Controlled student follow-ups

`python -m experiments.student_followups --baseline-root PREVIOUS_REPO --teacher ORIGINAL_TEACHER`
waits for the overnight queue, then alternates four-pass and one-pass pilots
alongside the independent clean-holdout teacher. Outputs, source hashes, recipe,
status and best full training states are in `results/nov03/student-followups/`.

The one-pass arms fork the original 5k full state, reuse the 16-pairs/class cache
and fixed5 recipe, and change only the active GAN-phase regression coefficient
from .025 to .05 or .10. Each gets the same 100 score-only warmup loops and at
most 650 generator updates, evaluated at 5250/5500/5750. From 5500, an arm stops
if its FID exceeds the matched cache16 control by more than two points. Explicit
DMD forks may change this coefficient; normal resumes must match exactly.

The four-pass arms start from the same tuned submission student and fresh Adam
as the completed `few5e6` control. At LR 5e-6, one changes training intervals
40 to 80; the other changes EMA decay .95 to .995. Each trains at most 1,000
updates with checks every 250; from 500, stop if FID exceeds the matched control
by 1.5 points. The inference schedule remains [80, 5, .5, .02], exactly four
network calls. These are hypotheses about target discretization and checkpoint
smoothing, not established improvements. Selection uses the legacy training-only
monitor, not official validation, and does not make a clean-holdout claim.

## Four-pass trajectory pilot

`python -m experiments.student_followups --task trajectory --baseline-root PREVIOUS_REPO --teacher ORIGINAL_TEACHER`
queues two matched arms after the existing student queue. Both start from the
same submitted four-pass checkpoint with fresh Adam, LR 5e-6, EMA .95, 80 training
intervals, the same seed and at most 1,000 updates. Outputs and best full states
are in `results/nov03/trajectory-four/`.

Half the microbatches retain ordinary consistency training. On the other half,
uniformly choose one of the four inference sigmas [80, 5, .5, .02]. The
`schedule-real` arm corrupts a training image at that sigma. The
`schedule-student` arm runs the online student's preceding inference steps in
FP32 without gradients, including the actual re-noising variance and no
intermediate clipping, to obtain the selected step's input. Only that step is
differentiated. The target is one guided FP32 Heun step to the nearest strictly
lower point on the 80-interval grid, followed by the stopped EMA consistency
network. Stage selection is uniform per microbatch. Prefix weights use eval
mode, restored afterward; these students have zero dropout.

This adapts backward simulation from [DMD2 section 4.5](https://arxiv.org/abs/2405.14867)
to our local consistency loss. It retains a regression-style target and does
not implement full multi-step DMD2's distribution matching/GAN objective. Whether
the frozen teacher gives useful targets on these generated states is an
experimental question. The schedule-real control isolates the input-distribution
change from the changed sigma sampling. Evaluate at 250/500/750/1000; from 500,
stop if FID exceeds the matched control by 1.5. If schedule-real stops early,
the remaining student arm uses the completed grid80 control as a safety bound.
All selection uses the legacy training-only monitor, not official validation.
Normal training defaults, inference architecture and four-call sampler are
unchanged; nonzero trajectory settings are part of strict resume identity.

## Overnight one-pass queue

`python -m experiments.overnight_one --baseline-root PREVIOUS_REPO --teacher ORIGINAL_TEACHER`
waits for the current `cache16` pilot, preserving its best evaluated full state
while the live checkpoint is atomically replaced. If its endpoint is its best
and improves the starting monitoring FID by at least one point, it gets at most
500 extra generator updates, checked every 250 and stopped on deterioration.

Next, the queue generates a 64-pairs/class teacher cache (9,664 fitting pairs
plus 151 synthetic monitoring pairs). It forks the same original 5k full state
and uses the same recipe/control as the cache16 experiment: 100 score-only
warmup loops, up to 1,000 generator updates, fixed regression .025, five fake
updates, and unchanged teacher/GAN weights. FID is checked at 5250, 5500, 5750,
6000 and 6100; from 5500 onward, an arm stops if it exceeds its matched control
by three points. Only a winning cache64 endpoint receives at most 500 further
updates. Best inference checkpoints and matching full optimizer/fake-score/GAN
states are preserved. Outputs are in `results/nov03/overnight-one/`.

This queue uses the legacy experiment's training-only reference, which its
parent weights may have seen. It is independent of the fresh, untouched
holdout-v1 teacher, and does not load the official validation reference.
