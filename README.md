# FastGen midterm submission — Team 20

**Team:** 20 — brave_panda, brave_rabbit, happy_fox.

This folder contains the complete source and the two required trained student
checkpoints. It is ready to evaluate from the directory containing `model.py`.
The submission rules are at <https://generative-models.kaist.ac.kr/fastgen/>.

## Checkpoints and measured results

| Required path | Actual NFE | Parameters | Self-measured FID |
|---|---:|---:|---:|
| `checkpoints/one_nfe.ckpt` | 1 | 34,798,755 | 38.6546 |
| `checkpoints/few_nfe.ckpt` | 4 | 34,798,755 | 26.3088 |

These scores were reproduced with the unchanged root `evaluate.py`, all 3,020
validation references, and 20 generated images per class. They are our measured
results; TA evaluation determines the official scores. Seeds/batch sizes were
10789/8 for one-pass and 0/32 for four-pass. Unseeded CLI runs may differ.

## Evaluate in the TA environment

Use the supplied course environment and the unchanged `requirements.txt`.
No additional packages beyond that list are required. The fixed split files
are included; TAs may supply the matching image dataset. It is not bundled.
The provided downloader handles missing images and may need Kaggle access.

```bash
python evaluate.py --model_checkpoint ./checkpoints/one_nfe.ckpt \
  --evaluate_mode one_nfe --batch_size 8 \
  --reference_dir ./results/reference --output_dir ./results/one
python evaluate.py --model_checkpoint ./checkpoints/few_nfe.ckpt \
  --evaluate_mode few_nfe --batch_size 32 \
  --reference_dir ./results/reference --output_dir ./results/four
```

Use fresh output directories for each evaluation. `Model.load_checkpoint()`
automatically constructs both architectures; no manual configuration, teacher,
refiner, classifier, or decoder checkpoint is needed at sampling time. All
outputs have shape `[batch, 3, 64, 64]` in `[-1,1]`.

```bash
python check_submission.py             # CPU: hashes, imports, sampling, caps, NFE
python check_submission.py --device cuda
python verify.py                       # Tiny synthetic training/resume/export checks
```

## Train from scratch

Full commands, hyperparameters, provenance limitations, and resume instructions
are in `TRAINING.md`; fixed settings are in `recipes.json`.

```bash
python train_teacher.py
python train_few_nfe.py
python train_one_nfe.py
```

All teacher and student optimization uses the official training manifest.
The four-pass model is an EMA consistency student refined at lower LR and uses
the fixed noise grid `[80,5,.5,.02]`. The one-pass model initializes from that
refined student's weights and uses DMD followed by adversarial refinement.
Training-only fake-score/GAN components are excluded from the inference export.
Full training can take substantially longer than evaluation; no pretrained
teacher checkpoint is needed or bundled for retraining from random weights.
The exact historical teacher restart sequence is unavailable, as documented
in `TRAINING.md`, so an exact bitwise recreation of its weights is not claimed.

## Source and checks

- `dataset.py`, `download_dataset.py`, `evaluate.py`, `src/utils.py`,
  `requirements.txt`, and fixed data files match the official base revision.
- The base `Model` class is unchanged. Its NFE-specific subclasses instantiate
  the submitted networks and use the original checkpoint-loading interface.
- `SUBMISSION_MANIFEST.json` records checkpoint/protected-file hashes,
  parameter counts, training-framework budgets, and environment versions.
- External code, methods, and adaptations are credited in `CITATIONS.md`.
  The NVIDIA source license and notice remain under `src/edm/`.

The midterm asks for code and checkpoints; the final submission additionally
requires the specified two-page write-up. This package is for the midterm and
is not uploaded to KLMS automatically.
