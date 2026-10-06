# Hybrid V3

The supplied completed-run results motivate combining the deeper convolutional
hierarchy with a modest transformer component:

| Backbone | Midpoint, 16 NFE | Midpoint, 64 NFE |
| --- | ---: | ---: |
| Pure EDM U-Net | 89.18 | 76.19 |
| Hybrid V2 | 64.59 | 54.20 |
| U-Net minus V2 | 24.59 | 21.99 |

The gap persists at 64 NFE, supporting an architecture/training limitation
beyond coarse solver discretization. These two experiments do not isolate the
transformer as the sole cause: depth, widths, normalization, and capacity
allocation also differ. V3 tests the proposed combination directly.

## Architecture

| Encoder resolution | Channels | Operations |
| --- | ---: | --- |
| 64×64 | 128 | Stem, ResBlock ×2 |
| 32×32 | 256 | Strided convolution, ResBlock ×2 |
| 16×16 | 384 | Strided convolution, ResBlock, Attention, ResBlock |
| 8×8 | 512 | Strided convolution, ResBlock ×2, flatten to 64 tokens, DiT ×2, LayerNorm, reshape |
| 4×4 | 512 | Strided convolution, ResBlock ×2 |
| 4×4 bottleneck | 512 | ResBlock, Attention |

The symmetric convolutional decoder visits 4, 8, 16, 32, and 64 pixels,
concatenating each corresponding encoder skip, with two residual blocks at
every stage and attention between the 16×16 blocks. The 8×8 skip includes the
transformer output. DiT runs only on the encoder path.

DiT uses eight heads, MLP ratio four, 8×8 2D RoPE, and zero-initialized adaLN
modulation. Time, class, and optional interval embeddings condition both
convolutional and DiT blocks. The final velocity head starts at zero.
Flow matching, MeanFlow, checkpoint metadata, and NFE counting use existing
project interfaces.

## Parameter budget

| Allocation | Parameters |
| --- | ---: |
| DiT blocks | 9,452,544 |
| Transformer output LayerNorm | 1,024 |
| Spatial path including FiLM | 88,066,819 |
| Global conditioning | 865,792 |
| **Total** | **98,386,179** |

Adding DiT to the unchanged 94,179,587-parameter U-Net would total 103,633,155.
Removing its second *extra bottleneck* residual block saves 5,246,976 parameters;
the two requested 4×4 encoder residual blocks remain intact.

## Experiment criteria

Train a fresh few-NFE flow-matching checkpoint with the same training budget
and evaluation settings as V2. Compare midpoint at 16 and 64 actual NFE using
the existing solver sweep (midpoint-64 uses 32 two-evaluation steps).
The regular training FID uses the submission's four-evaluation Euler sampler,
so it does not measure this high-NFE target.

Primary success criterion: **midpoint-64 FID < 54.20**. FID in the 40s would
motivate few-step compression experiments; FID in the 30s would strengthen
the case for prioritizing BézierFlow/distillation. These are decision thresholds,
not measured V3 results. Keep the data split, reference images, sample count,
sampling seed, training duration, and checkpoint selection policy comparable.
