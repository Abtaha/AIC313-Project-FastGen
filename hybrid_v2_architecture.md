# Hybrid-v2 parameter allocation

Hybrid-v2 uses **99,595,363 total trainable parameters**, including every global
conditioning module, residual-block FiLM projection, normalization, and RGB output.
The remaining budget is **404,637 parameters**. RoPE adds no learned parameters.

| Component | Exact parameters |
|---|---:|
| Eight Transformer blocks, including adaLN modulation | 59,059,200 |
| Transformer output normalization | 1,280 |
| Spatial convolutions, residual blocks, FiLM, and RGB output | 39,288,803 |
| Global category, timestep, and interval conditioning | 1,246,080 |
| **Total** | **99,595,363** |

Transformer-block share falls from **75.1565%** (73,824,000 / 98,227,043) to
**59.2991%**. Everything outside those blocks totals **40,536,163 parameters**.

## Final layout

- 64x64: RGB stem to 160 channels, **2 encoder ResBlocks**.
- Downsample to 32x32: 320 channels, **3 encoder ResBlocks**.
- Downsample to 16x16: 640 channels, **1 pre-Transformer ResBlock**.
- Flatten to 256 tokens; **8 adaLN-Zero Transformer blocks**, width 640,
  8 heads, head dimension 80, MLP ratio 4.
- Reshape to 16x16; **1 post-Transformer ResBlock** at 640 channels.
- Upsample to 32x32, concatenate encoder skip; **3 decoder ResBlocks**.
- Upsample to 64x64, concatenate encoder skip; **3 decoder ResBlocks**.
- GroupNorm, SiLU, 3x3 convolution to RGB.

Q and K use 2D rotary position embeddings; V is unchanged. Each head allocates
40 dimensions to x (20 adjacent pairs) and 40 to y (20 adjacent pairs). Token
coordinates follow the convolutional feature map's row-major flattening. The
rotary frequency base is 10,000, with fixed buffers rebuilt from configuration.
There is no learned absolute position parameter in Hybrid-v2.

The fully preferred layout (three encoder blocks at both resolutions, including
both 640-channel bottleneck blocks) totals **100,262,243**, just **262,243** over
the hard cap. Removing its extra 160-channel encoder block saves **666,880**, giving
**99,595,363**. This minimal block-count adjustment keeps both expensive bottleneck
blocks and all six high-resolution decoder blocks. Width and Transformer depth
remain 640 and 8. Because the preferred layout is only marginally over budget,
removing an entire 8,197,120-parameter bottleneck block is unnecessary.

## Detailed disjoint counts

| Module | Parameters |
|---|---:|
| Category embedding | 96,640 |
| Timestep embedding | 574,720 |
| Interval embedding | 574,720 |
| RGB stem | 4,480 |
| 64x64 encoder | 1,333,760 |
| 64 to 32 downsample | 461,120 |
| 32x32 encoder | 6,766,080 |
| 32 to 16 downsample | 1,843,840 |
| Pre-Transformer bottleneck | 8,197,120 |
| Transformer blocks | 59,059,200 |
| Transformer output norm | 1,280 |
| Post-Transformer bottleneck | 8,197,120 |
| 16 to 32 upsample convolution | 1,843,520 |
| 32x32 decoder | 7,893,440 |
| 32 to 64 upsample convolution | 460,960 |
| 64x64 decoder | 2,282,720 |
| Output norm | 320 |
| RGB output | 4,323 |
| **Total** | **99,595,363** |

## Controlled training comparison

```sh
python train.py --mode few_nfe --backbone hybrid_v2 --device cuda \
  --epochs 100 --batch_size 32 \
  --checkpoint_dir checkpoints/hybrid_v2 --output_dir runs/hybrid_v2
```

Start a fresh run. Existing Hybrid weights are a different architecture. New
checkpoints save `backbone=hybrid_v2` and reconstruct the correct architecture
through the existing evaluator and resume APIs. Both the backbone and the model
wrapper enforce the 100M total parameter limit, including custom configurations.

The FM loss, noise-to-data convention, sampling direction, four-step Euler solver,
optimizer, learning rate, uniform FM timestep sampling, timestep embedding and its
`*1000` scale, class conditioning, MeanFlow implementation, dataset preprocessing,
and FID implementation are unchanged. The original Hybrid configuration, tensor
names, and architecture remain supported.

## Validation

Tests check the exact full-size parameter budget and layout, absence of absolute
position parameters, rotary axis allocation and norm preservation, relative
position dot products, Q/K-only rotation, finite gradients, MeanFlow JVP
compatibility, checkpoint reconstruction, four sampler evaluations, and exact
training continuation in both NFE modes. Training/resume tests use reduced widths
for fast verification; they do not replace the full-size budget calculation.

Executed validation: **17 tests passed** (5 Hybrid-v2 tests, 5 existing Hybrid
checks, 7 existing MeanFlow checks). The full-size model also completed a real
MPS FP32 forward/backward pass with output shape `[1,3,64,64]` and all gradients
finite. The original Hybrid model was compared against the Git version with
identical seeded weights and a nonzero output head: tensor names, weights, and
forward outputs were exactly identical. No training run or FID evaluation of
Hybrid-v2 has been performed yet.
