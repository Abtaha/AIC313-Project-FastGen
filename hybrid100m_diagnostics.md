# Correction: flow-convention audit (2026-10-04)

The original diagnostic output below is retained as a historical record. Its
real-data endpoint interpretation is invalid: it used a data-to-noise path and
`noise - images`, whereas this checkpoint was trained on a noise-to-data path
with `images - noise`. Thus the earlier negative cosine and large MSE were
computed against the wrong training convention. The earlier `t=1` noise-endpoint
and smoothness conclusions should not be used to choose architecture changes.
The derivative was measured along that mismatched interpolation path and does
not establish a defect along the trained path.

The actual checkpoint metadata is `mode=few_nfe`, `objective=flow_matching`.
Its saved epoch mean FM loss is `2259.9997947216034 / 17056 = 0.132505`.
The wrapper's `forward` directly returns the backbone output, without changing
time or velocity sign. Training, sampling, and `evaluate_solver_sweep.py` agree.
There is no `sample.py` in this repository; the integration loop is in `model.py`.

The eight requested convention details, from `train.py:392-414` and
`model.py:174-178,215-223`, are:

1. Training start endpoint (mathematical x0): `noise = torch.randn_like(images)`.
2. Training end endpoint (mathematical x1): real normalized `images`.
3. Interpolation: `(1 - mix) * noise + mix * images`.
4. Target velocity: `images - noise`.
5. Sampling start distribution: `torch.randn(...)` (Gaussian noise).
6. Sampling start time: `0.0`.
7. Sampling end time: `1.0`.
8. Update: `x = x + (end - start) * self.backbone(x, t, category=category)`.
   Default nodes: `0, .25, .5, .75, 1`; four evaluations, then clamp to [-1,1].

Exact relevant training code:

```python
images = images.to(device, non_blocking=True)
noise = torch.randn_like(images)
t = torch.rand(images.shape[0], device=device)
mix = t[:, None, None, None]
interpolated = (1 - mix) * noise + mix * images
prediction = model(interpolated, t, category=categories)
loss = F.mse_loss(prediction.float(), images - noise)
```

Exact sampling loop:

```python
x, category = self._noise(shape, device, category, generator)
schedule = (0.0, *self._validate_knots(self.knots if knots is None else knots), 1.0)
for start, end in zip(schedule[:-1], schedule[1:]):
    t = torch.full((shape[0],), start, device=x.device, dtype=x.dtype)
    x = x + (end - start) * self.backbone(x, t, category=category)
return x.clamp(-1, 1)
```

Corrected real-data diagnostic: 32 validation images (16 batches of 2), seed
1234, MPS FP32, checkpoint `checkpoints/few_nfe_epoch_0090.ckpt`. This is a small
validation probe, not a full validation-loss or FID evaluation. Each image/noise
pair is fixed across times. In this table, labels follow the pasted request:
**x0=real, x1=noise**, so **x0-x1 is the actual FM training target**.

| t | cos(x1-x0) | MSE(x1-x0) | cos(x0-x1) | MSE(x0-x1) | Training relative RMSE |
|---|---:|---:|---:|---:|---:|
| 0.00 | -0.85180 | 4.818482 | +0.85180 | 0.437523 | 0.53082 |
| 0.01 | -0.87460 | 4.980175 | +0.87460 | 0.372953 | 0.49009 |
| 0.10 | -0.94027 | 5.712746 | +0.94027 | 0.174726 | 0.33545 |
| 0.50 | -0.96728 | 5.935324 | +0.96728 | 0.094733 | 0.24700 |
| 0.90 | -0.95435 | 5.836368 | +0.95435 | 0.127402 | 0.28644 |
| 0.99 | -0.85231 | 5.052194 | +0.85231 | 0.378271 | 0.49357 |
| 1.00 | -0.57027 | 3.190768 | +0.57027 | 1.008599 | 0.80595 |

These results support Case A: the diagnostic convention was backwards. They do
not show a sampler sign mismatch or explain the poor FID by themselves. Error
rises at both endpoints. At t=1 (pure data), the independent noise in the paired
FM target is unobservable, so nonzero target error there is expected even for
a correct conditional velocity field. The zero-velocity baseline and cosine
checks are screening heuristics, not proof of model quality. No architecture,
trainer, or sampler changes were made.

The updated diagnostic reads objective/config from checkpoint metadata, uses
the matching interpolation, reports both signs, and checks absolute error before
making any smoothness inference. The optional FM loss-vs-time and synthetic path
checks use the same interpolation helper. MeanFlow uses its separate data-to-noise
convention and is checked at r=t (interval=0), not against its weighted finite
interval objective.

Reproduce:

```sh
.venv/bin/python inspect_model.py \
  --checkpoint checkpoints/few_nfe_epoch_0090.ckpt \
  --device mps --batch-size 2 --num-batches 16 --num-workers 0 \
  --real-data-only --seed 1234
```

Validation: 3 new diagnostic regression tests and 7 existing MeanFlow tests
passed. The regression tests cover interpolation/target conventions, exact and
sign-reversed analytic velocities, and rejection of a smoothness conclusion when
absolute error is already large.

---

# Original diagnostic output (superseded interpretation)

Using device: mps

================================================================================
CHECKPOINT LOADING
================================================================================

Loaded checkpoint file:
checkpoints/few_nfe_epoch_0090.ckpt

Extracted 202 tensors/entries from checkpoint.

Inferred model config:
width : 640
depth : 10
heads : 8
mlp_ratio : 4.0
dropout : 0.0
base_channels : 160
backbone : hybrid

✓ Checkpoint loaded successfully.

================================================================================
PARAMETER BREAKDOWN
================================================================================

TOTAL: 98,227,043 parameters = 98.23M

category embedding 96,640 0.10M 0.10%
time embedding 574,720 0.57M 0.59%
interval embedding 574,720 0.57M 0.59%
stem 4,480 0.00M 0.00%
encoder 64 1,333,760 1.33M 1.36%
down 64->32 461,120 0.46M 0.47%
encoder 32 4,510,720 4.51M 4.59%
down 32->16 1,843,840 1.84M 1.88%
bottleneck conv 8,197,120 8.20M 8.35%
transformer 73,824,000 73.82M 75.16%
transformer norm 1,280 0.00M 0.00%
up32 conv 1,843,520 1.84M 1.88%
decoder 32 3,382,720 3.38M 3.44%
up64 conv 460,960 0.46M 0.47%
decoder 64 948,960 0.95M 0.97%
output norm 320 0.00M 0.00%
output 4,323 0.00M 0.00%

================================================================================
WEIGHT HEALTH
================================================================================
Overall exact-zero fraction: 0.00000000
NaN parameters: none
Inf parameters: none

================================================================================
CONDITIONING MAGNITUDES
================================================================================

Mean L2 norm per sample:
category mean= 1.03430 std= 0.17573 min= 0.72042 max= 1.18247
time mean= 14.09514 std= 1.68586 min= 10.58088 max= 15.47756
interval(t-r=0) mean= 3.70424 std= 0.00000 min= 3.70424 max= 3.70424
total condition mean= 16.54732 std= 1.57963 min= 13.21541 max= 17.90756

RMS activation:
category 0.041397
time 0.560636
interval(t-r=0) 0.146423
total condition 0.656693

Ratios:
category / time = 0.073380
category / interval = 0.279219

================================================================================
adaLN-ZERO TRANSFORMER GATES
================================================================================

Block 00:
attention gate |g1| mean = 0.65363246
attention gate RMS = 0.82935482
MLP gate |g2| mean = 0.78943253
MLP gate RMS = 0.95582789
shift1 RMS = 0.49713892
scale1 RMS = 1.01018643
shift2 RMS = 0.45787519
scale2 RMS = 1.21549058

Block 01:
attention gate |g1| mean = 0.51884156
attention gate RMS = 0.70895660
MLP gate |g2| mean = 0.94642752
MLP gate RMS = 1.12040865
shift1 RMS = 0.34630895
scale1 RMS = 0.86917263
shift2 RMS = 0.43119675
scale2 RMS = 1.24631023

Block 02:
attention gate |g1| mean = 0.69680929
attention gate RMS = 0.92741936
MLP gate |g2| mean = 1.17916548
MLP gate RMS = 1.33592045
shift1 RMS = 0.33662641
scale1 RMS = 0.87596077
shift2 RMS = 0.43393266
scale2 RMS = 1.26926601

Block 03:
attention gate |g1| mean = 0.85704672
attention gate RMS = 1.10263264
MLP gate |g2| mean = 1.28653514
MLP gate RMS = 1.48305523
shift1 RMS = 0.29451519
scale1 RMS = 0.77433419
shift2 RMS = 0.40251157
scale2 RMS = 1.36993825

Block 04:
attention gate |g1| mean = 1.16611159
attention gate RMS = 1.41825974
MLP gate |g2| mean = 1.52946615
MLP gate RMS = 1.79937589
shift1 RMS = 0.23522410
scale1 RMS = 0.70363009
shift2 RMS = 0.43326366
scale2 RMS = 1.46785176

Block 05:
attention gate |g1| mean = 0.79729271
attention gate RMS = 1.00357592
MLP gate |g2| mean = 1.25811803
MLP gate RMS = 1.56072426
shift1 RMS = 0.24496736
scale1 RMS = 0.83922446
shift2 RMS = 0.50937730
scale2 RMS = 1.90731418

Block 06:
attention gate |g1| mean = 0.69231808
attention gate RMS = 0.87530428
MLP gate |g2| mean = 1.38222766
MLP gate RMS = 1.75615573
shift1 RMS = 0.20276612
scale1 RMS = 0.81481099
shift2 RMS = 0.53616911
scale2 RMS = 1.96597970

Block 07:
attention gate |g1| mean = 0.17490430
attention gate RMS = 0.25498685
MLP gate |g2| mean = 1.02892208
MLP gate RMS = 1.35388649
shift1 RMS = 0.22821973
scale1 RMS = 0.85848999
shift2 RMS = 0.48413223
scale2 RMS = 1.96714151

Block 08:
attention gate |g1| mean = 0.24005885
attention gate RMS = 0.34638241
MLP gate |g2| mean = 0.08944620
MLP gate RMS = 0.15881607
shift1 RMS = 0.23010466
scale1 RMS = 1.08146238
shift2 RMS = 0.27461684
scale2 RMS = 0.93161803

Block 09:
attention gate |g1| mean = 0.03277105
attention gate RMS = 0.09241029
MLP gate |g2| mean = 0.01351795
MLP gate RMS = 0.09077331
shift1 RMS = 0.74000567
scale1 RMS = 0.66756856
shift2 RMS = 0.60145199
scale2 RMS = 0.96645248

---

AVERAGE |attention gate| = 0.58297866
AVERAGE |MLP gate| = 0.95032588

================================================================================
TRANSFORMER MODULATION PARAMETER NORMS
================================================================================
Block 00: W RMS=0.01228619 W norm=19.26073 B RMS=0.00957858
Block 01: W RMS=0.01215831 W norm=19.06027 B RMS=0.00964345
Block 02: W RMS=0.01267804 W norm=19.87503 B RMS=0.01072920
Block 03: W RMS=0.01329091 W norm=20.83580 B RMS=0.01139220
Block 04: W RMS=0.01447720 W norm=22.69552 B RMS=0.01319533
Block 05: W RMS=0.01384326 W norm=21.70170 B RMS=0.01216192
Block 06: W RMS=0.01345377 W norm=21.09112 B RMS=0.01245115
Block 07: W RMS=0.01094722 W norm=17.16167 B RMS=0.01116251
Block 08: W RMS=0.00794615 W norm=12.45697 B RMS=0.00835864
Block 09: W RMS=0.00786277 W norm=12.32625 B RMS=0.01158985

================================================================================
VELOCITY OUTPUT STATISTICS
================================================================================
mean: 0.01012484
std: 1.08604896
abs mean: 0.86507702
RMS: 1.08609056
min: -4.99868536
max: 4.94017935

================================================================================
CATEGORY CONDITIONING SENSITIVITY
================================================================================
Output RMS : 1.10257936
Category-change RMS diff : 0.05299968
Relative category effect : 0.04806881
Output cosine similarity : 0.99885833

================================================================================
TIMESTEP SENSITIVITY
================================================================================
t=0.00: RMS=1.03671670, abs mean=0.82579058
t=0.10: RMS=1.05019999, abs mean=0.83691531
t=0.25: RMS=1.09563696, abs mean=0.87319279
t=0.50: RMS=1.09979689, abs mean=0.87693816
t=0.75: RMS=1.10267818, abs mean=0.87773126
t=0.90: RMS=1.09109652, abs mean=0.86718386
t=1.00: RMS=0.72007138, abs mean=0.56909752

Adjacent timestep differences:
0.00 -> 0.10: RMS difference=0.28945619, relative=0.279205
0.10 -> 0.25: RMS difference=0.23128459, relative=0.220229
0.25 -> 0.50: RMS difference=0.54951483, relative=0.501548
0.50 -> 0.75: RMS difference=0.24549596, relative=0.223219
0.75 -> 0.90: RMS difference=0.17075615, relative=0.154856
0.90 -> 1.00: RMS difference=0.43518177, relative=0.398848

================================================================================
INTERVAL CONDITIONING SENSITIVITY
================================================================================
interval=0.00: velocity RMS=1.11038291, diff from 0=0.00000000, relative=0.000000
interval=0.05: velocity RMS=1.10619104, diff from 0=0.04215195, relative=0.038105
interval=0.10: velocity RMS=1.09839225, diff from 0=0.05593012, relative=0.050920
interval=0.25: velocity RMS=1.06859815, diff from 0=0.09033325, relative=0.084534
interval=0.50: velocity RMS=1.07738721, diff from 0=0.08658618, relative=0.080367

================================================================================
FLOW PATH SMOOTHNESS
================================================================================

## t |v| RMS |Δv|/Δt

0.00 0.83098 nan
0.05 0.76243 3.70547
0.10 0.75135 2.09395
0.15 0.76739 2.13836
0.20 0.78544 2.00491
0.25 0.81452 2.05560
0.30 0.85141 2.14629
0.35 0.89545 2.21775
0.40 0.94622 2.44053
0.45 0.99482 2.41621
0.50 1.02588 3.76538
0.55 1.00087 4.90018
0.60 0.98839 4.51320
0.65 1.00089 3.22428
0.70 1.01356 2.47236
0.75 1.02152 2.23294
0.80 1.03114 1.89349
0.85 1.04119 1.82115
0.90 1.04863 1.79654
0.95 1.07148 1.93738
1.00 0.71801 33.27250

Velocity temporal derivative:
mean : 3.11499
max : 33.27250
max near t = 1.000

Preparing real validation data...
Dataset already exists: data/pokemon-generation-one-22k/PokemonData
Dataset already exists: data/pokemon-generation-one-22k/PokemonData

====================================================================================================
REAL-DATA ENDPOINT DIAGNOSTIC
====================================================================================================

Validation batch:
shape : (8, 3, 64, 64)
x0 min : -1.000000
x0 max : 1.000000
x0 mean : 0.565507
x0 std : 0.567066
categories : [0, 0, 0, 0, 0, 0, 0, 0]

## t pred_RMS target_RMS MSE rel_RMSE cosine |dv/dt|

0.9000 1.05302 1.23480 4.266970 1.67367 -0.63044 nan
0.9200 1.06323 1.23480 4.274136 1.67514 -0.61914 1.7547
0.9400 1.07064 1.23480 4.276903 1.67573 -0.61015 1.8228
0.9600 1.07781 1.23480 4.278667 1.67613 -0.60120 1.9436
0.9700 1.07326 1.23480 4.247031 1.66994 -0.59558 2.3887
0.9800 1.06431 1.23480 4.195718 1.65983 -0.58841 3.0872
0.9850 1.05010 1.23480 4.132716 1.64733 -0.58365 4.5904
0.9900 1.02636 1.23480 4.031857 1.62709 -0.57676 6.8446
0.9925 1.00369 1.23480 3.939158 1.60827 -0.57086 11.4253
0.9950 0.96506 1.23480 3.787221 1.57692 -0.56169 18.1647
0.9975 0.89083 1.23480 3.509972 1.51804 -0.54473 33.2872
0.9990 0.78947 1.23480 3.161485 1.44061 -0.52270 72.7939
1.0000 0.72122 1.23480 2.944153 1.39014 -0.50766 73.0110

====================================================================================================
ENDPOINT SUMMARY
====================================================================================================

Relative RMSE @ t=0.999 : 1.440606
Relative RMSE @ t=1.000 : 1.390138
Endpoint error ratio : 0.965x
Cosine @ t=0.999 : -0.522696
Cosine @ t=1.000 : -0.507657
Cosine drop : -0.015039
|dv/dt| near t=1 : 73.011009

> > > DIAGNOSIS:
> > > Prediction error remains reasonably stable, but the vector field changes extremely rapidly near t=1.
> > > This supports a FIELD-SMOOTHNESS / FEW-NFE problem.

Next experiment:

- reduce timestep embedding scale
- test time_scale = 100, 30, 10
- compare FID@4 versus FID@32

================================================================================
DIAGNOSTICS COMPLETE

Using device: mps

================================================================================
CHECKPOINT LOADING
================================================================================

Loaded checkpoint file:
  checkpoints/few_nfe_epoch_0090.ckpt

Extracted 202 tensors/entries from checkpoint.

Model config (checkpoint metadata preferred):
  width          : 640
  depth          : 10
  heads          : 8
  mlp_ratio      : 4.0
  dropout        : 0.0
  base_channels  : 160
  backbone       : hybrid

Checkpoint objective: flow_matching

✓ Checkpoint loaded successfully.

================================================================================
PARAMETER BREAKDOWN
================================================================================

TOTAL: 98,227,043 parameters = 98.23M

category embedding             96,640     0.10M    0.10%
time embedding                574,720     0.57M    0.59%
interval embedding            574,720     0.57M    0.59%
stem                            4,480     0.00M    0.00%
encoder 64                  1,333,760     1.33M    1.36%
down 64->32                   461,120     0.46M    0.47%
encoder 32                  4,510,720     4.51M    4.59%
down 32->16                 1,843,840     1.84M    1.88%
bottleneck conv             8,197,120     8.20M    8.35%
transformer                73,824,000    73.82M   75.16%
transformer norm                1,280     0.00M    0.00%
up32 conv                   1,843,520     1.84M    1.88%
decoder 32                  3,382,720     3.38M    3.44%
up64 conv                     460,960     0.46M    0.47%
decoder 64                    948,960     0.95M    0.97%
output norm                       320     0.00M    0.00%
output                          4,323     0.00M    0.00%

================================================================================
WEIGHT HEALTH
================================================================================
Overall exact-zero fraction: 0.00000000
NaN parameters: none
Inf parameters: none

================================================================================
CONDITIONING MAGNITUDES
================================================================================

Mean L2 norm per sample:
category             mean=  1.03968 std=  0.22131 min=  0.75459 max=  1.38912
time                 mean= 13.77672 std=  1.70171 min= 11.05999 max= 15.49489
interval(t-r=0)      mean=  3.70424 std=  0.00000 min=  3.70424 max=  3.70424
total condition      mean= 16.29063 std=  1.53375 min= 13.78628 max= 17.74037

RMS activation:
category             0.041904
time                 0.548196
interval(t-r=0)      0.146423
total condition      0.646436

Ratios:
category / time     = 0.075466
category / interval = 0.280673

================================================================================
adaLN-ZERO TRANSFORMER GATES
================================================================================

Block 00:
  attention gate |g1| mean = 0.65933162
  attention gate RMS       = 0.83874726
  MLP gate       |g2| mean = 0.78219020
  MLP gate       RMS       = 0.94873691
  shift1 RMS               = 0.49195561
  scale1 RMS               = 0.98972255
  shift2 RMS               = 0.46391323
  scale2 RMS               = 1.20370340

Block 01:
  attention gate |g1| mean = 0.52021599
  attention gate RMS       = 0.70980680
  MLP gate       |g2| mean = 0.93424034
  MLP gate       RMS       = 1.10824764
  shift1 RMS               = 0.34410942
  scale1 RMS               = 0.86120081
  shift2 RMS               = 0.43683293
  scale2 RMS               = 1.24058318

Block 02:
  attention gate |g1| mean = 0.69304967
  attention gate RMS       = 0.92431867
  MLP gate       |g2| mean = 1.16950119
  MLP gate       RMS       = 1.32731760
  shift1 RMS               = 0.33381221
  scale1 RMS               = 0.86410058
  shift2 RMS               = 0.44149241
  scale2 RMS               = 1.26446128

Block 03:
  attention gate |g1| mean = 0.84777755
  attention gate RMS       = 1.09484625
  MLP gate       |g2| mean = 1.27248061
  MLP gate       RMS       = 1.46996975
  shift1 RMS               = 0.29383421
  scale1 RMS               = 0.76713639
  shift2 RMS               = 0.41071865
  scale2 RMS               = 1.36914122

Block 04:
  attention gate |g1| mean = 1.14707971
  attention gate RMS       = 1.39829910
  MLP gate       |g2| mean = 1.51910853
  MLP gate       RMS       = 1.79192102
  shift1 RMS               = 0.23512426
  scale1 RMS               = 0.70124018
  shift2 RMS               = 0.44279790
  scale2 RMS               = 1.47807264

Block 05:
  attention gate |g1| mean = 0.78271943
  attention gate RMS       = 0.98598790
  MLP gate       |g2| mean = 1.21566653
  MLP gate       RMS       = 1.51763546
  shift1 RMS               = 0.24755846
  scale1 RMS               = 0.83417004
  shift2 RMS               = 0.50675666
  scale2 RMS               = 1.90268958

Block 06:
  attention gate |g1| mean = 0.68256199
  attention gate RMS       = 0.86409211
  MLP gate       |g2| mean = 1.35089684
  MLP gate       RMS       = 1.72230065
  shift1 RMS               = 0.20482506
  scale1 RMS               = 0.81589037
  shift2 RMS               = 0.52661079
  scale2 RMS               = 1.94614947

Block 07:
  attention gate |g1| mean = 0.16892275
  attention gate RMS       = 0.24987219
  MLP gate       |g2| mean = 0.98221481
  MLP gate       RMS       = 1.31158185
  shift1 RMS               = 0.23120581
  scale1 RMS               = 0.84929961
  shift2 RMS               = 0.46990028
  scale2 RMS               = 1.91922200

Block 08:
  attention gate |g1| mean = 0.23398420
  attention gate RMS       = 0.34019613
  MLP gate       |g2| mean = 0.08634718
  MLP gate       RMS       = 0.15676768
  shift1 RMS               = 0.22841820
  scale1 RMS               = 1.07184315
  shift2 RMS               = 0.26974115
  scale2 RMS               = 0.92539829

Block 09:
  attention gate |g1| mean = 0.03274031
  attention gate RMS       = 0.09196837
  MLP gate       |g2| mean = 0.01305664
  MLP gate       RMS       = 0.09006739
  shift1 RMS               = 0.73427510
  scale1 RMS               = 0.66991162
  shift2 RMS               = 0.60016239
  scale2 RMS               = 0.96014655

--------------------------------------------------------------------------------
AVERAGE |attention gate| = 0.57683832
AVERAGE |MLP gate|       = 0.93257029

================================================================================
TRANSFORMER MODULATION PARAMETER NORMS
================================================================================
Block 00: W RMS=0.01228619  W norm=19.26073  B RMS=0.00957858
Block 01: W RMS=0.01215831  W norm=19.06027  B RMS=0.00964345
Block 02: W RMS=0.01267804  W norm=19.87503  B RMS=0.01072920
Block 03: W RMS=0.01329091  W norm=20.83580  B RMS=0.01139220
Block 04: W RMS=0.01447720  W norm=22.69552  B RMS=0.01319533
Block 05: W RMS=0.01384326  W norm=21.70170  B RMS=0.01216192
Block 06: W RMS=0.01345377  W norm=21.09112  B RMS=0.01245115
Block 07: W RMS=0.01094722  W norm=17.16167  B RMS=0.01116251
Block 08: W RMS=0.00794615  W norm=12.45697  B RMS=0.00835864
Block 09: W RMS=0.00786277  W norm=12.32625  B RMS=0.01158985

================================================================================
VELOCITY OUTPUT STATISTICS
================================================================================
mean:     0.00488717
std:      1.09738708
abs mean: 0.87407273
RMS:      1.09739232
min:      -4.78417730
max:      4.60629225

================================================================================
CATEGORY CONDITIONING SENSITIVITY
================================================================================
Output RMS                 : 1.09878182
Category-change RMS diff   : 0.05121504
Relative category effect   : 0.04661075
Output cosine similarity   : 0.99893880

================================================================================
TIMESTEP SENSITIVITY
================================================================================
t=0.00: RMS=1.04700959, abs mean=0.83455950
t=0.10: RMS=1.05611336, abs mean=0.84142774
t=0.25: RMS=1.10329974, abs mean=0.87927109
t=0.50: RMS=1.10558581, abs mean=0.88054806
t=0.75: RMS=1.10723841, abs mean=0.88116837
t=0.90: RMS=1.09611499, abs mean=0.87060452
t=1.00: RMS=0.71659464, abs mean=0.56579679

Adjacent timestep differences:
0.00 -> 0.10: RMS difference=0.29005107, relative=0.277028
0.10 -> 0.25: RMS difference=0.25730217, relative=0.243631
0.25 -> 0.50: RMS difference=0.56179827, relative=0.509198
0.50 -> 0.75: RMS difference=0.25079027, relative=0.226839
0.75 -> 0.90: RMS difference=0.16974784, relative=0.153307
0.90 -> 1.00: RMS difference=0.44340917, relative=0.404528

================================================================================
INTERVAL CONDITIONING SENSITIVITY
================================================================================
interval=0.00: velocity RMS=1.10661530, diff from 0=0.00000000, relative=0.000000
interval=0.05: velocity RMS=1.10223413, diff from 0=0.04224517, relative=0.038327
interval=0.10: velocity RMS=1.09351707, diff from 0=0.05672977, relative=0.051878
interval=0.25: velocity RMS=1.06218064, diff from 0=0.09302365, relative=0.087578
interval=0.50: velocity RMS=1.07124412, diff from 0=0.08879037, relative=0.082885

================================================================================
FLOW PATH SMOOTHNESS
================================================================================

t       |v| RMS      |Δv|/Δt
---------------------------------------------
0.00       1.03048           nan
0.05       1.02004       3.22678
0.10       1.00741       1.79600
0.15       1.00455       1.55881
0.20       1.00079       1.36647
0.25       1.00143       1.42596
0.30       1.00502       1.31029
0.35       1.01113       1.46671
0.40       1.01907       1.70897
0.45       1.02465       1.88038
0.50       1.02992       2.14551
0.55       1.02698       2.60288
0.60       1.01005       3.48671
0.65       0.97786       3.87823
0.70       0.95448       4.12706
0.75       0.94331       3.86333
0.80       0.91846       3.62642
0.85       0.88399       3.42931
0.90       0.83821       3.52197
0.95       0.84373       2.82286
1.00       0.52593      30.18091

Velocity temporal derivative:
mean : 2.99930
max  : 30.18091
max near t = 1.000

Preparing real validation data...
Dataset already exists: data/pokemon-generation-one-22k/PokemonData
Dataset already exists: data/pokemon-generation-one-22k/PokemonData

REAL-DATA FLOW-CONVENTION DIAGNOSTIC
Sign labels: x0=real, x1=noise
Objective: flow_matching
Training: xt=(1-t)*x1+t*x0; target=x0-x1; sample t=0 -> 1
Validation: shape=(8, 3, 64, 64), range=[-1.0000, 1.0000]

Using device: mps

================================================================================
CHECKPOINT LOADING
================================================================================

Loaded checkpoint file:
  checkpoints/few_nfe_epoch_0090.ckpt

Extracted 202 tensors/entries from checkpoint.

Model config (checkpoint metadata preferred):
  width          : 640
  depth          : 10
  heads          : 8
  mlp_ratio      : 4.0
  dropout        : 0.0
  base_channels  : 160
  backbone       : hybrid

Checkpoint objective: flow_matching

✓ Checkpoint loaded successfully.

Preparing real validation data...
Dataset already exists: data/pokemon-generation-one-22k/PokemonData
Dataset already exists: data/pokemon-generation-one-22k/PokemonData

REAL-DATA FLOW-CONVENTION DIAGNOSTIC
Sign labels: x0=real, x1=noise
Objective: flow_matching
Training: xt=(1-t)*x1+t*x0; target=x0-x1; sample t=0 -> 1
Validation: shape=(8, 3, 64, 64), range=[-1.0000, 1.0000]

t       pred_RMS target_RMS cos(x1-x0) MSE(x1-x0) cos(x0-x1) MSE(x0-x1) train_rel_RMSE |dv/dt|path
0.0000   1.03387    1.23445   -0.85712   4.770916   +0.85712   0.414579        0.52159         nan
0.0100   1.05123    1.23445   -0.87592   4.896130   +0.87592   0.361777        0.48724     14.5300
0.1000   1.16351    1.23445   -0.93664   5.570996   +0.93664   0.184265        0.34773      4.4894
0.5000   1.19327    1.23445   -0.96470   5.792991   +0.96470   0.102514        0.25937      0.9229
0.9000   1.18060    1.23445   -0.95243   5.700454   +0.95243   0.134890        0.29752      0.8359
0.9900   1.06072    1.23445   -0.85076   4.907690   +0.85076   0.390310        0.50609      5.2939
1.0000   0.71959    1.23445   -0.56400   3.076465   +0.56400   1.006876        0.81286     77.0655

Evaluated 128 images in 16 batches.
Zero-velocity baseline: relative RMSE=1; training MSE must be compared at matched times.

No negative-alignment or zero-baseline failure detected on this sample. Temporal variation alone does not establish the cause of poor FID; compare validation loss and sampler results before changing architecture.

================================================================================
PARAMETER BREAKDOWN
================================================================================

TOTAL: 98,227,043 parameters = 98.23M

category embedding             96,640     0.10M    0.10%
time embedding                574,720     0.57M    0.59%
interval embedding            574,720     0.57M    0.59%
stem                            4,480     0.00M    0.00%
encoder 64                  1,333,760     1.33M    1.36%
down 64->32                   461,120     0.46M    0.47%
encoder 32                  4,510,720     4.51M    4.59%
down 32->16                 1,843,840     1.84M    1.88%
bottleneck conv             8,197,120     8.20M    8.35%
transformer                73,824,000    73.82M   75.16%
transformer norm                1,280     0.00M    0.00%
up32 conv                   1,843,520     1.84M    1.88%
decoder 32                  3,382,720     3.38M    3.44%
up64 conv                     460,960     0.46M    0.47%
decoder 64                    948,960     0.95M    0.97%
output norm                       320     0.00M    0.00%
output                          4,323     0.00M    0.00%

================================================================================
WEIGHT HEALTH
================================================================================
Overall exact-zero fraction: 0.00000000
NaN parameters: none
Inf parameters: none

================================================================================
CONDITIONING MAGNITUDES
================================================================================

Mean L2 norm per sample:
category             mean=  1.18901 std=  0.11305 min=  1.05948 max=  1.44905
time                 mean= 14.21331 std=  1.07173 min= 12.39787 max= 15.68639
interval(t-r=0)      mean=  3.70424 std=  0.00000 min=  3.70424 max=  3.70424
total condition      mean= 16.71458 std=  0.89281 min= 15.23641 max= 18.00357

RMS activation:
category             0.047212
time                 0.563425
interval(t-r=0)      0.146423
total condition      0.661644

Ratios:
category / time     = 0.083655
category / interval = 0.320986

================================================================================
adaLN-ZERO TRANSFORMER GATES
================================================================================

Block 00:
  attention gate |g1| mean = 0.66705012
  attention gate RMS       = 0.84443963
  MLP gate       |g2| mean = 0.80667371
  MLP gate       RMS       = 0.97523367
  shift1 RMS               = 0.50872469
  scale1 RMS               = 1.04688895
  shift2 RMS               = 0.46931931
  scale2 RMS               = 1.24535739

Block 01:
  attention gate |g1| mean = 0.53020257
  attention gate RMS       = 0.72230393
  MLP gate       |g2| mean = 0.97100914
  MLP gate       RMS       = 1.14431000
  shift1 RMS               = 0.35390839
  scale1 RMS               = 0.88751596
  shift2 RMS               = 0.43994832
  scale2 RMS               = 1.27711344

Block 02:
  attention gate |g1| mean = 0.71562237
  attention gate RMS       = 0.94698870
  MLP gate       |g2| mean = 1.20281565
  MLP gate       RMS       = 1.36011887
  shift1 RMS               = 0.34639707
  scale1 RMS               = 0.89834297
  shift2 RMS               = 0.43832991
  scale2 RMS               = 1.29629111

Block 03:
  attention gate |g1| mean = 0.87779939
  attention gate RMS       = 1.12441027
  MLP gate       |g2| mean = 1.31634736
  MLP gate       RMS       = 1.51280236
  shift1 RMS               = 0.30349153
  scale1 RMS               = 0.79710394
  shift2 RMS               = 0.40586635
  scale2 RMS               = 1.39530766

Block 04:
  attention gate |g1| mean = 1.19596577
  attention gate RMS       = 1.44908416
  MLP gate       |g2| mean = 1.55860746
  MLP gate       RMS       = 1.83053064
  shift1 RMS               = 0.24155407
  scale1 RMS               = 0.72286648
  shift2 RMS               = 0.43895766
  scale2 RMS               = 1.49223757

Block 05:
  attention gate |g1| mean = 0.81742400
  attention gate RMS       = 1.02348006
  MLP gate       |g2| mean = 1.29795301
  MLP gate       RMS       = 1.59369850
  shift1 RMS               = 0.25098979
  scale1 RMS               = 0.85703701
  shift2 RMS               = 0.52340943
  scale2 RMS               = 1.94008279

Block 06:
  attention gate |g1| mean = 0.70961684
  attention gate RMS       = 0.89558297
  MLP gate       |g2| mean = 1.42424703
  MLP gate       RMS       = 1.79516590
  shift1 RMS               = 0.21010776
  scale1 RMS               = 0.83413655
  shift2 RMS               = 0.54897571
  scale2 RMS               = 2.00690198

Block 07:
  attention gate |g1| mean = 0.18445377
  attention gate RMS       = 0.26631069
  MLP gate       |g2| mean = 1.06930971
  MLP gate       RMS       = 1.39663315
  shift1 RMS               = 0.23348644
  scale1 RMS               = 0.87954044
  shift2 RMS               = 0.49884796
  scale2 RMS               = 2.01850748

Block 08:
  attention gate |g1| mean = 0.24826717
  attention gate RMS       = 0.35713869
  MLP gate       |g2| mean = 0.09514949
  MLP gate       RMS       = 0.16687931
  shift1 RMS               = 0.24102432
  scale1 RMS               = 1.10350025
  shift2 RMS               = 0.28049412
  scale2 RMS               = 0.95333451

Block 09:
  attention gate |g1| mean = 0.03451570
  attention gate RMS       = 0.09491117
  MLP gate       |g2| mean = 0.01406605
  MLP gate       RMS       = 0.09293406
  shift1 RMS               = 0.75659090
  scale1 RMS               = 0.68953544
  shift2 RMS               = 0.61507714
  scale2 RMS               = 0.98869246

--------------------------------------------------------------------------------
AVERAGE |attention gate| = 0.59809177
AVERAGE |MLP gate|       = 0.97561786

================================================================================
TRANSFORMER MODULATION PARAMETER NORMS
================================================================================
Block 00: W RMS=0.01228619  W norm=19.26073  B RMS=0.00957858
Block 01: W RMS=0.01215831  W norm=19.06027  B RMS=0.00964345
Block 02: W RMS=0.01267804  W norm=19.87503  B RMS=0.01072920
Block 03: W RMS=0.01329091  W norm=20.83580  B RMS=0.01139220
Block 04: W RMS=0.01447720  W norm=22.69552  B RMS=0.01319533
Block 05: W RMS=0.01384326  W norm=21.70170  B RMS=0.01216192
Block 06: W RMS=0.01345377  W norm=21.09112  B RMS=0.01245115
Block 07: W RMS=0.01094722  W norm=17.16167  B RMS=0.01116251
Block 08: W RMS=0.00794615  W norm=12.45697  B RMS=0.00835864
Block 09: W RMS=0.00786277  W norm=12.32625  B RMS=0.01158985

================================================================================
VELOCITY OUTPUT STATISTICS
================================================================================
mean:     -0.01545741
std:      1.09864306
abs mean: 0.87728930
RMS:      1.09874630
min:      -4.73735952
max:      5.25621605

================================================================================
CATEGORY CONDITIONING SENSITIVITY
================================================================================
Output RMS                 : 1.10074186
Category-change RMS diff   : 0.05318967
Relative category effect   : 0.04832165
Output cosine similarity   : 0.99885827

================================================================================
TIMESTEP SENSITIVITY
================================================================================
t=0.00: RMS=1.04082167, abs mean=0.82988673
t=0.10: RMS=1.05686116, abs mean=0.84241849
t=0.25: RMS=1.09953296, abs mean=0.87710398
t=0.50: RMS=1.10398507, abs mean=0.88011724
t=0.75: RMS=1.10856581, abs mean=0.88307571
t=0.90: RMS=1.09546375, abs mean=0.87140626
t=1.00: RMS=0.71627212, abs mean=0.56672817

Adjacent timestep differences:
0.00 -> 0.10: RMS difference=0.25072622, relative=0.240893
0.10 -> 0.25: RMS difference=0.25705367, relative=0.243224
0.25 -> 0.50: RMS difference=0.55180913, relative=0.501858
0.50 -> 0.75: RMS difference=0.24802119, relative=0.224660
0.75 -> 0.90: RMS difference=0.16816996, relative=0.151700
0.90 -> 1.00: RMS difference=0.44222361, relative=0.403686

================================================================================
INTERVAL CONDITIONING SENSITIVITY
Skipped: FM trains only interval=0; nonzero intervals are untrained.

================================================================================
FLOW PATH SMOOTHNESS
================================================================================
Synthetic path: random uniform images; this is not validation data.
Finite differences below follow the path, with x changing alongside t.

t       |v| RMS      |Δv|/Δt along path
0.00       1.03549           nan
0.05       1.01298       3.15918
0.10       1.00251       1.71590
0.15       1.00169       1.60092
0.20       0.99848       1.44112
0.25       0.99962       1.53517
0.30       1.00329       1.34734
0.35       1.00876       1.47181
0.40       1.01582       1.72679
0.45       1.02006       1.88978
0.50       1.02391       2.11688
0.55       1.01934       2.50623
0.60       0.99893       3.61281
0.65       0.96615       3.80782
0.70       0.94543       4.07911
0.75       0.93494       3.66277
0.80       0.91037       3.59578
0.85       0.87618       3.38527
0.90       0.83498       3.42394
0.95       0.84790       2.83943
1.00       0.53019      30.45923

Velocity derivative along synthetic path:
mean : 3.00397
max  : 30.45923
max near t = 1.000

================================================================================
DIAGNOSTICS COMPLETE
