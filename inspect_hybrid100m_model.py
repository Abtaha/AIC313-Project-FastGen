"""
inspect_model.py

Diagnostics for HybridFlowNet checkpoints.

Checks:
1. Checkpoint loading sanity
2. Parameter count / distribution
3. Conditioning magnitudes
4. adaLN-Zero transformer gate magnitudes
5. Transformer modulation weight norms
6. Velocity output statistics
7. Category-conditioning sensitivity
8. Timestep sensitivity
9. Interval-conditioning sensitivity
10. Optional flow-matching validation loss vs timestep

Example:
    uv run inspect_model.py \
        --checkpoint checkpoints/few_nfe_epoch_0090.ckpt

CUDA:
    uv run inspect_model.py \
        --checkpoint checkpoints/few_nfe_epoch_0090.ckpt \
        --device cuda

Mac:
    uv run inspect_model.py \
        --checkpoint checkpoints/few_nfe_epoch_0090.ckpt \
        --device mps
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
import torch.nn.functional as F

# =========================================================
# CHANGE THIS IMPORT IF YOUR MODEL FILE HAS ANOTHER NAME
# =========================================================

from src.models.hybrid import HybridFlowNet, HybridConfig
from dataset import PokemonDataModule


# =========================================================
# CHECKPOINT LOADING
# =========================================================


def extract_state_dict(checkpoint):
    """
    Extract HybridFlowNet weights from a checkpoint.

    Your checkpoint currently stores the network as:

        backbone.position
        backbone.category.weight
        backbone.transformer.0.qkv.weight
        ...

    while HybridFlowNet itself expects:

        position
        category.weight
        transformer.0.qkv.weight
        ...

    So we strip the `backbone.` prefix.
    """

    if not isinstance(checkpoint, dict):
        state = checkpoint

    else:
        state = None

        # Common checkpoint formats.
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "ema_state_dict",
            "ema",
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break

        if state is None:
            state = checkpoint

    cleaned = {}

    for key, value in state.items():

        # Lightning / custom metadata.
        if key == "_extra_state":
            continue

        # Strip common wrapper prefixes repeatedly.
        changed = True

        while changed:
            changed = False

            for prefix in (
                "module.",
                "_orig_mod.",
                "model.",
            ):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True

        # Your actual checkpoint wrapper.
        if key.startswith("backbone."):
            key = key[len("backbone.") :]

        cleaned[key] = value

    return cleaned


def infer_config_from_state_dict(state):
    """
    Infer the major HybridConfig dimensions directly from checkpoint shapes.

    This is safer than blindly assuming HybridConfig() defaults.
    """

    if "category.weight" not in state:
        raise RuntimeError("Could not infer width: category.weight missing.")

    width = state["category.weight"].shape[1]

    # stem.weight shape:
    # [base_channels, 3, 3, 3]
    if "stem.weight" not in state:
        raise RuntimeError("Could not infer base_channels: stem.weight missing.")

    base_channels = state["stem.weight"].shape[0]

    # Count transformer blocks.
    block_indices = set()

    prefix = "transformer."

    for key in state:
        if key.startswith(prefix):
            parts = key.split(".")

            if len(parts) >= 3:
                try:
                    block_indices.add(int(parts[1]))
                except ValueError:
                    pass

    depth = max(block_indices) + 1 if block_indices else 0

    if depth == 0:
        raise RuntimeError("Could not infer transformer depth.")

    # Heads cannot be inferred uniquely from qkv shape,
    # because qkv shape only tells us width.
    #
    # Your architecture uses 8 by default.
    heads = 8

    # Infer MLP ratio.
    mlp_key = "transformer.0.mlp.0.weight"

    if mlp_key in state:
        hidden = state[mlp_key].shape[0]
        mlp_ratio = hidden / width
    else:
        mlp_ratio = 4.0

    return HybridConfig(
        width=width,
        depth=depth,
        heads=heads,
        mlp_ratio=mlp_ratio,
        dropout=0.0,
        base_channels=base_channels,
        backbone="hybrid",
    )


def load_model(
    checkpoint_path,
    device,
):
    print("\n" + "=" * 80)
    print("CHECKPOINT LOADING")
    print("=" * 80)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    print(f"\nLoaded checkpoint file:\n" f"  {checkpoint_path}")

    state = extract_state_dict(
        checkpoint,
    )

    print(f"\nExtracted " f"{len(state):,} tensors/entries " f"from checkpoint.")

    # -----------------------------------------------------
    # Infer actual architecture from checkpoint.
    # -----------------------------------------------------

    config = infer_config_from_state_dict(
        state,
    )

    print("\nInferred model config:")

    for key, value in asdict(config).items():
        print(f"  {key:15s}: {value}")

    model = HybridFlowNet(config).to(device)

    # -----------------------------------------------------
    # Load strictly enough to detect mistakes.
    # -----------------------------------------------------

    result = model.load_state_dict(
        state,
        strict=False,
    )

    if result.missing_keys:
        print("\nMISSING KEYS:")

        for key in result.missing_keys:
            print(f"  {key}")

    if result.unexpected_keys:
        print("\nUNEXPECTED KEYS:")

        for key in result.unexpected_keys:
            print(f"  {key}")

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "\nCheckpoint did NOT cleanly load.\n"
            "Stopping diagnostics because results "
            "would be misleading."
        )

    print("\n✓ Checkpoint loaded successfully.")

    model.eval()

    return model, checkpoint


# =========================================================
# PARAMETER COUNT
# =========================================================


def count_params(module):
    return sum(parameter.numel() for parameter in module.parameters())


def print_parameter_breakdown(model):
    print("\n" + "=" * 80)
    print("PARAMETER BREAKDOWN")
    print("=" * 80)

    total = count_params(model)

    groups = {
        "category embedding": model.category,
        "time embedding": model.time,
        "interval embedding": model.interval,
        "stem": model.stem,
        "encoder 64": model.enc64,
        "down 64->32": model.down64,
        "encoder 32": model.enc32,
        "down 32->16": model.down32,
        "bottleneck conv": model.bottleneck,
        "transformer": model.transformer,
        "transformer norm": model.transformer_norm,
        "up32 conv": model.up32_conv,
        "decoder 32": model.dec32,
        "up64 conv": model.up64_conv,
        "decoder 64": model.dec64,
        "output norm": model.output_norm,
        "output": model.output,
    }

    print(f"\nTOTAL: " f"{total:,} parameters " f"= {total / 1e6:.2f}M\n")

    for name, module in groups.items():

        n = count_params(module)

        percent = 100.0 * n / total

        print(f"{name:25s}" f"{n:12,d} " f"{n / 1e6:8.2f}M " f"{percent:7.2f}%")


@torch.no_grad()
def inspect_real_endpoint(
    model,
    dataloader,
    device,
    num_batches=4,
):
    """
    Diagnose the flow field near the noise endpoint using REAL validation data.

    Training convention assumed:

        x0 = real image
        x1 = Gaussian noise

        x_t = (1 - t) * x0 + t * x1

        target velocity = x1 - x0

    Reports:
        - predicted velocity RMS
        - target velocity RMS
        - FM MSE
        - relative RMSE
        - cosine(pred, target)
        - temporal derivative |dv/dt|
        - change in predicted velocity between adjacent t values

    This specifically targets t ~= 1 because generation starts there.
    """

    print("\n" + "=" * 100)
    print("REAL-DATA ENDPOINT DIAGNOSTIC")
    print("=" * 100)

    # Dense near the endpoint.
    times = torch.tensor(
        [
            0.9000,
            0.9200,
            0.9400,
            0.9600,
            0.9700,
            0.9800,
            0.9850,
            0.9900,
            0.9925,
            0.9950,
            0.9975,
            0.9990,
            1.0000,
        ],
        device=device,
        dtype=torch.float32,
    )

    # Accumulate statistics over several real batches.
    accumulated = {
        float(t): {
            "pred_rms": [],
            "target_rms": [],
            "mse": [],
            "relative_rmse": [],
            "cosine": [],
            "dvdt": [],
        }
        for t in times.tolist()
    }

    batches_used = 0

    for batch_index, batch in enumerate(dataloader):
        if batch_index >= num_batches:
            break

        # =====================================================
        # BATCH PARSING
        # =====================================================
        #
        # Handles the most common dataset formats.
        #
        # If your PokemonDataModule returns something different,
        # this is the only section you'll need to change.
        #

        if isinstance(batch, dict):
            # Try common names.
            image_key = None
            category_key = None

            for candidate in (
                "image",
                "images",
                "pixel_values",
                "x",
            ):
                if candidate in batch:
                    image_key = candidate
                    break

            for candidate in (
                "category",
                "categories",
                "label",
                "labels",
                "y",
            ):
                if candidate in batch:
                    category_key = candidate
                    break

            if image_key is None:
                raise RuntimeError(
                    f"Could not find image tensor in batch keys: "
                    f"{list(batch.keys())}"
                )

            if category_key is None:
                raise RuntimeError(
                    f"Could not find category tensor in batch keys: "
                    f"{list(batch.keys())}"
                )

            x0 = batch[image_key]
            category = batch[category_key]

        elif isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise RuntimeError(
                    "Expected validation batch to contain " "(images, categories)."
                )

            x0 = batch[0]
            category = batch[1]

        else:
            raise RuntimeError(f"Unsupported validation batch type: {type(batch)}")

        x0 = x0.to(
            device=device,
            dtype=torch.float32,
        )

        category = category.to(
            device=device,
            dtype=torch.long,
        )

        # Flatten category if dataset gives shape [B, 1].
        category = category.reshape(-1)

        batch_size = x0.shape[0]

        # -----------------------------------------------------
        # DATA SANITY
        # -----------------------------------------------------

        if batch_index == 0:
            print("\nValidation batch:")

            print(f"  shape      : {tuple(x0.shape)}")

            print(f"  x0 min     : {x0.min().item():.6f}")

            print(f"  x0 max     : {x0.max().item():.6f}")

            print(f"  x0 mean    : {x0.mean().item():.6f}")

            print(f"  x0 std     : {x0.std().item():.6f}")

            print(f"  categories : " f"{category[:min(10, batch_size)].tolist()}")

            if tuple(x0.shape[1:]) != (3, 64, 64):
                raise RuntimeError(
                    "Expected real images with shape [B, 3, 64, 64], "
                    f"got {tuple(x0.shape)}"
                )

        # =====================================================
        # FIX ONE NOISE ENDPOINT FOR THE WHOLE TRAJECTORY
        # =====================================================

        x1 = torch.randn_like(x0)

        # Under straight-line flow matching this target is constant
        # along the entire path.
        target_velocity = x1 - x0

        target_rms_scalar = target_velocity.square().mean().sqrt().item()

        previous_velocity = None
        previous_t = None

        # =====================================================
        # DENSE ENDPOINT SWEEP
        # =====================================================

        for t_scalar in times:
            t_value = float(t_scalar.item())

            t = torch.full(
                (batch_size,),
                t_value,
                device=device,
                dtype=x0.dtype,
            )

            t_view = t[
                :,
                None,
                None,
                None,
            ]

            xt = (1.0 - t_view) * x0 + t_view * x1

            predicted_velocity = model(
                xt,
                t,
                category=category,
            )

            # -------------------------------------------------
            # FM ERROR
            # -------------------------------------------------

            error = predicted_velocity - target_velocity

            mse = error.square().mean().item()

            rmse = mse**0.5

            relative_rmse = rmse / (target_rms_scalar + 1e-12)

            pred_rms = predicted_velocity.square().mean().sqrt().item()

            # Per-example cosine, then mean.
            cosine = (
                F.cosine_similarity(
                    predicted_velocity.flatten(1),
                    target_velocity.flatten(1),
                    dim=1,
                    eps=1e-8,
                )
                .mean()
                .item()
            )

            # -------------------------------------------------
            # TEMPORAL VELOCITY DERIVATIVE
            # -------------------------------------------------

            if previous_velocity is None:
                dvdt = float("nan")

            else:
                dt = t_value - previous_t

                dv = predicted_velocity - previous_velocity

                dv_rms = dv.square().mean().sqrt().item()

                dvdt = dv_rms / dt

            stats = accumulated[t_value]

            stats["pred_rms"].append(pred_rms)

            stats["target_rms"].append(target_rms_scalar)

            stats["mse"].append(mse)

            stats["relative_rmse"].append(relative_rmse)

            stats["cosine"].append(cosine)

            if not torch.isnan(torch.tensor(dvdt)):
                stats["dvdt"].append(dvdt)

            previous_velocity = predicted_velocity

            previous_t = t_value

        batches_used += 1

    if batches_used == 0:
        raise RuntimeError("Validation dataloader produced no batches.")

    # =========================================================
    # PRINT RESULTS
    # =========================================================

    def mean(values):
        if not values:
            return float("nan")

        return sum(values) / len(values)

    print(
        "\n"
        "t       "
        "pred_RMS   "
        "target_RMS "
        "MSE        "
        "rel_RMSE   "
        "cosine     "
        "|dv/dt|"
    )

    print("-" * 100)

    rows = []

    for t_scalar in times:
        t_value = float(t_scalar.item())

        stats = accumulated[t_value]

        row = {
            "t": t_value,
            "pred_rms": mean(stats["pred_rms"]),
            "target_rms": mean(stats["target_rms"]),
            "mse": mean(stats["mse"]),
            "relative_rmse": mean(stats["relative_rmse"]),
            "cosine": mean(stats["cosine"]),
            "dvdt": mean(stats["dvdt"]),
        }

        rows.append(row)

        print(
            f"{row['t']:0.4f}  "
            f"{row['pred_rms']:9.5f}  "
            f"{row['target_rms']:10.5f}  "
            f"{row['mse']:9.6f}  "
            f"{row['relative_rmse']:9.5f}  "
            f"{row['cosine']:9.5f}  "
            f"{row['dvdt']:9.4f}"
        )

    # =========================================================
    # AUTOMATIC INTERPRETATION
    # =========================================================

    print("\n" + "=" * 100)

    print("ENDPOINT SUMMARY")

    print("=" * 100)

    endpoint = rows[-1]
    before_endpoint = rows[-2]

    endpoint_error_ratio = endpoint["relative_rmse"] / (
        before_endpoint["relative_rmse"] + 1e-12
    )

    cosine_drop = before_endpoint["cosine"] - endpoint["cosine"]

    print(f"\nRelative RMSE @ t=0.999 : " f"{before_endpoint['relative_rmse']:.6f}")

    print(f"Relative RMSE @ t=1.000 : " f"{endpoint['relative_rmse']:.6f}")

    print(f"Endpoint error ratio     : " f"{endpoint_error_ratio:.3f}x")

    print(f"Cosine @ t=0.999         : " f"{before_endpoint['cosine']:.6f}")

    print(f"Cosine @ t=1.000         : " f"{endpoint['cosine']:.6f}")

    print(f"Cosine drop              : " f"{cosine_drop:.6f}")

    print(f"|dv/dt| near t=1        : " f"{endpoint['dvdt']:.6f}")

    # ---------------------------------------------------------
    # Simple diagnosis
    # ---------------------------------------------------------

    if endpoint_error_ratio > 1.25 or cosine_drop > 0.05:
        print("\n>>> DIAGNOSIS:")

        print("Prediction quality deteriorates sharply " "at the exact noise endpoint.")

        print("This supports an ENDPOINT-LEARNING problem.")

        print("\nNext experiment:")

        print("  - oversample t close to 1")

        print("  - explicitly include t=1 examples")

        print("  - then re-evaluate FID@4")

    elif endpoint["dvdt"] > 10.0:
        print("\n>>> DIAGNOSIS:")

        print(
            "Prediction error remains reasonably stable, "
            "but the vector field changes extremely rapidly "
            "near t=1."
        )

        print("This supports a FIELD-SMOOTHNESS / FEW-NFE problem.")

        print("\nNext experiment:")

        print("  - reduce timestep embedding scale")

        print("  - test time_scale = 100, 30, 10")

        print("  - compare FID@4 versus FID@32")

    else:
        print("\n>>> DIAGNOSIS:")

        print("No catastrophic endpoint-specific failure detected.")

        print(
            "The FID problem is likely distributed across "
            "the trajectory or caused by general model quality."
        )

    return rows


# =========================================================
# CONDITIONING
# =========================================================


@torch.no_grad()
def inspect_conditioning(
    model,
    batch_size,
    device,
):
    print("\n" + "=" * 80)
    print("CONDITIONING MAGNITUDES")
    print("=" * 80)

    t = torch.rand(
        batch_size,
        device=device,
    )

    category = torch.randint(
        0,
        model.category.num_embeddings,
        (batch_size,),
        device=device,
    )

    interval = torch.zeros_like(t)

    category_embedding = model.category(category)

    time_embedding = model.time(t)

    interval_embedding = model.interval(interval)

    total_condition = category_embedding + time_embedding + interval_embedding

    tensors = {
        "category": category_embedding,
        "time": time_embedding,
        "interval(t-r=0)": interval_embedding,
        "total condition": total_condition,
    }

    print("\nMean L2 norm per sample:")

    for name, tensor in tensors.items():

        norm = tensor.norm(dim=-1)

        print(
            f"{name:20s} "
            f"mean={norm.mean().item():9.5f} "
            f"std={norm.std().item():9.5f} "
            f"min={norm.min().item():9.5f} "
            f"max={norm.max().item():9.5f}"
        )

    print("\nRMS activation:")

    for name, tensor in tensors.items():

        rms = tensor.square().mean().sqrt()

        print(f"{name:20s} " f"{rms.item():.6f}")

    category_norm = category_embedding.norm(dim=-1).mean()

    time_norm = time_embedding.norm(dim=-1).mean()

    interval_norm = interval_embedding.norm(dim=-1).mean()

    print("\nRatios:")

    category_time_ratio = (category_norm / (time_norm + 1e-12)).item()

    category_interval_ratio = (category_norm / (interval_norm + 1e-12)).item()

    print(f"category / time     = {category_time_ratio:.6f}")

    print(f"category / interval = {category_interval_ratio:.6f}")

    return total_condition

    return total_condition


@torch.no_grad()
def inspect_path_smoothness(
    model,
    batch_size,
    device,
    steps=101,
):
    print("\n" + "=" * 80)
    print("FLOW PATH SMOOTHNESS")
    print("=" * 80)

    # Synthetic x0 here should ideally be replaced
    # with REAL normalized validation images.
    #
    # For a first diagnostic this still gives us structure,
    # but real x0 is strongly preferred.
    x0 = torch.empty(
        batch_size,
        3,
        64,
        64,
        device=device,
    ).uniform_(-1, 1)

    x1 = torch.randn_like(x0)

    category = torch.randint(
        0,
        model.category.num_embeddings,
        (batch_size,),
        device=device,
    )

    times = torch.linspace(
        0,
        1,
        steps,
        device=device,
    )

    velocities = []

    for t_scalar in times:

        t = torch.full(
            (batch_size,),
            t_scalar.item(),
            device=device,
        )

        t_view = t[:, None, None, None]

        xt = (1 - t_view) * x0 + t_view * x1

        velocity = model(
            xt,
            t,
            category=category,
        )

        velocities.append(velocity)

    print("\nt       |v| RMS      |Δv|/Δt")

    print("-" * 45)

    derivative_values = []

    for i in range(len(times)):

        velocity_rms = velocities[i].square().mean().sqrt().item()

        if i == 0:
            derivative = float("nan")

        else:
            dt = (times[i] - times[i - 1]).item()

            derivative = (
                velocities[i] - velocities[i - 1]
            ).square().mean().sqrt().item() / dt

            derivative_values.append(derivative)

        if i % 5 == 0 or i == len(times) - 1:
            print(
                f"{times[i].item():.2f}    "
                f"{velocity_rms:10.5f}    "
                f"{derivative:10.5f}"
            )

    derivative_tensor = torch.tensor(derivative_values)

    print("\nVelocity temporal derivative:")

    print(f"mean : " f"{derivative_tensor.mean().item():.5f}")

    print(f"max  : " f"{derivative_tensor.max().item():.5f}")

    max_index = derivative_tensor.argmax().item()

    print(f"max near t = " f"{times[max_index + 1].item():.3f}")


# =========================================================
# adaLN GATE INSPECTION
# =========================================================


@torch.no_grad()
def inspect_transformer_gates(
    model,
    condition,
):
    print("\n" + "=" * 80)
    print("adaLN-ZERO TRANSFORMER GATES")
    print("=" * 80)

    all_g1 = []
    all_g2 = []

    for index, block in enumerate(model.transformer):

        modulation = block.modulation(condition)

        (
            shift1,
            scale1,
            gate1,
            shift2,
            scale2,
            gate2,
        ) = modulation.chunk(
            6,
            dim=-1,
        )

        g1_abs = gate1.abs().mean().item()

        g2_abs = gate2.abs().mean().item()

        g1_rms = gate1.square().mean().sqrt().item()

        g2_rms = gate2.square().mean().sqrt().item()

        shift1_rms = shift1.square().mean().sqrt().item()

        scale1_rms = scale1.square().mean().sqrt().item()

        shift2_rms = shift2.square().mean().sqrt().item()

        scale2_rms = scale2.square().mean().sqrt().item()

        all_g1.append(g1_abs)

        all_g2.append(g2_abs)

        print(f"\nBlock {index:02d}:")

        print("  attention gate |g1| mean = " f"{g1_abs:.8f}")

        print("  attention gate RMS       = " f"{g1_rms:.8f}")

        print("  MLP gate       |g2| mean = " f"{g2_abs:.8f}")

        print("  MLP gate       RMS       = " f"{g2_rms:.8f}")

        print("  shift1 RMS               = " f"{shift1_rms:.8f}")

        print("  scale1 RMS               = " f"{scale1_rms:.8f}")

        print("  shift2 RMS               = " f"{shift2_rms:.8f}")

        print("  scale2 RMS               = " f"{scale2_rms:.8f}")

    average_g1 = sum(all_g1) / len(all_g1)

    average_g2 = sum(all_g2) / len(all_g2)

    print("\n" + "-" * 80)

    print("AVERAGE |attention gate| = " f"{average_g1:.8f}")

    print("AVERAGE |MLP gate|       = " f"{average_g2:.8f}")

    if average_g1 < 0.02 and average_g2 < 0.02:
        print("\n!!! WARNING !!!")

        print("Transformer gates are very small.")

        print(
            "The transformer may still be "
            "close to its zero-initialized "
            "identity state."
        )


# =========================================================
# TRANSFORMER PARAMETER NORMS
# =========================================================


@torch.no_grad()
def inspect_modulation_parameters(
    model,
):
    print("\n" + "=" * 80)
    print("TRANSFORMER MODULATION PARAMETER NORMS")
    print("=" * 80)

    for index, block in enumerate(model.transformer):

        layer = block.modulation[-1]

        weight = layer.weight
        bias = layer.bias

        weight_rms = weight.square().mean().sqrt().item()

        weight_norm = weight.norm().item()

        bias_rms = bias.square().mean().sqrt().item()

        print(
            f"Block {index:02d}: "
            f"W RMS={weight_rms:.8f}  "
            f"W norm={weight_norm:.5f}  "
            f"B RMS={bias_rms:.8f}"
        )


# =========================================================
# OUTPUT STATISTICS
# =========================================================


@torch.no_grad()
def inspect_output_statistics(
    model,
    batch_size,
    device,
):
    print("\n" + "=" * 80)
    print("VELOCITY OUTPUT STATISTICS")
    print("=" * 80)

    x = torch.randn(
        batch_size,
        3,
        64,
        64,
        device=device,
    )

    t = torch.rand(
        batch_size,
        device=device,
    )

    category = torch.randint(
        0,
        model.category.num_embeddings,
        (batch_size,),
        device=device,
    )

    velocity = model(
        x,
        t,
        category=category,
    )

    print(f"mean:     " f"{velocity.mean().item():.8f}")

    print(f"std:      " f"{velocity.std().item():.8f}")

    print(f"abs mean: " f"{velocity.abs().mean().item():.8f}")

    print(f"RMS:      " f"{velocity.square().mean().sqrt().item():.8f}")

    print(f"min:      " f"{velocity.min().item():.8f}")

    print(f"max:      " f"{velocity.max().item():.8f}")


# =========================================================
# CATEGORY SENSITIVITY
# =========================================================


@torch.no_grad()
def inspect_category_sensitivity(
    model,
    batch_size,
    device,
):
    print("\n" + "=" * 80)
    print("CATEGORY CONDITIONING SENSITIVITY")
    print("=" * 80)

    # Same x and same timestep.
    # ONLY category changes.

    x = torch.randn(
        batch_size,
        3,
        64,
        64,
        device=device,
    )

    t = torch.full(
        (batch_size,),
        0.5,
        device=device,
    )

    max_category = model.category.num_embeddings - 1

    category_a = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=device,
    )

    category_b = torch.full(
        (batch_size,),
        min(100, max_category),
        dtype=torch.long,
        device=device,
    )

    velocity_a = model(
        x,
        t,
        category=category_a,
    )

    velocity_b = model(
        x,
        t,
        category=category_b,
    )

    difference = velocity_a - velocity_b

    difference_rms = difference.square().mean().sqrt()

    output_rms = 0.5 * (
        velocity_a.square().mean().sqrt() + velocity_b.square().mean().sqrt()
    )

    relative_difference = difference_rms / (output_rms + 1e-12)

    cosine_similarity = F.cosine_similarity(
        velocity_a.flatten(1),
        velocity_b.flatten(1),
        dim=1,
    ).mean()

    print("Output RMS                 : " f"{output_rms.item():.8f}")

    print("Category-change RMS diff   : " f"{difference_rms.item():.8f}")

    print("Relative category effect   : " f"{relative_difference.item():.8f}")

    print("Output cosine similarity   : " f"{cosine_similarity.item():.8f}")

    if relative_difference.item() < 0.01:
        print("\n!!! WARNING !!!")

        print("Changing category barely changes " "the predicted velocity.")

        print("The model may be largely ignoring " "class conditioning.")


# =========================================================
# TIME SENSITIVITY
# =========================================================


@torch.no_grad()
def inspect_time_sensitivity(
    model,
    batch_size,
    device,
):
    print("\n" + "=" * 80)
    print("TIMESTEP SENSITIVITY")
    print("=" * 80)

    x = torch.randn(
        batch_size,
        3,
        64,
        64,
        device=device,
    )

    category = torch.randint(
        0,
        model.category.num_embeddings,
        (batch_size,),
        device=device,
    )

    times = (
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        1.00,
    )

    outputs = {}

    for time_value in times:

        t = torch.full(
            (batch_size,),
            time_value,
            device=device,
        )

        velocity = model(
            x,
            t,
            category=category,
        )

        outputs[time_value] = velocity

        rms = velocity.square().mean().sqrt().item()

        abs_mean = velocity.abs().mean().item()

        print(f"t={time_value:4.2f}: " f"RMS={rms:.8f}, " f"abs mean={abs_mean:.8f}")

    print("\nAdjacent timestep differences:")

    for first, second in zip(
        times[:-1],
        times[1:],
    ):

        difference = outputs[second] - outputs[first]

        rms = difference.square().mean().sqrt().item()

        reference_rms = outputs[first].square().mean().sqrt().item()

        relative = rms / (reference_rms + 1e-12)

        print(
            f"{first:4.2f} -> {second:4.2f}: "
            f"RMS difference={rms:.8f}, "
            f"relative={relative:.6f}"
        )


# =========================================================
# INTERVAL SENSITIVITY
# =========================================================


@torch.no_grad()
def inspect_interval_sensitivity(
    model,
    batch_size,
    device,
):
    print("\n" + "=" * 80)
    print("INTERVAL CONDITIONING SENSITIVITY")
    print("=" * 80)

    x = torch.randn(
        batch_size,
        3,
        64,
        64,
        device=device,
    )

    t = torch.full(
        (batch_size,),
        0.75,
        device=device,
    )

    category = torch.randint(
        0,
        model.category.num_embeddings,
        (batch_size,),
        device=device,
    )

    intervals = (
        0.00,
        0.05,
        0.10,
        0.25,
        0.50,
    )

    baseline = None

    for interval_value in intervals:

        interval = torch.full(
            (batch_size,),
            interval_value,
            device=device,
        )

        velocity = model(
            x,
            t,
            category=category,
            interval=interval,
        )

        if baseline is None:
            baseline = velocity

        difference = velocity - baseline

        difference_rms = difference.square().mean().sqrt().item()

        velocity_rms = velocity.square().mean().sqrt().item()

        relative = difference_rms / (velocity_rms + 1e-12)

        print(
            f"interval={interval_value:4.2f}: "
            f"velocity RMS={velocity_rms:.8f}, "
            f"diff from 0={difference_rms:.8f}, "
            f"relative={relative:.6f}"
        )


# =========================================================
# FLOW MATCHING LOSS VS TIMESTEP
# =========================================================


@torch.no_grad()
def fm_loss_vs_timestep(
    model,
    dataloader,
    device,
    max_batches=100,
):
    """
    Optional diagnostic.

    Assumes each batch is either:

        x0, category

    or:

        {
            "image": ...,
            "category": ...
        }

    Adjust batch parsing if your dataset uses different names.
    """

    print("\n" + "=" * 80)
    print("FLOW MATCHING LOSS VS TIMESTEP")
    print("=" * 80)

    num_bins = 10

    loss_sum = torch.zeros(
        num_bins,
        device=device,
        dtype=torch.float64,
    )

    counts = torch.zeros(
        num_bins,
        device=device,
        dtype=torch.float64,
    )

    for batch_index, batch in enumerate(dataloader):

        if batch_index >= max_batches:
            break

        if isinstance(
            batch,
            dict,
        ):

            x0 = batch["image"].to(device)

            category = batch["category"].to(device)

        else:
            x0, category = batch

            x0 = x0.to(device)

            category = category.to(device)

        batch_size = x0.shape[0]

        x1 = torch.randn_like(x0)

        t = torch.rand(
            batch_size,
            device=device,
            dtype=x0.dtype,
        )

        t_view = t[
            :,
            None,
            None,
            None,
        ]

        xt = (1.0 - t_view) * x0 + t_view * x1

        target_velocity = x1 - x0

        predicted_velocity = model(
            xt,
            t,
            category=category,
        )

        per_sample_loss = (
            (predicted_velocity - target_velocity).square().flatten(1).mean(dim=1)
        )

        bin_indices = torch.clamp(
            (t * num_bins).long(),
            max=num_bins - 1,
        )

        for bin_index in range(num_bins):

            mask = bin_indices == bin_index

            if mask.any():

                loss_sum[bin_index] += per_sample_loss[mask].double().sum()

                counts[bin_index] += mask.sum().double()

    print()

    for bin_index in range(num_bins):

        lower = bin_index / num_bins

        upper = (bin_index + 1) / num_bins

        if counts[bin_index] > 0:

            average_loss = loss_sum[bin_index] / counts[bin_index]

            print(
                f"t in [{lower:.1f}, {upper:.1f}): "
                f"MSE={average_loss.item():.8f} "
                f"(n={int(counts[bin_index].item())})"
            )


# =========================================================
# BASIC WEIGHT HEALTH
# =========================================================


@torch.no_grad()
def inspect_weight_health(
    model,
):
    print("\n" + "=" * 80)
    print("WEIGHT HEALTH")
    print("=" * 80)

    total_elements = 0
    total_zeros = 0

    nan_parameters = []
    inf_parameters = []

    for name, parameter in model.named_parameters():

        tensor = parameter.detach()

        total_elements += tensor.numel()

        total_zeros += (tensor == 0).sum().item()

        if torch.isnan(tensor).any():
            nan_parameters.append(name)

        if torch.isinf(tensor).any():
            inf_parameters.append(name)

    zero_fraction = total_zeros / total_elements

    print(f"Overall exact-zero fraction: " f"{zero_fraction:.8f}")

    if nan_parameters:
        print("\nNaNs found in:")

        for name in nan_parameters:
            print(f"  {name}")

    else:
        print("NaN parameters: none")

    if inf_parameters:
        print("\nInfs found in:")

        for name in inf_parameters:
            print(f"  {name}")

    else:
        print("Inf parameters: none")


# =========================================================
# MAIN
# =========================================================


def main():
    parser = argparse.ArgumentParser(
        description=("Inspect a trained HybridFlowNet checkpoint.")
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file.",
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"\nUsing device: {device}")

    model, checkpoint = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print_parameter_breakdown(model)

    inspect_weight_health(model)

    condition = inspect_conditioning(
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    inspect_transformer_gates(
        model=model,
        condition=condition,
    )

    inspect_modulation_parameters(model)

    inspect_output_statistics(
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    inspect_category_sensitivity(
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    inspect_time_sensitivity(
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    inspect_interval_sensitivity(
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    inspect_path_smoothness(
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    print("\nPreparing real validation data...")

    data_module = PokemonDataModule(
        batch_size=args.batch_size,
        num_workers=4,
        return_category=True,
    )

    validation_loader = data_module.val_dataloader()

    inspect_real_endpoint(
        model=model,
        dataloader=validation_loader,
        device=device,
        num_batches=16,
    )

    print("\n" + "=" * 80)

    print("DIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    main()
