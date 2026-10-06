#!/usr/bin/env python3

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import torch


# =====================================================================
# CHECKPOINT LOADING
# =====================================================================


def load_checkpoint(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def looks_like_state_dict(obj):
    if not isinstance(obj, dict) or len(obj) == 0:
        return False

    tensor_count = sum(torch.is_tensor(v) for v in obj.values())

    return tensor_count >= max(1, len(obj) // 2)


def extract_state_dict(ckpt):
    """
    Locate the actual model state_dict inside a checkpoint.
    """

    if looks_like_state_dict(ckpt):
        return ckpt, "<root>"

    candidates = (
        "model_state_dict",
        "state_dict",
        "model",
        "network",
        "net",
        "ema_state_dict",
        "ema",
    )

    if isinstance(ckpt, dict):
        for key in candidates:
            if key in ckpt and looks_like_state_dict(ckpt[key]):
                return ckpt[key], key

        # Search one level deeper.
        for key, value in ckpt.items():
            if not isinstance(value, dict):
                continue

            for subkey in candidates:
                if subkey in value and looks_like_state_dict(value[subkey]):
                    return value[subkey], f"{key}.{subkey}"

    raise RuntimeError(
        "Could not locate model state_dict.\n"
        f"Top-level object: {type(ckpt)}\n"
        f"Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
    )


def strip_uniform_prefix(state):
    """
    Remove common wrapper prefixes only when every key has the prefix.

    We deliberately DO NOT remove backbone.* here because we want to
    inspect the outer FewNFE/OneNFE checkpoint exactly as stored.
    """

    state = dict(state)

    for prefix in (
        "module.",
        "model.",
        "network.",
        "net.",
    ):
        tensor_keys = [k for k, v in state.items() if torch.is_tensor(v)]

        if tensor_keys and all(k.startswith(prefix) for k in tensor_keys):
            state = {
                k[len(prefix) :] if k.startswith(prefix) else k: v
                for k, v in state.items()
            }

    return state


# =====================================================================
# GENERIC STATISTICS
# =====================================================================


def rms(x):
    x = x.detach().float()

    if x.numel() == 0:
        return 0.0

    return x.square().mean().sqrt().item()


def tensor_stats(x):
    x = x.detach().float().reshape(-1)

    if x.numel() == 0:
        return {}

    finite = torch.isfinite(x)
    xf = x[finite]

    result = {
        "numel": x.numel(),
        "finite_pct": 100.0 * finite.float().mean().item(),
        "zero_pct": 100.0 * (x == 0).float().mean().item(),
    }

    if xf.numel():
        result.update(
            mean=xf.mean().item(),
            std=xf.std(unbiased=False).item(),
            rms=xf.square().mean().sqrt().item(),
            abs_mean=xf.abs().mean().item(),
            min=xf.min().item(),
            max=xf.max().item(),
        )

    return result


def human_num(n):
    if n >= 1e9:
        return f"{n / 1e9:.3f}B"
    if n >= 1e6:
        return f"{n / 1e6:.3f}M"
    if n >= 1e3:
        return f"{n / 1e3:.3f}K"
    return str(n)


def group_for_key(key):
    groups = (
        "category",
        "time",
        "interval",
        "stem",
        "encoder",
        "downsample",
        "transformer",
        "transformer_norm",
        "bottleneck",
        "decoder",
        "upsample",
        "output_norm",
        "output",
    )

    # Must check transformer_norm before transformer.
    if "transformer_norm" in key:
        return "transformer_norm"

    for group in groups:
        if re.search(
            rf"(^|\.){re.escape(group)}(\.|$)",
            key,
        ):
            return group

    return key.split(".")[0]


# =====================================================================
# CHECKPOINT METADATA
# =====================================================================


def print_metadata(ckpt):
    print()
    print("=" * 80)
    print("CHECKPOINT METADATA")
    print("=" * 80)

    if not isinstance(ckpt, dict):
        print(f"Root object: {type(ckpt).__name__}")
        return

    print("Top-level keys:")

    for key, value in ckpt.items():
        if torch.is_tensor(value):
            desc = f"Tensor {tuple(value.shape)}"

        elif isinstance(value, dict):
            desc = f"dict ({len(value)} keys)"

        elif isinstance(
            value,
            (int, float, str, bool, type(None)),
        ):
            desc = repr(value)

        else:
            desc = type(value).__name__

        print(f"  {key:30s} {desc}")

    interesting = (
        "epoch",
        "step",
        "global_step",
        "batch_in_epoch",
        "total_steps",
        "best_loss",
        "loss",
        "val_loss",
        "best_fid",
        "best_fid_seed",
        "fid",
        "mode",
        "backbone",
        "objective",
        "precision",
        "elapsed_seconds",
        "peak_cuda_bytes",
    )

    print()
    print("Likely training metadata:")

    found = False

    for key in interesting:
        if key in ckpt:
            print(f"  {key:20s}: {ckpt[key]}")
            found = True

    if not found:
        print("  No obvious scalar training metadata found.")

    # Show selected arguments.
    args = ckpt.get("args")

    if isinstance(args, dict):
        interesting_args = (
            "backbone",
            "mode",
            "epochs",
            "batch_size",
            "lr",
            "learning_rate",
            "weight_decay",
            "objective",
            "solver",
            "nfe",
            "seed",
        )

        print()
        print("Saved training arguments:")

        for key in interesting_args:
            if key in args:
                print(f"  {key:20s}: {args[key]}")

    # Optimizer.
    for key in (
        "optimizer",
        "optimizer_state_dict",
    ):
        optimizer = ckpt.get(key)

        if not isinstance(optimizer, dict):
            continue

        print()
        print(f"Optimizer: {key}")

        groups = optimizer.get(
            "param_groups",
            [],
        )

        for i, group in enumerate(groups):
            print(
                f"  group {i}: "
                f"lr={group.get('lr')} "
                f"weight_decay={group.get('weight_decay')} "
                f"betas={group.get('betas')}"
            )


# =====================================================================
# MODEL STATE INSPECTION
# =====================================================================


def inspect_state_dict(state):
    print()
    print("=" * 80)
    print("MODEL STATE")
    print("=" * 80)

    tensor_values = [v for v in state.values() if torch.is_tensor(v)]

    total = sum(v.numel() for v in tensor_values)

    bytes_total = sum(v.numel() * v.element_size() for v in tensor_values)

    print(f"Tensor entries:   " f"{len(tensor_values)}")

    print(f"Parameters/items: " f"{total:,} ({human_num(total)})")

    print(f"Tensor storage:   " f"{bytes_total / 1024**2:.2f} MiB")

    nonfinite = []

    for key, value in state.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue

        finite = torch.isfinite(value)

        if not finite.all():
            nonfinite.append(
                (
                    key,
                    int((~finite).sum()),
                    value.numel(),
                )
            )

    print()
    print("Non-finite check:")

    if nonfinite:
        for key, bad, count in nonfinite:
            print(f"  BAD {key}: " f"{bad}/{count} NaN/Inf values")
    else:
        print("  ✓ No NaN/Inf values found")


def print_group_summary(state):
    print()
    print("=" * 80)
    print("PARAMETER BREAKDOWN")
    print("=" * 80)

    info = defaultdict(
        lambda: {
            "numel": 0,
            "sumsq": 0.0,
            "zeros": 0,
        }
    )

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue

        group = group_for_key(key)

        x = value.detach().float()

        info[group]["numel"] += x.numel()
        info[group]["sumsq"] += x.square().sum().item()
        info[group]["zeros"] += (x == 0).sum().item()

    print(f"{'group':22s} " f"{'params':>12s} " f"{'RMS':>12s} " f"{'zero%':>10s}")

    print("-" * 60)

    for group, values in sorted(
        info.items(),
        key=lambda x: -x[1]["numel"],
    ):
        n = values["numel"]

        if n == 0:
            continue

        group_rms = math.sqrt(values["sumsq"] / n)

        zero_pct = 100.0 * values["zeros"] / n

        print(
            f"{group:22s} "
            f"{human_num(n):>12s} "
            f"{group_rms:12.6g} "
            f"{zero_pct:9.3f}%"
        )


# =====================================================================
# adaLN-ZERO INSPECTION
# =====================================================================


def inspect_adaln(state):
    print()
    print("=" * 80)
    print("adaLN-ZERO TRANSFORMER DIAGNOSTICS")
    print("=" * 80)

    names = (
        "shift1",
        "scale1",
        "gate1",
        "shift2",
        "scale2",
        "gate2",
    )

    found = False

    for key, weight in sorted(state.items()):
        if not torch.is_tensor(weight):
            continue

        if not re.search(
            r"transformer\.\d+\.modulation\.\d+\.weight$",
            key,
        ):
            continue

        if weight.ndim != 2 or weight.shape[0] % 6 != 0:
            continue

        found = True

        match = re.search(
            r"transformer\.(\d+)",
            key,
        )

        block_idx = match.group(1) if match else "?"

        print()
        print(f"Transformer block {block_idx}")
        print(f"  modulation weight: {key}")
        print(f"  shape: {tuple(weight.shape)}")

        chunks = weight.float().chunk(
            6,
            dim=0,
        )

        for name, chunk in zip(
            names,
            chunks,
        ):
            stats = tensor_stats(chunk)

            print(
                f"    {name:7s}: "
                f"RMS={stats['rms']:.6g}  "
                f"|mean|={stats['abs_mean']:.6g}  "
                f"zero={stats['zero_pct']:.2f}%"
            )

        bias_key = key[: -len("weight")] + "bias"

        if bias_key in state:
            bias = state[bias_key].float()

            bias_chunks = bias.chunk(
                6,
                dim=0,
            )

            print("  modulation bias:")

            for name, chunk in zip(
                names,
                bias_chunks,
            ):
                print(
                    f"    {name:7s}: "
                    f"RMS={rms(chunk):.6g}  "
                    f"|mean|={chunk.abs().mean().item():.6g}"
                )

        gate1_rms = rms(chunks[2])
        gate2_rms = rms(chunks[5])

        if gate1_rms < 1e-5 and gate2_rms < 1e-5:
            print("  !!! Gates are essentially zero.")
            print("      Transformer block may be effectively inactive.")

        elif gate1_rms < 1e-3 and gate2_rms < 1e-3:
            print("  !! Gates remain very small.")
            print("     Transformer contribution may be weak.")

        else:
            print("  ✓ adaLN gates have clearly moved away from zero.")

    if not found:
        print("Could not find Transformer modulation weights.")


# =====================================================================
# CRITICAL LAYERS
# =====================================================================


def inspect_critical_layers(state):
    print()
    print("=" * 80)
    print("CRITICAL LAYERS")
    print("=" * 80)

    patterns = (
        "output.weight",
        "output.bias",
        "transformer_norm.weight",
        "transformer_norm.bias",
        "category.weight",
    )

    found = 0

    for suffix in patterns:
        for key, value in state.items():
            if not key.endswith(suffix) or not torch.is_tensor(value):
                continue

            stats = tensor_stats(value)

            print(key)

            print(
                f"  shape={tuple(value.shape)} "
                f"RMS={stats.get('rms', float('nan')):.6g} "
                f"|mean|={stats.get('abs_mean', float('nan')):.6g} "
                f"zero={stats['zero_pct']:.3f}% "
                f"finite={stats['finite_pct']:.3f}%"
            )

            found += 1

    if not found:
        print("No expected critical-layer names found.")


# =====================================================================
# CHECKPOINT COMPARISON
# =====================================================================


def compare_states(
    old_state,
    new_state,
    top=25,
):
    print()
    print("=" * 80)
    print("CHECKPOINT DELTA")
    print("=" * 80)

    common = sorted(set(old_state) & set(new_state))

    rows = []

    group_sums = defaultdict(lambda: [0.0, 0])

    for key in common:
        a = old_state[key]
        b = new_state[key]

        if (
            not torch.is_tensor(a)
            or not torch.is_tensor(b)
            or a.shape != b.shape
            or not a.is_floating_point()
        ):
            continue

        a = a.float()
        b = b.float()

        delta = b - a

        delta_rms = rms(delta)
        old_rms = rms(a)
        new_rms = rms(b)

        relative = delta_rms / (old_rms + 1e-12)

        rows.append(
            (
                relative,
                delta_rms,
                old_rms,
                new_rms,
                key,
            )
        )

        group = group_for_key(key)

        group_sums[group][0] += delta.square().sum().item()

        group_sums[group][1] += delta.numel()

    rows.sort(reverse=True)

    print()
    print(f"Top {top} parameters " f"by relative RMS change:")

    print(
        f"{'rel Δ':>11s} "
        f"{'Δ RMS':>11s} "
        f"{'old RMS':>11s} "
        f"{'new RMS':>11s}  parameter"
    )

    print("-" * 110)

    for (
        relative,
        delta_rms,
        old_rms,
        new_rms,
        key,
    ) in rows[:top]:

        print(
            f"{relative:11.4g} "
            f"{delta_rms:11.4g} "
            f"{old_rms:11.4g} "
            f"{new_rms:11.4g}  "
            f"{key}"
        )

    print()
    print("RMS checkpoint movement by module:")

    for group, (
        sumsq,
        count,
    ) in sorted(
        group_sums.items(),
        key=lambda item: -(item[1][0] / max(1, item[1][1])),
    ):
        print(f"  {group:22s}: " f"Δ RMS=" f"{math.sqrt(sumsq / count):.6g}")


# =====================================================================
# BACKBONE EXTRACTION
# =====================================================================


def extract_backbone_state(state):
    """
    Training checkpoints save the HybridV3FlowNet inside a wrapper:

        backbone.category.weight
        backbone.encoder....
        backbone.transformer....
        backbone.output....

    HybridV3FlowNet itself expects:

        category.weight
        encoder....
        transformer....
        output....

    Strip ONLY the backbone. prefix here.
    """

    backbone_state = {}

    for key, value in state.items():
        if key.startswith("backbone."):
            new_key = key[len("backbone.") :]
            backbone_state[new_key] = value

    return backbone_state


# =====================================================================
# FORWARD DIAGNOSTIC
# =====================================================================


def forward_diagnostic(
    state,
    ckpt,
    batch_size=4,
):
    print()
    print("=" * 80)
    print("FORWARD-PASS DIAGNOSTIC")
    print("=" * 80)

    try:
        from src.models.hybrid_v3 import (
            HybridV3FlowNet,
            HybridV3Config,
        )
    except Exception as exc:
        print("Could not import HybridV3FlowNet:")
        print(exc)
        return

    # -------------------------------------------------------------
    # Extract actual trained backbone
    # -------------------------------------------------------------

    backbone_state = extract_backbone_state(state)

    print(f"Outer checkpoint entries: " f"{len(state)}")

    print(f"Extracted backbone entries: " f"{len(backbone_state)}")

    if not backbone_state:
        print()
        print("ERROR: No backbone.* keys found.")
        print("Cannot perform reliable forward diagnostic.")
        return

    # -------------------------------------------------------------
    # Recover architecture configuration if possible
    # -------------------------------------------------------------

    config = None

    args = ckpt.get("args") if isinstance(ckpt, dict) else None

    possible_sources = []

    if isinstance(ckpt, dict):
        for key in (
            "architecture_config",
            "model_config",
            "config",
            "backbone_config",
        ):
            if isinstance(
                ckpt.get(key),
                dict,
            ):
                possible_sources.append(ckpt[key])

    if isinstance(args, dict):
        possible_sources.append(args)

    allowed_config_keys = {
        "base_channels",
        "width",
        "depth",
        "heads",
        "dropout",
        "mlp_ratio",
        "backbone",
    }

    for source in possible_sources:
        candidate = {
            key: value for key, value in source.items() if key in allowed_config_keys
        }

        # Force the correct backbone type.
        candidate["backbone"] = "hybrid_v3"

        if "base_channels" in candidate or "width" in candidate or "depth" in candidate:
            config = candidate
            break

    # -------------------------------------------------------------
    # Instantiate model
    # -------------------------------------------------------------

    if config:
        print()
        print("Recovered architecture config:")

        for key, value in config.items():
            print(f"  {key:20s}: {value}")

        try:
            model = HybridV3FlowNet(config)

        except Exception as exc:
            print()
            print("Saved config could not instantiate model:")
            print(exc)

            print("Falling back to default HybridV3FlowNet().")

            model = HybridV3FlowNet()

    else:
        print()
        print("No architecture config recovered.")
        print("Using default HybridV3FlowNet().")

        model = HybridV3FlowNet()

    # -------------------------------------------------------------
    # Load trained weights
    # -------------------------------------------------------------

    result = model.load_state_dict(
        backbone_state,
        strict=False,
    )

    print()
    print(f"Missing keys:    " f"{len(result.missing_keys)}")

    for key in result.missing_keys[:20]:
        print(f"  missing:    {key}")

    print(f"Unexpected keys: " f"{len(result.unexpected_keys)}")

    for key in result.unexpected_keys[:20]:
        print(f"  unexpected: {key}")

    # -------------------------------------------------------------
    # Validate actual tensor equality
    # -------------------------------------------------------------

    print()
    print("LOAD VERIFICATION")
    print("-" * 80)

    verification_pairs = (
        (
            "output.weight",
            model.output.weight,
        ),
        (
            "category.weight",
            model.category.weight,
        ),
        (
            "transformer.0.modulation.1.weight",
            model.transformer[0].modulation[-1].weight,
        ),
    )

    load_verified = True

    for key, model_tensor in verification_pairs:
        if key not in backbone_state:
            print(f"{key:45s} NOT FOUND")
            load_verified = False
            continue

        checkpoint_tensor = backbone_state[key].detach().cpu()

        actual_tensor = model_tensor.detach().cpu()

        if checkpoint_tensor.shape != actual_tensor.shape:
            print(f"{key:45s} SHAPE MISMATCH")

            load_verified = False
            continue

        max_error = (
            (checkpoint_tensor.float() - actual_tensor.float()).abs().max().item()
        )

        exact = torch.equal(
            checkpoint_tensor,
            actual_tensor,
        )

        print(f"{key:45s} " f"exact={str(exact):5s} " f"max_error={max_error:.6g}")

        if max_error != 0:
            load_verified = False

    # Missing persistent buffers may occasionally be harmless.
    #
    # But if lots of weights are missing, STOP instead of producing
    # misleading diagnostics.
    serious_missing = [
        key for key in result.missing_keys if not key.endswith("frequencies")
    ]

    if serious_missing or result.unexpected_keys or not load_verified:
        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("WARNING: MODEL DID NOT LOAD CLEANLY.")
        print("Forward diagnostics would be misleading.")
        print("Stopping forward analysis.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    print()
    print("✓ Trained backbone loaded successfully.")

    # -------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------

    model.eval()

    block_effects = {}
    gate_effects = {}

    handles = []

    for index, block in enumerate(model.transformer):

        def block_hook(
            module,
            inputs,
            output,
            index=index,
        ):
            x = inputs[0].detach().float()

            y = output.detach().float()

            delta = y - x

            input_rms = rms(x)
            output_rms = rms(y)
            delta_rms = rms(delta)

            block_effects[index] = {
                "input_rms": input_rms,
                "output_rms": output_rms,
                "delta_rms": delta_rms,
                "relative": (delta_rms / (input_rms + 1e-12)),
                "cosine": torch.nn.functional.cosine_similarity(
                    x.reshape(x.shape[0], -1),
                    y.reshape(y.shape[0], -1),
                    dim=1,
                )
                .mean()
                .item(),
            }

        def modulation_hook(
            module,
            inputs,
            output,
            index=index,
        ):
            values = (
                output.detach()
                .float()
                .chunk(
                    6,
                    dim=-1,
                )
            )

            gate_effects[index] = {
                "shift1": rms(values[0]),
                "scale1": rms(values[1]),
                "gate1": rms(values[2]),
                "shift2": rms(values[3]),
                "scale2": rms(values[4]),
                "gate2": rms(values[5]),
                "gate1_abs_mean": (values[2].abs().mean().item()),
                "gate2_abs_mean": (values[5].abs().mean().item()),
            }

        handles.append(block.register_forward_hook(block_hook))

        handles.append(block.modulation.register_forward_hook(modulation_hook))

    # -------------------------------------------------------------
    # Deterministic probe
    # -------------------------------------------------------------

    torch.manual_seed(0)

    x = torch.randn(
        batch_size,
        3,
        64,
        64,
    )

    timestep = torch.linspace(
        0.1,
        0.9,
        batch_size,
    )

    category = torch.arange(batch_size) % 151

    interval = torch.zeros_like(timestep)

    with torch.no_grad():
        output = model(
            x,
            timestep,
            category=category,
            interval=interval,
        )

    for handle in handles:
        handle.remove()

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print()
    print("FORWARD SUMMARY")
    print("-" * 80)

    print(f"Input RMS:              " f"{rms(x):.6f}")

    print(f"Predicted velocity RMS: " f"{rms(output):.6f}")

    print(f"Predicted velocity abs mean: " f"{output.abs().mean().item():.6f}")

    print(f"Predicted velocity max abs:  " f"{output.abs().max().item():.6f}")

    print()
    print("ACTUAL TRANSFORMER CONTRIBUTION")

    print("-" * 80)

    for index in sorted(block_effects):
        values = block_effects[index]

        print()
        print(f"Transformer block {index}")

        print(f"  input RMS:       " f"{values['input_rms']:.6f}")

        print(f"  output RMS:      " f"{values['output_rms']:.6f}")

        print(f"  residual Δ RMS:  " f"{values['delta_rms']:.6f}")

        print(f"  Δ / input:       " f"{100 * values['relative']:.3f}%")

        print(f"  input/output cos:" f" {values['cosine']:.6f}")

        if index in gate_effects:
            gates = gate_effects[index]

            print(f"  gate1 RMS:       " f"{gates['gate1']:.6f}")

            print(f"  gate2 RMS:       " f"{gates['gate2']:.6f}")

            print(f"  gate1 |mean|:    " f"{gates['gate1_abs_mean']:.6f}")

            print(f"  gate2 |mean|:    " f"{gates['gate2_abs_mean']:.6f}")

            print(f"  scale1 RMS:      " f"{gates['scale1']:.6f}")

            print(f"  scale2 RMS:      " f"{gates['scale2']:.6f}")

            print(f"  shift1 RMS:      " f"{gates['shift1']:.6f}")

            print(f"  shift2 RMS:      " f"{gates['shift2']:.6f}")

    print()
    print("=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)

    print(
        """
The most important value is Δ / input.

Very rough interpretation:

    < 0.1%     Transformer is nearly an identity map.
    0.1-1%     Small contribution.
    1-5%       Clearly contributing.
    5-20%      Strong contribution.
    > 20%      Very strong transformation.

gate1 / gate2 tell us whether the adaLN-Zero residual branches
opened during training, but Δ/input is more directly useful because
it measures the actual change produced by the complete block.

This probe uses synthetic random inputs. It diagnoses whether the
trained modules are alive; it does NOT directly measure generation
quality or FID.
"""
    )


# =====================================================================
# MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description=("Inspect HybridV3 / FewNFE checkpoints.")
    )

    parser.add_argument(
        "checkpoint",
        help="Checkpoint to inspect.",
    )

    parser.add_argument(
        "--compare",
        help=(
            "Optional second checkpoint. "
            "The first checkpoint is treated as OLD "
            "and --compare as NEW."
        ),
    )

    parser.add_argument(
        "--forward",
        action="store_true",
        help=(
            "Instantiate HybridV3FlowNet, load the trained backbone, "
            "and measure actual Transformer activity."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help=("Number of parameter deltas " "to display when using --compare."),
    )

    args = parser.parse_args()

    path = Path(args.checkpoint)

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"Loading: {path}")

    print(f"File size: " f"{path.stat().st_size / 1024**2:.2f} MiB")

    ckpt = load_checkpoint(path)

    state, source = extract_state_dict(ckpt)

    state = strip_uniform_prefix(state)

    print(f"State dict source: " f"{source}")

    print_metadata(ckpt)

    inspect_state_dict(state)

    print_group_summary(state)

    inspect_adaln(state)

    inspect_critical_layers(state)

    if args.compare:
        other_path = Path(args.compare)

        if not other_path.exists():
            raise FileNotFoundError(other_path)

        other_ckpt = load_checkpoint(other_path)

        other_state, _ = extract_state_dict(other_ckpt)

        other_state = strip_uniform_prefix(other_state)

        compare_states(
            state,
            other_state,
            top=args.top,
        )

    if args.forward:
        forward_diagnostic(
            state,
            ckpt,
        )


if __name__ == "__main__":
    main()
