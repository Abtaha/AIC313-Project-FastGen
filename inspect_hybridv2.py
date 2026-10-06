#!/usr/bin/env python3

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


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
    if not isinstance(obj, dict) or not obj:
        return False

    tensor_count = sum(torch.is_tensor(v) for v in obj.values())

    return tensor_count >= max(
        1,
        len(obj) // 2,
    )


def extract_state_dict(ckpt):
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

        for key, value in ckpt.items():
            if not isinstance(value, dict):
                continue

            for subkey in candidates:
                if subkey in value and looks_like_state_dict(value[subkey]):
                    return value[subkey], f"{key}.{subkey}"

    raise RuntimeError(
        "Could not locate state_dict.\n"
        f"Top-level keys: "
        f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
    )


def strip_uniform_prefix(state):
    state = dict(state)

    for prefix in (
        "module.",
        "model.",
        "network.",
        "net.",
    ):
        tensor_keys = [key for key, value in state.items() if torch.is_tensor(value)]

        if tensor_keys and all(key.startswith(prefix) for key in tensor_keys):
            state = {
                (key[len(prefix) :] if key.startswith(prefix) else key): value
                for key, value in state.items()
            }

    return state


def extract_backbone_state(state):
    """
    Training checkpoints store the actual backbone below `backbone.*`.

    Example:
        backbone.category.weight
        backbone.transformer.0.qkv.weight

    HybridV2FlowNet expects:
        category.weight
        transformer.0.qkv.weight
    """

    backbone_state = {}

    for key, value in state.items():
        if key.startswith("backbone."):
            backbone_state[key[len("backbone.") :]] = value

    # Support a checkpoint containing the backbone directly.
    if not backbone_state:
        backbone_state = {
            key: value for key, value in state.items() if key != "_extra_state"
        }

    return backbone_state


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
    if "transformer_norm" in key:
        return "transformer_norm"

    groups = (
        "category",
        "time",
        "interval",
        "stem",
        "enc64",
        "down64",
        "enc32",
        "down32",
        "bottleneck",
        "post_bottleneck",
        "transformer",
        "up32_conv",
        "dec32",
        "up64_conv",
        "dec64",
        "output_norm",
        "output",
    )

    for group in groups:
        if re.search(
            rf"(^|\.){re.escape(group)}(\.|$)",
            key,
        ):
            return group

    return key.split(".")[0]


# =====================================================================
# METADATA
# =====================================================================


def print_metadata(ckpt):
    print()
    print("=" * 90)
    print("CHECKPOINT METADATA")
    print("=" * 90)

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
            (
                int,
                float,
                str,
                bool,
                type(None),
            ),
        ):
            desc = repr(value)

        else:
            desc = type(value).__name__

        print(f"  {key:30s} " f"{desc}")

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
        "best_fid_epoch",
        "best_fid_step",
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
            print(f"  {key:20s}: " f"{ckpt[key]}")
            found = True

    if not found:
        print("  No obvious training metadata found.")

    args = ckpt.get("args")

    if isinstance(args, dict):
        print()
        print("Saved training arguments:")

        wanted = (
            "backbone",
            "mode",
            "epochs",
            "batch_size",
            "lr",
            "learning_rate",
            "weight_decay",
            "objective",
            "seed",
            "width",
            "depth",
            "heads",
            "base_channels",
            "knots",
            "grad_clip",
            "fid_every",
            "fid_batch_size",
            "fid_seed",
        )

        for key in wanted:
            if key in args:
                print(f"  {key:20s}: " f"{args[key]}")

    optimizer = ckpt.get("optimizer")

    if isinstance(optimizer, dict):
        print()
        print("Optimizer:")

        for i, group in enumerate(optimizer.get("param_groups", [])):
            print(
                f"  group {i}: "
                f"lr={group.get('lr')} "
                f"weight_decay={group.get('weight_decay')} "
                f"betas={group.get('betas')}"
            )


# =====================================================================
# STATE OVERVIEW
# =====================================================================


def inspect_state_dict(state):
    print()
    print("=" * 90)
    print("MODEL STATE")
    print("=" * 90)

    tensors = [value for value in state.values() if torch.is_tensor(value)]

    total = sum(value.numel() for value in tensors)

    storage = sum(value.numel() * value.element_size() for value in tensors)

    print(f"Tensor entries:   " f"{len(tensors)}")

    print(f"Parameters/items: " f"{total:,} ({human_num(total)})")

    print(f"Tensor storage:   " f"{storage / 1024**2:.2f} MiB")

    bad = []

    for key, value in state.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue

        finite = torch.isfinite(value)

        if not finite.all():
            bad.append(
                (
                    key,
                    int((~finite).sum()),
                    value.numel(),
                )
            )

    print()
    print("Non-finite check:")

    if bad:
        for key, count, total_count in bad:
            print(f"  BAD {key}: " f"{count}/{total_count}")
    else:
        print("  ✓ No NaN/Inf values found")


def print_parameter_breakdown(state):
    print()
    print("=" * 90)
    print("PARAMETER BREAKDOWN")
    print("=" * 90)

    groups = defaultdict(
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

        groups[group]["numel"] += x.numel()

        groups[group]["sumsq"] += x.square().sum().item()

        groups[group]["zeros"] += (x == 0).sum().item()

    print(f"{'group':24s} " f"{'params':>12s} " f"{'RMS':>12s} " f"{'zero%':>10s}")

    print("-" * 65)

    for group, values in sorted(
        groups.items(),
        key=lambda item: -item[1]["numel"],
    ):
        n = values["numel"]

        if n == 0:
            continue

        group_rms = math.sqrt(values["sumsq"] / n)

        zero_pct = 100.0 * values["zeros"] / n

        print(
            f"{group:24s} "
            f"{human_num(n):>12s} "
            f"{group_rms:12.6g} "
            f"{zero_pct:9.3f}%"
        )


# =====================================================================
# STATIC adaLN-ZERO INSPECTION
# =====================================================================


def inspect_adaln(state):
    print()
    print("=" * 90)
    print("adaLN-ZERO TRANSFORMER DIAGNOSTICS")
    print("=" * 90)

    names = (
        "shift1",
        "scale1",
        "gate1",
        "shift2",
        "scale2",
        "gate2",
    )

    blocks_found = 0

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

        match = re.search(
            r"transformer\.(\d+)",
            key,
        )

        block_idx = int(match.group(1)) if match else -1

        blocks_found += 1

        print()
        print(f"Transformer block {block_idx}")

        print(f"  modulation weight: {key}")

        print(f"  shape: {tuple(weight.shape)}")

        chunks = (
            weight.detach()
            .float()
            .chunk(
                6,
                dim=0,
            )
        )

        for name, chunk in zip(
            names,
            chunks,
        ):
            stats = tensor_stats(chunk)

            print(
                f"    {name:7s}: "
                f"RMS={stats['rms']:.6f}  "
                f"|mean|={stats['abs_mean']:.6f}  "
                f"zero={stats['zero_pct']:.2f}%"
            )

        bias_key = key[: -len("weight")] + "bias"

        if bias_key in state:
            bias_chunks = (
                state[bias_key]
                .detach()
                .float()
                .chunk(
                    6,
                    dim=0,
                )
            )

            print("  modulation bias:")

            for name, chunk in zip(
                names,
                bias_chunks,
            ):
                print(
                    f"    {name:7s}: "
                    f"RMS={rms(chunk):.6f}  "
                    f"|mean|="
                    f"{chunk.abs().mean().item():.6f}"
                )

    print()
    print(f"Found {blocks_found} " f"Transformer blocks.")

    if blocks_found != 8:
        print("WARNING: Hybrid v2 normally " "expects 8 Transformer blocks.")


# =====================================================================
# CHECKPOINT COMPARISON
# =====================================================================


def compare_states(
    old_state,
    new_state,
    top=30,
):
    print()
    print("=" * 90)
    print("CHECKPOINT DELTA")
    print("=" * 90)

    common = sorted(set(old_state) & set(new_state))

    rows = []

    group_delta = defaultdict(lambda: [0.0, 0])

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

        group_delta[group][0] += delta.square().sum().item()

        group_delta[group][1] += delta.numel()

    rows.sort(reverse=True)

    print()
    print(f"Top {top} tensors " f"by relative RMS movement:")

    print(
        f"{'rel Δ':>11s} "
        f"{'Δ RMS':>11s} "
        f"{'old RMS':>11s} "
        f"{'new RMS':>11s} "
        f"parameter"
    )

    print("-" * 115)

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
            f"{new_rms:11.4g} "
            f"{key}"
        )

    print()
    print("RMS checkpoint movement " "by module:")

    for group, (
        sumsq,
        count,
    ) in sorted(
        group_delta.items(),
        key=lambda item: -(
            item[1][0]
            / max(
                1,
                item[1][1],
            )
        ),
    ):
        if count == 0:
            continue

        print(f"  {group:24s}: " f"Δ RMS=" f"{math.sqrt(sumsq / count):.6g}")


# =====================================================================
# FORWARD DIAGNOSTIC
# =====================================================================


def forward_diagnostic(
    state,
    ckpt,
    batch_size=4,
):
    print()
    print("=" * 90)
    print("FORWARD-PASS DIAGNOSTIC")
    print("=" * 90)

    try:
        from src.models.hybrid_v2 import (
            HybridV2FlowNet,
        )
    except Exception as exc:
        print("Could not import HybridV2FlowNet:")
        print(exc)
        return

    backbone_state = extract_backbone_state(state)

    print(f"Outer checkpoint entries: " f"{len(state)}")

    print(f"Extracted backbone entries: " f"{len(backbone_state)}")

    # -------------------------------------------------------------
    # Recover architecture configuration
    # -------------------------------------------------------------

    config = None

    args = ckpt.get("args") if isinstance(ckpt, dict) else None

    candidates = []

    if isinstance(ckpt, dict):
        for key in (
            "architecture_config",
            "model_config",
            "config",
            "backbone_config",
        ):
            value = ckpt.get(key)

            if isinstance(
                value,
                dict,
            ):
                candidates.append(value)

    # Most of your checkpoints store CLI config here.
    if isinstance(args, dict):
        candidates.append(args)

    # Best source is _extra_state if available.
    extra = state.get("_extra_state")

    if isinstance(extra, dict) and isinstance(
        extra.get("config"),
        dict,
    ):
        candidates.insert(
            0,
            extra["config"],
        )

    allowed = {
        "base_channels",
        "width",
        "depth",
        "heads",
        "dropout",
        "mlp_ratio",
        "backbone",
    }

    for source in candidates:
        candidate = {key: value for key, value in source.items() if key in allowed}

        candidate["backbone"] = "hybrid_v2"

        if "width" in candidate or "base_channels" in candidate or "depth" in candidate:
            config = candidate
            break

    if config:
        print()
        print("Recovered architecture config:")

        for key, value in config.items():
            print(f"  {key:20s}: " f"{value}")

        try:
            model = HybridV2FlowNet(config)

        except Exception as exc:
            print()
            print("Config instantiation failed:")
            print(exc)

            print("Using default HybridV2FlowNet().")

            model = HybridV2FlowNet()

    else:
        print()
        print("No saved architecture config found.")
        print("Using default HybridV2FlowNet().")

        model = HybridV2FlowNet()

    # -------------------------------------------------------------
    # Load trained backbone
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
    # Verify exact tensor loading
    # -------------------------------------------------------------

    print()
    print("LOAD VERIFICATION")

    print("-" * 90)

    checks = [
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
        (
            "transformer.7.modulation.1.weight",
            model.transformer[-1].modulation[-1].weight,
        ),
    ]

    verified = True

    for key, actual in checks:
        if key not in backbone_state:
            print(f"{key:48s} " f"NOT FOUND")

            verified = False
            continue

        expected = backbone_state[key].detach().cpu()

        actual = actual.detach().cpu()

        if expected.shape != actual.shape:
            print(
                f"{key:48s} "
                f"SHAPE MISMATCH "
                f"{tuple(expected.shape)} vs "
                f"{tuple(actual.shape)}"
            )

            verified = False
            continue

        max_error = (expected.float() - actual.float()).abs().max().item()

        exact = torch.equal(
            expected,
            actual,
        )

        print(f"{key:48s} " f"exact={str(exact):5s} " f"max_error=" f"{max_error:.6g}")

        if max_error != 0:
            verified = False

    serious_missing = [
        key for key in result.missing_keys if not key.endswith("frequencies")
    ]

    if serious_missing or result.unexpected_keys or not verified:
        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: CHECKPOINT DID NOT LOAD CLEANLY.")
        print("Refusing to report misleading forward diagnostics.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    print()
    print("✓ Trained Hybrid v2 backbone " "loaded successfully.")

    model.eval()

    # -------------------------------------------------------------
    # Forward hooks
    # -------------------------------------------------------------

    block_stats = {}
    modulation_stats = {}
    qkv_stats = {}
    mlp_stats = {}

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

            x_rms = rms(x)
            y_rms = rms(y)
            delta_rms = rms(delta)

            cosine = (
                F.cosine_similarity(
                    x.reshape(
                        x.shape[0],
                        -1,
                    ),
                    y.reshape(
                        y.shape[0],
                        -1,
                    ),
                    dim=1,
                )
                .mean()
                .item()
            )

            block_stats[index] = {
                "input_rms": x_rms,
                "output_rms": y_rms,
                "delta_rms": delta_rms,
                "relative": (delta_rms / (x_rms + 1e-12)),
                "cosine": cosine,
            }

        def modulation_hook(
            module,
            inputs,
            output,
            index=index,
        ):
            chunks = (
                output.detach()
                .float()
                .chunk(
                    6,
                    dim=-1,
                )
            )

            modulation_stats[index] = {
                "shift1": rms(chunks[0]),
                "scale1": rms(chunks[1]),
                "gate1": rms(chunks[2]),
                "shift2": rms(chunks[3]),
                "scale2": rms(chunks[4]),
                "gate2": rms(chunks[5]),
                "gate1_abs": (chunks[2].abs().mean().item()),
                "gate2_abs": (chunks[5].abs().mean().item()),
            }

        def qkv_hook(
            module,
            inputs,
            output,
            index=index,
        ):
            qkv_stats[index] = {
                "input_rms": rms(inputs[0]),
                "output_rms": rms(output),
            }

        def mlp_hook(
            module,
            inputs,
            output,
            index=index,
        ):
            mlp_stats[index] = {
                "input_rms": rms(inputs[0]),
                "output_rms": rms(output),
            }

        handles.append(block.register_forward_hook(block_hook))

        handles.append(block.modulation.register_forward_hook(modulation_hook))

        handles.append(block.qkv.register_forward_hook(qkv_hook))

        handles.append(block.mlp.register_forward_hook(mlp_hook))

    # -------------------------------------------------------------
    # Deterministic synthetic probe
    # -------------------------------------------------------------

    torch.manual_seed(0)

    x = torch.randn(
        batch_size,
        3,
        64,
        64,
    )

    if batch_size == 4:
        timestep = torch.tensor(
            [
                0.05,
                0.25,
                0.50,
                0.90,
            ],
            dtype=torch.float32,
        )
    else:
        timestep = torch.linspace(
            0.05,
            0.95,
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
    # Forward summary
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print("FORWARD SUMMARY")
    print("=" * 90)

    print(f"Input RMS:              " f"{rms(x):.6f}")

    print(f"Predicted velocity RMS: " f"{rms(output):.6f}")

    print(f"Velocity abs mean:      " f"{output.abs().mean().item():.6f}")

    print(f"Velocity max abs:       " f"{output.abs().max().item():.6f}")

    # -------------------------------------------------------------
    # Per-block table
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print("TRANSFORMER BLOCK ACTIVITY")
    print("=" * 90)

    if not block_stats:
        print("No Transformer forward hooks fired.")
        return

    print(
        f"{'blk':>3s} "
        f"{'in RMS':>10s} "
        f"{'out RMS':>10s} "
        f"{'Δ RMS':>10s} "
        f"{'Δ/in':>10s} "
        f"{'cos':>9s} "
        f"{'gate1':>10s} "
        f"{'gate2':>10s}"
    )

    print("-" * 90)

    for index in sorted(block_stats):
        block = block_stats[index]

        modulation = modulation_stats.get(
            index,
            {},
        )

        print(
            f"{index:3d} "
            f"{block['input_rms']:10.4f} "
            f"{block['output_rms']:10.4f} "
            f"{block['delta_rms']:10.4f} "
            f"{100 * block['relative']:9.2f}% "
            f"{block['cosine']:9.4f} "
            f"{modulation.get('gate1', float('nan')):10.4f} "
            f"{modulation.get('gate2', float('nan')):10.4f}"
        )

    # -------------------------------------------------------------
    # Detailed diagnostics
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print("DETAILED BLOCK DIAGNOSTICS")
    print("=" * 90)

    for index in sorted(block_stats):
        block = block_stats[index]

        modulation = modulation_stats.get(
            index,
            {},
        )

        qkv = qkv_stats.get(
            index,
            {},
        )

        mlp = mlp_stats.get(
            index,
            {},
        )

        print()
        print(f"Transformer block {index}")

        print(f"  input RMS:           " f"{block['input_rms']:.6f}")

        print(f"  output RMS:          " f"{block['output_rms']:.6f}")

        print(f"  residual Δ RMS:      " f"{block['delta_rms']:.6f}")

        print(f"  Δ / input:           " f"{100 * block['relative']:.3f}%")

        print(f"  input/output cosine: " f"{block['cosine']:.6f}")

        if modulation:
            print(f"  gate1 RMS:           " f"{modulation['gate1']:.6f}")

            print(f"  gate2 RMS:           " f"{modulation['gate2']:.6f}")

            print(f"  gate1 |mean|:        " f"{modulation['gate1_abs']:.6f}")

            print(f"  gate2 |mean|:        " f"{modulation['gate2_abs']:.6f}")

            print(f"  scale1 RMS:          " f"{modulation['scale1']:.6f}")

            print(f"  scale2 RMS:          " f"{modulation['scale2']:.6f}")

            print(f"  shift1 RMS:          " f"{modulation['shift1']:.6f}")

            print(f"  shift2 RMS:          " f"{modulation['shift2']:.6f}")

        if qkv:
            print(f"  QKV input RMS:       " f"{qkv['input_rms']:.6f}")

            print(f"  QKV output RMS:      " f"{qkv['output_rms']:.6f}")

        if mlp:
            print(f"  MLP input RMS:       " f"{mlp['input_rms']:.6f}")

            print(f"  MLP output RMS:      " f"{mlp['output_rms']:.6f}")

    # -------------------------------------------------------------
    # FIXED: accumulation across depth
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print("DEPTH ACCUMULATION")
    print("=" * 90)

    indices = sorted(block_stats.keys())

    if not indices:
        print("No Transformer block statistics " "were captured.")
        return

    first_index = indices[0]
    last_index = indices[-1]

    first_rms = block_stats[first_index]["input_rms"]

    last_rms = block_stats[last_index]["output_rms"]

    total_ratio = last_rms / (first_rms + 1e-12)

    mean_delta = sum(block_stats[index]["relative"] for index in indices) / len(indices)

    max_index = max(
        indices,
        key=lambda index: (block_stats[index]["relative"]),
    )

    min_index = min(
        indices,
        key=lambda index: (block_stats[index]["relative"]),
    )

    print(f"RMS entering block {first_index}: " f"{first_rms:.6f}")

    print(f"RMS leaving block {last_index}: " f"{last_rms:.6f}")

    print(f"Total RMS ratio:       " f"{total_ratio:.4f}x")

    print(f"Mean per-block Δ/input:" f" {100.0 * mean_delta:.3f}%")

    print(
        f"Strongest block:       "
        f"{max_index} "
        f"({100.0 * block_stats[max_index]['relative']:.3f}%)"
    )

    print(
        f"Mildest block:         "
        f"{min_index} "
        f"({100.0 * block_stats[min_index]['relative']:.3f}%)"
    )

    print()
    print("Interpretation:")

    print(
        """
For comparison, Hybrid v3 gave approximately:

    block 0: Δ/input = 127.5%
    block 1: Δ/input = 341.5%

Hybrid v2 has eight 16x16 Transformer blocks. We want to determine
whether it performs a sequence of smaller refinements instead of the
two enormous representation rewrites seen in Hybrid v3.

Very rough guide:

    < 5%       very mild refinement
    5-20%      moderate contribution
    20-50%     strong contribution
    50-100%    very strong rewrite
    > 100%     residual change exceeds the input RMS

Do not interpret these synthetic-probe values as FID. They are only
architectural diagnostics.
"""
    )

    branch_residual_diagnostic(
        model,
        x,
        timestep,
        category,
        interval,
    )

    channel_pathology_diagnostic(
        model,
        x,
        timestep,
        category,
        interval,
        block_indices=(6, 7),
        topk=20,
    )


def branch_residual_diagnostic(
    backbone,
    x,
    timestep,
    category,
    interval=None,
):
    """
    Measure the exact attention and MLP residual contributions for every
    Transformer block.

    For each block:

        x1  = x0 + gate1 * attn(...)
        out = x1 + gate2 * mlp(...)

    Therefore:

        block_delta = attention_residual + mlp_residual

    This diagnostic captures proj/mlp/modulation outputs with hooks and
    reconstructs those two residual branches exactly.
    """

    print()
    print("=" * 90)
    print("ATTENTION / MLP RESIDUAL CONTRIBUTIONS")
    print("=" * 90)

    blocks = list(backbone.transformer)
    captures = [dict() for _ in blocks]
    handles = []

    def make_block_hook(index):
        def hook(module, inputs, output):
            x_in = inputs[0]

            captures[index]["input"] = x_in.detach().float().cpu()
            captures[index]["output"] = output.detach().float().cpu()

        return hook

    def make_modulation_hook(index):
        def hook(module, inputs, output):
            captures[index]["modulation"] = output.detach().float().cpu()

        return hook

    def make_proj_hook(index):
        def hook(module, inputs, output):
            # This is the attention branch output immediately before
            # gate1 is applied.
            captures[index]["attention_raw"] = output.detach().float().cpu()

        return hook

    def make_mlp_hook(index):
        def hook(module, inputs, output):
            # This is the MLP branch output immediately before
            # gate2 is applied.
            captures[index]["mlp_raw"] = output.detach().float().cpu()

        return hook

    for index, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_block_hook(index)))

        handles.append(
            block.modulation.register_forward_hook(make_modulation_hook(index))
        )

        handles.append(block.proj.register_forward_hook(make_proj_hook(index)))

        handles.append(block.mlp.register_forward_hook(make_mlp_hook(index)))

    was_training = backbone.training
    backbone.eval()

    try:
        with torch.inference_mode():
            kwargs = {
                "category": category,
            }

            if interval is not None:
                kwargs["interval"] = interval

            _ = backbone(
                x,
                timestep,
                **kwargs,
            )

    finally:
        for handle in handles:
            handle.remove()

        backbone.train(was_training)

    def rms(tensor):
        return tensor.square().mean().sqrt().item()

    def cosine(a, b):
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)

        value = torch.nn.functional.cosine_similarity(
            a,
            b,
            dim=1,
            eps=1e-12,
        )

        return value.mean().item()

    rows = []

    for index, capture in enumerate(captures):
        required = {
            "input",
            "output",
            "modulation",
            "attention_raw",
            "mlp_raw",
        }

        missing = required - capture.keys()

        if missing:
            print(f"block {index}: missing captures: " f"{sorted(missing)}")
            continue

        x_in = capture["input"]
        x_out = capture["output"]

        modulation = capture["modulation"]

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

        # [B, width] -> [B, 1, width]
        gate1 = gate1[:, None]
        gate2 = gate2[:, None]

        attention_raw = capture["attention_raw"]
        mlp_raw = capture["mlp_raw"]

        # backbone is in eval mode, so Transformer dropout is identity.
        attention_residual = gate1 * attention_raw

        mlp_residual = gate2 * mlp_raw

        measured_delta = x_out - x_in

        reconstructed_delta = attention_residual + mlp_residual

        reconstruction_error = measured_delta - reconstructed_delta

        input_rms = rms(x_in)

        attn_rms = rms(attention_residual)
        mlp_rms = rms(mlp_residual)

        total_rms = rms(measured_delta)
        reconstructed_rms = rms(reconstructed_delta)

        attn_ratio = 100.0 * attn_rms / input_rms if input_rms > 0 else float("nan")

        mlp_ratio = 100.0 * mlp_rms / input_rms if input_rms > 0 else float("nan")

        total_ratio = 100.0 * total_rms / input_rms if input_rms > 0 else float("nan")

        branch_cosine = cosine(
            attention_residual,
            mlp_residual,
        )

        attn_delta_cosine = cosine(
            attention_residual,
            measured_delta,
        )

        mlp_delta_cosine = cosine(
            mlp_residual,
            measured_delta,
        )

        max_reconstruction_error = reconstruction_error.abs().max().item()

        relative_reconstruction_error = rms(reconstruction_error) / max(
            total_rms, 1e-12
        )

        rows.append(
            {
                "index": index,
                "input_rms": input_rms,
                "attn_rms": attn_rms,
                "mlp_rms": mlp_rms,
                "total_rms": total_rms,
                "reconstructed_rms": reconstructed_rms,
                "attn_ratio": attn_ratio,
                "mlp_ratio": mlp_ratio,
                "total_ratio": total_ratio,
                "branch_cosine": branch_cosine,
                "attn_delta_cosine": attn_delta_cosine,
                "mlp_delta_cosine": mlp_delta_cosine,
                "max_reconstruction_error": (max_reconstruction_error),
                "relative_reconstruction_error": (relative_reconstruction_error),
                "gate1_rms": rms(gate1),
                "gate2_rms": rms(gate2),
                "attention_raw_rms": rms(attention_raw),
                "mlp_raw_rms": rms(mlp_raw),
            }
        )

    print()
    print(
        f"{'blk':>3} "
        f"{'input':>9} "
        f"{'attn Δ':>9} "
        f"{'attn/in':>9} "
        f"{'MLP Δ':>9} "
        f"{'MLP/in':>9} "
        f"{'total Δ':>9} "
        f"{'total/in':>9} "
        f"{'A·M cos':>9}"
    )

    print("-" * 90)

    for row in rows:
        print(
            f"{row['index']:>3} "
            f"{row['input_rms']:>9.4f} "
            f"{row['attn_rms']:>9.4f} "
            f"{row['attn_ratio']:>8.2f}% "
            f"{row['mlp_rms']:>9.4f} "
            f"{row['mlp_ratio']:>8.2f}% "
            f"{row['total_rms']:>9.4f} "
            f"{row['total_ratio']:>8.2f}% "
            f"{row['branch_cosine']:>9.4f}"
        )

    print()
    print("=" * 90)
    print("DETAILED BRANCH DIAGNOSTICS")
    print("=" * 90)

    for row in rows:
        print()
        print(f"Transformer block {row['index']}")

        print(f"  input RMS:                  " f"{row['input_rms']:.6f}")

        print(f"  raw attention output RMS:   " f"{row['attention_raw_rms']:.6f}")

        print(f"  gate1 RMS:                  " f"{row['gate1_rms']:.6f}")

        print(f"  attention residual RMS:     " f"{row['attn_rms']:.6f}")

        print(f"  attention Δ / input:        " f"{row['attn_ratio']:.3f}%")

        print(f"  raw MLP output RMS:         " f"{row['mlp_raw_rms']:.6f}")

        print(f"  gate2 RMS:                  " f"{row['gate2_rms']:.6f}")

        print(f"  MLP residual RMS:           " f"{row['mlp_rms']:.6f}")

        print(f"  MLP Δ / input:              " f"{row['mlp_ratio']:.3f}%")

        print(f"  total block Δ RMS:          " f"{row['total_rms']:.6f}")

        print(f"  total Δ / input:            " f"{row['total_ratio']:.3f}%")

        print(f"  attention/MLP cosine:       " f"{row['branch_cosine']:.6f}")

        print(f"  attention/total Δ cosine:   " f"{row['attn_delta_cosine']:.6f}")

        print(f"  MLP/total Δ cosine:         " f"{row['mlp_delta_cosine']:.6f}")

        print(
            f"  reconstruction max error:   " f"{row['max_reconstruction_error']:.3e}"
        )

        print(
            f"  reconstruction relative RMS:"
            f" {100 * row['relative_reconstruction_error']:.6f}%"
        )

    if rows:
        print()
        print("=" * 90)
        print("BRANCH SUMMARY")
        print("=" * 90)

        strongest_attn = max(
            rows,
            key=lambda row: row["attn_ratio"],
        )

        strongest_mlp = max(
            rows,
            key=lambda row: row["mlp_ratio"],
        )

        print(
            "Strongest attention branch: "
            f"block {strongest_attn['index']} "
            f"({strongest_attn['attn_ratio']:.3f}% of input RMS)"
        )

        print(
            "Strongest MLP branch:       "
            f"block {strongest_mlp['index']} "
            f"({strongest_mlp['mlp_ratio']:.3f}% of input RMS)"
        )

        print()
        print("A·M cosine interpretation:")
        print("  > 0  attention and MLP residuals reinforce each other")
        print("  ~ 0  branches modify mostly independent directions")
        print("  < 0  branches partially cancel each other")

        worst_reconstruction = max(
            rows,
            key=lambda row: row["relative_reconstruction_error"],
        )

        print()
        print(
            "Worst residual reconstruction error: "
            f"block {worst_reconstruction['index']} = "
            f"{100 * worst_reconstruction['relative_reconstruction_error']:.6f}%"
        )

        if worst_reconstruction["relative_reconstruction_error"] < 1e-5:
            print(
                "✓ attention Δ + MLP Δ exactly reconstructs "
                "the measured block residual."
            )
        else:
            print(
                "⚠ Branch reconstruction is not exact. "
                "Check dropout or block implementation."
            )

    return rows


def channel_pathology_diagnostic(
    backbone,
    x,
    timestep,
    category,
    interval=None,
    block_indices=(6, 7),
    topk=20,
):
    """
    Diagnose channel-wise MLP/gate pathology in selected Transformer blocks.

    Answers:
      1. Are a few channels responsible for most gated MLP residual energy?
      2. Do large gate2 channels coincide with large MLP channels?
      3. How strong is gate/MLP alignment?
      4. Is the explosion specific to certain samples/timesteps/categories?
    """

    print()
    print("=" * 100)
    print("CHANNEL-WISE MLP / GATE PATHOLOGY")
    print("=" * 100)

    captures = {}
    handles = []

    valid_indices = [i for i in block_indices if 0 <= i < len(backbone.transformer)]

    if not valid_indices:
        print("No valid Transformer block indices.")
        return {}

    # ------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------

    for index in valid_indices:
        block = backbone.transformer[index]

        captures[index] = {}

        def modulation_hook(
            module,
            inputs,
            output,
            index=index,
        ):
            captures[index]["modulation"] = output.detach().float().cpu()

        def mlp_hook(
            module,
            inputs,
            output,
            index=index,
        ):
            captures[index]["mlp_raw"] = output.detach().float().cpu()

        handles.append(block.modulation.register_forward_hook(modulation_hook))

        handles.append(block.mlp.register_forward_hook(mlp_hook))

    was_training = backbone.training
    backbone.eval()

    try:
        with torch.inference_mode():
            kwargs = {
                "category": category,
            }

            if interval is not None:
                kwargs["interval"] = interval

            _ = backbone(
                x,
                timestep,
                **kwargs,
            )

    finally:
        for handle in handles:
            handle.remove()

        backbone.train(was_training)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def local_rms(tensor):
        tensor = tensor.float()
        return tensor.square().mean().sqrt().item()

    def pearson(a, b):
        a = a.float().flatten()
        b = b.float().flatten()

        a = a - a.mean()
        b = b - b.mean()

        denom = a.square().mean().sqrt() * b.square().mean().sqrt()

        if denom.item() < 1e-12:
            return float("nan")

        return ((a * b).mean() / denom).item()

    def ranks(x):
        x = x.float().flatten()

        order = torch.argsort(x)

        result = torch.empty(
            x.numel(),
            dtype=torch.float32,
        )

        result[order] = torch.arange(
            x.numel(),
            dtype=torch.float32,
        )

        return result

    def spearman(a, b):
        return pearson(
            ranks(a),
            ranks(b),
        )

    def quantiles(x):
        x = x.float().flatten()

        qs = torch.tensor(
            [
                0.50,
                0.90,
                0.95,
                0.99,
                0.999,
            ]
        )

        values = torch.quantile(
            x,
            qs,
        )

        return {
            "p50": values[0].item(),
            "p90": values[1].item(),
            "p95": values[2].item(),
            "p99": values[3].item(),
            "p99.9": values[4].item(),
            "max": x.max().item(),
        }

    def print_quantiles(
        label,
        values,
    ):
        q = quantiles(values)

        print(
            f"  {label:24s}"
            f" p50={q['p50']:.5f}"
            f" p90={q['p90']:.5f}"
            f" p95={q['p95']:.5f}"
            f" p99={q['p99']:.5f}"
            f" p99.9={q['p99.9']:.5f}"
            f" max={q['max']:.5f}"
        )

    def concentration(
        energy,
        counts=(1, 5, 10, 20, 50),
    ):
        energy = energy.float()

        sorted_energy = torch.sort(
            energy,
            descending=True,
        ).values

        total = sorted_energy.sum().item() + 1e-12

        result = {}

        for count in counts:
            count = min(
                count,
                sorted_energy.numel(),
            )

            result[count] = 100.0 * sorted_energy[:count].sum().item() / total

        return result

    def make_ranks(values):
        order = torch.argsort(
            values,
            descending=True,
        )

        result = torch.empty(
            values.numel(),
            dtype=torch.long,
        )

        result[order] = (
            torch.arange(
                values.numel(),
            )
            + 1
        )

        return result

    results = {}

    # ------------------------------------------------------------
    # Analyze each block
    # ------------------------------------------------------------

    for index in valid_indices:
        capture = captures[index]

        if "modulation" not in capture or "mlp_raw" not in capture:
            print(f"\nBlock {index}: " "required hooks did not fire.")
            continue

        modulation = capture["modulation"]
        mlp_raw = capture["mlp_raw"]

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

        # gate2: [B, C]
        # mlp_raw: [B, T, C]
        # residual: [B, T, C]

        residual = gate2[:, None, :] * mlp_raw

        batch, tokens, channels = residual.shape

        # ========================================================
        # Global RMS
        # ========================================================

        gate_global_rms = local_rms(gate2)

        mlp_global_rms = local_rms(mlp_raw)

        residual_global_rms = local_rms(residual)

        # If gate and MLP were uncorrelated/unstructured,
        # this would be roughly ~1.
        alignment_amplification = residual_global_rms / (
            gate_global_rms * mlp_global_rms + 1e-12
        )

        # ========================================================
        # Per-channel statistics
        # ========================================================

        gate_channel_rms = gate2.square().mean(dim=0).sqrt()

        gate_channel_abs_mean = gate2.abs().mean(dim=0)

        gate_channel_abs_max = gate2.abs().amax(dim=0)

        mlp_channel_rms = mlp_raw.square().mean(dim=(0, 1)).sqrt()

        residual_channel_energy = residual.square().mean(dim=(0, 1))

        residual_channel_rms = residual_channel_energy.sqrt()

        # Per-channel alignment factor.
        channel_alignment = residual_channel_rms / (
            gate_channel_rms * mlp_channel_rms + 1e-12
        )

        # ========================================================
        # Correlations
        # ========================================================

        gate_mlp_pearson = pearson(
            gate_channel_rms,
            mlp_channel_rms,
        )

        gate_residual_pearson = pearson(
            gate_channel_rms,
            residual_channel_rms,
        )

        mlp_residual_pearson = pearson(
            mlp_channel_rms,
            residual_channel_rms,
        )

        gate_mlp_spearman = spearman(
            gate_channel_rms,
            mlp_channel_rms,
        )

        gate_residual_spearman = spearman(
            gate_channel_rms,
            residual_channel_rms,
        )

        mlp_residual_spearman = spearman(
            mlp_channel_rms,
            residual_channel_rms,
        )

        # ========================================================
        # Energy concentration
        # ========================================================

        energy_concentration = concentration(residual_channel_energy)

        # ========================================================
        # Channel rankings
        # ========================================================

        gate_rank = make_ranks(gate_channel_rms)

        mlp_rank = make_ranks(mlp_channel_rms)

        residual_rank = make_ranks(residual_channel_rms)

        actual_topk = min(
            topk,
            channels,
        )

        top_residual_channels = torch.topk(
            residual_channel_rms,
            actual_topk,
        ).indices

        # ========================================================
        # Print block-level summary
        # ========================================================

        print()
        print("=" * 100)
        print(f"TRANSFORMER BLOCK {index}")
        print("=" * 100)

        print()
        print("GLOBAL SCALE")

        print(f"  gate2 RMS:                " f"{gate_global_rms:.6f}")

        print(f"  raw MLP RMS:              " f"{mlp_global_rms:.6f}")

        print(f"  gated MLP residual RMS:   " f"{residual_global_rms:.6f}")

        print(f"  alignment amplification: " f"{alignment_amplification:.3f}x")

        print()
        print("  amplification = " "RMS(gate × MLP) / " "(RMS(gate) × RMS(MLP))")

        print("  ~1 = roughly independent")

        print("  >>1 = large gates and large " "MLP activations strongly coincide")

        # ========================================================
        # Distribution tails
        # ========================================================

        print()
        print("MAGNITUDE DISTRIBUTIONS")

        print_quantiles(
            "|gate2|",
            gate2.abs(),
        )

        print_quantiles(
            "|raw MLP activation|",
            mlp_raw.abs(),
        )

        print_quantiles(
            "|gated residual|",
            residual.abs(),
        )

        print()
        print("PER-CHANNEL RMS DISTRIBUTIONS")

        print_quantiles(
            "gate channel RMS",
            gate_channel_rms,
        )

        print_quantiles(
            "MLP channel RMS",
            mlp_channel_rms,
        )

        print_quantiles(
            "residual channel RMS",
            residual_channel_rms,
        )

        print_quantiles(
            "channel alignment",
            channel_alignment,
        )

        # ========================================================
        # Correlations
        # ========================================================

        print()
        print("CHANNEL CORRELATIONS")

        print(
            f"  gate ↔ MLP RMS:"
            f"       Pearson={gate_mlp_pearson:+.4f}"
            f"  Spearman={gate_mlp_spearman:+.4f}"
        )

        print(
            f"  gate ↔ residual RMS:"
            f"  Pearson={gate_residual_pearson:+.4f}"
            f"  Spearman={gate_residual_spearman:+.4f}"
        )

        print(
            f"  MLP ↔ residual RMS:"
            f"   Pearson={mlp_residual_pearson:+.4f}"
            f"  Spearman={mlp_residual_spearman:+.4f}"
        )

        # ========================================================
        # Energy concentration
        # ========================================================

        print()
        print("RESIDUAL ENERGY CONCENTRATION")

        for count, fraction in energy_concentration.items():
            print(f"  top {count:2d} channels:" f" {fraction:7.3f}%")

        # ========================================================
        # Top pathological channels
        # ========================================================

        print()
        print(f"TOP {actual_topk} CHANNELS " "BY GATED-MLP RESIDUAL RMS")

        print(
            f"{'ch':>4} "
            f"{'res RMS':>10} "
            f"{'gate RMS':>10} "
            f"{'MLP RMS':>10} "
            f"{'align':>9} "
            f"{'gate rk':>8} "
            f"{'MLP rk':>7} "
            f"{'energy%':>9}"
        )

        print("-" * 82)

        total_energy = residual_channel_energy.sum().item() + 1e-12

        for channel_tensor in top_residual_channels:
            channel = int(channel_tensor.item())

            energy_pct = 100.0 * residual_channel_energy[channel].item() / total_energy

            print(
                f"{channel:4d} "
                f"{residual_channel_rms[channel]:10.4f} "
                f"{gate_channel_rms[channel]:10.4f} "
                f"{mlp_channel_rms[channel]:10.4f} "
                f"{channel_alignment[channel]:9.3f} "
                f"{int(gate_rank[channel]):8d} "
                f"{int(mlp_rank[channel]):7d} "
                f"{energy_pct:8.3f}%"
            )

        # ========================================================
        # Top-set overlap
        # ========================================================

        print()
        print("TOP-CHANNEL OVERLAP")

        for count in (
            5,
            10,
            20,
            50,
        ):
            count = min(
                count,
                channels,
            )

            gate_set = set(
                torch.topk(
                    gate_channel_rms,
                    count,
                ).indices.tolist()
            )

            mlp_set = set(
                torch.topk(
                    mlp_channel_rms,
                    count,
                ).indices.tolist()
            )

            residual_set = set(
                torch.topk(
                    residual_channel_rms,
                    count,
                ).indices.tolist()
            )

            res_gate = len(residual_set & gate_set)

            res_mlp = len(residual_set & mlp_set)

            all_three = len(residual_set & gate_set & mlp_set)

            print(
                f"  top {count:2d}: "
                f"res∩gate={res_gate:2d}/{count}   "
                f"res∩MLP={res_mlp:2d}/{count}   "
                f"all-three={all_three:2d}/{count}"
            )

        # ========================================================
        # Per-sample diagnosis
        # ========================================================

        print()
        print("PER-SAMPLE DIAGNOSIS")

        print(
            f"{'n':>2} "
            f"{'t':>7} "
            f"{'cat':>5} "
            f"{'gate':>9} "
            f"{'MLP':>9} "
            f"{'resid':>9} "
            f"{'align':>9} "
            f"{'top10 E':>9}"
        )

        print("-" * 80)

        timestep_cpu = timestep.detach().float().cpu()

        category_cpu = category.detach().cpu()

        for sample in range(batch):
            sample_gate = gate2[sample]

            sample_mlp = mlp_raw[sample]

            sample_residual = residual[sample]

            sample_gate_rms = local_rms(sample_gate)

            sample_mlp_rms = local_rms(sample_mlp)

            sample_residual_rms = local_rms(sample_residual)

            sample_alignment = sample_residual_rms / (
                sample_gate_rms * sample_mlp_rms + 1e-12
            )

            # Energy per channel for this sample.
            sample_energy = sample_residual.square().mean(dim=0)

            sorted_energy = torch.sort(
                sample_energy,
                descending=True,
            ).values

            count = min(
                10,
                channels,
            )

            sample_top10 = (
                100.0
                * sorted_energy[:count].sum().item()
                / (sorted_energy.sum().item() + 1e-12)
            )

            print(
                f"{sample:2d} "
                f"{timestep_cpu[sample].item():7.3f} "
                f"{int(category_cpu[sample]):5d} "
                f"{sample_gate_rms:9.4f} "
                f"{sample_mlp_rms:9.4f} "
                f"{sample_residual_rms:9.4f} "
                f"{sample_alignment:9.2f} "
                f"{sample_top10:8.2f}%"
            )

        results[index] = {
            "gate_rms": gate_global_rms,
            "mlp_rms": mlp_global_rms,
            "residual_rms": residual_global_rms,
            "alignment_amplification": (alignment_amplification),
            "energy_concentration": (energy_concentration),
            "gate_mlp_pearson": (gate_mlp_pearson),
            "gate_residual_pearson": (gate_residual_pearson),
            "mlp_residual_pearson": (mlp_residual_pearson),
        }

    # ------------------------------------------------------------
    # Final interpretation hints
    # ------------------------------------------------------------

    print()
    print("=" * 100)
    print("WHAT TO LOOK FOR")
    print("=" * 100)

    print(
        """
1. ALIGNMENT AMPLIFICATION
   ~1-2x:
       gate and MLP magnitudes are not strongly aligned.

   >5x:
       suspicious channel/sample alignment.

   >10x:
       severe alignment pathology.

2. RESIDUAL ENERGY CONCENTRATION
   If top 5-20 / 640 channels contain a huge fraction of energy,
   the blow-up is channel-localized rather than a general FFN scale issue.

3. TOP-CHANNEL OVERLAP
   If residual-top channels are also gate-top AND MLP-top channels,
   adaLN gating is amplifying exactly the already-large FFN channels.

4. PER-SAMPLE DIAGNOSIS
   If only one timestep/sample explodes, the issue may be conditioning/time
   dependent rather than universally architectural.

5. BLOCK 6 -> BLOCK 7
   If the same channels appear in both blocks, the pathology is propagating
   through depth rather than appearing independently in block 7.
"""
    )

    return results


# =====================================================================
# MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description=("Inspect a trained Hybrid v2 FastGen checkpoint.")
    )

    parser.add_argument(
        "checkpoint",
        help="Checkpoint to inspect.",
    )

    parser.add_argument(
        "--forward",
        action="store_true",
        help=("Load HybridV2FlowNet and measure " "actual Transformer activity."),
    )

    parser.add_argument(
        "--compare",
        help=(
            "Optional second checkpoint. " "First checkpoint is OLD, --compare is NEW."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help=("Number of parameter movements " "to show with --compare."),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=("Synthetic diagnostic batch size."),
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

    print_parameter_breakdown(state)

    inspect_adaln(state)

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
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
