"""Full, balanced clean-FID checks using the provided image export routines."""

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import torch
from cleanfid import fid

from evaluate import SAMPLES_PER_CATEGORY, generate_samples, prepare_reference_set


def reference_set(data, work_dir, worker_init_fn=None):
    """Cache only a complete reference export matching this validation dataset."""
    count = len(data.val_dataset)
    if count < 2:
        raise ValueError("FID requires at least two validation images")
    signature = hashlib.sha256(json.dumps({
        "version": "rgb64-normalized-png-v1",
        "root": str(Path(getattr(data, "data_root", ".")).resolve()),
        "paths": data.val_dataset.paths,
        "transform": repr(getattr(data.val_dataset, "transform", None)),
        "categories": data.category_to_id,
    }, sort_keys=True).encode()).hexdigest()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    destination = work_dir / f"reference_{signature[:16]}"
    marker = destination / "ready.json"
    expected = {f"{i:04d}.png" for i in range(count)} | {"ready.json"}
    try:
        ready = json.loads(marker.read_text())
        if ready == {"signature": signature, "count": count} and {
            path.name for path in destination.iterdir()
        } == expected:
            return destination
    except (OSError, ValueError):
        pass
    # Build outside the final directory so an interrupted export is never reused.
    with tempfile.TemporaryDirectory(prefix="reference_tmp_", dir=work_dir) as tmp:
        exported = Path(tmp) / "images"
        loader = data.val_dataloader()
        loader.worker_init_fn = worker_init_fn
        loader.generator = torch.Generator().manual_seed(0)
        prepare_reference_set(argparse.Namespace(val_dataloader=lambda: loader), exported)
        if {path.name for path in exported.iterdir()} != expected - {"ready.json"}:
            raise RuntimeError("Incomplete FID reference export")
        (exported / "ready.json").write_text(json.dumps({"signature": signature, "count": count}))
        if destination.exists():
            shutil.rmtree(destination)
        exported.replace(destination)
    return destination


def run_fid_check(model, data, work_dir, *, batch_size=32, seed=1234, worker_init_fn=None):
    """20 images/category against all validation images, matching official FID.

    The caller preserves training RNG and mode. Generated PNGs are temporary;
    each check starts empty, so stale samples cannot contaminate the score.
    """
    sampling_device = next(model.parameters()).device
    metric_device = sampling_device if sampling_device.type == "cuda" else torch.device("cpu")
    reference_dir = reference_set(data, work_dir, worker_init_fn)
    torch.manual_seed(seed)
    categories = torch.arange(len(data.category_to_id), dtype=torch.long).repeat_interleave(SAMPLES_PER_CATEGORY)
    with tempfile.TemporaryDirectory(prefix="generated_", dir=work_dir) as tmp:
        generated_dir = generate_samples(model, tmp, categories, batch_size=batch_size, device=sampling_device)
        if len(list(generated_dir.glob("*.png"))) != len(categories):
            raise RuntimeError("Incomplete FID sample generation")
        score = float(fid.compute_fid(
            str(generated_dir), str(reference_dir), mode="clean",
            model_name="inception_v3", device=metric_device, batch_size=batch_size,
            # Single-GPU course setup; no worker subprocess inherits stop handlers.
            num_workers=0, use_dataparallel=False,
        ))
    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite FID score: {score}")
    return {"fid": score, "generated_images": len(categories),
            "reference_images": len(data.val_dataset), "samples_per_category": SAMPLES_PER_CATEGORY,
            "seed": seed, "fid_device": str(metric_device)}
