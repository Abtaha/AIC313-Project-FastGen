"""Export the fixed teacher, 1-NFE, and 4-NFE endpoints without training auxiliaries."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from _common import ROOT, RECIPES
from distillation.model import ConsistencyStudent
from distillation.train import atomic_save
from teacher import EDMTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=ROOT / "runs")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--outdir", type=Path, default=ROOT / "submission")
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError("Choose an empty output directory to preserve existing exports")
    sources = {"teacher": args.teacher or args.workdir / "teacher/teacher.ckpt",
               "one_nfe": args.workdir / "one-adversarial/snapshots/one_nfe-005000.ckpt",
               "few_nfe": args.workdir / "few-refine/snapshots/student-002000.ckpt"}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, source in sources.items():
        payload = torch.load(source, map_location="cpu", weights_only=True)
        expected_step = {"teacher": 100000, "one_nfe": 5000, "few_nfe": 2000}[name]
        if payload.get("step") != expected_step:
            raise ValueError(f"{name} must be the fixed endpoint at step {expected_step}")
        if name == "one_nfe" and payload.get("training_recipe") != "DMD2-inspired-gan-v1":
            raise ValueError("One-pass export must come from the completed adversarial recipe")
        if name == "teacher":
            network = EDMTeacher.load_checkpoint(source)
        else:
            network = ConsistencyStudent.load_checkpoint(source)
            if name == "few_nfe":
                network.set_four_sigmas(RECIPES["four_sigmas"])
                payload["state_dict"] = network.export(category_to_id=payload["category_to_id"],
                    step=payload["step"], guidance=payload["teacher_guidance"])["state_dict"]
                payload["sampling_selection"] = "fixed-synthetic-training-pair-selected-grid"
        count = sum(parameter.numel() for parameter in network.parameters())
        if count > 100_000_000:
            raise ValueError(f"{name} exceeds 100M parameters")
        path = args.outdir / (name + ".ckpt")
        atomic_save(payload, path)
        metadata[name] = {"source": str(source.resolve()), "parameters": count,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "export_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.outdir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
