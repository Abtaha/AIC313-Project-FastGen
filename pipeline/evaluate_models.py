"""Report official pooled CleanFID using fresh reference and generated directories."""
import argparse
import json
from pathlib import Path
import tempfile

import torch

from _common import ROOT
from dataset import PokemonDataModule
from evaluate import evaluate, prepare_reference_set
from model import Model
from teacher import EDMTeacher


class TeacherSampler(torch.nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        self.teacher = EDMTeacher.load_checkpoint(checkpoint, device=device)

    def sample(self, shape, *, device, category):
        return self.teacher.sample(shape, device=device, category=category,
                                   num_steps=40, guidance=2.0, solver="heun")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("teacher", "one_nfe", "few_nfe"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/pokemon-generation-one-22k")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/pokemon-generation-one-22k")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError("Use a new evaluation outdir")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.kind == "teacher":
        network = TeacherSampler(args.checkpoint, args.device)
    else:
        network = Model.load_checkpoint(args.checkpoint, evaluate_mode=args.kind, device=args.device)
    if sum(p.numel() for p in network.parameters()) > 100_000_000:
        raise ValueError("Model exceeds 100M")
    metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    module = PokemonDataModule(data_root=args.data_root, split_dir=args.split_dir,
                               batch_size=args.batch_size, num_workers=2)
    if module.category_to_id != metadata["category_to_id"]:
        raise ValueError("Class mapping differs from checkpoint")
    categories = torch.arange(len(module.category_to_id)).repeat_interleave(20)
    with tempfile.TemporaryDirectory(dir=args.outdir, prefix="fid-") as work:
        reference = prepare_reference_set(module, Path(work) / "reference")
        if len(list(reference.glob("*.png"))) != len(categories):
            raise ValueError("Expected all 3020 official validation references")
        torch.manual_seed(args.seed)
        score = evaluate(network, reference, Path(work) / "generated", categories,
                         batch_size=args.batch_size, device=args.device)
    result = {"kind": args.kind, "competition_fid": score, "seed": args.seed,
              "batch_size": args.batch_size, "count": len(categories),
              "nfe": {"teacher": 158, "one_nfe": 1, "few_nfe": 4}[args.kind]}
    (args.outdir / "fid.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
