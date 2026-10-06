"""Compare guided EMA teachers using training-holdout FID and paired seeds."""

import argparse
import json
from pathlib import Path

import torch

from dataset import PokemonDataset
from teacher import EDMTeacher
from teacher_monitor import holdout_fid, split_fingerprint


def main(args):
    run = Path(args.run)
    checkpoint_path = args.checkpoint or str(run / "best-fid.ckpt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    teacher = EDMTeacher.load_checkpoint(checkpoint_path, device=args.device)
    config = json.loads((run / "config.json").read_text())
    split = json.loads((run / "monitor-split.json").read_text())
    data_root = args.data_root or config["training"]["data_root"]
    split_dir = Path(args.split_dir or config["training"]["split_dir"])
    mapping = json.loads((split_dir / "category_to_id.json").read_text())
    if checkpoint["category_to_id"] != mapping:
        raise ValueError("Dataset category mapping differs from checkpoint")
    dataset = PokemonDataset(data_root, split_dir / "train_split.txt",
                             return_category=True, category_to_id=mapping)
    if split_fingerprint(dataset) != split["train_manifest_sha256"]:
        raise ValueError("Training manifest differs from monitoring split")
    scores = []
    for guidance in args.guidance_values:
        print(f"Evaluating guidance={guidance}", flush=True)
        score = holdout_fid(
            teacher, dataset, split["holdout_indices"], checkpoint["class_probabilities"],
            outdir=run, device=torch.device(args.device), batch_size=args.batch_size,
            count=args.count, steps=args.steps, guidance=guidance, seed=args.seed,
        )
        scores.append({"guidance": guidance, "holdout_fid": score})
        print(scores[-1], flush=True)
    payload = {"checkpoint": checkpoint_path, "count": args.count, "steps": args.steps,
               "seed": args.seed, "scores": scores,
               "best_guidance": min(scores, key=lambda row: row["holdout_fid"])["guidance"]}
    (run / "guidance-sweep.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="checkpoints/teacher")
    p.add_argument("--checkpoint")
    p.add_argument("--data-root")
    p.add_argument("--split-dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--count", type=int, default=5000)
    p.add_argument("--steps", type=int, default=18)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--guidance-values", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    main(p.parse_args())
