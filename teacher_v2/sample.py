"""Export class-conditional samples from a locally trained EMA teacher."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from .model import EDMTeacherV2


def main(args):
    teacher = EDMTeacherV2.load_checkpoint(args.checkpoint, device=args.device)
    if not 0 <= args.category < teacher.config.num_classes:
        raise ValueError("category is outside the teacher's category mapping")
    if args.count < 1 or args.batch_size < 1:
        raise ValueError("count and batch-size must be positive")
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    for start in range(0, args.count, args.batch_size):
        count = min(args.batch_size, args.count - start)
        labels = torch.full((count,), args.category, device=args.device, dtype=torch.long)
        samples = teacher.sample(
            (count, teacher.config.channels, teacher.config.resolution, teacher.config.resolution),
            device=args.device, category=labels, num_steps=args.steps,
            solver=args.solver, generator=generator, guidance=args.guidance,
        )
        for offset, image in enumerate(samples):
            save_image((image.cpu() + 1) / 2, output / f"{start + offset:06d}.png")
    (output / "sampling.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    nfe = (2 * args.steps - 1 if args.solver == "heun" else args.steps) * (2 if args.guidance != 1 else 1)
    print(f"Saved {args.count} samples to {output}; {nfe} NFE per image")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/edm2-112/teacher.ckpt")
    p.add_argument("--outdir", default="results/teacher-v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--category", type=int, default=0)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--solver", choices=("euler", "heun"), default="heun")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=1.0)
    main(p.parse_args())
