#!/usr/bin/env python3

"""
Evaluate an EDM teacher checkpoint using the competition FID protocol.

Protocol:
  - load official PokemonDataModule validation split
  - export ALL validation images as reference PNGs
  - generate exactly 20 images for every category
  - 151 * 20 = 3020 generated images
  - compute ONE pooled CleanFID score

Important:
  - generated/reference directories are recreated each run
    so stale PNGs cannot contaminate FID.
"""

import argparse
import shutil
from pathlib import Path

import torch
from cleanfid import fid
from tqdm import tqdm

from dataset import PokemonDataModule
from teacher import EDMTeacher


SAMPLES_PER_CATEGORY = 20


def reset_dir(path: Path):
    """Delete directory if it exists, then recreate it."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_reference_set(data_module, reference_dir: Path):
    """Export the official validation split as RGB PNGs."""
    from torchvision.transforms.functional import to_pil_image

    reset_dir(reference_dir)

    count = 0
    category_counts = {}

    print("Exporting official validation reference set...")

    for batch in tqdm(
        data_module.val_dataloader(),
        desc="Reference",
    ):
        if isinstance(batch, (tuple, list)):
            images = batch[0]

            # Usually batch[1] is category.
            labels = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            labels = None

        for i, image in enumerate(images):
            image = ((image.float().cpu() + 1) / 2).clamp(0, 1)

            to_pil_image(image).convert("RGB").save(reference_dir / f"{count:05d}.png")

            if labels is not None:
                category = int(labels[i])
                category_counts[category] = category_counts.get(category, 0) + 1

            count += 1

    print(f"Reference images: {count}")

    if category_counts:
        counts = list(category_counts.values())

        print(
            f"Reference categories: {len(category_counts)} | "
            f"min/class={min(counts)} | "
            f"max/class={max(counts)}"
        )

        if len(category_counts) == 151 and min(counts) == 20 and max(counts) == 20:
            print("Validation set is exactly balanced: " "20 real images/category.")
        else:
            print("NOTE: validation set is not exactly " "20 images/category.")

    return count


@torch.no_grad()
def generate_samples(
    teacher,
    output_dir: Path,
    num_categories: int,
    batch_size: int,
    device: torch.device,
    steps: int,
    solver: str,
    guidance: float,
    seed: int,
):
    """Generate exactly 20 samples for each category."""
    from torchvision.transforms.functional import to_pil_image

    reset_dir(output_dir)

    categories = torch.arange(num_categories, dtype=torch.long).repeat_interleave(
        SAMPLES_PER_CATEGORY
    )

    total = len(categories)

    print()
    print(
        f"Generating {SAMPLES_PER_CATEGORY} samples/category "
        f"x {num_categories} categories "
        f"= {total} images"
    )
    print(f"Steps:    {steps}")
    print(f"Solver:   {solver}")
    print(f"Guidance: {guidance}")
    print(f"Seed:     {seed}")
    print()

    generator = torch.Generator(device=device).manual_seed(seed)

    image_counts = {}

    teacher.eval()

    for start in tqdm(
        range(0, total, batch_size),
        desc="Generating",
    ):
        batch_categories = categories[start : start + batch_size].to(device)

        count = len(batch_categories)

        samples = teacher.sample(
            (count, 3, 64, 64),
            device=device,
            category=batch_categories,
            num_steps=steps,
            solver=solver,
            generator=generator,
            guidance=guidance,
        )

        for sample, category in zip(
            samples,
            batch_categories.tolist(),
        ):
            image_idx = image_counts.get(category, 0)
            image_counts[category] = image_idx + 1

            image = ((sample.float().cpu() + 1) / 2).clamp(0, 1)

            to_pil_image(image).convert("RGB").save(
                output_dir / f"{category:04d}_{image_idx:02d}.png"
            )

    # Sanity checks.
    expected = num_categories * SAMPLES_PER_CATEGORY
    actual = len(list(output_dir.glob("*.png")))

    if actual != expected:
        raise RuntimeError(f"Expected {expected} generated images, " f"found {actual}.")

    bad_categories = {
        c: image_counts.get(c, 0)
        for c in range(num_categories)
        if image_counts.get(c, 0) != SAMPLES_PER_CATEGORY
    }

    if bad_categories:
        raise RuntimeError(f"Incorrect per-category counts: {bad_categories}")

    print(f"Generated exactly {actual} images " f"({SAMPLES_PER_CATEGORY}/category).")


def compute_fid(
    generated_dir: Path,
    reference_dir: Path,
    device: str,
    batch_size: int,
):
    """Same pooled CleanFID used by competition evaluator."""
    print()
    print("=" * 72)
    print("COMPUTING CLEANFID")
    print("=" * 72)

    print(f"Generated images: " f"{len(list(generated_dir.glob('*.png')))}")

    print(f"Reference images: " f"{len(list(reference_dir.glob('*.png')))}")

    score = fid.compute_fid(
        str(generated_dir),
        str(reference_dir),
        device=device,
        batch_size=batch_size,
        num_workers=4,
    )

    return float(score)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("COMPETITION-PROTOCOL EDM TEACHER FID")
    print("=" * 72)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    print()

    # ---------------------------------------------------------
    # Load teacher
    # ---------------------------------------------------------

    teacher = EDMTeacher.load_checkpoint(
        args.checkpoint,
        device=device,
    )

    # ---------------------------------------------------------
    # Dataset / official validation split
    # ---------------------------------------------------------

    data_module = PokemonDataModule(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_categories = len(data_module.category_to_id)

    print(f"Categories: {num_categories}")

    if num_categories != 151:
        print(f"WARNING: expected 151 categories, " f"found {num_categories}.")

    root = Path(args.workdir)

    reference_dir = root / "reference"
    generated_dir = root / "generated"

    # ---------------------------------------------------------
    # Export exact validation reference set
    # ---------------------------------------------------------

    reference_count = prepare_reference_set(
        data_module,
        reference_dir,
    )

    # ---------------------------------------------------------
    # Generate competition-style samples
    # ---------------------------------------------------------

    generate_samples(
        teacher=teacher,
        output_dir=generated_dir,
        num_categories=num_categories,
        batch_size=args.batch_size,
        device=device,
        steps=args.steps,
        solver=args.solver,
        guidance=args.guidance,
        seed=args.seed,
    )

    # ---------------------------------------------------------
    # Pooled FID
    # ---------------------------------------------------------

    score = compute_fid(
        generated_dir,
        reference_dir,
        device=str(device),
        batch_size=args.batch_size,
    )

    # Heun NFE accounting.
    if args.solver == "heun":
        base_nfe = 2 * args.steps - 1
    else:
        base_nfe = args.steps

    if args.guidance != 1.0:
        nfe = 2 * base_nfe
    else:
        nfe = base_nfe

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Reference count : {reference_count}")
    print(f"Generated count : " f"{num_categories * SAMPLES_PER_CATEGORY}")
    print(f"Samples/class   : " f"{SAMPLES_PER_CATEGORY}")
    print(f"Steps           : {args.steps}")
    print(f"Solver          : {args.solver}")
    print(f"Guidance        : {args.guidance}")
    print(f"Approx NFE      : {nfe}")

    print()
    print(f"ACTUAL COMPETITION-PROTOCOL FID: " f"{score:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        default="./data/pokemon-generation-one-22k",
    )

    parser.add_argument(
        "--split-dir",
        default="./data/pokemon-generation-one-22k",
    )

    parser.add_argument(
        "--workdir",
        default="./results/teacher_actual_fid",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--solver",
        choices=("euler", "heun"),
        default="heun",
    )

    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    main(parser.parse_args())
