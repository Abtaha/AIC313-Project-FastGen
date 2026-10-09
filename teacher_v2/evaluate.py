"""Independent V2 FID using the population and PNG conversion in evaluate.py."""

import argparse
import json
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

SAMPLES_PER_CATEGORY = 20
FID_PROTOCOL = "fastgen-full-validation-balanced20-cleanfid-v1"


def export_image(image, path):
    image = ((image.float().cpu() + 1) / 2).clamp(0, 1)
    to_pil_image(image).convert("RGB").save(path)


@torch.no_grad()
def export_reference(dataset, directory, *, num_classes, batch_size):
    counts = torch.zeros(num_classes, dtype=torch.long)
    count = 0
    for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        for image, label in zip(images, labels.tolist()):
            if not 0 <= label < num_classes:
                raise ValueError("Validation category is outside the model mapping")
            export_image(image, directory / f"{count:04d}.png")
            counts[label] += 1
            count += 1
    if not (counts == SAMPLES_PER_CATEGORY).all():
        raise ValueError("Competition FID requires the complete validation set: 20 real images per category")
    return count


@torch.no_grad()
def export_generated(model, directory, *, num_classes, batch_size, device, steps,
                     guidance, seed):
    labels = torch.arange(num_classes, dtype=torch.long).repeat_interleave(SAMPLES_PER_CATEGORY)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for start in range(0, len(labels), batch_size):
        categories = labels[start:start + batch_size].to(device)
        samples = model.sample((len(categories), 3, 64, 64), device=device, category=categories,
                               num_steps=steps, guidance=guidance, generator=generator)
        if samples.shape != (len(categories), 3, 64, 64) or not torch.isfinite(samples).all():
            raise ValueError("Teacher returned invalid FID samples")
        for offset, (image, label) in enumerate(zip(samples, categories.tolist())):
            # Same category-aware filenames and quantization as evaluate.py.
            image_index = (start + offset) % SAMPLES_PER_CATEGORY
            export_image(image, directory / f"{label:04d}_{image_index:02d}.png")
    if len(list(directory.glob("*.png"))) != len(labels):
        raise RuntimeError("Generated population is incomplete")
    return len(labels)


@torch.no_grad()
def competition_fid(model, validation_dataset, *, num_classes, outdir, device,
                    batch_size=32, steps=18, guidance=1.0, seed=0):
    from cleanfid import fid

    if batch_size < 1 or num_classes < 1:
        raise ValueError("batch_size and num_classes must be positive")
    if model.config.channels != 3 or model.config.resolution != 64:
        raise ValueError("Competition FID requires 64x64 RGB samples")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # A fresh pair of directories ensures old files cannot enter the pooled score.
    with tempfile.TemporaryDirectory(prefix="v2-fid-", dir=outdir) as temporary:
        reference, generated = Path(temporary) / "reference", Path(temporary) / "generated"
        reference.mkdir()
        generated.mkdir()
        real_count = export_reference(validation_dataset, reference, num_classes=num_classes,
                                      batch_size=batch_size)
        generated_count = export_generated(model, generated, num_classes=num_classes,
                                            batch_size=batch_size, device=device, steps=steps,
                                            guidance=guidance, seed=seed)
        print(f"FID: {generated_count} generated images, {real_count} full-validation reference images", flush=True)
        # One pooled score, matching the clean-fid call in evaluate.py.
        return float(fid.compute_fid(str(generated), str(reference), device=device,
                                    batch_size=batch_size, num_workers=4))


def main(args):
    from dataset import PokemonDataset
    from .model import EDMTeacherV2

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    mapping = json.loads((Path(args.split_dir) / "category_to_id.json").read_text())
    if checkpoint.get("category_to_id") != mapping:
        raise ValueError("Checkpoint and dataset category mappings differ")
    teacher = EDMTeacherV2.load_checkpoint(args.checkpoint, device=args.device)
    validation = PokemonDataset(args.data_root, Path(args.split_dir) / "val_split.txt",
                                return_category=True, category_to_id=mapping)
    score = competition_fid(teacher, validation, num_classes=len(mapping), outdir=args.outdir,
                            device=torch.device(args.device), batch_size=args.batch_size,
                            steps=args.steps, guidance=args.guidance, seed=args.seed)
    result = {"checkpoint": args.checkpoint, "competition_fid": score,
              "fid_protocol": FID_PROTOCOL, "count": len(mapping) * SAMPLES_PER_CATEGORY,
              "steps": args.steps, "guidance": args.guidance, "seed": args.seed,
              "nfe": (2 * args.steps - 1) * (2 if args.guidance != 1 else 1)}
    (Path(args.outdir) / "fid.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/edm2-112/best-fid.ckpt")
    p.add_argument("--data-root", default="data/pokemon-generation-one-22k")
    p.add_argument("--split-dir", default="data/pokemon-generation-one-22k")
    p.add_argument("--outdir", default="results/teacher-v2-fid")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=18)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
