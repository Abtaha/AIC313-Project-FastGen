"""Train-only sampling and pixel statistics for teacher experiments."""

import torch
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from torch.utils.data import WeightedRandomSampler, DataLoader, Subset


def balanced_sampler(category_ids, *, num_classes, generator):
    """P(image i)=1/(number_of_classes * number_of_images_in_class_i)."""
    labels = torch.as_tensor(category_ids, dtype=torch.long)
    if labels.numel() == 0:
        raise ValueError("Cannot balance an empty training set")
    counts = torch.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or (counts == 0).any():
        raise ValueError("Every configured category needs fitting images for class balancing")
    weights = counts[labels].double().reciprocal()
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=generator)


@torch.no_grad()
def pixel_statistics(loader):
    count = 0
    sums = squares = None
    for batch in loader:
        images = batch[0].double()
        s = images.sum(dim=(0, 2, 3))
        sq = images.square().sum(dim=(0, 2, 3))
        sums = s if sums is None else sums + s
        squares = sq if squares is None else squares + sq
        count += len(images) * images.shape[2] * images.shape[3]
    if not count:
        raise ValueError("No images for statistics")
    mean = sums / count
    second = squares / count
    return {"channel_mean": mean.tolist(), "channel_rms": second.sqrt().tolist(),
            "channel_std": (second - mean.square()).clamp_min(0).sqrt().tolist(),
            "global_mean": float(mean.mean()), "global_rms": float(second.mean().sqrt()),
            "global_std": float((second.mean() - mean.mean().square()).clamp_min(0).sqrt())}


def stratified_holdout(category_ids, fraction, seed):
    if not 0 <= fraction < 1:
        raise ValueError("holdout-fraction must be in [0, 1)")
    groups = defaultdict(list)
    for index, category in enumerate(category_ids):
        groups[category].append(index)
    generator = torch.Generator().manual_seed(seed)
    train, holdout = [], []
    for category in sorted(groups):
        indices = groups[category]
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[i] for i in order]
        count = min(max(round(len(indices) * fraction), 1), len(indices) - 1) if fraction else 0
        holdout.extend(indices[:count])
        train.extend(indices[count:])
    return sorted(train), sorted(holdout)


def split_fingerprint(dataset):
    return hashlib.sha256("\n".join(dataset.paths).encode()).hexdigest()


@torch.no_grad()
def denoising_losses(model, dataset, indices, *, device, batch_size, seed, limit=512):
    """Fixed noise, fixed sigma evaluation; no label dropout or augmentation."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    # Choose a fixed random subset instead of the first (category-sorted) rows.
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    selected = [indices[i] for i in order[:limit].tolist()]
    loader = DataLoader(Subset(dataset, selected), batch_size=batch_size, num_workers=0)
    totals = {sigma: 0.0 for sigma in (0.1, 0.5, 2.0, 10.0)}
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        noise = torch.randn(images.shape, device=device, generator=generator)
        for sigma in totals:
            denoised = model(images + sigma * noise, sigma, labels)
            weight = (sigma ** 2 + model.config.sigma_data ** 2) / (sigma * model.config.sigma_data) ** 2
            totals[sigma] += float((weight * (denoised - images).square()).mean()) * len(images)
        count += len(images)
    if not count:
        raise ValueError("Loss monitoring requires nonempty indices")
    return {str(sigma): value / count for sigma, value in totals.items()}


def save_monitor_split(dataset, train_indices, holdout_indices, output):
    payload = {"train_manifest_sha256": split_fingerprint(dataset),
               "fit_indices": train_indices, "holdout_indices": holdout_indices,
               "holdout_paths": [dataset.paths[i] for i in holdout_indices]}
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")


def main(args):
    from dataset import PokemonDataset

    split_dir = Path(args.split_dir)
    mapping = json.loads((split_dir / "category_to_id.json").read_text())
    dataset = PokemonDataset(args.data_root, split_dir / "train_split.txt",
                             return_category=True, category_to_id=mapping)
    fit, _ = stratified_holdout(dataset.category_ids, args.holdout_fraction, args.seed)
    result = pixel_statistics(DataLoader(Subset(dataset, fit), batch_size=args.batch_size,
                                        num_workers=args.num_workers))
    counts = {name: sum(dataset.category_ids[i] == category for i in fit)
              for name, category in mapping.items()}
    result.update(image_count=len(fit), class_counts=counts, holdout_fraction=args.holdout_fraction,
                  seed=args.seed, normalization="RGB [-1,1] after provided dataset transform")
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/pokemon-generation-one-22k")
    p.add_argument("--split-dir", default="data/pokemon-generation-one-22k")
    p.add_argument("--holdout-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output")
    main(p.parse_args())
