"""Teacher feedback using a fixed holdout from the permitted training manifest.

The official val_split.txt is never opened for selection. FID uses clean-fid,
already included in the provided requirements, and its evaluation Inception.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image


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


@torch.no_grad()
def holdout_fid(model, dataset, holdout_indices, class_probabilities, *, outdir,
                device, batch_size=16, count=5000, steps=18, guidance=1.0, seed=1234):
    from cleanfid import fid

    if count < 2 or len(holdout_indices) < 2:
        raise ValueError("FID requires at least two reference and generated images")
    model.eval()
    # Temporary directories prevent stale samples contaminating future scores.
    with tempfile.TemporaryDirectory(prefix="teacher-fid-", dir=outdir) as directory:
        reference = Path(directory) / "reference"
        generated = Path(directory) / "generated"
        reference.mkdir()
        generated.mkdir()
        loader = DataLoader(Subset(dataset, holdout_indices), batch_size=batch_size, num_workers=0)
        index = 0
        for images, _ in loader:
            for image in images:
                save_image((image + 1) / 2, reference / f"{index:06d}.png")
                index += 1
        # Reset the seed for comparable labels and noise across guidance sweeps.
        label_generator = torch.Generator().manual_seed(seed)
        labels = torch.multinomial(class_probabilities.cpu(), count,
                                   replacement=True, generator=label_generator)
        noise_generator = torch.Generator(device=device).manual_seed(seed)
        for start in range(0, count, batch_size):
            categories = labels[start:start + batch_size].to(device)
            images = model.sample(
                (len(categories), model.config.channels, model.config.resolution, model.config.resolution),
                device=device, category=categories, num_steps=steps,
                guidance=guidance, generator=noise_generator,
            )
            for offset, image in enumerate(images):
                save_image((image.cpu() + 1) / 2, generated / f"{start + offset:06d}.png")
        return float(fid.compute_fid(str(generated), str(reference), device=device,
                                    batch_size=batch_size, num_workers=0))


def save_monitor_split(dataset, train_indices, holdout_indices, output):
    payload = {"train_manifest_sha256": split_fingerprint(dataset),
               "fit_indices": train_indices, "holdout_indices": holdout_indices,
               "holdout_paths": [dataset.paths[i] for i in holdout_indices]}
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")
