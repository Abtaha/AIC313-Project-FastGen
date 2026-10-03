"""Train the existing category-conditioned models with linear flow matching.

This is a baseline for both samplers, not one-step distillation. Only the
official training split is used; validation images are reserved for FID.
"""

import argparse
import hashlib
import json
import math
import random
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Sampler
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import PokemonDataModule
from model import ModelFewNFE, ModelOneNFE
from src.models.dit import DiTConfig


def select_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


RESUME_VERSION = 1
_STOP_REQUESTED = False


def request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\nStop requested; saving after the current optimizer step.", flush=True)


def configure_worker(worker_id):
    # Linux fork workers inherit the parent's graceful-stop handlers. They must
    # still terminate normally when DataLoader shuts them down.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def rng_state(device):
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        # Restore corresponding devices; migration may change GPU count.
        for index, value in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value, index)
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


class EpochSampler(Sampler):
    """Rebuild the same shuffle and skip completed batches without reading them."""

    def __init__(self, size, seed, skip=0):
        self.size, self.seed, self.skip = size, seed, skip

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        return iter(torch.randperm(self.size, generator=generator).tolist()[self.skip:])

    def __len__(self):
        return self.size - self.skip


def data_signature(data):
    # Relocation is allowed, but manifest content/order and category IDs must match.
    payload = {
        "train_paths": data.train_dataset.paths,
        "category_to_id": data.category_to_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def trim_log(path, step):
    """Discard entries newer than the saved state after a crash/rollback."""
    if not path.exists():
        return
    temporary = path.with_suffix(".jsonl.tmp")
    with path.open() as source, temporary.open("w") as target:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # A crash may leave a partial final line.
            if record["step"] <= step:
                target.write(json.dumps(record) + "\n")
    temporary.replace(path)


def train_mode(args, data, device, mode):
    args = argparse.Namespace(**vars(args))
    path = Path(args.checkpoint_dir) / f"{mode}.ckpt"
    resume_path = None
    if args.resume == "auto":
        resume_path = path if path.exists() else None
    elif args.resume:
        resume_path = Path(args.resume)
    checkpoint = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("resume_version") != RESUME_VERSION:
            raise ValueError("Checkpoint lacks complete resume state; use a checkpoint saved by this trainer")
        if checkpoint["state_dict"]["_extra_state"]["mode"] != mode:
            raise ValueError(f"Checkpoint does not belong to {mode}")
        if checkpoint["data_signature"] != data_signature(data):
            raise ValueError("Training manifest/category mapping differs from the checkpoint")
        # Restore training settings unless explicitly supplied. Changing these
        # mid-run would invalidate the saved batch position or learning trajectory.
        for name in ("batch_size", "seed", "width", "depth", "heads", "patch_size",
                     "knots", "lr", "weight_decay", "grad_clip", "sample_every", "epochs"):
            saved = checkpoint["args"][name]
            current = getattr(args, name)
            if name == "knots":
                current, saved = tuple(current), tuple(saved)
            if name in args.specified and current != saved:
                raise ValueError(f"--{name} must match the resumed run ({saved!r})")
            setattr(args, name, saved)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = DiTConfig(
        width=args.width, depth=args.depth, heads=args.heads,
        patch_size=args.patch_size,
    )
    model_class = ModelOneNFE if mode == "one_nfe" else ModelFewNFE
    model = model_class(device=device, config=config, knots=args.knots)
    batches_per_epoch = len(data.train_dataset) // args.batch_size
    if not batches_per_epoch:
        raise ValueError("No training batches: reduce --batch_size (drop_last=True)")
    # max_steps is a stopping boundary, not the cosine schedule's horizon.
    total_steps = args.epochs * batches_per_epoch
    stop_step = min(total_steps, args.max_steps) if args.max_steps else total_steps
    precision = args.precision
    if checkpoint and "precision" not in args.specified:
        precision = checkpoint["precision"]
    if precision == "auto":
        precision = (
            "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else "fp16" if device.type == "cuda" else "fp32"
        )
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed precision is supported here only on CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; use fp16 or fp32")
    if checkpoint and precision != checkpoint["precision"]:
        raise ValueError(f"Resume requires --precision {checkpoint['precision']}")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if checkpoint:
        # Load before constructing the optimizer: loading may rebuild the backbone.
        model.load_state_dict(checkpoint["state_dict"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    step, start_epoch, start_batch = 0, 1, 0
    previous_seconds, previous_peak = 0.0, 0
    if checkpoint:
        if checkpoint["total_steps"] != total_steps:
            raise ValueError("Training length differs from the checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["batch_in_epoch"]
        if start_batch == batches_per_epoch:
            start_epoch += 1
            start_batch = 0
        previous_seconds = checkpoint["elapsed_seconds"]
        previous_peak = checkpoint["peak_cuda_bytes"] or 0
        restore_rng(checkpoint["rng_state"], device)
        print(f"Resuming {mode} from {resume_path}: step {step}, epoch {start_epoch}, batch {start_batch}")
    if step >= stop_step:
        print(f"{mode}: already reached step {step}; nothing to train")
        return

    # The data module was constructed with CLI defaults; use restored batch size.
    data.batch_size = args.batch_size
    path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.output_dir) / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.jsonl"
    if checkpoint:
        trim_log(log_path, step)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"{mode}: {model.num_parameters:,} parameters, {device}, {precision}")
    print("Objective: MSE(v((1-t)*noise + t*image, t, category), image-noise)")
    started = time.monotonic()

    def elapsed():
        return previous_seconds + time.monotonic() - started

    def save_checkpoint(epoch, batch, loss_sum, image_count):
        state = {
            "resume_version": RESUME_VERSION,
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "batch_in_epoch": batch, "step": step,
            "epoch_loss_sum": loss_sum, "epoch_image_count": image_count,
            "total_steps": total_steps, "precision": precision,
            "category_to_id": data.category_to_id, "data_signature": data_signature(data),
            "args": vars(args), "rng_state": rng_state(device),
            "elapsed_seconds": elapsed(), "peak_cuda_bytes": previous_peak or None,
        }
        temporary = path.with_suffix(".ckpt.tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    with log_path.open("a" if checkpoint else "w") as log:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            offset = start_batch if epoch == start_epoch else 0
            loss_sum = checkpoint["epoch_loss_sum"] if checkpoint and offset else 0.0
            image_count = checkpoint["epoch_image_count"] if checkpoint and offset else 0
            sampler = EpochSampler(len(data.train_dataset), args.seed + epoch, offset * args.batch_size)
            loader = data.train_dataloader(sampler=sampler)
            loader.worker_init_fn = configure_worker
            # Worker seeding must not consume the noise/time RNG restored above.
            loader.generator = torch.Generator().manual_seed(args.seed + epoch)
            progress = tqdm(loader, initial=offset, total=batches_per_epoch,
                            desc=f"{mode} epoch {epoch}/{args.epochs}")
            for batch, (images, categories) in enumerate(progress, start=offset + 1):
                images = images.to(device, non_blocking=True)
                categories = categories.to(device, non_blocking=True)
                noise = torch.randn_like(images)
                t = torch.rand(images.shape[0], device=device)
                mix = t[:, None, None, None]
                interpolated = (1 - mix) * noise + mix * images
                optimizer.zero_grad(set_to_none=True)
                context = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                           if precision != "fp32" else nullcontext())
                with context:
                    prediction = model(interpolated, t, category=categories)
                    loss = F.mse_loss(prediction.float(), images - noise)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                lr = optimizer.param_groups[0]["lr"]
                scheduler.step()
                step += 1
                peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                previous_peak = max(previous_peak, peak_bytes)
                if peak_bytes >= 20_000_000_000:
                    raise RuntimeError("Peak allocated CUDA memory reached 20 GB; reduce batch size/model")
                value = loss.item()
                loss_sum += value * images.shape[0]
                image_count += images.shape[0]
                record = {"epoch": epoch, "step": step, "loss": value, "lr": lr,
                          "elapsed_seconds": elapsed(), "peak_cuda_bytes": previous_peak or None}
                log.write(json.dumps(record) + "\n")
                progress.set_postfix(loss=f"{value:.4f}", step=step)
                if step % args.save_every == 0:
                    log.flush()
                    save_checkpoint(epoch, batch, loss_sum, image_count)
                if step >= stop_step or _STOP_REQUESTED:
                    break
            log.flush()
            save_checkpoint(epoch, batch, loss_sum, image_count)
            if not _STOP_REQUESTED and (epoch % args.sample_every == 0 or epoch == 1 or step >= stop_step):
                # Previews must not affect the resumed training RNG trajectory.
                state = rng_state(device)
                try:
                    model.eval()
                    with torch.no_grad():
                        categories = torch.arange(8, device=device, dtype=torch.long)
                        samples = model.sample((8, 3, 64, 64), device=device, category=categories)
                    save_image(((samples.float().cpu() + 1) / 2).clamp(0, 1),
                               run_dir / f"epoch_{epoch:04d}.png", nrow=4)
                finally:
                    restore_rng(state, device)
            print(f"Epoch {epoch}: mean loss={loss_sum / image_count:.6f}; saved {path}")
            if step >= stop_step or _STOP_REQUESTED:
                break


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("one_nfe", "few_nfe", "both"), default="both")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--output_dir", default="./runs")
    parser.add_argument("--resume", nargs="?", const="auto", help="Resume a checkpoint path, or auto-resume each selected mode when no path is given")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps, plus epoch ends and graceful stops")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, help="Stop at this absolute global step per model (does not change the LR schedule)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--patch_size", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--knots", type=float, nargs=3, default=(0.25, 0.5, 0.75))
    parser.add_argument("--sample_every", type=int, default=10)
    args = parser.parse_args()
    args.specified = [token.split("=", 1)[0][2:] for token in sys.argv[1:] if token.startswith("--")]
    if args.resume and args.resume != "auto" and args.mode == "both":
        parser.error("A checkpoint path requires --mode one_nfe or --mode few_nfe; use --resume alone for both")
    for name in ("epochs", "batch_size", "sample_every", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max_steps must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be nonnegative")
    for name in ("lr", "grad_clip", "weight_decay"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
            parser.error(f"--{name} has an invalid value")
    return args


def main():
    args = parse_args()
    device = select_device(args.device)
    data = PokemonDataModule(
        data_root=args.data_root, split_dir=args.split_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        return_category=True,
    )
    if sorted(data.category_to_id.values()) != list(range(151)):
        raise ValueError("Expected the official mapping with contiguous category IDs 0..150")
    modes = ("one_nfe", "few_nfe") if args.mode == "both" else (args.mode,)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    for mode in modes:
        if _STOP_REQUESTED:
            break
        train_mode(args, data, device, mode)


if __name__ == "__main__":
    main()
