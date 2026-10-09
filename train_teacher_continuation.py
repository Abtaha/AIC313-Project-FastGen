"""EDM teacher continuation on every official TRAIN image; no validation use or model selection."""

import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import PokemonDataset
from teacher import EDMTeacher, TeacherConfig
from teacher_monitor import split_fingerprint


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def update_ema(ema, model, beta):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source.detach(), 1 - beta)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.microbatch < 1:
        raise ValueError("steps, batch-size, and microbatch must be positive")
    if args.batch_size % args.microbatch:
        raise ValueError("batch-size must be divisible by microbatch")
    if min(args.save_every, args.log_every, args.sample_every) < 1:
        raise ValueError("logging and checkpoint intervals must be positive")
    if args.lr <= 0 or args.ema_halflife_kimg <= 0 or args.memory_limit_gb <= 0:
        raise ValueError("lr, EMA half-life, and memory limit must be positive")
    if args.sample_steps < 2:
        raise ValueError("sample-steps must be >= 2")
    if args.grad_clip <= 0:
        raise ValueError("grad-clip must be positive")
    if args.continuation_lr is not None and (not args.resume or args.continuation_lr <= 0):
        raise ValueError("--continuation-lr requires --resume and a positive learning rate")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a smoke test")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("AMP is supported on CUDA only; use --amp none on CPU")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unavailable on this GPU; use --amp fp16 or --amp none")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(args.outdir)
    if args.resume and output.resolve() == Path(args.resume).resolve().parent:
        raise ValueError("Outdir must differ from original checkpoint directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to protect existing files")
    output.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)
    with (split_dir / "category_to_id.json").open() as f:
        category_to_id = json.load(f)
    if sorted(category_to_id.values()) != list(range(len(category_to_id))):
        raise ValueError("category_to_id.json must contain contiguous IDs from zero")
    # train_split.txt is the entire official training set; validation never backpropagates.
    dataset = PokemonDataset(
        args.data_root, split_dir / "train_split.txt", return_category=True,
        category_to_id=category_to_id,
    )
    fit_indices = list(range(len(dataset)))
    holdout_indices = []
    if len(fit_indices) < args.microbatch:
        raise ValueError("Training dataset is smaller than microbatch")
    print(f"Training on all {len(dataset):,} official training images; validation never loaded", flush=True)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Subset(dataset, fit_indices), batch_size=args.microbatch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    config = TeacherConfig(
        num_classes=len(category_to_id), model_channels=args.model_channels,
        num_blocks=args.num_blocks, dropout=args.dropout, label_dropout=args.label_dropout,
    )
    model = EDMTeacher(config).to(device).train()
    # EMA is a separate frozen model, never a registered submodule of the teacher.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    # Also enforce 100M for the combined live+EMA framework conservatively.
    if 2 * model.num_parameters > 100_000_000:
        raise ValueError("Live teacher plus EMA exceed 100M; reduce model-channels or num-blocks")
    print(f"Teacher: {model.num_parameters:,} parameters; "
          f"live + EMA: {2 * model.num_parameters:,}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    start_step = 0
    seen_images = 0
    elapsed_before = 0.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["config"] != asdict(config) or state["category_to_id"] != category_to_id:
            raise ValueError("Resume architecture/category mapping differs from this run")
        if state["batch_size"] != args.batch_size:
            raise ValueError("Resume batch-size differs from checkpoint")
        if state["train_manifest_sha256"] != split_fingerprint(dataset):
            raise ValueError("Training manifest differs from checkpoint")
        # The original checkpoint used a 90/10 internal holdout. This intentionally
        # changes the training subset; its old monitoring scores are not comparable.
        previous_fit = state.get("fit_indices")
        if previous_fit is not None:
            if set(previous_fit) - set(fit_indices):
                raise ValueError("Saved fit indices are outside the official training split")
            print(f"Switching training subset from {len(previous_fit)} to {len(fit_indices)} images", flush=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step, seen_images = state["step"], state["seen_images"]
        if args.steps <= start_step:
            raise ValueError(f"--steps={args.steps} must exceed resumed step {start_step}")
        if Path(args.resume).resolve().parent == output.resolve():
            raise ValueError("Use a NEW --outdir when resuming to protect your original checkpoints")
        if (output / "training-state.pt").exists() or (output / "teacher.ckpt").exists():
            raise ValueError("Continuation output already contains checkpoints; use an empty --outdir")
        print(f"Resuming from step {start_step:,} to step {args.steps:,}; "
              f"learning rate = {args.continuation_lr or args.lr:g}", flush=True)
        elapsed_before = state["elapsed_seconds"]
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_python"])
        loader_generator.set_state(state["rng_loader"])
        if device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    (output / "config.json").write_text(json.dumps(
        {"teacher": asdict(config), "training": vars(args),
         "category_to_id": category_to_id}, indent=2
    ) + "\n")
    (output / "data-split.json").write_text(json.dumps({
        "train_count": len(dataset),
        "train_manifest_sha256": split_fingerprint(dataset),
        "validation_loaded": False,
        "validation_used_for_training_or_selection": False,
    }, indent=2) + "\n")
    class_probabilities = torch.bincount(
        torch.tensor([dataset.category_ids[i] for i in fit_indices]), minlength=config.num_classes
    ).float()
    class_probabilities /= class_probabilities.sum()

    def batches():
        while True:
            yield from loader

    iterator = iter(batches())
    accumulation = args.batch_size // args.microbatch
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    started = time.monotonic()
    peak_allocated = peak_reserved = 0.0

    def check_memory():
        nonlocal peak_allocated, peak_reserved
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1e9
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9
            if peak_allocated >= args.memory_limit_gb:
                raise RuntimeError(
                    f"Peak VRAM allocated={peak_allocated:.2f}, reserved={peak_reserved:.2f} "
                    f"GB exceeds {args.memory_limit_gb} GB; reduce --microbatch"
                )

    def save(step):
        metadata = {"category_to_id": category_to_id, "step": step,
                    "seen_images": seen_images, "class_probabilities": class_probabilities,
                    "train_manifest_sha256": split_fingerprint(dataset)}
        atomic_save({**ema.checkpoint(), **metadata}, output / "teacher.ckpt")
        atomic_save({
            "config": asdict(config), **metadata, "model": model.state_dict(),
            "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "batch_size": args.batch_size,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "fit_indices": fit_indices, "holdout_indices": holdout_indices,
            "best_fid": None, "best_holdout_loss": None,
            "stale_checks": 0,
            "selection_config": {"guidance": args.guidance},
            "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
            "rng_loader": loader_generator.get_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, output / "training-state.pt")

    for step in range(start_step + 1, args.steps + 1):
        # For continuation, do not redo warmup or inherit the old 2e-4 LR.
        if args.resume and args.continuation_lr is not None:
            lr = args.continuation_lr
        else:
            lr = args.lr * min((seen_images + args.batch_size) / max(args.lr_rampup_kimg * 1000, 1), 1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accumulation):
            images, category = next(iterator)
            images = images.to(device, non_blocking=True)
            category = category.to(device, non_blocking=True)
            if args.hflip:
                mask = torch.rand((len(images), 1, 1, 1), device=device) < 0.5
                images = torch.where(mask, images.flip(-1), images)
            context = torch.autocast("cuda", dtype=amp_dtype) if args.amp != "none" else nullcontext()
            with context:
                loss = model.loss(images, category)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            scaler.scale(loss / accumulation).backward()
            loss_sum += loss.detach().item() / accumulation
            check_memory()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        scaler.step(optimizer)
        scaler.update()
        seen_images += args.batch_size
        half_life = min(args.ema_halflife_kimg * 1000, seen_images * 0.05)
        update_ema(ema, model, 0.5 ** (args.batch_size / max(half_life, 1)))
        check_memory()
        if step % args.log_every == 0 or step == start_step + 1:
            record = {"step": step, "kimg": seen_images / 1000, "loss": loss_sum,
                      "lr": lr, "grad_norm": float(grad_norm),
                      "elapsed_seconds": elapsed_before + time.monotonic() - started,
                      "peak_allocated_gb": peak_allocated, "peak_reserved_gb": peak_reserved}
            with (output / "stats.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.sample_every == 0:
            from torchvision.utils import save_image
            labels = torch.arange(8, device=device) % config.num_classes
            samples = ema.sample((8, 3, 64, 64), device=device, category=labels,
                                 generator=torch.Generator(device=device).manual_seed(args.seed),
                                 num_steps=args.sample_steps, guidance=args.guidance)
            (output / "samples").mkdir(exist_ok=True)
            save_image((samples.cpu() + 1) / 2, output / "samples" / f"step-{step:07d}.png", nrow=4)
            check_memory()
        # Checkpoint cadence is predetermined, never based on validation metrics.
        if step % args.save_every == 0 or step == args.steps:
            save(step)



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/pokemon-generation-one-22k")
    p.add_argument("--split-dir", default="data/pokemon-generation-one-22k")
    p.add_argument("--outdir", default="checkpoints/teacher")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=16)
    p.add_argument("--model-channels", type=int, default=96)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--label-dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-rampup-kimg", type=float, default=100)
    p.add_argument("--ema-halflife-kimg", type=float, default=500)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    p.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", help="Full training-state.pt, NOT the inference teacher.ckpt")
    p.add_argument("--continuation-lr", type=float, default=None,
                   help="Constant LR used after resuming; retains Adam moments and EMA")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=40)
    p.add_argument("--memory-limit-gb", type=float, default=20.0)
    p.add_argument("--guidance", type=float, default=1.0)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
