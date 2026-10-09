"""Independent V2 training: balanced MP U-Net with competition-protocol FID."""

import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import PokemonDataset
from .model import EDMTeacherV2, TeacherV2Config
from .data import balanced_sampler
from .data import (denoising_losses, save_monitor_split,
                               split_fingerprint, stratified_holdout)
from .evaluate import competition_fid, FID_PROTOCOL, SAMPLES_PER_CATEGORY


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


def learning_rate_schedule(seen_images, *, ref_lr, rampup_kimg, decay_ref_kimg):
    """EDM2 inverse-square-root decay with a dataset-scaled reference point."""
    lr = ref_lr / math.sqrt(max(seen_images / (decay_ref_kimg * 1000), 1)) if decay_ref_kimg else ref_lr
    return lr * min(seen_images / (rampup_kimg * 1000), 1) if rampup_kimg else lr


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.microbatch < 1:
        raise ValueError("steps, batch-size, and microbatch must be positive")
    if args.batch_size % args.microbatch:
        raise ValueError("batch-size must be divisible by microbatch")
    if min(args.save_every, args.log_every, args.sample_every) < 1:
        raise ValueError("logging and checkpoint intervals must be positive")
    if args.lr <= 0 or args.ema_halflife_kimg <= 0 or args.memory_limit_gb <= 0:
        raise ValueError("lr, EMA half-life, and memory limit must be positive")
    if args.lr_rampup_kimg < 0 or args.lr_decay_ref_kimg < 0 or args.snapshot_every < 0:
        raise ValueError("LR reference points and snapshot interval must be nonnegative")
    if args.fid_batch_size < 1:
        raise ValueError("fid-batch-size must be positive")
    if args.loss_every < 0 or args.fid_every < 0 or args.loss_count < 1:
        raise ValueError("monitor intervals must be nonnegative and loss-count positive")
    if args.fid_steps < 2 or args.sample_steps < 2:
        raise ValueError("fid-steps and sample-steps must be >= 2")
    if args.grad_clip <= 0 or args.early_stop_patience < 0:
        raise ValueError("grad-clip must be positive and early-stop-patience nonnegative")
    if args.early_stop_patience and not args.loss_every:
        raise ValueError("Early stopping requires loss monitoring")
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
    if not args.resume and (output / "training-state.pt").exists():
        raise ValueError("This output directory already has a run; use --resume or a new --outdir")
    output.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)
    with (split_dir / "category_to_id.json").open() as f:
        category_to_id = json.load(f)
    if sorted(category_to_id.values()) != list(range(len(category_to_id))):
        raise ValueError("category_to_id.json must contain contiguous IDs from zero")
    # Optimization uses only train_split.txt; validation is used only for FID.
    dataset = PokemonDataset(
        args.data_root, split_dir / "train_split.txt", return_category=True,
        category_to_id=category_to_id,
    )
    fit_indices, holdout_indices = stratified_holdout(
        dataset.category_ids, args.holdout_fraction, args.seed
    )
    if args.loss_every and len(holdout_indices) < 2:
        raise ValueError("Monitoring needs a holdout; or set --loss-every 0 --fid-every 0")
    if len(fit_indices) < args.microbatch:
        raise ValueError("training split has fewer images than microbatch")
    validation_dataset = PokemonDataset(
        args.data_root, split_dir / "val_split.txt", return_category=True,
        category_to_id=category_to_id,
    ) if args.fid_every else None
    loader_generator = torch.Generator().manual_seed(args.seed)
    sampler = balanced_sampler(
        [dataset.category_ids[i] for i in fit_indices], num_classes=len(category_to_id),
        generator=loader_generator,
    ) if args.class_sampling == "balanced" else None
    loader = DataLoader(
        Subset(dataset, fit_indices), batch_size=args.microbatch, shuffle=sampler is None,
        sampler=sampler, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    config = TeacherV2Config(
        num_classes=len(category_to_id), model_channels=args.model_channels,
        num_blocks=args.num_blocks, dropout=args.dropout, label_dropout=args.label_dropout,
        channel_mult=tuple(args.channel_mult),
        attn_resolutions=tuple(args.attn_resolutions), channels_per_head=args.channels_per_head,
        p_mean=args.p_mean, p_std=args.p_std, sigma_data=args.sigma_data,
        logvar_channels=args.logvar_channels,
    )
    model = EDMTeacherV2(config).to(device).train()
    if 2 * model.num_parameters > 100_000_000:
        raise ValueError("Teacher plus full EMA, including both uncertainty heads, exceed 100M")
    # Full EMA copy is included in the conservative framework parameter count.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"Teacher: {model.num_parameters:,} parameters; "
          f"live + EMA: {2 * model.num_parameters:,}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    start_step = 0
    seen_images = 0
    elapsed_before = 0.0
    best_fid = None
    best_holdout_loss = None
    stale_checks = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("format") != "fastgen-edm-training-v2" or state.get("fid_protocol") != FID_PROTOCOL:
            raise ValueError("Expected an independent V2 state with the competition FID protocol")
        if state.get("recipe") != "edm2-uncertainty-v1":
            raise ValueError("Start a fresh experiment; this state predates the EDM2 uncertainty recipe")
        if state["lr_config"] != {key: getattr(args, key) for key in ("lr", "lr_rampup_kimg", "lr_decay_ref_kimg")}:
            raise ValueError("Resume LR configuration differs from this run")
        if TeacherV2Config(**state["config"]) != config or state["category_to_id"] != category_to_id:
            raise ValueError("Resume architecture/category mapping differs from this run")
        if state.get("class_sampling", "natural") != args.class_sampling:
            raise ValueError("Resume class sampling differs; start a separate experiment")
        if state["batch_size"] != args.batch_size:
            raise ValueError("Resume batch-size differs from checkpoint")
        if (state["train_manifest_sha256"] != split_fingerprint(dataset)
                or state["fit_indices"] != fit_indices or state["holdout_indices"] != holdout_indices):
            raise ValueError("Resume training manifest or holdout split differs from checkpoint")
        if state["selection_config"] != {key: getattr(args, key) for key in (
            "guidance", "fid_steps", "fid_batch_size", "fid_seed", "loss_count"
        )}:
            raise ValueError("Resume monitoring/guidance config differs; use a new run")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step, seen_images = state["step"], state["seen_images"]
        elapsed_before = state["elapsed_seconds"]
        best_fid = state["best_fid"]
        best_holdout_loss = state["best_holdout_loss"]
        stale_checks = state["stale_checks"]
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_python"])
        loader_generator.set_state(state["rng_loader"])
        if device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    (output / "config.json").write_text(json.dumps(
        {"teacher": asdict(config), "training": vars(args),
         "category_to_id": category_to_id}, indent=2
    ) + "\n")
    save_monitor_split(dataset, fit_indices, holdout_indices, output / "monitor-split.json")
    class_probabilities = torch.bincount(
        torch.tensor([dataset.category_ids[i] for i in fit_indices]), minlength=config.num_classes
    ).float()
    class_probabilities /= class_probabilities.sum()
    training_class_probabilities = (torch.ones_like(class_probabilities) / config.num_classes
                                    if args.class_sampling == "balanced" else class_probabilities)

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
                    "training_class_probabilities": training_class_probabilities,
                    "train_manifest_sha256": split_fingerprint(dataset)}
        atomic_save({**ema.checkpoint(), **metadata}, output / "teacher.ckpt")
        atomic_save({
            "format": "fastgen-edm-training-v2", "fid_protocol": FID_PROTOCOL,
            "recipe": "edm2-uncertainty-v1",
            "lr_config": {key: getattr(args, key) for key in ("lr", "lr_rampup_kimg", "lr_decay_ref_kimg")},
            "config": asdict(config), **metadata, "model": model.state_dict(),
            "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "batch_size": args.batch_size,
            "class_sampling": args.class_sampling,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "fit_indices": fit_indices, "holdout_indices": holdout_indices,
            "best_fid": best_fid, "best_holdout_loss": best_holdout_loss,
            "stale_checks": stale_checks,
            "selection_config": {key: getattr(args, key) for key in (
                "guidance", "fid_steps", "fid_batch_size", "fid_seed", "loss_count"
            )},
            "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
            "rng_loader": loader_generator.get_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, output / "training-state.pt")

    for step in range(start_step + 1, args.steps + 1):
        step_started = time.monotonic()
        lr = learning_rate_schedule(seen_images + args.batch_size, ref_lr=args.lr,
                                    rampup_kimg=args.lr_rampup_kimg,
                                    decay_ref_kimg=args.lr_decay_ref_kimg)
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
        model.normalize_weights()
        seen_images += args.batch_size
        half_life = min(args.ema_halflife_kimg * 1000, seen_images * 0.05)
        update_ema(ema, model, 0.5 ** (args.batch_size / max(half_life, 1)))
        check_memory()
        if step % args.log_every == 0 or step == start_step + 1:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record = {"step": step, "kimg": seen_images / 1000, "loss": loss_sum,
                      "lr": lr, "grad_norm": float(grad_norm),
                      "elapsed_seconds": elapsed_before + time.monotonic() - started,
                      "peak_allocated_gb": peak_allocated, "peak_reserved_gb": peak_reserved,
                      "train_step_seconds": time.monotonic() - step_started}
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
        monitor_record = {"step": step}
        if args.loss_every and step % args.loss_every == 0:
            for name, indices in (("fit", fit_indices), ("holdout", holdout_indices)):
                monitor_record[f"{name}_loss_by_sigma"] = denoising_losses(
                    ema, dataset, indices, device=device, batch_size=args.microbatch,
                    seed=args.seed + 1234, limit=args.loss_count,
                )
            value = sum(monitor_record["holdout_loss_by_sigma"].values()) / 4
            if best_holdout_loss is None or value < best_holdout_loss:
                best_holdout_loss = value
                stale_checks = 0
                atomic_save({**ema.checkpoint(), "step": step, "holdout_loss": value,
                             "category_to_id": category_to_id, "class_probabilities": class_probabilities},
                            output / "best-loss.ckpt")
            else:
                stale_checks += 1
        if args.fid_every and step % args.fid_every == 0:
            print(f"Step {step}: competition FID ({config.num_classes * SAMPLES_PER_CATEGORY} samples)", flush=True)
            score = competition_fid(
                ema, validation_dataset, num_classes=config.num_classes, outdir=output,
                device=device, batch_size=args.fid_batch_size, steps=args.fid_steps,
                guidance=args.guidance, seed=args.fid_seed,
            )
            if not math.isfinite(score):
                raise RuntimeError("Non-finite competition FID")
            monitor_record.update(competition_fid=score, fid_protocol=FID_PROTOCOL, guidance=args.guidance,
                                  fid_count=config.num_classes * SAMPLES_PER_CATEGORY, fid_steps=args.fid_steps)
            if best_fid is None or score < best_fid:
                best_fid = score
                atomic_save({**ema.checkpoint(), "step": step, "competition_fid": score, "fid_protocol": FID_PROTOCOL,
                             "guidance": args.guidance, "category_to_id": category_to_id,
                             "class_probabilities": class_probabilities}, output / "best-fid.ckpt")
        if len(monitor_record) > 1:
            check_memory()
            with (output / "monitor.jsonl").open("a") as f:
                f.write(json.dumps(monitor_record) + "\n")
            print(json.dumps(monitor_record), flush=True)
        stop = args.early_stop_patience > 0 and stale_checks >= args.early_stop_patience
        if step % args.save_every == 0 or step == args.steps or stop:
            save(step)
        if args.snapshot_every and (step % args.snapshot_every == 0 or step == args.steps or stop):
            snapshots = output / "snapshots"
            snapshots.mkdir(exist_ok=True)
            metadata = {"step": step, "seen_images": seen_images, "category_to_id": category_to_id,
                        "recipe": "edm2-uncertainty-v1"}
            atomic_save({**model.checkpoint(), **metadata}, snapshots / f"raw-{step:07d}.ckpt")
            atomic_save({**ema.checkpoint(), **metadata}, snapshots / f"ema-{step:07d}.ckpt")
        if stop:
            print("Stopped after training-holdout loss failed to improve", flush=True)
            break


def v2_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/pokemon-generation-one-22k")
    p.add_argument("--split-dir", default="data/pokemon-generation-one-22k")
    p.add_argument("--outdir", default="checkpoints/edm2-112")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--model-channels", type=int, default=112)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--class-sampling", choices=("natural", "balanced"), default="balanced")
    p.add_argument("--channel-mult", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--attn-resolutions", type=int, nargs="+", default=[16])
    p.add_argument("--channels-per-head", type=int, default=56)
    p.add_argument("--p-mean", type=float, default=-0.4)
    p.add_argument("--p-std", type=float, default=1.0)
    p.add_argument("--logvar-channels", type=int, default=128)
    p.add_argument("--sigma-data", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--label-dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-rampup-kimg", type=float, default=100)
    p.add_argument("--lr-decay-ref-kimg", type=float, default=128,
                   help="Inverse-sqrt decay starts here; 128 kimg = 2000 updates at batch 64; 0 disables")
    p.add_argument("--ema-halflife-kimg", type=float, default=500)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    p.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=1000, help="Retain raw/EMA snapshots; 0 disables")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=40)
    p.add_argument("--memory-limit-gb", type=float, default=20.0)
    p.add_argument("--holdout-fraction", type=float, default=0.10)
    p.add_argument("--loss-every", type=int, default=2000, help="0 disables loss monitoring")
    p.add_argument("--loss-count", type=int, default=512)
    p.add_argument("--fid-every", type=int, default=5000, help="0 disables FID monitoring")
    p.add_argument("--fid-batch-size", type=int, default=32)
    p.add_argument("--fid-steps", type=int, default=18)
    p.add_argument("--fid-seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop after this many loss checks without improvement; 0 disables")
    return p


if __name__ == "__main__":
    train(v2_parser().parse_args())
