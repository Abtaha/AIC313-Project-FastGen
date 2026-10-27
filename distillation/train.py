"""Train a conditional consistency student using all official training images."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import PokemonDataset
from teacher import EDMTeacher
from .loss import consistency_loss, noise_grid
from .model import ConsistencyStudent


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def update_ema(target, student, decay):
    for dest, source in zip(target.parameters(), student.parameters()):
        dest.lerp_(source, 1 - decay)
    for dest, source in zip(target.buffers(), student.buffers()):
        dest.copy_(source)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_memory(device, limit_gb):
    if device.type == "cuda" and torch.cuda.max_memory_allocated(device) / 1e9 > limit_gb:
        raise RuntimeError("Allocated training memory exceeds limit; reduce microbatch")


def train(args):
    if min(args.steps, args.batch_size, args.microbatch, args.save_every, args.log_every,
           args.sample_every, args.num_intervals) < 1 or args.batch_size % args.microbatch:
        raise ValueError("Positive intervals/batch sizes required; microbatch must divide batch-size")
    if args.num_intervals < 2 or args.lr <= 0 or args.grad_clip <= 0 or args.memory_limit_gb <= 0:
        raise ValueError("Invalid noise grid, learning rate, clipping, or memory limit")
    if not 0 <= args.ema_decay < 1 or not 0 <= args.guidance < float("inf") or args.warmup_steps < 0:
        raise ValueError("Invalid EMA, guidance, or warmup")
    device = torch.device(args.device)
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 requires a supported CUDA GPU; use --amp none on CPU")
    output = Path(args.outdir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Choose an empty outdir, or explicitly resume training-state.pt")
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("Resume into the existing run's outdir")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    teacher_metadata = torch.load(args.teacher, map_location="cpu", weights_only=True)
    teacher = EDMTeacher.load_checkpoint(args.teacher, device=device)
    student = ConsistencyStudent.from_teacher(teacher).train()
    target = copy.deepcopy(student).eval().requires_grad_(False)
    student_parameters = sum(p.numel() for p in student.parameters())
    teacher_parameters = sum(p.numel() for p in teacher.parameters())
    if 2 * student_parameters > 100_000_000 or teacher_parameters > 100_000_000:
        raise ValueError("Student including EMA or separate teacher exceeds the 100M limit")
    mapping = json.loads((Path(args.split_dir) / "category_to_id.json").read_text())
    if mapping != teacher_metadata.get("category_to_id"):
        raise ValueError("Training categories differ from teacher checkpoint")
    del teacher_metadata
    dataset = PokemonDataset(args.data_root, Path(args.split_dir) / "train_split.txt",
                             return_category=True, category_to_id=mapping)
    if len(dataset) < args.microbatch:
        raise ValueError("Training set is smaller than microbatch")
    loader_rng = torch.Generator().manual_seed(args.seed)
    counts = torch.bincount(torch.tensor(dataset.category_ids), minlength=len(mapping))
    if (counts == 0).any():
        raise ValueError("Every category requires training examples")
    sampler = WeightedRandomSampler(counts[dataset.category_ids].double().reciprocal(),
                                    len(dataset), replacement=True, generator=loader_rng)
    loader = DataLoader(dataset, batch_size=args.microbatch, sampler=sampler, drop_last=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        generator=loader_rng)
    identity = {"teacher_sha256": file_hash(args.teacher),
                "train_manifest_sha256": hashlib.sha256("\n".join(dataset.paths).encode()).hexdigest(),
                "config": asdict(student.config), "category_to_id": mapping,
                "recipe": {key: getattr(args, key) for key in (
                    "guidance", "batch_size", "num_intervals", "lr", "ema_decay", "loss",
                    "warmup_steps", "amp", "hflip", "seed")}}
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start, elapsed_before = 0, 0.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("format") != "fastgen-consistency-training-v1" or state["identity"] != identity:
            raise ValueError("Resume teacher, data, configuration, or training recipe differs")
        student.load_state_dict(state["student"])
        target.load_state_dict(state["target"])
        optimizer.load_state_dict(state["optimizer"])
        start, elapsed_before = state["step"], state["elapsed_seconds"]
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_python"])
        loader_rng.set_state(state["rng_loader"])
        if device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        del state
    if args.steps <= start:
        raise ValueError("--steps must exceed resumed step")
    (output / "config.json").write_text(json.dumps({**identity, "training": vars(args),
        "train_count": len(dataset), "validation_loaded": False,
        "student_parameters": student_parameters, "student_plus_ema_parameters": 2 * student_parameters,
        "teacher_parameters": teacher_parameters}, indent=2) + "\n")
    print(f"All {len(dataset)} training images; balanced classes; no validation loaded. "
          f"Student {student_parameters:,}; student + EMA {2*student_parameters:,}; "
          f"separate frozen teacher {teacher_parameters:,}", flush=True)
    grid = noise_grid(student.config, args.num_intervals, device)
    iterator = iter(loader)
    clock = time.monotonic()
    accumulation = args.batch_size // args.microbatch
    for step in range(start + 1, args.steps + 1):
        step_clock = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        rate = args.lr * min(step / max(args.warmup_steps, 1), 1)
        for group in optimizer.param_groups:
            group["lr"] = rate
        total = 0.0
        for _ in range(accumulation):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, labels = next(iterator)
            images, labels = images.to(device), labels.to(device)
            if args.hflip:
                flip = torch.rand((len(images), 1, 1, 1), device=device) < 0.5
                images = torch.where(flip, images.flip(-1), images)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
                loss = consistency_loss(student, target, teacher, images, labels,
                    grid=grid, guidance=args.guidance, loss_type=args.loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        update_ema(target, student, args.ema_decay)
        check_memory(device, args.memory_limit_gb)
        if step == start + 1 or step % args.log_every == 0:
            row = {"step": step, "kimg": step * args.batch_size / 1000, "loss": total,
                   "lr": rate, "grad_norm": float(norm), "step_seconds": time.monotonic() - step_clock,
                   "elapsed_seconds": elapsed_before + time.monotonic() - clock,
                   "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0}
            print(json.dumps(row), flush=True)
            with (output / "stats.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        if step % args.sample_every == 0 or step == args.steps:
            from torchvision.utils import save_image
            labels = torch.arange(min(8, len(mapping)), device=device)
            for nfe in (1, 4):
                rng = torch.Generator(device=device).manual_seed(args.seed + 1000)
                samples = target.sample((len(labels), student.config.channels, student.config.resolution,
                    student.config.resolution), device=device, category=labels, nfe=nfe, generator=rng)
                save_image((samples + 1) / 2, output / f"samples-{step:06d}-{nfe}nfe.png", nrow=4)
            check_memory(device, args.memory_limit_gb)
        if step % args.save_every == 0 or step == args.steps:
            atomic_save(target.export(category_to_id=mapping, step=step, guidance=args.guidance), output / "student.ckpt")
            atomic_save({"format": "fastgen-consistency-training-v1", "identity": identity,
                "step": step, "student": student.state_dict(), "target": target.state_dict(),
                "optimizer": optimizer.state_dict(), "elapsed_seconds": elapsed_before + time.monotonic() - clock,
                "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
                "rng_loader": loader_rng.get_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                output / "training-state.pt")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True)
    p.add_argument("--outdir", default="checkpoints/distillation-cd")
    p.add_argument("--resume")
    p.add_argument("--data-root", default="data/pokemon-generation-one-22k")
    p.add_argument("--split-dir", default="data/pokemon-generation-one-22k")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--num-intervals", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--ema-decay", type=float, default=0.95)
    p.add_argument("--loss", choices=("mse", "pseudo-huber"), default="pseudo-huber")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--memory-limit-gb", type=float, default=20.0)
    p.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
