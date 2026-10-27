"""Train and report frozen-teacher LD3/Bézier scheduler pilots."""

import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import torch

from teacher import EDMTeacher
from .schedulers import LearnedSampler, matching_loss, project_latents
from .train import atomic_save, check_memory, file_hash


def target_cache(args, teacher, teacher_hash, mapping, device):
    path = Path(args.cache)
    identity = {"teacher_sha256": teacher_hash, "guidance": args.guidance,
                "teacher_steps": args.teacher_steps, "seed": args.seed,
                "train_per_class": args.train_per_class, "category_to_id": mapping}
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("identity") != identity:
            raise ValueError("Cache provenance differs; choose a different cache path")
        return data
    c = teacher.config
    if sorted(mapping.values()) != list(range(c.num_classes)):
        raise ValueError("Teacher categories must be contiguous")
    # Independent synthetic noise for fitting and monitoring; no validation images.
    per_class = args.train_per_class + 1
    labels = torch.arange(c.num_classes).repeat_interleave(per_class)
    noise = torch.randn((len(labels), c.channels, c.resolution, c.resolution),
                        generator=torch.Generator().manual_seed(args.seed))
    heldout = torch.arange(len(labels)) % per_class == args.train_per_class
    targets = torch.empty_like(noise)
    for start in range(0, len(noise), args.microbatch):
        end = min(start+args.microbatch, len(noise))
        targets[start:end] = teacher.sample(noise[start:end].shape, device=device,
            latents=noise[start:end].to(device), category=labels[start:end].to(device),
            num_steps=args.teacher_steps, guidance=args.guidance).cpu()
        check_memory(device, args.memory_limit_gb)
        if start == 0 or end == len(noise) or start // args.microbatch % 20 == 0:
            print(f"Teacher cache {end}/{len(noise)}", flush=True)
    data = {"identity": identity, "latents": noise, "labels": labels,
            "targets": targets, "heldout": heldout}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(data, path)
    return data


@torch.no_grad()
def monitor(model, cache, batch_size, device):
    indices = cache["heldout"].nonzero().flatten()
    total = 0.0
    for ids in indices.split(batch_size):
        prediction = model(cache["latents"][ids].to(device), cache["labels"][ids].to(device),
                           rematerialize=False)
        value = matching_loss(prediction, cache["targets"][ids].to(device))
        total += float(value) * len(ids)
    return total / len(indices)


@torch.no_grad()
def sample_grid(model, cache, path, device):
    from torchvision.utils import save_image
    ids = cache["heldout"].nonzero().flatten()[:8]
    # Keep this microbatched too: it runs concurrently with existing experiments.
    images = [model.sample(cache["latents"][i:i+1].shape, device=device,
        latents=cache["latents"][i:i+1], category=cache["labels"][i:i+1].to(device)) for i in ids.tolist()]
    save_image((torch.cat(images)+1)/2, path, nrow=4)


def report_fid(args, output, device):
    # Reuse the official image conversion, 20/class sampling, and pooled CleanFID.
    from dataset import PokemonDataModule
    from evaluate import prepare_reference_set, evaluate
    model = LearnedSampler.load_checkpoint(output / "best.ckpt", device=device)
    module = PokemonDataModule(data_root=args.data_root, split_dir=args.split_dir,
                               batch_size=args.microbatch, num_workers=2)
    if module.category_to_id != torch.load(args.cache, weights_only=True)["identity"]["category_to_id"]:
        raise ValueError("Evaluation categories differ from teacher cache")
    categories = torch.arange(len(module.category_to_id)).repeat_interleave(20)
    torch.manual_seed(args.seed + 10000)
    with tempfile.TemporaryDirectory(prefix="fid-", dir=output) as directory:
        reference = prepare_reference_set(module, Path(directory) / "reference")
        if len(list(reference.glob("*.png"))) != len(categories):
            raise ValueError("Expected full balanced official validation reference")
        score = evaluate(model, reference, Path(directory) / "generated", categories,
                         batch_size=args.microbatch, device=device)
    row = {"competition_fid": score, "nfe": args.nfe, "guidance": args.guidance,
           "method": args.method, "seed": args.seed+10000,
           "checkpoint_selection": "synthetic-heldout-matching-loss", "fid_count": len(categories)}
    (output / "fid.json").write_text(json.dumps(row, indent=2)+"\n")
    print(json.dumps(row), flush=True)


def train(args):
    if min(args.steps, args.batch_size, args.microbatch, args.train_per_class,
           args.eval_every, args.log_every, args.degree) < 1:
        raise ValueError("Steps, batches, intervals, and train-per-class must be positive")
    if args.teacher_steps < 2 or args.lr <= 0 or args.latent_lr < 0 or args.soft_radius < 0:
        raise ValueError("Invalid solver, learning rate, or soft teacher forcing radius")
    device = torch.device(args.device)
    if args.memory_fraction is not None:
        if device.type != "cuda" or not 0 < args.memory_fraction <= 1:
            raise ValueError("Memory fraction requires CUDA and a value in (0,1]")
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, index)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(args.outdir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Choose an empty output directory or use --resume")
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("Resume in the original output directory")
    output.mkdir(parents=True, exist_ok=True)
    teacher_data = torch.load(args.teacher, map_location="cpu", weights_only=True)
    mapping = teacher_data["category_to_id"]
    del teacher_data
    teacher = EDMTeacher.load_checkpoint(args.teacher, device=device)
    model = LearnedSampler(teacher, method=args.method, nfe=args.nfe,
                           guidance=args.guidance, degree=args.degree).to(device).train()
    identity = {"teacher_sha256": file_hash(args.teacher), "recipe": {k: getattr(args, k) for k in (
        "method", "nfe", "guidance", "teacher_steps", "seed", "train_per_class", "degree",
        "lr", "latent_lr", "soft_radius", "batch_size")}, "category_to_id": mapping}
    (output / "config.json").write_text(json.dumps({**identity, "arguments": vars(args),
        "parameters": model.count_parameters(), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "loss": "multiscale-L2 (not LPIPS)", "validation_used_for_training_or_selection": False}, indent=2)+"\n")
    print(f"{args.method}: {args.nfe} actual NFE; {model.steps} Euler updates; "
          f"{model.count_parameters():,} total parameters; only scheduler parameters train", flush=True)
    cache = target_cache(args, teacher, identity["teacher_sha256"], mapping, device)
    indices = (~cache["heldout"]).nonzero().flatten()
    perturbations = cache["latents"].clone()
    optimizer = torch.optim.Adam(model.schedule.parameters(), lr=args.lr)
    start, best = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("identity") != identity:
            raise ValueError("Resume recipe or teacher differs")
        model.schedule.load_state_dict(state["schedule"])
        optimizer.load_state_dict(state["optimizer"])
        perturbations, start, best = state["latents"], state["step"], state["best"]
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    if start >= args.steps:
        raise ValueError("--steps must exceed resumed step")
    if not args.resume:
        best = monitor(model, cache, args.microbatch, device)
        if not math.isfinite(best):
            raise FloatingPointError("Nonfinite initial monitoring loss")
        atomic_save(model.export(), output / "initial.ckpt")
        atomic_save(model.export(), output / "best.ckpt")
        sample_grid(model, cache, output / "samples-initial.png", device)
        print(json.dumps({"step": 0, "heldout_matching_loss": best}), flush=True)
    started = time.monotonic()
    for step in range(start+1, args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        selected = indices[torch.randperm(len(indices))[:args.batch_size]]
        total = 0.0
        for ids in selected.split(args.microbatch):
            original = cache["latents"][ids].to(device)
            soft = args.method == "ld3" and args.soft_radius > 0
            latent = perturbations[ids].to(device).detach().requires_grad_(soft)
            prediction = model(latent, cache["labels"][ids].to(device))
            loss = matching_loss(prediction, cache["targets"][ids].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            (loss * len(ids) / len(selected)).backward()
            total += float(loss.detach()) * len(ids) / len(selected)
            if soft:
                proposal = latent.detach() - args.latent_lr * len(selected) * latent.grad
                perturbations[ids] = project_latents(proposal, original, args.soft_radius).cpu()
        norm = torch.nn.utils.clip_grad_norm_(model.schedule.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        check_memory(device, args.memory_limit_gb)
        if step == start+1 or step % args.log_every == 0:
            row = {"step": step, "matching_loss": total, "grad_norm": float(norm),
                   "elapsed_seconds": time.monotonic()-started,
                   "peak_allocated_gb": torch.cuda.max_memory_allocated(device)/1e9 if device.type == "cuda" else 0}
            print(json.dumps(row), flush=True)
            with (output / "stats.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
        if step % args.eval_every == 0 or step == args.steps:
            value = monitor(model, cache, args.microbatch, device)
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite monitoring loss")
            if value < best:
                best = value
                atomic_save(model.export(), output / "best.ckpt")
            atomic_save(model.export(), output / "latest.ckpt")
            sample_grid(model, cache, output / "samples-latest.png", device)
            atomic_save({"identity": identity, "step": step, "schedule": model.schedule.state_dict(),
                "optimizer": optimizer.state_dict(), "latents": perturbations, "best": best,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, output / "training-state.pt")
            print(json.dumps({"step": step, "heldout_matching_loss": value, "best": best}), flush=True)
    if args.report_fid:
        report_fid(args, output, device)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--method", choices=("ld3", "bezier"), required=True)
    p.add_argument("--nfe", type=int, choices=(1, 4), default=4)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--degree", type=int, default=8)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--train-per-class", type=int, default=2)
    p.add_argument("--teacher-steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--soft-radius", type=float, default=0.19)
    p.add_argument("--latent-lr", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-fraction", type=float)
    p.add_argument("--memory-limit-gb", type=float, default=20.0)
    p.add_argument("--resume")
    p.add_argument("--report-fid", action="store_true")
    p.add_argument("--data-root", default="data/pokemon-generation-one-22k")
    p.add_argument("--split-dir", default="data/pokemon-generation-one-22k")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
