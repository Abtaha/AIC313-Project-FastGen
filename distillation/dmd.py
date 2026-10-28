"""EDM-space DMD pilot: guided real score, conditional fake score, paired L2.

Implements the score-gradient weighting of arXiv:2311.18828 equations 7/8.
Generator starts from our consistency checkpoint; fake score starts from V1.
No pretrained perceptual network or generator EMA is introduced.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from teacher import EDMTeacher
from .model import ConsistencyStudent
from .schedulers import matching_loss
from .train import atomic_save, check_memory, file_hash
from .train_scheduler import report_fid


def generate(generator, noise, labels):
    return generator(noise * generator.config.sigma_max, generator.config.sigma_max, labels)


@torch.no_grad()
def distribution_gradient(images, labels, teacher, fake, *, guidance=2.0, sigma=None, noise=None):
    if sigma is None:
        sigma = (torch.randn(len(images), device=images.device)*1.2-1.2).exp().clamp(.05, 20)
    if noise is None:
        noise = torch.randn_like(images)
    noisy = images.detach().float() + sigma[:, None, None, None]*noise
    previous_mode = fake.training
    fake.eval()
    with torch.autocast(device_type=images.device.type, enabled=False):
        real_prediction = teacher.guided(noisy, sigma, labels, guidance)
        fake_prediction = fake(noisy, sigma, labels)
        scale = (images.detach()-real_prediction).abs().flatten(1).mean(1).clamp_min(.01)
        # sigma^2 cancels against the paper's weight. Sign is fake minus real:
        # descending this direction moves toward the real score prediction.
        gradient = (fake_prediction-real_prediction)/scale[:, None, None, None]
    fake.train(previous_mode)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite distribution-matching gradient")
    return gradient


def gradient_surrogate(images, gradient):
    # Its gradient w.r.t. images is gradient/numel; scalar is not a KL estimate.
    return 0.5 * (images - (images.detach()-gradient).detach()).square().mean()


def compatible_identity(saved, requested, *, fork=False):
    """Forks may change either generator LR or update ratio, one at a time."""
    saved = {**saved, "recipe": {"fake_updates": 1, **saved["recipe"]}}
    requested = {**requested, "recipe": {"fake_updates": 1, **requested["recipe"]}}
    if fork:
        differences = 0
        for key in ("fake_updates", "lr"):
            differences += saved["recipe"].pop(key) != requested["recipe"].pop(key)
        if differences > 1:
            return False
    return saved == requested


def train(args):
    if min(args.steps, args.batch_size, args.microbatch, args.save_every, args.log_every, args.fake_updates) < 1 or args.batch_size % args.microbatch:
        raise ValueError("Positive intervals required; microbatch must divide batch-size")
    if min(args.lr, args.fake_lr) <= 0 or args.reg_weight <= 0 or args.score_warmup < 0:
        raise ValueError("Positive rates/regression weight and nonnegative warmup required")
    if args.gan_weight < 0 or args.gan_reg_weight < 0 or (args.gan_weight and args.gan_start < args.score_warmup):
        raise ValueError("Nonnegative GAN weights required; GAN must start after score warmup")
    device = torch.device(args.device)
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 requires CUDA support; use --amp none on CPU")
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(args.outdir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Choose an empty output directory or resume")
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("Resume in original outdir")
    output.mkdir(parents=True, exist_ok=True)
    source = torch.load(args.student, map_location="cpu", weights_only=True)
    base = torch.load(args.teacher, map_location="cpu", weights_only=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    mapping = source["category_to_id"]
    if mapping != base["category_to_id"] or mapping != cache["identity"]["category_to_id"]:
        raise ValueError("Teacher, student, and cache mappings differ")
    if source.get("teacher_guidance") != args.guidance:
        raise ValueError("Student initialization and requested teacher guidance differ")
    teacher_hash = file_hash(args.teacher)
    if cache["identity"]["teacher_sha256"] != teacher_hash or cache["identity"]["guidance"] != args.guidance:
        raise ValueError("Cache teacher or guidance differs")
    generator = ConsistencyStudent.load_checkpoint(args.student, device).train()
    teacher = EDMTeacher.load_checkpoint(args.teacher, device=device)
    if replace(teacher.config, dropout=0, label_dropout=0) != generator.config:
        raise ValueError("Teacher and generator configurations differ")
    fake = EDMTeacher(replace(teacher.config, dropout=0, label_dropout=0)).to(device).train()
    fake.backbone.load_state_dict(teacher.backbone.state_dict())
    head, real_images = None, None
    if args.gan_weight:
        from .adversarial import GANHead, BalancedTrainingImages
        head = GANHead(generator.config).to(device).train()
        real_images = BalancedTrainingImages(args.data_root, args.split_dir, mapping)
    del source, base
    critic_parameters = list(fake.parameters()) + (list(head.parameters()) if head is not None else [])
    framework_parameters = sum(p.numel() for p in list(generator.parameters()) + critic_parameters)
    if framework_parameters > 100_000_000 or teacher.num_parameters > 100_000_000:
        raise ValueError("Generator including auxiliary fake score or teacher exceeds 100M")
    identity = {"teacher_sha256": teacher_hash, "student_sha256": file_hash(args.student),
                "cache_sha256": file_hash(args.cache), "recipe": {k: getattr(args,k) for k in (
                    "guidance", "lr", "fake_lr", "reg_weight", "score_warmup", "batch_size", "amp", "seed", "fake_updates")}}
    if head is not None:
        identity["recipe"].update({key: getattr(args, key) for key in (
            "gan_weight", "gan_start", "gan_reg_weight")})
        identity["gan_train_manifest_sha256"] = file_hash(Path(args.split_dir)/"train_split.txt")
        identity["adversarial_recipe"] = "noisy-standard-song-middle-128-hflip-v1"
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(.9,.999))
    fake_optimizer = torch.optim.Adam(critic_parameters, lr=args.fake_lr, betas=(.9,.999))
    start = 0
    restore = args.resume or args.fork_from
    if restore:
        state = torch.load(restore, map_location="cpu", weights_only=True)
        if not compatible_identity(state["identity"], identity, fork=bool(args.fork_from)):
            raise ValueError("Resume recipe differs")
        generator.load_state_dict(state["generator"])
        fake.load_state_dict(state["fake"])
        if head is not None:
            head.load_state_dict(state["gan_head"])
        optimizer.load_state_dict(state["optimizer"])
        fake_optimizer.load_state_dict(state["fake_optimizer"])
        if args.fork_from:
            # Keep Adam moments/counters, but apply the requested fork LR.
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        start = state["step"]
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        del state
    if args.steps <= start:
        raise ValueError("Steps must exceed resumed step")
    (output/"config.json").write_text(json.dumps({**identity, "arguments": vars(args),
        "framework_parameters": framework_parameters, "teacher_parameters": teacher.num_parameters,
        "generator_ema": False, "validation_loaded_for_training": False,
        "fork_parent": args.fork_from, "fork_parent_sha256": file_hash(args.fork_from) if args.fork_from else None,
        "gan_training_images": len(real_images.dataset) if real_images is not None else 0,
        "adaptations": "consistency initialization; lognormal EDM sigma .05..20; multiscale L2 regression"}, indent=2)+"\n")
    if not args.resume:
        atomic_save(generator.export(category_to_id=mapping, step=start, guidance=args.guidance), output/"initial.ckpt")
    fit = (~cache["heldout"]).nonzero().flatten()
    c = generator.config
    clock = time.monotonic()
    print(f"DMD: generator + fake score {framework_parameters:,}; separate teacher {teacher.num_parameters:,}; "
          f"no EMA; {args.fake_updates} fake-score updates per generator update; start step {start}", flush=True)
    for step in range(start+1, args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        fake_optimizer.zero_grad(set_to_none=True)
        gan_active = head is not None and step > args.gan_start
        reg_weight = args.gan_reg_weight if gan_active else args.reg_weight
        gan_g_total, gan_d_total = 0.0, 0.0
        real_accuracy, fake_accuracy = 0.0, 0.0
        image_std, saturation = 0.0, 0.0
        generated, dm_total, reg_total = [], 0.0, 0.0
        for _ in range(args.batch_size//args.microbatch):
            noise = torch.randn(args.microbatch,c.channels,c.resolution,c.resolution,device=device)
            labels = torch.randint(c.num_classes,(args.microbatch,),device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp=="bf16"):
                images = generate(generator, noise, labels)
                if step > args.score_warmup:
                    grad = distribution_gradient(images, labels, teacher, fake, guidance=args.guidance)
                    dm = gradient_surrogate(images.float(), grad)
                    reg = images.new_zeros(())
                    if reg_weight:
                        ids = fit[torch.randint(len(fit),(args.microbatch,))]
                        paired = generate(generator, cache["latents"][ids].to(device), cache["labels"][ids].to(device))
                        reg = matching_loss(paired.float(), cache["targets"][ids].to(device))
                    loss = dm + reg_weight*reg
                    if gan_active:
                        from .adversarial import frozen_parameters, random_flip
                        with frozen_parameters(fake, head):
                            gan_g = F.softplus(-head(fake, random_flip(images), labels)).mean()
                        loss = loss + args.gan_weight*gan_g
                        gan_g_total += float(gan_g.detach())*args.microbatch/args.batch_size
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite generator loss")
                    (loss*args.microbatch/args.batch_size).backward()
                    dm_total += float(dm.detach())*args.microbatch/args.batch_size
                    reg_total += float(reg.detach())*args.microbatch/args.batch_size
            generated.append((images.detach(),labels))
            image_std += float(images.detach().float().std())*args.microbatch/args.batch_size
            saturation += float((images.detach().abs()>1).float().mean())*args.microbatch/args.batch_size
        norm = torch.nn.utils.clip_grad_norm_(generator.parameters(),1.0,error_if_nonfinite=True)
        if step > args.score_warmup:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        fake_total = 0.0
        for _ in range(args.fake_updates):
            fake_optimizer.zero_grad(set_to_none=True)
            # Reuse the detached generated batch, with fresh EDM corruption on
            # every critic update. Generator and teacher receive no gradients.
            for images, labels in generated:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp=="bf16"):
                    if gan_active:
                        # Fresh samples after the generator update on every critic pass.
                        with torch.no_grad():
                            labels = torch.randint(c.num_classes, labels.shape, device=device)
                            images = generate(generator, torch.randn_like(images), labels)
                    loss = fake.loss(images.float(),labels)
                    fake_total += float(loss.detach())*args.microbatch/args.batch_size/args.fake_updates
                    if gan_active:
                        from .adversarial import random_flip
                        real, real_labels = real_images.sample(len(images), device)
                        real_logits = head(fake, random_flip(real), real_labels)
                        fake_logits = head(fake, random_flip(images.detach()), labels)
                        gan_d = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
                        loss = loss + args.gan_weight*gan_d
                        gan_d_total += float(gan_d.detach())*args.microbatch/args.batch_size/args.fake_updates
                        real_accuracy += float((real_logits.detach()>0).float().mean())*args.microbatch/args.batch_size/args.fake_updates
                        fake_accuracy += float((fake_logits.detach()<0).float().mean())*args.microbatch/args.batch_size/args.fake_updates
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite fake-score loss")
                (loss*args.microbatch/args.batch_size).backward()
            torch.nn.utils.clip_grad_norm_(critic_parameters,1.0,error_if_nonfinite=True)
            fake_optimizer.step()
        check_memory(device,args.memory_limit_gb)
        if step == start+1 or step % args.log_every == 0:
            row = {"step":step,"dm_surrogate":dm_total,"reg_loss":reg_total,"fake_loss":fake_total,"fake_updates":args.fake_updates,
                   "gan_active":gan_active,"reg_weight":reg_weight,"gan_g":gan_g_total,"gan_d":gan_d_total,
                   "gan_real_accuracy":real_accuracy,"gan_fake_accuracy":fake_accuracy,
                   "image_std":image_std,"out_of_range_fraction":saturation,
                   "grad_norm":float(norm),"elapsed_seconds":time.monotonic()-clock,
                   "peak_allocated_gb":torch.cuda.max_memory_allocated(device)/1e9 if device.type=="cuda" else 0}
            print(json.dumps(row),flush=True)
            with (output/"stats.jsonl").open("a") as stream: stream.write(json.dumps(row)+"\n")
        if step % args.save_every == 0 or step == args.steps:
            from torchvision.utils import save_image
            generator.eval()
            ids = cache["heldout"].nonzero().flatten()[:8]
            with torch.no_grad():
                samples = generator.sample(cache["latents"][ids].shape,device=device,nfe=1,
                    category=cache["labels"][ids].to(device),latents=cache["latents"][ids].to(device))
            save_image((samples+1)/2,output/f"samples-{step:06d}.png",nrow=4)
            exported = generator.export(category_to_id=mapping,step=step,guidance=args.guidance)
            exported["training_recipe"] = "DMD-adapted-v1"
            if head is not None:
                exported["training_recipe"] = "DMD2-inspired-gan-v1"
            atomic_save(exported,output/"one_nfe.ckpt")
            snapshots = output/"snapshots"
            snapshots.mkdir(exist_ok=True)
            atomic_save(exported,snapshots/f"one_nfe-{step:06d}.ckpt")
            atomic_save({"identity":identity,"step":step,"generator":generator.state_dict(),
                **({"gan_head":head.state_dict()} if head is not None else {}),
                "fake":fake.state_dict(),"optimizer":optimizer.state_dict(),"fake_optimizer":fake_optimizer.state_dict(),
                "rng_cpu":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all() if device.type=="cuda" else None},output/"training-state.pt")
            generator.train()
    if args.report_fid:
        report_fid(args,output,device,checkpoint=output/"one_nfe.ckpt",mode="one_nfe",selection="fixed-final-pilot-step")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher","student","cache","outdir"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--steps",type=int,default=2000)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--microbatch",type=int,default=8)
    p.add_argument("--lr",type=float,default=5e-6)
    p.add_argument("--fake-lr",type=float,default=1e-4)
    p.add_argument("--fake-updates",type=int,default=1)
    p.add_argument("--guidance",type=float,default=2.0)
    p.add_argument("--reg-weight",type=float,default=.25)
    p.add_argument("--score-warmup",type=int,default=50)
    p.add_argument("--gan-weight",type=float,default=0,help="Enable shared-backbone adversarial head; 0 preserves legacy DMD")
    p.add_argument("--gan-start",type=int,default=2000,help="Enable GAN after this absolute generator-loop step")
    p.add_argument("--gan-reg-weight",type=float,default=.025,help="Regression weight once the GAN is active; 0 removes regression")
    p.add_argument("--amp",choices=("bf16","none"),default="bf16")
    p.add_argument("--device",default="cuda")
    p.add_argument("--memory-fraction",type=float)
    p.add_argument("--memory-limit-gb",type=float,default=20)
    p.add_argument("--seed",type=int,default=789)
    p.add_argument("--log-every",type=int,default=25)
    p.add_argument("--save-every",type=int,default=500)
    restore = p.add_mutually_exclusive_group()
    restore.add_argument("--resume")
    restore.add_argument("--fork-from")
    p.add_argument("--report-fid",action="store_true")
    p.add_argument("--data-root",default="data/pokemon-generation-one-22k")
    p.add_argument("--split-dir",default="data/pokemon-generation-one-22k")
    p.set_defaults(method="dmd",nfe=1)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
