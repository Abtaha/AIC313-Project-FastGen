"""Teacher target cache, multiscale regression, and official endpoint FID only."""
import json
import tempfile
from pathlib import Path
import torch
from .train import atomic_save, check_memory

def matching_loss(prediction, target):
    """Dependency-free multiscale L2; explicitly not LPIPS."""
    loss = prediction.new_zeros(())
    for factor in (1, 2, 4):
        if factor == 1:
            left, right = prediction, target
        else:
            left = torch.nn.functional.avg_pool2d(prediction, factor)
            right = torch.nn.functional.avg_pool2d(target, factor)
        loss = loss + (left-right).square().mean() / 3
    return loss


def teacher_feature_loss(teacher, prediction, target, labels):
    """Spatial cosine anchor using only our frozen teacher, not external features.

    This is an experimental alternative supervision signal, not LPIPS. Identical
    sigma=.5 corruption is used on each pair; gradients reach prediction only.
    """
    from .adversarial import bottleneck_features, frozen_parameters
    sigma=prediction.new_full((len(prediction),),.5,dtype=torch.float32)
    noise=torch.randn_like(prediction,dtype=torch.float32)*sigma[:,None,None,None]
    mode=teacher.training
    try:
        teacher.eval()
        with frozen_parameters(teacher), torch.autocast(device_type=prediction.device.type,enabled=False):
            with torch.no_grad():
                expected=bottleneck_features(teacher,target.float()+noise,sigma,labels)
            actual=bottleneck_features(teacher,prediction.float()+noise,sigma,labels)
            return (1-torch.nn.functional.cosine_similarity(actual,expected,dim=1,eps=1e-8)).mean()
    finally:
        teacher.train(mode)

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

def report_fid(args, output, device, *, checkpoint=None, mode=None,
               selection="synthetic-heldout-matching-loss"):
    # Reuse the official image conversion, 20/class sampling, and pooled CleanFID.
    from dataset import PokemonDataModule
    from evaluate import prepare_reference_set, evaluate
    checkpoint = checkpoint or output / "best.ckpt"
    if mode not in ("one_nfe", "few_nfe"):
        raise ValueError("Select one_nfe or few_nfe")
    from model import Model
    model = Model.load_checkpoint(checkpoint, evaluate_mode=mode, device=device)
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
    row = {"competition_fid": score, "nfe": args.nfe,
           "guidance": args.guidance if mode is None else 1.0, "teacher_guidance": args.guidance,
           "method": args.method, "seed": args.seed+10000,
           "checkpoint_selection": selection, "fid_count": len(categories)}
    (output / "fid.json").write_text(json.dumps(row, indent=2)+"\n")
    print(json.dumps(row), flush=True)
