"""Select four-pass noise levels using synthetic teacher pairs, never validation FID."""

import argparse
import json
from pathlib import Path

import torch

from .model import ConsistencyStudent
from .schedulers import matching_loss
from .train import atomic_save, file_hash
from .train_scheduler import report_fid


@torch.no_grad()
def score(model, cache, ids, batch_size, device, seed):
    total = 0.0
    for chunk in ids.split(batch_size):
        generator = torch.Generator(device=device).manual_seed(seed + int(chunk[0]))
        noise = cache["latents"][chunk].to(device)
        samples = model.sample(noise.shape, category=cache["labels"][chunk].to(device),
            device=device, latents=noise, nfe=4, generator=generator)
        total += float(matching_loss(samples, cache["targets"][chunk].to(device))) * len(chunk)
    return total / len(ids)


def run(args):
    device = torch.device(args.device)
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    output = Path(args.outdir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    source = torch.load(args.student, map_location="cpu", weights_only=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if source["category_to_id"] != cache["identity"]["category_to_id"] or source["teacher_guidance"] != cache["identity"]["guidance"]:
        raise ValueError("Student and teacher cache have different classes or guidance")
    args.guidance = source["teacher_guidance"]
    model = ConsistencyStudent.load_checkpoint(args.student, device).requires_grad_(False)
    c = model.config
    candidates = {"original": model.four_sigmas.tolist()}
    for rho in (3.0, 5.0, 9.0):
        ramp = torch.arange(4).float()/4
        candidates[f"rho{rho:g}"] = ((c.sigma_max**(1/rho) + ramp *
            (c.sigma_min**(1/rho)-c.sigma_max**(1/rho))) ** rho).tolist()
    # Relative levels also allow the tiny CPU model to exercise this search.
    for name, ratios in [("a", [.15, .0125, .000625]), ("b", [.10, .01, .000625]),
                         ("c", [.25, .025, .00125]), ("d", [.0625, .00625, .00025])]:
        candidates[name] = [c.sigma_max] + [max(c.sigma_min, c.sigma_max*r) for r in ratios]
    fit, heldout = (~cache["heldout"]).nonzero().flatten(), cache["heldout"].nonzero().flatten()
    rows = []
    for name, levels in candidates.items():
        model.set_four_sigmas(levels)
        row = {"name": name, "sigmas": levels,
               "fit_matching_loss": score(model, cache, fit, args.microbatch, device, args.seed),
               "heldout_matching_loss": score(model, cache, heldout, args.microbatch, device, args.seed)}
        print(json.dumps(row), flush=True)
        rows.append(row)
    # Monitor scores are reported but are not used to pick the winner.
    best = min(rows, key=lambda row: row["fit_matching_loss"])
    model.set_four_sigmas(best["sigmas"])
    exported = model.export(category_to_id=source["category_to_id"], step=source["step"], guidance=args.guidance)
    exported["sampling_selection"] = "synthetic-fit-pair-matching-loss"
    atomic_save(exported, output / "few_nfe.ckpt")
    (output / "summary.json").write_text(json.dumps({"source_sha256": file_hash(args.student),
        "cache_sha256": file_hash(args.cache), "results": rows, "selected": best,
        "validation_used_for_selection": False}, indent=2)+"\n")
    if args.report_fid:
        report_fid(args, output, device, checkpoint=output/"few_nfe.ckpt", mode="few_nfe",
                   selection="synthetic-fit-pair-matching-loss")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--microbatch", type=int, default=32)
    p.add_argument("--seed", type=int, default=456)
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-fraction", type=float)
    p.add_argument("--report-fid", action="store_true")
    p.add_argument("--data-root", default="data/pokemon-generation-one-22k")
    p.add_argument("--split-dir", default="data/pokemon-generation-one-22k")
    p.set_defaults(method="consistency-schedule-search", nfe=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
