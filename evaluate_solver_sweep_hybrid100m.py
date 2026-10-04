"""Sweep ODE solvers / NFE counts for an existing few-NFE FM checkpoint.

Examples:

    # Submission-budget comparison only: exactly 4 model evaluations.
    uv run --no-sync evaluate_solver_sweep.py \
        --model_checkpoint checkpoints/hybrid100m/few_nfe_epoch_0090.ckpt \
        --nfes 4

    # Full diagnostic sweep.
    uv run --no-sync evaluate_solver_sweep.py \
        --model_checkpoint checkpoints/hybrid100m/few_nfe_epoch_0090.ckpt

This does NOT retrain anything.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
from cleanfid import fid
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from dataset import PokemonDataModule
from model import Model


SAMPLES_PER_CATEGORY = 20

# Actual backbone evaluations required for ONE integration step.
SOLVER_COST = {
    "euler": 1,
    "heun": 2,
    "midpoint": 2,
    "rk4": 4,
}


def prepare_reference_set(data_module, reference_dir):
    """Export the validation split exactly once."""
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)

    existing = list(reference_dir.glob("*.png"))
    if existing:
        print(f"Using existing reference set: {len(existing)} images")
        return reference_dir

    print("Exporting validation reference images...")

    count = 0

    for batch in tqdm(data_module.val_dataloader(), desc="reference"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        for image in images:
            image = ((image.float().cpu() + 1) / 2).clamp(0, 1)

            to_pil_image(image).convert("RGB").save(reference_dir / f"{count:05d}.png")

            count += 1

    print(f"Reference images: {count}")
    return reference_dir


def compute_fid(
    generated_dir,
    reference_dir,
    device,
    batch_size,
    num_workers,
):
    return float(
        fid.compute_fid(
            str(generated_dir),
            str(reference_dir),
            device=str(device),
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )


def make_schedule(steps, power=1.0):
    """Create monotonically increasing nodes from t=0 to t=1.

    power > 1:
        more resolution near t=0

    power < 1:
        more resolution near t=1

    power = 1:
        uniform schedule
    """
    u = torch.linspace(0.0, 1.0, steps + 1)

    nodes = u.pow(power)

    # Be absolutely exact at the endpoints.
    nodes[0] = 0.0
    nodes[-1] = 1.0

    return [float(x) for x in nodes]


def velocity(model, x, t, categories):
    """Evaluate the standard FM vector field once."""
    timestep = torch.full(
        (x.shape[0],),
        float(t),
        device=x.device,
        dtype=x.dtype,
    )

    return model(
        x,
        timestep,
        category=categories,
    )


@torch.inference_mode()
def integrate(
    model,
    x,
    categories,
    *,
    solver,
    total_nfe,
    schedule_power=1.0,
):
    """Integrate noise(t=0) -> data(t=1) using exactly total_nfe calls."""

    cost = SOLVER_COST[solver]

    if total_nfe % cost != 0:
        raise ValueError(
            f"{solver} costs {cost} evaluations/step, "
            f"so total_nfe={total_nfe} is invalid"
        )

    steps = total_nfe // cost

    if steps < 1:
        raise ValueError("Need at least one integration step")

    nodes = make_schedule(
        steps,
        power=schedule_power,
    )

    eval_count = 0

    for start, end in zip(nodes[:-1], nodes[1:]):
        dt = end - start

        if solver == "euler":
            # 1 NFE
            k1 = velocity(
                model,
                x,
                start,
                categories,
            )

            x = x + dt * k1

            eval_count += 1

        elif solver == "midpoint":
            # Explicit midpoint / RK2.
            # 2 NFE per integration step.
            k1 = velocity(
                model,
                x,
                start,
                categories,
            )

            mid_t = start + 0.5 * dt
            mid_x = x + 0.5 * dt * k1

            k2 = velocity(
                model,
                mid_x,
                mid_t,
                categories,
            )

            x = x + dt * k2

            eval_count += 2

        elif solver == "heun":
            # Improved Euler / trapezoidal RK2.
            # 2 NFE per integration step.
            k1 = velocity(
                model,
                x,
                start,
                categories,
            )

            predicted = x + dt * k1

            k2 = velocity(
                model,
                predicted,
                end,
                categories,
            )

            x = x + 0.5 * dt * (k1 + k2)

            eval_count += 2

        elif solver == "rk4":
            # Classical RK4.
            # 4 NFE per integration step.
            half_t = start + 0.5 * dt

            k1 = velocity(
                model,
                x,
                start,
                categories,
            )

            k2 = velocity(
                model,
                x + 0.5 * dt * k1,
                half_t,
                categories,
            )

            k3 = velocity(
                model,
                x + 0.5 * dt * k2,
                half_t,
                categories,
            )

            k4 = velocity(
                model,
                x + dt * k3,
                end,
                categories,
            )

            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            eval_count += 4

        else:
            raise ValueError(f"Unknown solver {solver!r}")

    if eval_count != total_nfe:
        raise RuntimeError(
            f"NFE accounting error: got {eval_count}, " f"expected {total_nfe}"
        )

    return x.clamp(-1, 1)


@torch.inference_mode()
def generate_samples(
    model,
    output_dir,
    categories,
    *,
    solver,
    nfe,
    schedule_power,
    batch_size,
    device,
    seed,
):
    """Generate the same noise/category set for every experiment."""

    output_dir = Path(output_dir)

    # Critical: never allow stale images from an earlier experiment
    # to contaminate FID.
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    parameter = next(model.parameters())
    dtype = parameter.dtype

    # Reinitializing this generator for every experiment means every
    # solver receives exactly the same initial Gaussian noise.
    generator = torch.Generator(device=device).manual_seed(seed)

    image_counts = {}

    for start in tqdm(
        range(0, len(categories), batch_size),
        desc=f"{solver} NFE={nfe} p={schedule_power:g}",
    ):
        batch_categories = categories[start : start + batch_size].to(device)

        count = len(batch_categories)

        x = torch.randn(
            (count, 3, 64, 64),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        samples = integrate(
            model,
            x,
            batch_categories,
            solver=solver,
            total_nfe=nfe,
            schedule_power=schedule_power,
        )

        for sample, category in zip(
            samples,
            batch_categories.tolist(),
        ):
            image_idx = image_counts.get(
                category,
                0,
            )

            image_counts[category] = image_idx + 1

            image = ((sample.float().cpu() + 1) / 2).clamp(0, 1)

            to_pil_image(image).convert("RGB").save(
                output_dir / f"{category:04d}_{image_idx:02d}.png"
            )

    return output_dir


def experiment_key(solver, nfe, power):
    return f"{solver}" f"_nfe{nfe}" f"_p{power:g}"


def load_completed(log_path):
    completed = {}

    if not log_path.exists():
        return completed

    for line in log_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if record.get("status") == "ok":
            completed[record["key"]] = record

    return completed


def write_csv(path, records):
    fields = [
        "solver",
        "nfe",
        "steps",
        "schedule_power",
        "fid",
        "seed",
    ]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in sorted(
            records,
            key=lambda x: x["fid"],
        ):
            writer.writerow({name: record[name] for name in fields})


def build_experiments(args):
    experiments = []

    # Main solver/NFE sweep using uniform time grids.
    for nfe in args.nfes:
        for solver in args.solvers:
            cost = SOLVER_COST[solver]

            if nfe % cost:
                continue

            experiments.append((solver, nfe, 1.0))

    # Extra search specifically for the submission-budget Euler
    # sampler. Same 4 NFE, different placement of time points.
    for power in args.euler_powers:
        experiments.append(
            (
                "euler",
                args.power_sweep_nfe,
                power,
            )
        )

    # Deduplicate.
    seen = set()
    unique = []

    for experiment in experiments:
        if experiment not in seen:
            seen.add(experiment)
            unique.append(experiment)

    return unique


def main(args):
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print(f"Loading checkpoint: " f"{args.model_checkpoint}")

    model = Model.load_checkpoint(
        args.model_checkpoint,
        evaluate_mode="few_nfe",
        device=device,
    )

    print(f"Parameters: " f"{model.count_parameters():,}")

    objective = getattr(
        model,
        "objective",
        None,
    )

    if objective != "flow_matching":
        raise ValueError(
            "This evaluator expects the standard "
            f"few-NFE flow-matching model; got {objective!r}"
        )

    data = PokemonDataModule(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    reference_dir = prepare_reference_set(
        data,
        args.reference_dir,
    )

    categories = torch.arange(
        len(data.category_to_id),
        dtype=torch.long,
    ).repeat_interleave(SAMPLES_PER_CATEGORY)

    print(
        f"{len(data.category_to_id)} categories × "
        f"{SAMPLES_PER_CATEGORY} samples = "
        f"{len(categories)} generated images/run"
    )

    root = Path(args.output_dir)

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = root / "scores.jsonl"
    csv_path = root / "scores.csv"

    completed = {} if args.force else load_completed(log_path)

    experiments = build_experiments(args)

    print("\nExperiments:")
    for solver, nfe, power in experiments:
        steps = nfe // SOLVER_COST[solver]

        print(
            f"  {solver:8s} " f"NFE={nfe:2d} " f"steps={steps:2d} " f"power={power:g}"
        )

    for solver, nfe, power in experiments:
        key = experiment_key(
            solver,
            nfe,
            power,
        )

        if key in completed:
            print(f"\nSkipping {key}: " f"FID={completed[key]['fid']:.4f}")
            continue

        print(f"\n{'=' * 70}\n" f"{key}\n" f"{'=' * 70}")

        generated_dir = root / "generated" / key

        try:
            generate_samples(
                model,
                generated_dir,
                categories,
                solver=solver,
                nfe=nfe,
                schedule_power=power,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed,
            )

            score = compute_fid(
                generated_dir,
                reference_dir,
                device,
                args.fid_batch_size,
                args.num_workers,
            )

            record = {
                "key": key,
                "solver": solver,
                "nfe": nfe,
                "steps": (nfe // SOLVER_COST[solver]),
                "schedule_power": power,
                "fid": score,
                "seed": args.seed,
                "status": "ok",
            }

            print(f"\n>>> {key}: " f"FID = {score:.4f}")

            with log_path.open("a") as file:
                file.write(json.dumps(record) + "\n")

            completed[key] = record

            if not args.keep_images:
                shutil.rmtree(
                    generated_dir,
                    ignore_errors=True,
                )

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        except Exception as error:
            record = {
                "key": key,
                "solver": solver,
                "nfe": nfe,
                "schedule_power": power,
                "status": "error",
                "error": (f"{type(error).__name__}: " f"{error}"),
            }

            with log_path.open("a") as file:
                file.write(json.dumps(record) + "\n")

            raise

        successful = [
            record for record in completed.values() if record.get("status") == "ok"
        ]

        write_csv(
            csv_path,
            successful,
        )

    successful = sorted(
        (record for record in completed.values() if record.get("status") == "ok"),
        key=lambda x: x["fid"],
    )

    print("\n\n" + "=" * 72)
    print("FINAL RANKING")
    print("=" * 72)

    for rank, result in enumerate(
        successful,
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"{result['solver']:8s} "
            f"NFE={result['nfe']:2d} "
            f"steps={result['steps']:2d} "
            f"p={result['schedule_power']:<4g} "
            f"FID={result['fid']:.4f}"
        )

    print(f"\nResults written to: " f"{csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_checkpoint",
        default=("checkpoints/hybrid100m/" "few_nfe_epoch_0090.ckpt"),
    )

    parser.add_argument(
        "--data_root",
        default="./data/pokemon-generation-one-22k",
    )

    parser.add_argument(
        "--split_dir",
        default="./data/pokemon-generation-one-22k",
    )

    parser.add_argument(
        "--reference_dir",
        default="./results/solver_sweep/reference",
    )

    parser.add_argument(
        "--output_dir",
        default="./results/solver_sweep",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--fid_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--nfes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
    )

    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=tuple(SOLVER_COST),
        default=[
            "euler",
            "heun",
            "midpoint",
            "rk4",
        ],
    )

    # Additional 4-NFE Euler schedule search.
    parser.add_argument(
        "--power_sweep_nfe",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--euler_powers",
        type=float,
        nargs="+",
        default=[
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
        ],
    )

    parser.add_argument(
        "--keep_images",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    main(parser.parse_args())
