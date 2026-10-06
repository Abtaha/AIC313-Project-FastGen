"""Sweep ODE solvers / NFE counts for an existing few-NFE FM checkpoint.

Examples:

    # Default diagnostic sweep for whatever checkpoint is configured below.
    uv run --no-sync evaluate_solver_sweep.py

    # Explicit checkpoint.
    uv run --no-sync evaluate_solver_sweep.py \
        --model_checkpoint checkpoints/hybrid_v2/few_nfe_epoch_0090.ckpt

    # Submission-budget comparison only: exactly 4 model evaluations.
    uv run --no-sync evaluate_solver_sweep.py \
        --model_checkpoint checkpoints/hybrid_v2/few_nfe_epoch_0090.ckpt \
        --nfes 4

    # Larger diagnostic sweep.
    uv run --no-sync evaluate_solver_sweep.py \
        --model_checkpoint checkpoints/hybrid_v2/few_nfe_epoch_0090.ckpt \
        --nfes 4 8 16 32 64 \
        --solvers euler midpoint heun rk4

    # Override automatic result directory if desired.
    uv run --no-sync evaluate_solver_sweep.py \
        --model_checkpoint checkpoints/hybrid_v2/few_nfe_epoch_0090.ckpt \
        --output_dir ./results/custom_hybrid_v2_run

This does NOT retrain anything.

IMPORTANT:
    By default, each checkpoint gets its own output directory:

        checkpoints/hybrid_v2/few_nfe_epoch_0090.ckpt

    becomes:

        results/solver_sweep_hybrid_v2_few_nfe_epoch_0090/

    This prevents results from different models/checkpoints from being mixed
    or accidentally skipped because they share the same experiment keys.

    The validation reference images are shared across experiments.
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


# ---------------------------------------------------------------------------
# Solver cost in actual neural-network function evaluations.
# ---------------------------------------------------------------------------

SOLVER_COST = {
    "euler": 1,
    "heun": 2,
    "midpoint": 2,
    "rk4": 4,
}


# ---------------------------------------------------------------------------
# Reference dataset
# ---------------------------------------------------------------------------


def prepare_reference_set(data_module, reference_dir):
    """Export the validation split exactly once."""

    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)

    existing = list(reference_dir.glob("*.png"))

    if existing:
        print(f"Using existing reference set: " f"{len(existing)} images")
        return reference_dir

    print("Exporting validation reference images...")

    count = 0

    for batch in tqdm(
        data_module.val_dataloader(),
        desc="reference",
    ):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        for image in images:
            image = ((image.float().cpu() + 1) / 2).clamp(0, 1)

            to_pil_image(image).convert("RGB").save(reference_dir / f"{count:05d}.png")

            count += 1

    print(f"Reference images: {count}")

    return reference_dir


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Time schedules
# ---------------------------------------------------------------------------


def make_schedule(steps, power=1.0):
    """Create monotonically increasing nodes from t=0 to t=1.

    power > 1:
        more resolution near t=0

    power < 1:
        more resolution near t=1

    power = 1:
        uniform schedule
    """

    u = torch.linspace(
        0.0,
        1.0,
        steps + 1,
    )

    nodes = u.pow(power)

    # Be absolutely exact at endpoints.
    nodes[0] = 0.0
    nodes[-1] = 1.0

    return [float(x) for x in nodes]


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------


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
    """Integrate noise(t=0) -> data(t=1).

    total_nfe is the exact number of model forward passes.

    For example:

        Euler:
            NFE=4 -> 4 integration steps

        Midpoint:
            NFE=4 -> 2 integration steps

        Heun:
            NFE=4 -> 2 integration steps

        RK4:
            NFE=4 -> 1 integration step
    """

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

    for start, end in zip(
        nodes[:-1],
        nodes[1:],
    ):
        dt = end - start

        # ---------------------------------------------------------------
        # Euler
        # ---------------------------------------------------------------

        if solver == "euler":
            # 1 NFE per integration step.

            k1 = velocity(
                model,
                x,
                start,
                categories,
            )

            x = x + dt * k1

            eval_count += 1

        # ---------------------------------------------------------------
        # Explicit midpoint / RK2
        # ---------------------------------------------------------------

        elif solver == "midpoint":
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

        # ---------------------------------------------------------------
        # Heun / improved Euler
        # ---------------------------------------------------------------

        elif solver == "heun":
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

        # ---------------------------------------------------------------
        # Classical RK4
        # ---------------------------------------------------------------

        elif solver == "rk4":
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
            f"NFE accounting error: " f"got {eval_count}, " f"expected {total_nfe}"
        )

    return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


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

    # Never allow stale images from an earlier experiment
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

    # Resetting the generator for every experiment means every
    # solver/configuration receives exactly the same Gaussian noise.
    generator = torch.Generator(device=device).manual_seed(seed)

    image_counts = {}

    for start in tqdm(
        range(
            0,
            len(categories),
            batch_size,
        ),
        desc=(f"{solver} " f"NFE={nfe} " f"p={schedule_power:g}"),
    ):
        batch_categories = categories[start : start + batch_size].to(device)

        count = len(batch_categories)

        x = torch.randn(
            (
                count,
                3,
                64,
                64,
            ),
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


# ---------------------------------------------------------------------------
# Experiment bookkeeping
# ---------------------------------------------------------------------------


def experiment_key(
    solver,
    nfe,
    power,
):
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

    with path.open(
        "w",
        newline="",
    ) as file:
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


# ---------------------------------------------------------------------------
# Build sweep
# ---------------------------------------------------------------------------


def build_experiments(args):
    experiments = []

    # Main solver/NFE sweep using uniform time grids.
    for nfe in args.nfes:
        for solver in args.solvers:
            cost = SOLVER_COST[solver]

            if nfe % cost:
                continue

            experiments.append(
                (
                    solver,
                    nfe,
                    1.0,
                )
            )

    # Extra search specifically for the submission-budget
    # Euler sampler.
    #
    # Same number of NFEs, but different placement
    # of integration nodes.
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


# ---------------------------------------------------------------------------
# Automatic per-checkpoint output directory
# ---------------------------------------------------------------------------


def resolve_output_dir(args):
    """Return a unique result directory for the selected checkpoint.

    Example:

        checkpoints/hybrid_v2/few_nfe_epoch_0090.ckpt

    becomes:

        results/solver_sweep_hybrid_v2_few_nfe_epoch_0090/

    This prevents experiment keys from one checkpoint from colliding
    with those from another checkpoint.
    """

    if args.output_dir is not None:
        return Path(args.output_dir)

    checkpoint = Path(args.model_checkpoint)

    parent_name = checkpoint.parent.name
    checkpoint_name = checkpoint.stem

    run_name = f"{parent_name}_" f"{checkpoint_name}"

    return Path("./results") / f"solver_sweep_{run_name}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # Resolve a checkpoint-specific result directory.
    root = resolve_output_dir(args)

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SOLVER SWEEP")
    print("=" * 72)

    print(f"Checkpoint : " f"{args.model_checkpoint}")

    print(f"Results    : " f"{root}")

    print(f"Reference  : " f"{args.reference_dir}")

    print("=" * 72)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    print(f"\nLoading checkpoint: " f"{args.model_checkpoint}")

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
            "few-NFE flow-matching model; "
            f"got {objective!r}"
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

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

    # Exactly 20 samples per Pokémon class.
    categories = torch.arange(
        len(data.category_to_id),
        dtype=torch.long,
    ).repeat_interleave(SAMPLES_PER_CATEGORY)

    print(
        f"\n{len(data.category_to_id)} categories × "
        f"{SAMPLES_PER_CATEGORY} samples = "
        f"{len(categories)} generated images/run"
    )

    # ------------------------------------------------------------------
    # Result files
    # ------------------------------------------------------------------

    log_path = root / "scores.jsonl"

    csv_path = root / "scores.csv"

    completed = {} if args.force else load_completed(log_path)

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    experiments = build_experiments(args)

    print("\nExperiments:")

    for solver, nfe, power in experiments:
        steps = nfe // SOLVER_COST[solver]

        print(
            f"  {solver:8s} " f"NFE={nfe:2d} " f"steps={steps:2d} " f"power={power:g}"
        )

    # ------------------------------------------------------------------
    # Run sweep
    # ------------------------------------------------------------------

    for solver, nfe, power in experiments:
        key = experiment_key(
            solver,
            nfe,
            power,
        )

        if key in completed:
            print(f"\nSkipping {key}: " f"FID=" f"{completed[key]['fid']:.4f}")
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

        # Rewrite CSV after each successful run,
        # so partial progress is always preserved.
        successful = [
            record for record in completed.values() if record.get("status") == "ok"
        ]

        write_csv(
            csv_path,
            successful,
        )

    # ------------------------------------------------------------------
    # Final ranking
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_checkpoint",
        default=("checkpoints/hybrid_v2/" "few_nfe_epoch_0090.ckpt"),
    )

    parser.add_argument(
        "--data_root",
        default=("./data/" "pokemon-generation-one-22k"),
    )

    parser.add_argument(
        "--split_dir",
        default=("./data/" "pokemon-generation-one-22k"),
    )

    # Shared across all models.
    #
    # Keep this pointing at your existing reference directory
    # so we don't waste time exporting the validation set again.
    parser.add_argument(
        "--reference_dir",
        default=("./results/" "solver_sweep/" "reference"),
    )

    # None = automatically create a result directory
    # based on checkpoint name.
    parser.add_argument(
        "--output_dir",
        default=None,
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

    # Smaller diagnostic sweep by default.
    #
    # The most important question right now:
    #
    #     Is Hybrid V2 bad only at low NFE,
    #     or is its converged/high-NFE FID also bad?
    parser.add_argument(
        "--nfes",
        type=int,
        nargs="+",
        default=[
            4,
            16,
            64,
        ],
    )

    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=tuple(SOLVER_COST),
        default=[
            "euler",
            "midpoint",
        ],
    )

    # Additional schedule search at the actual
    # submission budget.
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

    # Re-run experiments even if that exact configuration
    # already exists for THIS checkpoint.
    parser.add_argument(
        "--force",
        action="store_true",
    )

    main(parser.parse_args())
