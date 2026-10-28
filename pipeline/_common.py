"""Shared command execution for the three fixed reproduction recipes."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RECIPES = json.loads((ROOT / "recipes.json").read_text())


def parser(description, *, teacher=True):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", type=Path, default=ROOT / "data/pokemon-generation-one-22k")
    p.add_argument("--split-dir", type=Path, default=ROOT / "data/pokemon-generation-one-22k")
    p.add_argument("--workdir", type=Path, default=ROOT / "runs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true", help="Resume incomplete stages; skip completed stages")
    p.add_argument("--dry-run", action="store_true", help="Print commands without training")
    if teacher:
        p.add_argument("--teacher", type=Path, help="Optional existing V1 teacher export")
    return p


def options(values):
    result = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            result.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            result.extend([flag, str(value)])
    return result


def paths(args):
    for name in ("data_root", "split_dir", "workdir"):
        setattr(args, name, getattr(args, name).resolve())
    return options({"data_root": args.data_root, "split_dir": args.split_dir, "device": args.device})


def teacher_path(args):
    return (args.teacher or args.workdir / "teacher/teacher.ckpt").resolve()


def execute(command, dry_run):
    command = [str(value) for value in command]
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def stage(module, arguments, output, steps, *, dry_run, resume=False, fork=None):
    state = output / "training-state.pt"
    restore = []
    if not dry_run and output.exists() and any(output.iterdir()) and not state.exists():
        raise FileExistsError(f"{output} is nonempty but has no complete training state; choose a new --workdir")
    if state.exists() and not dry_run:
        if not resume:
            raise FileExistsError(f"{output} exists; use --resume or a new --workdir")
        import torch
        saved = torch.load(state, map_location="cpu", weights_only=True)
        saved_step = saved["step"]
        del saved
        if saved_step >= steps:
            print(f"Already complete: {output} at step {saved_step}", flush=True)
            return
        restore = ["--resume", str(state)]
    elif dry_run and resume:
        restore = ["--resume", str(state)]
    elif fork:
        restore = ["--fork-from", str(fork)]
    execute([sys.executable, "-m", module, *arguments, "--outdir", output,
             "--steps", steps, *restore], dry_run)
