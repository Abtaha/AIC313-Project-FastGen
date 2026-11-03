"""Retain atomic teacher saves, including from an already-running trainer."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

import torch


def retain(source, directory, step, prefix):
    """Hard-link an immutable atomic save; subsequent replacements stay independent."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{prefix}-{step:07d}{source.suffix}"
    if not target.exists():
        os.link(source, target)
        print(json.dumps({"retained": str(target), "step": step}), flush=True)
    return target


def scan(output, seen, intervals):
    for name, prefix, interval in intervals:
        source = output / name
        try:
            identity = source.stat().st_ino
        except FileNotFoundError:
            continue
        if seen.get(name) == identity:
            continue
        # Pin the inode before reading: training may atomically replace source meanwhile.
        staging = output / f".retain-{uuid.uuid4().hex}{source.suffix}"
        try:
            os.link(source, staging)
            identity = staging.stat().st_ino
            state = torch.load(staging, map_location="cpu", weights_only=True)
            step = int(state["step"])
            del state
            if name not in seen or step % interval == 0:
                retain(staging, output / "snapshots", step, prefix)
            seen[name] = identity
        except FileNotFoundError:
            pass
        finally:
            staging.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--snapshot-every", type=int, default=5000)
    parser.add_argument("--state-snapshot-every", type=int, default=20000)
    parser.add_argument("--exit-status", type=Path, required=True)
    args = parser.parse_args()
    if min(args.snapshot_every, args.state_snapshot_every) < 1:
        parser.error("retention intervals must be positive")
    seen = {}
    intervals = [("teacher.ckpt", "teacher", args.snapshot_every),
                 ("training-state.pt", "training-state", args.state_snapshot_every)]
    while True:
        # Check completion before scanning to include the final save on exit.
        finished = args.exit_status.exists()
        scan(args.outdir, seen, intervals)
        if finished:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
