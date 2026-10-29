"""Train the best measured one-pass lineage: refined CD -> DMD -> adversarial refinement."""
import shutil
import sys
from _common import ROOT, RECIPES, execute, options, parser, paths, stage, teacher_path


def main():
    args = parser(__doc__).parse_args()
    data = paths(args)
    teacher = teacher_path(args)
    initial = args.workdir / "few-refine/snapshots/student-002000.ckpt"
    if not args.dry_run and not initial.exists():
        raise FileNotFoundError("Run train_few_nfe.py first: its refined 2k EMA snapshot initializes this recipe.")
    execute([sys.executable, ROOT / "prepare_cache.py", "--teacher", teacher,
             "--workdir", args.workdir, *data], args.dry_run)
    common = [*data, "--teacher", str(teacher), "--student", str(initial),
              "--cache", str(args.workdir / "teacher-pairs.pt"), *options(RECIPES["dmd"])]
    output = args.workdir / "one-adversarial"
    stage("distillation.dmd", common, output, 2000,
          dry_run=args.dry_run, resume=args.resume)
    backup = output / "stage1-step2000"
    if args.dry_run:
        print(f"Preserve full stage-1 state in {backup}")
    elif not backup.exists():
        import torch
        state = torch.load(output / "training-state.pt", map_location="cpu", weights_only=True)
        if state["step"] == 2000:
            backup.mkdir()
            for name in ("training-state.pt", "one_nfe.ckpt", "config.json"):
                shutil.copy2(output / name, backup / name)
        del state
    stage("distillation.dmd", common, output, 5000,
          dry_run=args.dry_run, resume=True)


if __name__ == "__main__":
    main()
