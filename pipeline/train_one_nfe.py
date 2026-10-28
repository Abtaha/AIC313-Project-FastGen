"""Train the best measured one-pass lineage: 1k CD -> 5k DMD -> 7k 3:1 DMD."""
import sys
from _common import ROOT, RECIPES, execute, options, parser, paths, stage, teacher_path


def main():
    args = parser(__doc__).parse_args()
    data = paths(args)
    teacher = teacher_path(args)
    initial = args.workdir / "consistency/snapshots/student-001000.ckpt"
    if not args.dry_run and not initial.exists():
        raise FileNotFoundError("Run train_few_nfe.py first: its saved 1k snapshot initializes this recipe.")
    execute([sys.executable, ROOT / "prepare_cache.py", "--teacher", teacher,
             "--workdir", args.workdir, *data], args.dry_run)
    common = [*data, "--teacher", str(teacher), "--student", str(initial),
              "--cache", str(args.workdir / "teacher-pairs.pt"), *options(RECIPES["dmd"])]
    parent = args.workdir / "one-dmd"
    stage("distillation.dmd", [*common, "--fake-updates", "1"], parent, 5000,
          dry_run=args.dry_run, resume=args.resume)
    stage("distillation.dmd", [*common, "--fake-updates", "3"], args.workdir / "one-nfe", 7000,
          dry_run=args.dry_run, resume=args.resume, fork=parent / "training-state.pt")


if __name__ == "__main__":
    main()
