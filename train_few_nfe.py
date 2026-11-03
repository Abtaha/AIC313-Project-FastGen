"""Train the 9k consistency student, then its 2k lower-LR refinement."""
from _common import RECIPES, options, parser, paths, stage, teacher_path


def main():
    args = parser(__doc__).parse_args()
    data = [*paths(args), "--teacher", str(teacher_path(args))]
    base = args.workdir / "consistency"
    stage("distillation.train", [*data, *options({**RECIPES["consistency"], "microbatch": 8})],
          base, 2000, dry_run=args.dry_run, resume=args.resume)
    stage("distillation.train", [*data, *options({**RECIPES["consistency"], "microbatch": 32})],
          base, 9000, dry_run=args.dry_run, resume=True)
    stage("distillation.train", [*data, *options({**RECIPES["consistency"], **RECIPES["refinement"]}),
          "--init-student", str(base / "snapshots/student-009000.ckpt")],
          args.workdir / "few-refine", 2000, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
