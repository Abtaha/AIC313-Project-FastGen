"""Train the best-performing teacher architecture from random initialization."""
from _common import RECIPES, options, parser, paths, stage


def main():
    args = parser(__doc__, teacher=False).parse_args()
    data = paths(args)
    stage("training.teacher", [*data, *options(RECIPES["teacher"])],
          args.workdir / "teacher", 100000, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
