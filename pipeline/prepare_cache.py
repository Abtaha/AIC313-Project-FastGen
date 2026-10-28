"""Generate the guided teacher pairs used by the best one-pass recipe."""
from _common import RECIPES, parser, paths, teacher_path


def main():
    args = parser(__doc__).parse_args()
    paths(args)
    if args.dry_run:
        print("Create/verify 453 deterministic teacher pairs (302 fitting, 151 synthetic monitoring).")
        return
    import torch
    from teacher import EDMTeacher
    from distillation.train import file_hash
    from distillation.support import target_cache
    for key, value in RECIPES["cache"].items():
        setattr(args, key, value)
    args.cache = str(args.workdir / "teacher-pairs.pt")
    args.memory_limit_gb = 20.0
    args.device = torch.device(args.device)
    teacher = EDMTeacher.load_checkpoint(teacher_path(args), device=args.device)
    metadata = torch.load(teacher_path(args), map_location="cpu", weights_only=True)
    target_cache(args, teacher, file_hash(teacher_path(args)), metadata["category_to_id"], args.device)


if __name__ == "__main__":
    main()
