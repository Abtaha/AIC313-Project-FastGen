"""Offline CPU verification of the standalone training package; no image download."""
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from _common import ROOT
import teacher
import dataset
import model
from teacher import EDMTeacher, TeacherConfig
from distillation.model import ConsistencyStudent
from distillation.train import parser as consistency_parser, train as train_consistency
from distillation.dmd import parser as dmd_parser, train as train_dmd
from distillation.support import target_cache
from training.teacher import parser as teacher_parser, train as train_teacher


def main():
    torch.set_num_threads(1)
    for module in (teacher, dataset, model):
        assert Path(module.__file__).resolve().is_relative_to(ROOT), module.__file__
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for name, digest in manifest.get("package_files", {}).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for name, entry in manifest["files"].items():
        if name.startswith("data/"):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == entry["sha256"]
    config = TeacherConfig(resolution=8, num_classes=3, model_channels=16,
                           channel_mult=(1,2), num_blocks=1, attn_resolutions=(4,), dropout=0)
    frozen = EDMTeacher(config).eval().requires_grad_(False)
    student = ConsistencyStudent.from_teacher(frozen).eval()
    student.set_four_sigmas([80,5,.5,.02])
    for nfe in (1,4):
        calls = []
        hook = student.backbone.register_forward_hook(lambda *args: calls.append(1))
        sample = student.sample((2,3,8,8), device="cpu", category=torch.tensor([0,2]), nfe=nfe)
        hook.remove()
        assert len(calls) == nfe and torch.isfinite(sample).all()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mapping = {"a":0,"b":1,"c":2}
        base = root / "teacher.ckpt"
        torch.save({**frozen.checkpoint(), "category_to_id":mapping}, base)
        (root / "category_to_id.json").write_text(json.dumps(mapping))
        fixture = torch.utils.data.TensorDataset(torch.randn(6,3,8,8), torch.tensor([0,1,2,0,1,2]))
        fixture.category_ids = [0,1,2,0,1,2]
        fixture.paths = [f"train/{index}.png" for index in range(6)]
        (root / "train_split.txt").write_text("train-only synthetic fixture\n")
        teacher_args = teacher_parser().parse_args(["--outdir",str(root/"teacher-run"),
            "--split-dir",str(root),"--steps","1","--batch-size","2","--microbatch","1",
            "--device","cpu","--amp","none","--num-workers","0","--loss-every","0","--fid-every","0"])
        with patch("training.teacher.PokemonDataset",return_value=fixture), patch("training.teacher.TeacherConfig",return_value=config):
            train_teacher(teacher_args)
            teacher_args.resume,teacher_args.steps = str(root/"teacher-run/training-state.pt"),2
            train_teacher(teacher_args)
        assert torch.load(root/"teacher-run/teacher.ckpt",weights_only=True)["step"] == 2
        args = consistency_parser().parse_args(["--teacher",str(base),"--outdir",str(root/"cd"),
            "--split-dir",str(root),"--steps","1","--batch-size","2","--microbatch","1",
            "--device","cpu","--amp","none","--num-workers","0","--warmup-steps","0"])
        with patch("distillation.train.PokemonDataset", return_value=fixture):
            train_consistency(args)
        args = dmd_parser().parse_args(["--teacher",str(base),"--student",str(root/"cd/student.ckpt"),
            "--cache",str(root/"pairs.pt"),"--outdir",str(root/"dmd"),"--steps","1",
            "--batch-size","2","--microbatch","1","--device","cpu","--amp","none","--score-warmup","0",
            "--fake-updates","3","--gan-weight",".003","--gan-start","1","--gan-reg-weight",".025"])
        args.teacher_steps,args.train_per_class = 2,2
        target_cache(args, frozen, hashlib.sha256(base.read_bytes()).hexdigest(), mapping, torch.device("cpu"))
        with patch("distillation.adversarial.PokemonDataset",return_value=fixture):
            train_dmd(args)
            before=torch.load(root/"dmd/training-state.pt",weights_only=True)
            args.resume,args.steps = str(root/"dmd/training-state.pt"),2
            train_dmd(args)
        state=torch.load(root/"dmd/training-state.pt",weights_only=True)
        assert state["step"] == 2
        assert int(next(iter(state["fake_optimizer"]["state"].values()))["step"]) == 6
        assert any(not torch.equal(value,state["gan_head"][key]) for key,value in before["gan_head"].items())
        rows=[json.loads(line) for line in (root/"dmd/stats.jsonl").read_text().splitlines()]
        assert [row["gan_active"] for row in rows] == [False,True]
        assert rows[-1]["reg_weight"] == .025
        # Export actual tiny checkpoint structures through the production entry point.
        import export_models
        for name,source,step in (("teacher/teacher.ckpt",base,100000),
                ("one-adversarial/snapshots/one_nfe-005000.ckpt",root/"dmd/one_nfe.ckpt",5000),
                ("few-refine/snapshots/student-002000.ckpt",root/"cd/student.ckpt",2000)):
            payload=torch.load(source,weights_only=True)
            payload["step"]=step
            destination=root/"exports-fixture"/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            torch.save(payload,destination)
        with patch("sys.argv",["export_models.py","--workdir",str(root/"exports-fixture"),"--outdir",str(root/"submission")]):
            export_models.main()
        exported=ConsistencyStudent.load_checkpoint(root/"submission/few_nfe.ckpt")
        torch.testing.assert_close(exported.four_sigmas,torch.tensor([80,5,.5,.02]))
        assert (root/"submission/manifest.json").exists()
    print("PASS: standalone imports, fixed splits, teacher train/resume, 1/4 actual NFE, consistency training, cache, adversarial stage transition/resume, and exports.")


if __name__ == "__main__":
    main()
