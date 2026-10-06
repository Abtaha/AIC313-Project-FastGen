"""Numerical, NFE, checkpoint, and training/resume checks; no dataset downloads."""

import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from teacher import EDMTeacher, TeacherConfig
from teacher_monitor import denoising_losses, holdout_fid, stratified_holdout
from train_teacher import parser, train, update_ema


class ZeroBackbone(nn.Module):
    def forward(self, x, noise, labels):
        return torch.zeros_like(x)


class TeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tiny(self):
        return EDMTeacher(TeacherConfig(resolution=8, num_classes=3, model_channels=16,
                                        channel_mult=(1, 2), num_blocks=1,
                                        attn_resolutions=(4,), dropout=0))

    def test_preconditioning_and_sigma_validation(self):
        model = self.tiny()
        model.backbone = ZeroBackbone()
        x = torch.randn(2, 3, 8, 8)
        sigma = torch.tensor([0.1, 2.0])
        expected = 0.25 / (sigma[:, None, None, None].square() + 0.25) * x
        torch.testing.assert_close(model(x, sigma, torch.tensor([0, 1])), expected)
        torch.testing.assert_close(model(x, sigma, None), expected)
        with self.assertRaises(ValueError):
            model(x, 0, None)

    def test_loss_backprop_and_ema(self):
        model = self.tiny().train()
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        before = next(ema.parameters()).clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = model.loss(torch.randn(2, 3, 8, 8), torch.tensor([0, 2]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        update_ema(ema, model, 0.5)
        torch.testing.assert_close(next(ema.parameters()), (before + next(model.parameters())) / 2)

    def test_cfg_labels_and_actual_nfe(self):
        model = self.tiny().eval()
        x = torch.randn(2, 3, 8, 8)
        category = torch.tensor([0, 2])
        labels = []
        handle = model.backbone.register_forward_pre_hook(lambda module, inputs: labels.append(inputs[2].clone()))
        conditional = model(x, 1.0, category)
        unconditional = model(x, 1.0, None)
        torch.testing.assert_close(model.guided(x, 1.0, category, 2.0),
                                   unconditional + 2 * (conditional - unconditional))
        self.assertTrue((labels[1] == 0).all())
        for solver, guidance, expected in (("heun", 1, 5), ("heun", 2, 10), ("euler", 1, 3)):
            labels.clear()
            result = model.sample(x.shape, latents=x, category=category, num_steps=3,
                                  solver=solver, guidance=guidance, return_trajectory=True)
            self.assertEqual(len(labels), expected)
            self.assertEqual(result["nfe"], expected)
            self.assertEqual(result["states"].shape, (4, 2, 3, 8, 8))
            self.assertEqual(result["sigmas"][-1], 0)
            self.assertTrue(torch.isfinite(result["samples"]).all())
            repeated = model.sample(x.shape, latents=x, category=category, num_steps=3,
                                    solver=solver, guidance=guidance, return_trajectory=True)
            torch.testing.assert_close(result["states"], repeated["states"])
        handle.remove()

    def test_checkpoint_roundtrip(self):
        model = self.tiny().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.ckpt"
            torch.save(model.checkpoint(), path)
            loaded = EDMTeacher.load_checkpoint(path)
            self.assertFalse(loaded.training)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            x = torch.randn(2, 3, 8, 8)
            torch.testing.assert_close(model(x, 0.5, torch.tensor([1, 2])),
                                       loaded(x, 0.5, torch.tensor([1, 2])))

    def test_default_framework_under_parameter_cap(self):
        model = EDMTeacher()
        self.assertLessEqual(2 * model.num_parameters, 100_000_000)

    def test_train_holdout_is_disjoint(self):
        train_ids, holdout = stratified_holdout([0] * 10 + [1] * 10, 0.2, 42)
        self.assertFalse(set(train_ids) & set(holdout))
        self.assertEqual(len(holdout), 4)
        self.assertEqual(sorted(train_ids + holdout), list(range(20)))

    def test_fixed_sigma_losses_reproducible_and_fid_png_pipeline(self):
        model = self.tiny().eval()
        dataset = [(torch.zeros(3, 8, 8), torch.tensor(i % 3)) for i in range(6)]
        kwargs = {"device": torch.device("cpu"), "batch_size": 2, "seed": 7, "limit": 4}
        first = denoising_losses(model, dataset, list(range(6)), **kwargs)
        self.assertEqual(first, denoising_losses(model, dataset, list(range(6)), **kwargs))
        self.assertEqual(set(first), {"0.1", "0.5", "2.0", "10.0"})

        def compute_fid(generated, reference, **kwargs):
            self.assertEqual(len(list(Path(generated).glob("*.png"))), 5)
            self.assertEqual(len(list(Path(reference).glob("*.png"))), 3)
            for file in Path(generated).glob("*.png"):
                with Image.open(file) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (8, 8))
            return 12.0

        fake_module = types.SimpleNamespace(fid=types.SimpleNamespace(compute_fid=compute_fid))
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"cleanfid": fake_module}):
                score = holdout_fid(model, dataset, [0, 1, 2], torch.ones(3) / 3,
                                    outdir=directory, device=torch.device("cpu"),
                                    batch_size=2, count=5, steps=2, guidance=2)
            self.assertEqual(score, 12.0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_training_monitor_and_resume_without_validation_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for i in range(8):
                category = "Abra" if i < 4 else "Pikachu"
                relative = f"PokemonData/{category}/{i}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 64), (i * 30, 100, 255)).save(path)
                paths.append(relative)
            (root / "train_split.txt").write_text("\n".join(paths))
            (root / "category_to_id.json").write_text(json.dumps({"Abra": 0, "Pikachu": 1}))
            (root / ".transparency_composited_on_white").touch()
            run = root / "run"
            argv = ["--device", "cpu", "--amp", "none", "--data-root", str(root),
                    "--split-dir", str(root), "--outdir", str(run), "--steps", "1",
                    "--batch-size", "2", "--microbatch", "1", "--model-channels", "16",
                    "--num-blocks", "1", "--num-workers", "0", "--loss-every", "1",
                    "--loss-count", "2", "--fid-every", "1", "--sample-every", "1",
                    "--sample-steps", "2", "--save-every", "1"]
            # Exercise selection/checkpoint plumbing without downloading Inception.
            with patch("train_teacher.holdout_fid", return_value=42.0):
                train(parser().parse_args(argv))
            self.assertTrue((run / "best-fid.ckpt").exists())
            self.assertTrue((run / "best-loss.ckpt").exists())
            loaded = EDMTeacher.load_checkpoint(run / "teacher.ckpt")
            self.assertEqual(loaded.config.num_classes, 2)
            argv[argv.index("--steps") + 1] = "2"
            argv += ["--resume", str(run / "training-state.pt")]
            with patch("train_teacher.holdout_fid", return_value=41.0):
                train(parser().parse_args(argv))
            state = torch.load(run / "training-state.pt", weights_only=True)
            self.assertEqual(state["step"], 2)
            self.assertEqual(state["seen_images"], 4)
            self.assertEqual(state["best_fid"], 41.0)
            self.assertFalse((root / "val_split.txt").exists())


if __name__ == "__main__":
    unittest.main()
