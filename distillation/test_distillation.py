"""CPU numerical, gradient, NFE, export, and training/resume checks."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from model import Model, ModelOneNFE, ModelFewNFE
from teacher import EDMTeacher, TeacherConfig
from .model import ConsistencyStudent
from .loss import consistency_loss, heun_target, noise_grid
from .train import parser, train


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def teacher(self):
        return EDMTeacher(TeacherConfig(resolution=8, num_classes=3, model_channels=16,
            channel_mult=(1, 2), num_blocks=1, attn_resolutions=(4,), dropout=0)).eval().requires_grad_(False)

    def test_boundary_and_teacher_initialization(self):
        teacher = self.teacher()
        student = ConsistencyStudent.from_teacher(teacher)
        x, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        torch.testing.assert_close(student(x, student.config.sigma_min, labels), x, rtol=0, atol=0)
        for left, right in zip(teacher.backbone.parameters(), student.backbone.parameters()):
            torch.testing.assert_close(left, right)
        self.assertEqual(student.config.label_dropout, 0)
        self.assertEqual(student.config.dropout, 0)

    def test_vector_heun_matches_teacher_scalar_steps(self):
        teacher = self.teacher()
        x, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        high, low = torch.tensor([2.0, 1.0]), torch.tensor([1.0, 0.5])
        actual = heun_target(teacher, x, high, low, labels, 2.0)
        expected = torch.cat([teacher.ode_step(x[i:i+1], high[i], low[i], labels[i:i+1], guidance=2.0)
                              for i in range(2)])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_loss_updates_only_student(self):
        teacher = self.teacher()
        student = ConsistencyStudent.from_teacher(teacher).train()
        target = copy.deepcopy(student).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        for loss_type in ("mse", "pseudo-huber"):
            optimizer.zero_grad()
            loss = consistency_loss(student, target, teacher, torch.randn(2, 3, 8, 8),
                torch.tensor([0, 2]), grid=noise_grid(student.config, 8, "cpu"), guidance=2.0,
                loss_type=loss_type)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [p.grad for p in student.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
            optimizer.step()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_actual_backbone_nfe_and_checkpoint_roundtrip(self):
        config = self.teacher().config
        images, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        for cls, mode, expected in ((ModelOneNFE, "one_nfe", 1), (ModelFewNFE, "few_nfe", 4)):
            model = cls(config=config).eval()
            calls = []
            hook = model.backbone.backbone.register_forward_hook(lambda *args: calls.append(1))
            sample = model.sample(images.shape, device="cpu", category=labels, latents=images,
                generator=torch.Generator().manual_seed(10))
            self.assertEqual(len(calls), expected)
            self.assertTrue(torch.isfinite(sample).all())
            self.assertLessEqual(float(sample.abs().max()), 1)
            hook.remove()
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "student.ckpt"
                torch.save(model.backbone.export(category_to_id={"a": 0, "b": 1, "c": 2}, step=1, guidance=2), checkpoint)
                loaded = Model.load_checkpoint(checkpoint, evaluate_mode=mode, config=config)
                repeated = loaded.sample(images.shape, device="cpu", category=labels, latents=images,
                    generator=torch.Generator().manual_seed(10))
                torch.testing.assert_close(sample, repeated)
                self.assertEqual(loaded.count_parameters(), model.count_parameters())

    def test_train_save_and_resume(self):
        teacher = self.teacher()
        dataset = TensorDataset(torch.randn(6, 3, 8, 8), torch.tensor([0, 1, 2, 0, 1, 2]))
        dataset.category_ids = [0, 1, 2, 0, 1, 2]
        dataset.paths = [f"train/{i}.png" for i in range(6)]
        mapping = {"a": 0, "b": 1, "c": 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "teacher.ckpt"
            torch.save({**teacher.checkpoint(), "category_to_id": mapping}, checkpoint)
            (root / "category_to_id.json").write_text(json.dumps(mapping))
            out = root / "run"
            args = parser().parse_args(["--teacher", str(checkpoint), "--outdir", str(out),
                "--split-dir", str(root), "--steps", "1", "--batch-size", "2", "--microbatch", "1",
                "--device", "cpu", "--amp", "none", "--num-workers", "0", "--warmup-steps", "0"])
            with patch("distillation.train.PokemonDataset", return_value=dataset):
                train(args)
                state = out / "training-state.pt"
                self.assertEqual(torch.load(state, weights_only=True)["step"], 1)
                args.resume, args.steps = str(state), 2
                train(args)
            self.assertEqual(torch.load(state, weights_only=True)["step"], 2)
            self.assertEqual(torch.load(out / "student.ckpt", weights_only=True)["step"], 2)
            self.assertFalse(json.loads((out / "config.json").read_text())["validation_loaded"])


if __name__ == "__main__":
    unittest.main()
