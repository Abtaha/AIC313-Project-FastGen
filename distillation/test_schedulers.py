"""Numerical and NFE tests for frozen-network learned samplers."""

from pathlib import Path
import tempfile
import unittest

import torch

from teacher import EDMTeacher, TeacherConfig
from .schedulers import (BezierSchedule, LD3Schedule, LearnedSampler,
                         matching_loss, project_latents)
from .train_scheduler import parser, train


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def teacher(self):
        return EDMTeacher(TeacherConfig(resolution=8, num_classes=3, model_channels=16,
            channel_mult=(1, 2), num_blocks=1, attn_resolutions=(4,), dropout=0)).eval().requires_grad_(False)

    def test_boundaries_monotonicity_and_derivatives(self):
        config = self.teacher().config
        schedule = BezierSchedule(config)
        with torch.no_grad():
            schedule.alpha_logits.normal_()
            schedule.beta_logits.normal_()
        time = torch.linspace(0, 1, 100, requires_grad=True)
        a, b, da, db = schedule.coefficients(time)
        self.assertTrue(torch.all(a[1:] > a[:-1]))
        self.assertTrue(torch.all(b[1:] < b[:-1]))
        torch.testing.assert_close(a[0], a.new_tensor(schedule.alpha_start))
        torch.testing.assert_close(a[-1], a.new_tensor(1))
        torch.testing.assert_close(b[0], b.new_tensor(schedule.beta_start))
        torch.testing.assert_close(b[-1], b.new_tensor(0))
        torch.testing.assert_close(torch.autograd.grad(a.sum(), time, retain_graph=True)[0], da, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.autograd.grad(b.sum(), time)[0], db, atol=1e-5, rtol=1e-5)
        ld3 = LD3Schedule(4, config)
        times = ld3.times()
        self.assertEqual(float(times[0]), config.sigma_max)
        self.assertEqual(float(times[-1]), 0)
        self.assertTrue(torch.all(times[:-1] > times[1:]))

    def test_ld3_matches_explicit_euler(self):
        teacher = self.teacher()
        model = LearnedSampler(teacher, method="ld3", nfe=4, guidance=1).eval()
        noise, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
        x = noise * teacher.config.sigma_max
        for current, following in zip(model.schedule.times()[:-1], model.schedule.times()[1:]):
            x = teacher.ode_step(x, current, following, labels, solver="euler")
        torch.testing.assert_close(model(noise, labels, rematerialize=False), x)

    def test_actual_nfe_and_roundtrip(self):
        for method, nfe, guidance in [("ld3", 4, 2), ("ld3", 4, 1), ("bezier", 4, 2), ("bezier", 1, 1)]:
            model = LearnedSampler(self.teacher(), method=method, nfe=nfe, guidance=guidance).eval()
            noise, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 2])
            calls = []
            handle = model.teacher.backbone.register_forward_hook(lambda *args: calls.append(1))
            output = model.sample(noise.shape, device="cpu", latents=noise, category=labels)
            self.assertEqual(len(calls), nfe)
            self.assertTrue(torch.isfinite(output).all())
            handle.remove()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sampler.ckpt"
                torch.save(model.export(), path)
                loaded = LearnedSampler.load_checkpoint(path)
                torch.testing.assert_close(output, loaded.sample(noise.shape, device="cpu", latents=noise, category=labels))
        with self.assertRaises(ValueError):
            LearnedSampler(self.teacher(), method="bezier", nfe=1, guidance=2)
        with self.assertRaises(ValueError):
            LearnedSampler(self.teacher(), method="ld3", nfe=1, guidance=1)

    def test_gradients_only_update_scheduler(self):
        for method, nfe, guidance in [("ld3", 4, 2), ("bezier", 4, 2), ("bezier", 1, 1)]:
            model = LearnedSampler(self.teacher(), method=method, nfe=nfe, guidance=guidance).train()
            frozen = [p.clone() for p in model.teacher.parameters()]
            noise = torch.randn(2, 3, 8, 8, requires_grad=True)
            loss = matching_loss(model(noise, torch.tensor([0, 2])), torch.randn_like(noise))
            loss.backward()
            gradients = [p.grad for p in model.schedule.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
            self.assertTrue(torch.isfinite(noise.grad).all())
            self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
            optimizer = torch.optim.Adam(model.schedule.parameters(), lr=0.01)
            optimizer.step()
            for old, current in zip(frozen, model.teacher.parameters()):
                torch.testing.assert_close(old, current, rtol=0, atol=0)
        proposal, original = torch.randn(3, 3, 8, 8), torch.randn(3, 3, 8, 8)
        projected = project_latents(proposal, original, 0.19)
        self.assertTrue(torch.all((projected-original).flatten(1).norm(dim=1) <= 0.190001))

    def test_train_cache_and_resume_without_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = self.teacher()
            teacher_file = root / "teacher.ckpt"
            torch.save({**teacher.checkpoint(), "category_to_id": {"a": 0, "b": 1, "c": 2}}, teacher_file)
            for method in ("ld3", "bezier"):
                args = parser().parse_args(["--teacher", str(teacher_file), "--outdir", str(root/method),
                    "--cache", str(root/"cache.pt"), "--method", method, "--steps", "1", "--batch-size", "2",
                    "--microbatch", "1", "--train-per-class", "1", "--teacher-steps", "2", "--device", "cpu"])
                train(args)
                args.steps, args.resume = 2, str(root/method/"training-state.pt")
                train(args)
                self.assertEqual(torch.load(args.resume, weights_only=True)["step"], 2)
                self.assertTrue((root/method/"best.ckpt").exists())


if __name__ == "__main__":
    unittest.main()
