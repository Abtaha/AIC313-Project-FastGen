"""Verify the MeanFlow identity, stop-gradient, sampling convention, and AMP."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from model import Model, ModelOneNFE
from src.meanflow import meanflow_loss, sample_times
from src.models.dit import DiTConfig
from src.models.hybrid import HybridConfig
from src.models.attention import math_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


class LinearVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.2))
        self.b = nn.Parameter(torch.tensor(0.3))
        self.c = nn.Parameter(torch.tensor(0.4))
        self.grad_modes = []

    def forward(self, z, t, *, category, interval):
        self.grad_modes.append(torch.is_grad_enabled())
        return self.a * z + self.b * t[:, None, None, None] + self.c * interval[:, None, None, None]


class MeanFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_identity_and_detached_target_and_weight(self):
        images = torch.arange(8).reshape(2, 1, 2, 2).float() / 10
        noise = torch.flip(images, (0,)) + 0.1
        r, t = torch.tensor([0.1, 0.2]), torch.tensor([0.6, 0.9])
        category = torch.tensor([0, 1])
        dt = (t - r)[:, None, None, None]
        z = (1 - t[:, None, None, None]) * images + t[:, None, None, None] * noise
        v = noise - images
        # du/dt = a*v + b + c, because interval=t-r also varies with t.
        derivative = 0.2 * v + 0.3 + 0.4
        prediction = 0.2 * z + 0.3 * t[:, None, None, None] + 0.4 * dt
        target = v - dt * derivative
        error = prediction - target
        per_sample = error.square().flatten(1).mean(1)
        for power in (0.0, 1.0):
            with self.subTest(power=power):
                model = LinearVelocity()
                loss, mse = meanflow_loss(model, images, category, noise, r, t, power=power)
                weight = (per_sample + 1e-3).pow(-power)
                torch.testing.assert_close(loss, (weight * per_sample).mean())
                torch.testing.assert_close(mse, per_sample.mean())
                self.assertFalse(mse.requires_grad)
                loss.backward()
                weighted_error = error * weight[:, None, None, None]
                torch.testing.assert_close(model.a.grad, (2 * weighted_error * z).mean())
                torch.testing.assert_close(model.b.grad, (2 * weighted_error * t[:, None, None, None]).mean())
                torch.testing.assert_close(model.c.grad, (2 * weighted_error * dt).mean())
                self.assertEqual(model.grad_modes, [False, True])

    def test_diagonal_reduces_to_flow_matching(self):
        model = LinearVelocity()
        images, noise = torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2)
        t = torch.tensor([0.2, 0.7])
        loss, _ = meanflow_loss(model, images, torch.tensor([0, 1]), noise, t, t, power=0)
        z = t[:, None, None, None] * noise
        expected = model(z, t, category=torch.tensor([0, 1]), interval=torch.zeros_like(t))
        torch.testing.assert_close(loss, (expected - (noise - images)).square().mean())

    def test_time_sampler_has_ordered_and_diagonal_times(self):
        r, t = sample_times(32, "cpu")
        self.assertTrue(((r > 0) & (r <= t) & (t < 1)).all())
        self.assertEqual((r == t).sum().item(), 24)
        self.assertEqual((r < t).sum().item(), 8)

    def test_full_interval_sampler_and_legacy_checkpoint(self):
        model = ModelOneNFE(config=DiTConfig(width=16, depth=1, heads=2, patch_size=8))
        with torch.no_grad():
            model.backbone.output.bias.fill_(0.25)
        category = torch.tensor([0, 150])
        noise = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(5))
        calls = []
        hook = model.backbone.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append((args[1].clone(), kwargs["interval"].clone())),
            with_kwargs=True,
        )
        output = model.sample(noise.shape, device="cpu", category=category,
                              generator=torch.Generator().manual_seed(5))
        hook.remove()
        torch.testing.assert_close(output, (noise - 0.25).clamp(-1, 1))
        self.assertEqual(len(calls), 1)
        self.assertTrue((calls[0][0] == 1).all() and (calls[0][1] == 1).all())
        with tempfile.TemporaryDirectory() as tmp:
            state = model.state_dict()
            del state["_extra_state"]["objective"]
            path = Path(tmp) / "legacy.ckpt"
            torch.save({"state_dict": state}, path)
            legacy = Model.load_checkpoint(str(path), evaluate_mode="one_nfe")
            self.assertEqual(legacy.objective, "flow_matching")
            output = legacy.sample(noise.shape, device="cpu", category=category,
                                   generator=torch.Generator().manual_seed(5))
            torch.testing.assert_close(output, (noise + 0.25).clamp(-1, 1))

    def test_cpu_bfloat16_training_for_both_backbones(self):
        for config in (DiTConfig(width=16, depth=1, heads=2, patch_size=8),
                       HybridConfig(width=128, base_channels=32, depth=1, heads=2)):
            with self.subTest(backbone=config.backbone):
                model = ModelOneNFE(config=config)
                images, noise = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                for _ in range(3):
                    optimizer.zero_grad()
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        loss, mse = meanflow_loss(model, images, torch.tensor([0, 150]), noise,
                                                 torch.tensor([0.0, 0.2]), torch.tensor([0.7, 0.9]))
                    self.assertTrue(torch.isfinite(loss) and torch.isfinite(mse))
                    loss.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    optimizer.step()

    def test_jvp_replays_dropout_rng(self):
        class DropoutVelocity(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.1))
                self.dropout = nn.Dropout(0.2)

            def forward(self, z, t, *, category, interval):
                return self.dropout(z) * self.scale

        model = DropoutVelocity()
        images, noise = torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2)
        r, t = torch.zeros(2), torch.ones(2)
        state = torch.get_rng_state()
        meanflow_loss(model, images, torch.tensor([0, 1]), noise, r, t)
        actual_state = torch.get_rng_state()
        torch.set_rng_state(state)
        model(noise, t, category=torch.tensor([0, 1]), interval=t - r)
        self.assertTrue(torch.equal(actual_state, torch.get_rng_state()))

    def test_primitive_attention_matches_sdpa_jvp(self):
        q, k, v = [torch.randn(2, 2, 4, 8) for _ in range(3)]
        tangent = tuple(torch.randn_like(value) for value in (q, k, v))
        actual = torch.func.jvp(math_attention, (q, k, v), tangent)
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.func.jvp(torch.nn.functional.scaled_dot_product_attention,
                                      (q, k, v), tangent)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
