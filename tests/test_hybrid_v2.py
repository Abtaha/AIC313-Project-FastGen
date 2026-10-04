"""Hybrid-v2 budget, rotary geometry, gradients, checkpoints, and trainer resume."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from model import Model, ModelFewNFE, build_backbone
from src.meanflow import meanflow_loss
from src.models.hybrid import HybridFlowNet, TransformerBlock
from src.models.hybrid_v2 import HybridV2Config, HybridV2FlowNet, Rotary2D
from src.utils import count_parameters
from train import parse_args
import test_train_resume


class HybridV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_default_budget_and_layout(self):
        with torch.device("meta"):
            model = build_backbone(HybridV2Config())
            legacy = HybridFlowNet()
            self.assertEqual(count_parameters(legacy, False), 98_227_043)
            parts = model.parameter_breakdown()
            self.assertEqual(parts["total"], 99_595_363)
            self.assertEqual(parts["transformer"], 59_059_200)
            self.assertEqual(sum(v for k, v in parts.items() if k != "total"), parts["total"])
            self.assertLessEqual(parts["total"], 100_000_000)
            self.assertEqual((len(model.enc64), len(model.enc32), len(model.dec32), len(model.dec64)), (2, 3, 3, 3))
            self.assertEqual(len(model.transformer), 8)
            self.assertFalse(hasattr(model, "position"))
            self.assertEqual(model.transformer[0].rotary.head_dim, 80)
            result = model(torch.empty(1, 3, 64, 64), torch.ones(1), category=torch.zeros(1, dtype=torch.long))
            self.assertEqual(result.shape, (1, 3, 64, 64))
            with self.assertRaisesRegex(ValueError, "100M"):
                HybridV2FlowNet(HybridV2Config(depth=9))
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            HybridV2Config(width=128, base_channels=32, heads=64)

    def test_rotary_geometry_and_relative_position(self):
        rope = Rotary2D(80)
        vector = torch.randn(1, 1, 1, 80).expand(1, 1, 256, 80)
        result = rope.rotate(vector)
        torch.testing.assert_close(result.norm(dim=-1), vector.norm(dim=-1))
        # Origin is unchanged. Moving right affects x's 40 dimensions only.
        torch.testing.assert_close(result[..., 0, :], vector[..., 0, :])
        torch.testing.assert_close(result[..., 1, 40:], result[..., 0, 40:])
        self.assertFalse(torch.allclose(result[..., 1, :40], result[..., 0, :40]))
        # Moving down affects y's 40 dimensions only.
        torch.testing.assert_close(result[..., 16, :40], result[..., 0, :40])
        self.assertFalse(torch.allclose(result[..., 16, 40:], result[..., 0, 40:]))
        # Same displacement gives the same Q/K dot product after translation.
        q = rope.rotate(torch.randn(1, 1, 1, 80).expand(1, 1, 256, 80))
        k = rope.rotate(torch.randn(1, 1, 1, 80).expand(1, 1, 256, 80))
        torch.testing.assert_close((q[..., 0, :]*k[..., 1, :]).sum(),
                                   (q[..., 16, :]*k[..., 17, :]).sum())

    def test_rotary_changes_qk_but_preserves_values(self):
        block = TransformerBlock(80, 1, rotary=Rotary2D(80))
        x, condition = torch.randn(1, 256, 80), torch.randn(1, 80)
        raw = block.qkv(block.norm1(x)).reshape(1, 256, 3, 1, 80).permute(2, 0, 3, 1, 4)
        captured = []
        def attention(q, k, v, **kwargs):
            captured.extend((q, k, v))
            return v
        with patch("src.models.hybrid.scaled_attention", side_effect=attention):
            # Zero modulation makes the attention input equal norm1(x).
            torch.nn.init.zeros_(block.modulation[-1].weight)
            torch.nn.init.zeros_(block.modulation[-1].bias)
            block(x, condition)
        torch.testing.assert_close(captured[0], block.rotary.rotate(raw[0]))
        torch.testing.assert_close(captured[1], block.rotary.rotate(raw[1]))
        torch.testing.assert_close(captured[2], raw[2])

    def test_gradients_meanflow_and_checkpoint_roundtrip(self):
        config = HybridV2Config(width=128, base_channels=32, depth=1, heads=2)
        model = ModelFewNFE(config=config)
        images = torch.randn(1, 3, 64, 64)
        noise = torch.randn_like(images)
        categories = torch.zeros(1, dtype=torch.long)
        t = torch.tensor([.5])
        prediction = model((1-t[:, None, None, None])*noise+t[:, None, None, None]*images, t, category=categories)
        self.assertEqual(torch.count_nonzero(prediction).item(), 0)
        loss = (prediction - (images-noise)).square().mean()
        loss.backward()
        self.assertGreater(model.backbone.output.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.zero_grad()
        mf_loss, mf_mse = meanflow_loss(model, images, categories, noise, torch.tensor([.25]), t)
        mf_loss.backward()
        self.assertTrue(torch.isfinite(mf_loss) and torch.isfinite(mf_mse))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"few.ckpt"
            torch.save({"state_dict": model.state_dict()}, path)
            loaded = Model.load_checkpoint(str(path), "few_nfe")
            self.assertIsInstance(loaded.backbone, HybridV2FlowNet)
            self.assertEqual(loaded.backbone.architecture_config(), model.backbone.architecture_config())
            calls = []
            hook = loaded.backbone.register_forward_hook(lambda *args: calls.append(1))
            sample = loaded.sample(images.shape, device="cpu", category=categories)
            hook.remove()
            self.assertEqual(len(calls), 4)
            self.assertTrue(torch.isfinite(sample).all())

    def test_cli_and_exact_resume(self):
        with patch.object(sys, "argv", ["train.py", "--backbone", "hybrid_v2"]):
            args = parse_args()
        self.assertEqual(args.backbone, "hybrid_v2")
        # Reuse the existing integration test's exact optimizer/RNG comparison.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            test_train_resume.ResumeTests().check_exact_continuation(dict(backbone="hybrid_v2", width=128,
                base_channels=32, depth=1, specified=["depth"]))


if __name__ == "__main__":
    unittest.main()
