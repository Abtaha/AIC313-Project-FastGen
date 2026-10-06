"""Check V3's actual token grid, trainability, metadata, and exact resume."""

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
from src.models.hybrid_v3 import HybridV3Config, HybridV3FlowNet
from train import parse_args, train_mode
import test_train_resume


class HybridV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def small_config(self):
        return HybridV3Config(base_channels=32, width=128, heads=2)

    def test_topology_tokens_and_budget(self):
        with torch.device("meta"):
            model = build_backbone(HybridV3Config())
            parts = model.parameter_breakdown()
            self.assertEqual(parts["total"], 98_386_179)
            self.assertEqual(sum(v for k, v in parts.items() if k != "total"), parts["total"])
            self.assertEqual(model.config.channels, (128, 256, 384, 512, 512))
            shapes, tokens, decoder_shapes = [], [], []
            for stage in model.encoder:
                stage.register_forward_hook(lambda m, a, out: shapes.append(tuple(out.shape[1:])))
            for stage in model.decoder:
                stage.register_forward_hook(lambda m, a, out: decoder_shapes.append(tuple(out.shape[1:])))
            for block in model.transformer:
                block.register_forward_pre_hook(lambda m, a: tokens.append(tuple(a[0].shape)))
                self.assertEqual(block.rotary.tokens, 64)
            result = model(torch.empty(1, 3, 64, 64), torch.ones(1), category=torch.zeros(1, dtype=torch.long))
            self.assertEqual(result.shape, (1, 3, 64, 64))
            self.assertEqual(shapes, [(128, 64, 64), (256, 32, 32), (384, 16, 16), (512, 8, 8), (512, 4, 4)])
            self.assertEqual(decoder_shapes, shapes[::-1])
            self.assertEqual(tokens, [(1, 64, 512)] * 2)
            self.assertTrue(all(len(s.blocks) == 2 for s in (*model.encoder, *model.decoder)))
            self.assertEqual(len(model.bottleneck.blocks), 1)
            with self.assertRaisesRegex(ValueError, "100M"):
                HybridV3FlowNet(HybridV3Config(depth=3))

    def test_gradients_conditioning_and_meanflow(self):
        model = ModelFewNFE(config=self.small_config())
        x, t, y = torch.randn(2, 3, 64, 64), torch.tensor([.2, .7]), torch.tensor([0, 150])
        self.assertEqual(torch.count_nonzero(model(x, t, category=y)).item(), 0)
        torch.nn.init.normal_(model.backbone.output.weight, std=.02)
        # Open adaLN-Zero gates to check QKV and the entire convolutional path.
        for block in model.backbone.transformer:
            torch.nn.init.normal_(block.modulation[-1].weight, std=.02)
        output = model(x, t, category=y, interval=t / 2)
        output.square().mean().backward()
        for p in (model.backbone.stem.weight, model.backbone.transformer[0].qkv.weight,
                  model.backbone.transformer[-1].qkv.weight, model.backbone.category.weight,
                  model.backbone.time.net[-1].weight, model.backbone.interval.net[-1].weight):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        with torch.no_grad():
            self.assertFalse(torch.allclose(output, model(x, t, category=y.flip(0), interval=t / 2)))
        model.zero_grad()
        loss, mse = meanflow_loss(model, x, y, torch.randn_like(x), t / 2, t)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(mse))

    def test_checkpoint_and_sampling(self):
        model = ModelFewNFE(config=self.small_config()).eval()
        torch.nn.init.normal_(model.backbone.output.weight, std=.02)
        x, t, y = torch.randn(1, 3, 64, 64), torch.ones(1), torch.tensor([150])
        with tempfile.TemporaryDirectory() as tmp:
            for wrapped in (False, True):
                path = Path(tmp) / f"{wrapped}.ckpt"
                state = model.state_dict()
                torch.save({"state_dict": state} if wrapped else state, path)
                loaded = Model.load_checkpoint(str(path), "few_nfe")
                self.assertIsInstance(loaded.backbone, HybridV3FlowNet)
                self.assertEqual(loaded.backbone.architecture_config(), model.backbone.architecture_config())
                torch.testing.assert_close(loaded(x, t, category=y), model(x, t, category=y))
                calls = []
                hook = loaded.backbone.register_forward_hook(lambda *args: calls.append(1))
                samples = loaded.sample(x.shape, device="cpu", category=y)
                hook.remove()
                self.assertEqual(len(calls), 4)
                self.assertTrue(torch.isfinite(samples).all())

    def test_cli_defaults_and_resume(self):
        with patch.object(sys, "argv", ["train.py", "--backbone", "hybrid_v3"]):
            args = parse_args()
        captured = []
        def capture(**kwargs):
            captured.append(kwargs["config"])
            raise RuntimeError("captured")
        with patch("train.ModelFewNFE", side_effect=capture), self.assertRaisesRegex(RuntimeError, "captured"):
            train_mode(args, test_train_resume.TinyDataModule(), torch.device("cpu"), "few_nfe")
        self.assertEqual(captured[0], HybridV3Config())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            test_train_resume.ResumeTests().check_exact_continuation(dict(
                backbone="hybrid_v3", width=128, base_channels=32, depth=1,
                specified=["width", "depth"],
            ))

    def test_invalid_config(self):
        for kwargs in (dict(base_channels=16), dict(width=640), dict(depth=0),
                       dict(heads=5), dict(heads=0), dict(dropout=1),
                       dict(mlp_ratio=float("nan")), dict(backbone="edm_unet")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HybridV3Config(**kwargs)
