"""Verify multiscale topology, conditioning, and trainer/evaluator integration."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from model import Model, ModelFewNFE, build_backbone
from src.models.edm_unet import EDMUNetConfig, EDMUNetFlowNet, SpatialAttention
from src.utils import count_parameters
from train import parse_args, train_mode
import test_train_resume


class EDMUNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def small_config(self):
        return EDMUNetConfig(base_channels=32, width=128, heads=2)

    def test_default_topology_and_budget(self):
        with torch.device("meta"):
            model = build_backbone(EDMUNetConfig())
            self.assertEqual(model.config.channels, (128, 256, 384, 512, 512))
            self.assertEqual(count_parameters(model, trainable_only=False), 94_179_587)
            self.assertLessEqual(count_parameters(model, trainable_only=False), 100_000_000)
            encoder_shapes, decoder_shapes, attention_shapes = [], [], []
            for stages, shapes in ((model.encoder, encoder_shapes), (model.decoder, decoder_shapes)):
                for stage in stages:
                    stage.register_forward_hook(lambda module, args, output, shapes=shapes: shapes.append(tuple(output.shape[1:])))
            for module in model.modules():
                if isinstance(module, SpatialAttention):
                    module.register_forward_hook(lambda module, args, output: attention_shapes.append(output.shape[-1]))
            output = model(torch.empty(1, 3, 64, 64), torch.ones(1), category=torch.zeros(1, dtype=torch.long))
            self.assertEqual(output.shape, (1, 3, 64, 64))
            self.assertEqual(encoder_shapes, [(128, 64, 64), (256, 32, 32), (384, 16, 16), (512, 8, 8), (512, 4, 4)])
            self.assertEqual(decoder_shapes, encoder_shapes[::-1])
            self.assertEqual(attention_shapes, [16, 4, 16])
            self.assertTrue(all(len(stage.blocks) == 2 for stage in (*model.encoder, *model.decoder)))
            self.assertEqual(len(model.bottleneck.blocks), 2)

    def test_conditioning_and_backward(self):
        model = build_backbone(self.small_config())
        x = torch.randn(2, 3, 64, 64)
        t, y = torch.tensor([0.2, 0.7]), torch.tensor([0, 150])
        self.assertEqual(torch.count_nonzero(model(x, t, category=y)).item(), 0)
        # Open the zero-initialized output head to test the entire gradient path.
        torch.nn.init.normal_(model.output.weight, std=0.02)
        prediction = model(x, t, category=y, interval=t / 2)
        prediction.square().mean().backward()
        for parameter in (model.stem.weight, model.category.weight,
                          model.time.net[-1].weight, model.interval.net[-1].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        with torch.no_grad():
            self.assertFalse(torch.allclose(prediction, model(x, t, category=y.flip(0), interval=t / 2)))
            self.assertFalse(torch.allclose(prediction, model(x, t.flip(0), category=y, interval=t / 2)))

    def test_checkpoint_and_four_nfe_sampling(self):
        model = ModelFewNFE(config=self.small_config()).eval()
        torch.nn.init.normal_(model.backbone.output.weight, std=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            for wrapped in (False, True):
                path = Path(tmp) / f"{wrapped}.ckpt"
                state = model.state_dict()
                torch.save({"state_dict": state} if wrapped else state, path)
                loaded = Model.load_checkpoint(str(path), "few_nfe", device="cpu")
                self.assertIsInstance(loaded.backbone, EDMUNetFlowNet)
                self.assertEqual(loaded.backbone.architecture_config(), model.backbone.architecture_config())
                x, t, y = torch.randn(1, 3, 64, 64), torch.ones(1), torch.tensor([150])
                torch.testing.assert_close(loaded(x, t, category=y), model(x, t, category=y))
                calls = []
                hook = loaded.backbone.register_forward_hook(lambda *args: calls.append(1))
                samples = loaded.sample(x.shape, device="cpu", category=y)
                hook.remove()
                self.assertEqual(len(calls), 4)
                self.assertTrue(torch.isfinite(samples).all())

    def test_cli_defaults_used_by_trainer(self):
        with patch.object(sys, "argv", ["train.py", "--backbone", "edm_unet"]):
            args = parse_args()
        self.assertEqual(args.backbone, "edm_unet")
        # Stop immediately after construction to check resolved defaults cheaply.
        captured = []
        def capture(**kwargs):
            captured.append(kwargs["config"])
            raise RuntimeError("captured config")
        with patch("train.ModelFewNFE", side_effect=capture), self.assertRaisesRegex(RuntimeError, "captured config"):
            train_mode(args, test_train_resume.TinyDataModule(), torch.device("cpu"), "few_nfe")
        self.assertEqual(captured[0], EDMUNetConfig())

    def test_invalid_configs_and_inputs(self):
        for kwargs in (dict(base_channels=16), dict(width=640), dict(depth=0),
                       dict(heads=5), dict(heads=0), dict(dropout=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EDMUNetConfig(**kwargs)
        model = build_backbone(self.small_config())
        for x, t, y, interval in (
            (torch.zeros(1, 3, 32, 32), torch.ones(1), torch.zeros(1, dtype=torch.long), None),
            (torch.zeros(1, 3, 64, 64), torch.ones(2), torch.zeros(1, dtype=torch.long), None),
            (torch.zeros(1, 3, 64, 64), torch.ones(1), torch.zeros(2, dtype=torch.long), None),
            (torch.zeros(1, 3, 64, 64), torch.ones(1), torch.zeros(1, dtype=torch.long), torch.ones(2)),
        ):
            with self.assertRaises(ValueError):
                model(x, t, category=y, interval=interval)

    def test_exact_training_continuation(self):
        test_train_resume.ResumeTests().check_exact_continuation(dict(
            backbone="edm_unet", width=128, base_channels=32, depth=1,
            specified=["width", "depth"],
        ))


if __name__ == "__main__":
    unittest.main()
