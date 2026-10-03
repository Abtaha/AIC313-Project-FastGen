"""Check architecture validation, dispatch, defaults, and evaluator integration."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluate import generate_samples
from model import Model, ModelFewNFE, build_backbone
from src.models.hybrid import HybridConfig, HybridFlowNet
from src.models.dit import DiTConfig
from src.utils import count_parameters
from train import parse_args


class HybridTests(unittest.TestCase):
    def test_invalid_configs(self):
        for settings in (dict(heads=0), dict(width=0), dict(depth=0),
                         dict(base_channels=16, width=64), dict(width=256),
                         dict(mlp_ratio=float("nan")), dict(dropout=1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                HybridConfig(**settings)

    def test_dispatch_and_default_budget(self):
        for config in (DiTConfig(), HybridConfig()):
            with self.subTest(backbone=config.backbone), torch.device("meta"):
                backbone = build_backbone(config)
                parameters = count_parameters(backbone, trainable_only=False)
                self.assertGreater(parameters, 98_000_000)
                self.assertLessEqual(parameters, 100_000_000)
                output = backbone(torch.empty(1, 3, 64, 64), torch.ones(1),
                                  category=torch.zeros(1, dtype=torch.long), interval=torch.ones(1))
                self.assertEqual(output.shape, (1, 3, 64, 64))
        with self.assertRaisesRegex(ValueError, "Unknown backbone"):
            build_backbone({"backbone": "unknown"})

    def test_evaluator_loads_wrapped_and_raw_hybrid_checkpoint(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            config = HybridConfig(width=128, base_channels=32, depth=1, heads=2)
            model = ModelFewNFE(config=config)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for wrapped in (False, True):
                    path = root / f"{wrapped}.ckpt"
                    state = model.state_dict()
                    torch.save({"state_dict": state} if wrapped else state, path)
                    loaded = Model.load_checkpoint(str(path), evaluate_mode="few_nfe", device="cpu")
                    self.assertIsInstance(loaded.backbone, HybridFlowNet)
                    self.assertEqual(loaded.backbone.architecture_config(), model.backbone.architecture_config())
                    out = root / str(wrapped)
                    generate_samples(loaded, out, torch.tensor([0, 150]), batch_size=2, device="cpu")
                    self.assertEqual(len(list(out.glob("*.png"))), 2)
        finally:
            torch.set_num_threads(old_threads)

    def test_cli_selects_hybrid(self):
        with patch.object(sys, "argv", ["train.py", "--backbone", "hybrid", "--width", "128", "--depth", "1"]):
            args = parse_args()
        self.assertEqual(args.backbone, "hybrid")
        self.assertIn("backbone", args.specified)
        self.assertIsNone(args.base_channels)

    def test_cli_defaults_match_dit_config(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = parse_args()
        config = DiTConfig()
        self.assertEqual((args.width, args.depth, args.heads, args.patch_size),
                         (config.width, config.depth, config.heads, config.patch_size))


if __name__ == "__main__":
    unittest.main()
