"""Multi-scale residual behavior, flow objectives, metadata, and exact resume."""

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
from src.models.hybrid import ResBlock
from src.models.inceptflow import InceptFlowConfig, InceptFlowNet, InceptionResBlock
from train import parse_args
import test_train_resume


class InceptFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_default_layout_budget_and_validation(self):
        with torch.device("meta"):
            net = build_backbone(InceptFlowConfig())
            parts = net.parameter_breakdown()
            self.assertLessEqual(parts["total"], 100_000_000)
            self.assertEqual(sum(v for k, v in parts.items() if k != "total"), parts["total"])
            self.assertEqual((net.config.width, net.config.heads, len(net.transformer)), (640, 8, 6))
            self.assertEqual((len(net.enc64), len(net.enc32), len(net.post_bottleneck)), (2, 2, 2))
            self.assertIsInstance(net.enc32[0], ResBlock)
            self.assertIsInstance(net.enc32[1], InceptionResBlock)
            self.assertIsInstance(net.dec32, InceptionResBlock)
            self.assertIsInstance(net.dec64, ResBlock)
            self.assertFalse(hasattr(net, "position"))
            self.assertEqual(net.transformer[0].rotary.head_dim, 80)
            result = net(torch.empty(1, 3, 64, 64), torch.ones(1), category=torch.zeros(1, dtype=torch.long))
            self.assertEqual(result.shape, (1, 3, 64, 64))
            with self.assertRaisesRegex(ValueError, "100M"):
                InceptFlowNet(InceptFlowConfig(depth=20))
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            InceptFlowConfig(width=128, base_channels=32, heads=64)

    def test_identity_skip_and_conditioned_branch_gradients(self):
        for in_channels in (32, 64):
            block = InceptionResBlock(32, 128, in_channels=in_channels)
            x, condition = torch.randn(2, in_channels, 8, 8), torch.randn(2, 128)
            torch.testing.assert_close(block(x, condition), block.skip(x), rtol=0, atol=0)
            self.assertEqual(block.branch4[-1].dilation, (2, 2))
            # Open the zero projection to test actual conditioning and all branches.
            torch.nn.init.xavier_uniform_(block.proj.weight)
            prediction = block(x, condition)
            self.assertFalse(torch.allclose(prediction, block(x, condition + 1)))
            prediction.square().mean().backward()
            for branch in (block.branch1, block.branch2, block.branch3, block.branch4, block.condition):
                grads = [p.grad for p in branch.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_flow_meanflow_and_checkpoint(self):
        config = InceptFlowConfig(width=128, base_channels=32, depth=1, heads=2)
        model = ModelFewNFE(config=config)
        images, categories = torch.randn(1, 3, 64, 64), torch.zeros(1, dtype=torch.long)
        noise, t = torch.randn_like(images), torch.tensor([.5])
        prediction = model(images, t, category=categories)
        self.assertEqual(torch.count_nonzero(prediction).item(), 0)
        (prediction - (images - noise)).square().mean().backward()
        self.assertGreater(model.backbone.output.weight.grad.abs().sum().item(), 0)
        model.zero_grad()
        # Exercise forward AD through active Inception/Transformer residual paths.
        torch.nn.init.normal_(model.backbone.output.weight, std=.01)
        for block in model.backbone.modules():
            if isinstance(block, InceptionResBlock):
                torch.nn.init.normal_(block.proj.weight, std=.01)
        torch.nn.init.normal_(model.backbone.transformer[0].modulation[-1].weight, std=.001)
        loss, mse = meanflow_loss(model, images, categories, noise, torch.tensor([.25]), t)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(mse))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "few.ckpt"
            torch.save({"state_dict": model.state_dict()}, path)
            loaded = Model.load_checkpoint(str(path), "few_nfe")
            self.assertIsInstance(loaded.backbone, InceptFlowNet)
            self.assertEqual(loaded.backbone.architecture_config(), model.backbone.architecture_config())
            torch.testing.assert_close(loaded(images, t, category=categories), model(images, t, category=categories))

    def test_cli_and_exact_resume(self):
        with patch.object(sys, "argv", ["train.py", "--backbone", "inceptflow"]):
            self.assertEqual(parse_args().backbone, "inceptflow")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            test_train_resume.ResumeTests().check_exact_continuation(dict(
                backbone="inceptflow", width=128, base_channels=32, depth=1,
                specified=["depth"]))


if __name__ == "__main__":
    unittest.main()
