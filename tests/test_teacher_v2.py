"""V2 architecture, balancing, compatibility, and synthetic resume checks."""

import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from teacher_v2.networks import MPConv, mp_sum, normalize
from teacher_v2 import EDMTeacherV2, TeacherV2Config
from teacher_v2.data import balanced_sampler, pixel_statistics
from teacher_v2.train import train, learning_rate_schedule
from teacher_v2.train import v2_parser
from teacher_v2.evaluate import competition_fid, export_reference, export_generated


class TeacherV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tiny(self):
        return EDMTeacherV2(TeacherV2Config(
            resolution=16, num_classes=3,
            model_channels=16, channel_mult=(1, 2), num_blocks=1,
            attn_resolutions=(8,), channels_per_head=8, dropout=0,
        ))

    def test_balanced_sampler_equal_class_probability(self):
        labels = torch.tensor([0] * 10 + [1] * 90)
        sampler = balanced_sampler(labels, num_classes=2,
                                   generator=torch.Generator().manual_seed(7))
        self.assertTrue(sampler.replacement)
        self.assertEqual(len(sampler), 100)
        self.assertAlmostEqual(float(sampler.weights[labels == 0].sum()), 1)
        self.assertAlmostEqual(float(sampler.weights[labels == 1].sum()), 1)
        draws = torch.multinomial(sampler.weights, 10000, replacement=True,
                                  generator=torch.Generator().manual_seed(123))
        self.assertLess(abs(float(labels[draws].float().mean()) - 0.5), 0.02)
        with self.assertRaises(ValueError):
            balanced_sampler([0, 0], num_classes=2, generator=None)

    def test_statistics_known_mean_rms_std(self):
        images = torch.stack([torch.zeros(3, 8, 8), torch.ones(3, 8, 8)])
        result = pixel_statistics(DataLoader(TensorDataset(images, torch.tensor([0, 1])), batch_size=1))
        self.assertEqual(result["global_mean"], 0.5)
        self.assertEqual(result["global_std"], 0.5)
        self.assertAlmostEqual(result["global_rms"], 2 ** -0.5)

    def test_sdpa_matches_explicit_attention_and_gradients(self):
        torch.manual_seed(1)
        q, k, v = [normalize(torch.randn(2, 3, 8, 16), dim=2).requires_grad_() for _ in range(3)]
        explicit = torch.einsum("nhcq,nhck->nhqk", q, k / (8 ** 0.5)).softmax(-1)
        expected = torch.einsum("nhqk,nhck->nhcq", explicit, v)
        actual = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        ).transpose(-2, -1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        probe = torch.randn_like(expected)
        grad_expected = torch.autograd.grad((expected * probe).sum(), (q, k, v))
        grad_actual = torch.autograd.grad((actual * probe).sum(), (q, k, v))
        for a, b in zip(grad_actual, grad_expected):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_mp_sum_controls_independent_branch_rms(self):
        torch.manual_seed(0)
        a, b = torch.randn(100000), torch.randn(100000)
        rms = mp_sum(a, b, 0.3).square().mean().sqrt()
        self.assertLess(abs(float(rms) - 1), 0.01)

    def test_uncertainty_loss_matches_equation_and_head_receives_gradients(self):
        model = self.tiny().eval()
        images = torch.randn(2, 3, 16, 16)
        categories = torch.tensor([0, 1])
        generator = torch.Generator().manual_seed(123)
        sigma = (torch.randn((2, 1, 1, 1), generator=generator) * model.config.p_std + model.config.p_mean).exp()
        noise = torch.randn(images.shape, generator=generator)
        prediction, logvar = model(images + sigma * noise, sigma, categories, return_logvar=True)
        self.assertEqual(logvar.shape, (2, 1, 1, 1))
        weight = (sigma.square() + model.config.sigma_data ** 2) / (sigma * model.config.sigma_data).square()
        expected = (weight * (-logvar).exp() * (prediction - images).square() + logvar).mean()
        actual = model.loss(images, categories, generator=torch.Generator().manual_seed(123))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(float(model.logvar_linear.weight.grad.abs().sum()), 0)
        calls = []
        hook = model.logvar_linear.register_forward_hook(lambda *args: calls.append(1))
        model.sample(images.shape, latents=images, category=categories, num_steps=2)
        hook.remove()
        self.assertEqual(len(calls), 0)

    def test_lr_warmup_and_inverse_square_root_decay(self):
        kwargs = {"ref_lr": 2e-4, "rampup_kimg": 100, "decay_ref_kimg": 128}
        self.assertEqual(learning_rate_schedule(0, **kwargs), 0)
        self.assertAlmostEqual(learning_rate_schedule(50000, **kwargs), 1e-4)
        self.assertAlmostEqual(learning_rate_schedule(100000, **kwargs), 2e-4)
        self.assertAlmostEqual(learning_rate_schedule(512000, **kwargs), 1e-4)

    def test_full_default_teacher_and_ema_with_heads_under_cap(self):
        model = EDMTeacherV2()
        self.assertLessEqual(2 * model.num_parameters, 100_000_000)
        self.assertGreater(model.num_parameters, 43_000_000)

    def test_v2_entrypoints_do_not_import_v1_modules(self):
        source = """
import sys
import teacher_v2.model, teacher_v2.train, teacher_v2.data
import teacher_v2.evaluate, teacher_v2.sample
for name in ('teacher', 'train_teacher', 'teacher_monitor', 'src.edm.networks'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", source], check=True, capture_output=True)

    def test_fid_population_and_pixels_match_provided_evaluator(self):
        # Pixel/protocol checks do not need Inception or scipy binaries.
        fake_fid = SimpleNamespace(compute_fid=Mock())
        with patch.dict("sys.modules", {"cleanfid": SimpleNamespace(fid=fake_fid)}):
            from evaluate import prepare_reference_set, generate_samples

        images = torch.linspace(-1, 1, 40 * 3 * 64 * 64).reshape(40, 3, 64, 64)
        categories = torch.arange(2).repeat_interleave(20)
        dataset = TensorDataset(images, categories)
        class FakeTeacher:
            config = SimpleNamespace(channels=3, resolution=64)
            def eval(self):
                return self
            def sample(self, shape, *, category, **kwargs):
                return (category.float()[:, None, None, None] * 0.456 - 0.123).expand(shape)
        model = FakeTeacher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref, gen = root / "ref", root / "gen"
            ref.mkdir(); gen.mkdir()
            export_reference(dataset, ref, num_classes=2, batch_size=7)
            export_generated(model, gen, num_classes=2, batch_size=7, device="cpu",
                             steps=18, guidance=1, seed=1234)
            provided_ref, provided_gen = root / "provided_ref", root / "provided_gen"
            module = SimpleNamespace(val_dataloader=lambda: DataLoader(dataset, batch_size=7))
            prepare_reference_set(module, provided_ref)
            generate_samples(model, provided_gen, categories, batch_size=7, device="cpu")
            for filename in ref.glob("*.png"):
                self.assertEqual(filename.read_bytes(), (provided_ref / filename.name).read_bytes())
            for filename in gen.glob("*.png"):
                self.assertEqual(filename.read_bytes(), (provided_gen / filename.name).read_bytes())
            def score(generated, reference, **kwargs):
                self.assertEqual(len(list(Path(generated).glob("*.png"))), 40)
                self.assertEqual(len(list(Path(reference).glob("*.png"))), 40)
                self.assertEqual(kwargs["num_workers"], 4)
                return 30.0
            metric = Mock(side_effect=score)
            with patch.dict("sys.modules", {"cleanfid": SimpleNamespace(fid=SimpleNamespace(compute_fid=metric))}):
                result = competition_fid(model, dataset, num_classes=2, outdir=root,
                                         device="cpu", batch_size=7)
            self.assertEqual(result, 30)
            self.assertEqual(metric.call_count, 1)
            self.assertEqual(list(root.glob("v2-fid-*")), [])

    def test_v2_accumulation_projection_dropout_and_checkpoint(self):
        model = self.tiny().train()
        model.backbone.out_gain.data.fill_(0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            model.loss(torch.randn(2, 3, 16, 16), torch.tensor([0, 1])).backward()
        attention_grad = model.backbone.enc["8x8_block0"].attn_qkv.weight.grad
        self.assertTrue(torch.isfinite(attention_grad).all())
        self.assertGreater(float(attention_grad.abs().sum()), 0)
        optimizer.step()
        model.normalize_weights()
        for module in model.modules():
            if isinstance(module, MPConv):
                rms = module.weight.square().flatten(1).mean(1).sqrt()
                torch.testing.assert_close(rms, torch.ones_like(rms), atol=2e-4, rtol=0)
        seen_labels = []
        hook = model.backbone.register_forward_pre_hook(lambda m, inputs: seen_labels.append(inputs[2]))
        with patch("teacher_v2.model.torch.rand", return_value=torch.zeros(2, 1)):
            model(torch.randn(2, 3, 16, 16), 1.0, torch.tensor([0, 1]))
        self.assertEqual(float(seen_labels[-1].sum()), 0)
        model.eval()(torch.randn(2, 3, 16, 16), 1.0, torch.tensor([0, 1]))
        self.assertEqual(float(seen_labels[-1].sum()), 2)
        hook.remove()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2.ckpt"
            torch.save(model.checkpoint(), path)
            restored = EDMTeacherV2.load_checkpoint(path)
            x = torch.randn(2, 3, 16, 16)
            labels = torch.tensor([0, 1])
            torch.testing.assert_close(model(x, 1.0, labels), restored(x, 1.0, labels))
            calls = []
            hook = restored.backbone.register_forward_hook(lambda *args: calls.append(1))
            result = restored.sample(x.shape, latents=x, category=labels, num_steps=2,
                                     guidance=2, return_trajectory=True)
            hook.remove()
            self.assertEqual(len(calls), 6)
            self.assertEqual(result["nfe"], 6)

    def test_v2_train_and_resume_on_synthetic_train_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for i in range(10):
                name = "Abra" if i < 3 else "Pikachu"
                relative = f"PokemonData/{name}/{i}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 64), (i * 20, 100, 255)).save(path)
                paths.append(relative)
            (root / "train_split.txt").write_text("\n".join(paths))
            (root / "val_split.txt").write_text("\n".join(paths))
            (root / "category_to_id.json").write_text(json.dumps({"Abra": 0, "Pikachu": 1}))
            (root / ".transparency_composited_on_white").touch()
            argv = ["--device", "cpu", "--amp", "none", "--data-root", str(root),
                    "--split-dir", str(root), "--outdir", str(root / "v2"), "--steps", "1",
                    "--batch-size", "2", "--microbatch", "1", "--model-channels", "16",
                    "--num-blocks", "1", "--channels-per-head", "8", "--num-workers", "0",
                    "--loss-every", "1", "--loss-count", "2", "--fid-every", "1",
                    "--sample-every", "1", "--sample-steps", "2", "--save-every", "1"]
            with patch("teacher_v2.train.competition_fid", return_value=42.0):
                train(v2_parser().parse_args(argv))
            argv[argv.index("--steps") + 1] = "2"
            argv += ["--resume", str(root / "v2/training-state.pt")]
            with patch("teacher_v2.train.competition_fid", return_value=40.0):
                train(v2_parser().parse_args(argv))
            state = torch.load(root / "v2/training-state.pt", weights_only=True)
            self.assertEqual(state["class_sampling"], "balanced")
            self.assertEqual(state["step"], 2)
            self.assertEqual(state["best_fid"], 40)
            loaded = EDMTeacherV2.load_checkpoint(root / "v2/teacher.ckpt")
            self.assertTrue((root / "val_split.txt").exists())
            self.assertTrue((root / "v2/snapshots/raw-0000002.ckpt").exists())
            self.assertEqual(state["optimizer"]["param_groups"][0]["betas"], (0.9, 0.99))


if __name__ == "__main__":
    unittest.main()
