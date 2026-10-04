"""Check diagnostic targets against analytic straight-line vector fields."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from inspect_model import (flow_path, inspect_real_endpoint, interpret_flow_rows,
                           load_model, fm_loss_vs_timestep)
from model import ModelFewNFE
from src.models.hybrid import HybridConfig


class FlowDiagnosticTests(unittest.TestCase):
    def test_path_matches_each_training_objective(self):
        images, noise = torch.tensor([[[[2.]]]]), torch.tensor([[[[-3.]]]])
        for objective, expected_start, expected_end, expected_target in (
            ("flow_matching", noise, images, images-noise),
            ("meanflow", images, noise, noise-images),
        ):
            model = type("Model", (), {"objective": objective})()
            for t, expected in ((0., expected_start), (1., expected_end)):
                xt, target = flow_path(model, images, noise, torch.tensor([t]))
                torch.testing.assert_close(xt, expected)
                torch.testing.assert_close(target, expected_target)

    def test_exact_velocity_and_reversed_velocity(self):
        # Zero data makes the exact conditional velocity reconstructible from xt.
        images = torch.zeros(2, 3, 64, 64)
        categories = torch.zeros(2, dtype=torch.long)
        for objective in ("flow_matching", "meanflow"):
            for sign in (1, -1):
                class AnalyticModel:
                    def __call__(self, x, t, **kwargs):
                        mix = t[:, None, None, None]
                        if self.objective == "flow_matching":
                            return torch.where(mix < 1, -x / (1-mix).clamp_min(1e-6), 0.) * sign
                        return torch.where(mix > 0, x / mix.clamp_min(1e-6), 0.) * sign
                model = AnalyticModel()
                model.objective = objective
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = inspect_real_endpoint(model, [(images, categories)], "cpu", 1)
                # At t=.5 both analytic velocities recover the known noise exactly.
                row = next(row for row in rows if row["t"] == .5)
                self.assertAlmostEqual(row["cosine"], sign, places=5)
                self.assertAlmostEqual(row["cos_forward"], -row["cos_reverse"], places=5)
                if sign == 1:
                    self.assertLess(row["mse"], 1e-10)
                else:
                    self.assertLess(row["mse_opposite"], 1e-10)
                    self.assertIn("Negative alignment", interpret_flow_rows([row]))

    def test_stable_large_error_does_not_imply_smoothness_problem(self):
        rows = [dict(cosine=.1, mse=4., mse_opposite=5., relative_rmse=1.67, dvdt=73.)]
        self.assertIn("Absolute FM error", interpret_flow_rows(rows))

    def test_nonfinite_metrics_are_not_reported_as_healthy(self):
        row = dict(cosine=float("nan"), mse=float("inf"), mse_opposite=1., relative_rmse=1.)
        self.assertIn("Non-finite", interpret_flow_rows([row]))

    def test_checkpoint_formats_preserve_metadata(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            original = ModelFewNFE(config=HybridConfig(width=128, base_channels=32, depth=1, heads=2))
            state = {"module."+k: v for k, v in original.state_dict().items()}
            with tempfile.TemporaryDirectory() as tmp:
                for key in (None, "state_dict", "model_state_dict", "model", "ema_state_dict", "ema"):
                    with self.subTest(key=key):
                        path = Path(tmp) / "test.ckpt"
                        torch.save(state if key is None else {key: state}, path)
                        with contextlib.redirect_stdout(io.StringIO()):
                            loaded, _ = load_model(path, "cpu")
                        self.assertEqual(loaded.config.heads, 2)
                        self.assertEqual(loaded.objective, "flow_matching")
                        torch.testing.assert_close(loaded.stem.weight, original.backbone.stem.weight)
        finally:
            torch.set_num_threads(old_threads)

    def test_binned_loss_runs_on_cpu_and_mps(self):
        class ExactFM:
            objective = "flow_matching"

            def __call__(self, x, t, **kwargs):
                return -x / (1-t[:, None, None, None])

        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        for device in devices:
            with self.subTest(device=device):
                batch = {"images": torch.zeros(4, 3, 64, 64),
                         "labels": torch.zeros(4, 1, dtype=torch.long)}
                with contextlib.redirect_stdout(io.StringIO()):
                    losses, counts = fm_loss_vs_timestep(ExactFM(), [batch], device, 1)
                self.assertEqual(counts.sum().item(), 4)
                self.assertLess(losses.sum().item(), 1e-10)
                self.assertEqual(losses.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
