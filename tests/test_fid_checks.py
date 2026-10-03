"""Check FID scheduling, archive retention, RNG isolation, and image protocol."""

import json
import tempfile
import signal
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import train
from cleanfid import fid
from torch.utils.data import DataLoader

from model import Model, ModelFewNFE
from src.fid_checks import reference_set, run_fid_check
from src.models.dit import DiTConfig
from test_train_resume import TinyDataModule, TinyDataset, arguments, load
from train import train_mode


class EvaluationData(TinyDataModule):
    def __init__(self):
        super().__init__()
        self.val_dataset = TinyDataset()
        self.category_to_id = {"a": 0, "b": 1}

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=2)


class FIDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_archive_and_fid_every_ten_complete_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Ten completed epochs is step 30; stop midway through epoch 11.
            args = arguments(root, epochs=11, max_steps=31, fid_every=10)
            def metric(model, data, work_dir, **kwargs):
                model.eval()
                torch.randn(50)  # Evaluation must not alter training RNG.
                return {"fid": 12.34, "seed": 1234}
            with patch("train.run_fid_check", side_effect=metric) as evaluate:
                train_mode(args, TinyDataModule(), torch.device("cpu"), "one_nfe")
                self.assertEqual(evaluate.call_count, 1)
            archived = root / "checkpoints/one_nfe_epoch_0010.ckpt"
            self.assertTrue(archived.exists())
            self.assertEqual(torch.load(archived, weights_only=False)["step"], 30)
            self.assertEqual(load(root, "one_nfe")["step"], 31)
            model = Model.load_checkpoint(str(archived), evaluate_mode="one_nfe")
            self.assertEqual(model.objective, "meanflow")
            records = [json.loads(line) for line in (root / "runs/one_nfe/fid.jsonl").read_text().splitlines()]
            self.assertEqual([(r["epoch"], r["step"], r["status"]) for r in records], [(10, 30, "ok")])
            self.assertEqual(records[0]["checkpoint"], str(archived))
            baseline = root / "baseline"
            train_mode(arguments(baseline, epochs=11, max_steps=31), TinyDataModule(), torch.device("cpu"), "one_nfe")
            for name, value in load(root, "one_nfe")["state_dict"].items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, load(baseline, "one_nfe")["state_dict"][name]), name)
            # The archived file is a complete resume checkpoint, not only weights.
            with patch("train.run_fid_check", side_effect=metric):
                train_mode(arguments(root, resume="auto", max_steps=33), TinyDataModule(), torch.device("cpu"), "one_nfe")
            records = [json.loads(line) for line in (root / "runs/one_nfe/fid.jsonl").read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in records], [10, 11])

    def test_partial_epoch_never_archived_or_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("train.run_fid_check") as evaluate:
                train_mode(arguments(root, epochs=10, max_steps=28, fid_every=10),
                           TinyDataModule(), torch.device("cpu"), "few_nfe")
                evaluate.assert_not_called()
            self.assertFalse((root / "checkpoints/few_nfe_epoch_0010.ckpt").exists())

    def test_fid_failure_keeps_checkpoint_and_retries_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = arguments(root, epochs=1, fid_every=10)
            with patch("train.run_fid_check", side_effect=RuntimeError("weights unavailable")):
                train_mode(args, TinyDataModule(), torch.device("cpu"), "few_nfe")
            self.assertEqual(load(root, "few_nfe")["step"], 3)
            log = root / "runs/few_nfe/fid.jsonl"
            self.assertEqual(json.loads(log.read_text())["status"], "error")
            args = arguments(root, resume="auto")  # Saved FID settings restored.
            with patch("train.run_fid_check", return_value={"fid": 9.5, "seed": 1234}) as evaluate:
                train_mode(args, TinyDataModule(), torch.device("cpu"), "few_nfe")
                self.assertEqual(evaluate.call_count, 1)
                train_mode(args, TinyDataModule(), torch.device("cpu"), "few_nfe")
                self.assertEqual(evaluate.call_count, 1)
            self.assertEqual([json.loads(line)["status"] for line in log.read_text().splitlines()], ["error", "ok"])

    def test_reference_cache_checks_completeness_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = EvaluationData()
            path = reference_set(data, tmp)
            self.assertEqual(len(list(path.glob("*.png"))), 6)
            (path / "0000.png").unlink()
            (path / "stale.jpg").write_bytes(b"stale")
            repaired = reference_set(data, tmp)
            self.assertEqual(repaired, path)
            self.assertEqual(len(list(path.glob("*.png"))), 6)
            self.assertFalse((path / "stale.jpg").exists())
            data.val_dataset.paths.reverse()
            changed = reference_set(data, tmp)
            self.assertNotEqual(changed, path)

    def test_stop_during_fid_keeps_completed_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def interrupt(*args, **kwargs):
                train.request_stop(signal.SIGTERM, None)
            try:
                with patch("train.run_fid_check", side_effect=interrupt):
                    train_mode(arguments(root, epochs=2, fid_every=1),
                               TinyDataModule(), torch.device("cpu"), "few_nfe")
                self.assertEqual(load(root, "few_nfe")["step"], 3)
                record = json.loads((root / "runs/few_nfe/fid.jsonl").read_text())
                self.assertEqual(record["status"], "interrupted")
                self.assertFalse(train._FID_ACTIVE)
            finally:
                train._STOP_REQUESTED = False
                train._FID_ACTIVE = False

    def test_full_image_protocol_with_real_cleanfid_backend(self):
        # Exercise clean-fid's real folder/covariance/FID pipeline with a small
        # feature extractor; unit tests do not download pretrained Inception.
        def small_features(batch):
            return batch.float().mean(dim=(-1, -2))
        original = fid.compute_fid
        observed = []
        def metric(generated, reference, **kwargs):
            observed.append((len(list(Path(generated).glob("*.png"))),
                             len(list(Path(reference).glob("*.png"))), kwargs["mode"]))
            kwargs["custom_feat_extractor"] = small_features
            return original(generated, reference, **kwargs)
        model = ModelFewNFE(config=DiTConfig(width=16, depth=1, heads=2, patch_size=8))
        with tempfile.TemporaryDirectory() as tmp, patch("src.fid_checks.fid.compute_fid", side_effect=metric):
            result = run_fid_check(model, EvaluationData(), tmp, batch_size=4)
            self.assertTrue(torch.isfinite(torch.tensor(result["fid"])))
            self.assertEqual(result["generated_images"], 40)
            self.assertEqual(result["reference_images"], 6)
            self.assertEqual(observed, [(40, 6, "clean")])
            self.assertFalse(list(Path(tmp).glob("generated_*")))


if __name__ == "__main__":
    unittest.main()
