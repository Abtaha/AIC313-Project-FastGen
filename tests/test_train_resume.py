"""Check exact CPU continuation through mid-epoch and epoch-boundary saves."""

import argparse
import json
import signal
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model import Model
from train import configure_worker, request_stop, train_mode


class TinyDataset(Dataset):
    def __init__(self):
        self.paths = [f"image-{i}" for i in range(6)]
        self.images = torch.randn(6, 3, 64, 64, generator=torch.Generator().manual_seed(7))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.images[index], torch.tensor(index, dtype=torch.long)


class TinyDataModule:
    def __init__(self):
        self.train_dataset = TinyDataset()
        self.category_to_id = {str(i): i for i in range(151)}
        self.batch_size = 2

    def train_dataloader(self, sampler):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          sampler=sampler, drop_last=True, num_workers=0)


def arguments(root, **overrides):
    values = dict(
        checkpoint_dir=str(root / "checkpoints"), output_dir=str(root / "runs"),
        resume=None, specified=[], seed=42, width=16, depth=1, heads=2,
        patch_size=8, knots=(0.25, 0.5, 0.75), batch_size=2, epochs=2,
        max_steps=None, precision="fp32", lr=1e-4, weight_decay=0.01,
        grad_clip=1.0, sample_every=10, save_every=1,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def load(root, mode):
    return torch.load(root / "checkpoints" / f"{mode}.ckpt",
                      map_location="cpu", weights_only=False)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_state_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_exact_continuation(self):
        for mode in ("one_nfe", "few_nfe"):
            for boundary in (2, 3):
                with self.subTest(mode=mode, boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                    whole, resumed = Path(tmp) / "whole", Path(tmp) / "resumed"
                    train_mode(arguments(whole), TinyDataModule(), torch.device("cpu"), mode)
                    train_mode(arguments(resumed, max_steps=boundary), TinyDataModule(), torch.device("cpu"), mode)
                    log = resumed / "runs" / mode / "train.jsonl"
                    # Emulate log writes after the last persisted checkpoint.
                    with log.open("a") as file:
                        file.write('{"step": 999, "loss": 0}\n{"partial":')
                    # Architecture/batch defaults must come from the checkpoint.
                    args = arguments(resumed, resume="auto", width=512, batch_size=32)
                    train_mode(args, TinyDataModule(), torch.device("cpu"), mode)
                    expected, actual = load(whole, mode), load(resumed, mode)
                    for key in ("state_dict", "optimizer", "scheduler", "scaler", "rng_state",
                                "epoch", "batch_in_epoch", "step", "epoch_loss_sum", "epoch_image_count"):
                        self.assert_state_equal(expected[key], actual[key])
                    records = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual([row["step"] for row in records], list(range(1, 7)))
                    original = [json.loads(line) for line in (whole / "runs" / mode / "train.jsonl").read_text().splitlines()]
                    self.assertEqual([(r["loss"], r["lr"]) for r in records],
                                     [(r["loss"], r["lr"]) for r in original])
                    model = Model.load_checkpoint(str(resumed / "checkpoints" / f"{mode}.ckpt"),
                                                  evaluate_mode=mode, device="cpu")
                    self.assertEqual(model.backbone.config.width, 16)
                    before = (resumed / "checkpoints" / f"{mode}.ckpt").stat().st_mtime_ns
                    train_mode(args, TinyDataModule(), torch.device("cpu"), mode)
                    self.assertEqual(before, (resumed / "checkpoints" / f"{mode}.ckpt").stat().st_mtime_ns)

    def test_incompatible_resume_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_mode(arguments(root, max_steps=1), TinyDataModule(), torch.device("cpu"), "one_nfe")
            with self.assertRaisesRegex(ValueError, "batch_size"):
                train_mode(arguments(root, resume="auto", batch_size=4, specified=["batch_size"]),
                           TinyDataModule(), torch.device("cpu"), "one_nfe")
            data = TinyDataModule()
            data.train_dataset.paths.reverse()
            with self.assertRaisesRegex(ValueError, "manifest"):
                train_mode(arguments(root, resume="auto"), data, torch.device("cpu"), "one_nfe")
            with self.assertRaisesRegex(ValueError, "belong"):
                train_mode(arguments(root, resume=str(root / "checkpoints/one_nfe.ckpt")),
                           TinyDataModule(), torch.device("cpu"), "few_nfe")

    def test_worker_does_not_inherit_graceful_stop_handlers(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
            configure_worker(0)
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
            self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    unittest.main()
