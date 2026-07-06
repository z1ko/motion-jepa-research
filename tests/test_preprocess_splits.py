import tempfile
import unittest
from pathlib import Path

import polars as pl

from motion_jepa.preprocess import _assign_dataset_splits, load_dataset_splits


class DatasetSplitTests(unittest.TestCase):
    def test_load_dataset_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "datasets.yaml"
            path.write_text(
                "\n".join([
                    "pretrain_train:",
                    "  - ACCAD",
                    "pretrain_validation:",
                    "  - BMLhandball",
                    "validation:",
                    "  - SOMA",
                    "  - HumanEva",
                ]),
                encoding="utf-8",
            )

            self.assertEqual(
                load_dataset_splits(path),
                {
                    "ACCAD": "train",
                    "BMLhandball": "val",
                    "SOMA": "eval",
                    "HumanEva": "eval",
                },
            )

    def test_load_dataset_splits_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "datasets.yaml"
            path.write_text(
                "\n".join([
                    "pretrain_train:",
                    "  - ACCAD",
                    "pretrain_validation:",
                    "  - ACCAD",
                ]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "multiple split lists"):
                load_dataset_splits(path)

    def test_assign_dataset_splits(self) -> None:
        samples = pl.DataFrame({
            "suid": ["a", "b", "c"],
            "dataset": ["ACCAD", "BMLhandball", "SOMA"],
            "num_frames": [100, 100, 100],
        })

        result = _assign_dataset_splits(
            samples,
            {
                "ACCAD": "train",
                "BMLhandball": "val",
                "SOMA": "eval",
            },
        )

        self.assertEqual(result.sort("suid")["split"].to_list(), ["train", "val", "eval"])

    def test_assign_dataset_splits_rejects_missing_config_dataset(self) -> None:
        samples = pl.DataFrame({
            "suid": ["a"],
            "dataset": ["UNKNOWN"],
            "num_frames": [100],
        })

        with self.assertRaisesRegex(ValueError, "missing from dataset split config"):
            _assign_dataset_splits(samples, {"ACCAD": "train"})


if __name__ == "__main__":
    unittest.main()
