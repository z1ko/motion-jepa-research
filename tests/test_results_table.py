import tempfile
import unittest
from pathlib import Path

import polars as pl

from motion_jepa.evaluation.results_table import _parse_version, flatten_result, upsert_results_table


def _result(**overrides) -> dict:
    base = {
        "dataset": "BABEL",
        "label_kind": "babel_action",
        "n_windows": 100,
        "n_subjects": 10,
        "root": "data/processed/motion",
        "split": "eval",
        "eval_window_size": 200,
        "eval_stride": 50,
        "eval_min_valid_frames": 200,
        "n_classes": 8,
        "n_folds": 10,
        "n_walks_total": 30,
        "chance_baseline": 0.125,
        "majority_baseline_mean": 0.27,
        "walk_majority_baseline_mean": 0.27,
        "accuracy_mean": 0.4,
        "accuracy_std": 0.1,
        "balanced_accuracy_mean": 0.38,
        "balanced_accuracy_std": 0.1,
        "f1_macro": 0.35,
        "knn_accuracy_mean": 0.3, "knn_accuracy_std": 0.1,
        "knn_balanced_accuracy_mean": 0.3, "knn_balanced_accuracy_std": 0.1, "knn_f1_macro": 0.28,
        "mlp_accuracy_mean": 0.3, "mlp_accuracy_std": 0.1,
        "mlp_balanced_accuracy_mean": 0.3, "mlp_balanced_accuracy_std": 0.1, "mlp_f1_macro": 0.28,
        "walk_accuracy_mean": 0.4, "walk_accuracy_std": 0.1, "walk_f1_macro": 0.36,
        "walk_knn_accuracy_mean": 0.3, "walk_knn_accuracy_std": 0.1, "walk_knn_f1_macro": 0.28,
        "walk_mlp_accuracy_mean": 0.3, "walk_mlp_accuracy_std": 0.1, "walk_mlp_f1_macro": 0.28,
    }
    base.update(overrides)
    return base


class VersionParseTests(unittest.TestCase):
    def test_well_formed_path(self) -> None:
        version, run_group = _parse_version(
            "runs/v2/version_3/checkpoints/best-epoch=0023-val_loss=2.1655.ckpt"
        )
        self.assertEqual(version, 3)
        self.assertEqual(run_group, "v2")

    def test_malformed_path_returns_none(self) -> None:
        version, run_group = _parse_version("checkpoint.ckpt")
        self.assertIsNone(version)
        self.assertIsNone(run_group)


class FlattenResultTests(unittest.TestCase):
    def test_missing_config_sections_dont_crash(self) -> None:
        # Mirrors runs/v1/version_16/evaluation_v16.json's real shape: no
        # `masking` section at all, and `optim` missing `loss`.
        run_info = {
            "checkpoint": "runs/v1/version_16/checkpoints/x.ckpt",
            "epoch": 10,
            "global_step": 1000,
            "config": {
                "data": {"window_size": 200, "stride": 400, "min_valid_frames": 200},
                "training": {"groups": {"pelvis": [0, 1]}},
                "architecture": {"encoder": {"depth": 4, "dropout": 0.1}, "predictor": {"depth": 2}},
                "optim": {"ema_momentum": 0.999, "learning_rate": 0.001, "weight_decay": 0.1},
            },
        }
        row = flatten_result(run_info, _result())
        self.assertIsNone(row["mamp_target_fraction"])
        self.assertIsNone(row["mamp_temperature"])
        self.assertIsNone(row["loss"])
        self.assertEqual(row["encoder_depth"], 4)
        self.assertEqual(row["n_groups"], 1)

    def test_random_baseline_fields_and_delta(self) -> None:
        run_info = {
            "checkpoint": "runs/v2/version_2/checkpoints/best-epoch=0038-val_loss=2.0957.ckpt",
            "epoch": 38,
            "global_step": 10413,
            "config": {},
        }
        result = _result(random_baseline=_result(f1_macro=0.28, walk_f1_macro=0.29))
        row = flatten_result(run_info, result)
        self.assertAlmostEqual(row["delta_f1_macro"], 0.35 - 0.28)
        self.assertAlmostEqual(row["delta_walk_f1_macro"], 0.36 - 0.29)
        self.assertAlmostEqual(row["random_f1_macro"], 0.28)

    def test_no_random_baseline_leaves_delta_none(self) -> None:
        run_info = {"checkpoint": "runs/v2/version_2/checkpoints/x.ckpt", "epoch": 1, "global_step": 1, "config": {}}
        row = flatten_result(run_info, _result())
        self.assertIsNone(row["delta_f1_macro"])
        self.assertIsNone(row["random_f1_macro"])


class UpsertResultsTableTests(unittest.TestCase):
    def test_round_trip_diagonal_concat_and_overwrite(self) -> None:
        run_info_a = {"checkpoint": "runs/v2/version_0/checkpoints/a.ckpt", "epoch": 1, "global_step": 1, "config": {}}
        run_info_b = {"checkpoint": "runs/v2/version_1/checkpoints/b.ckpt", "epoch": 2, "global_step": 2, "config": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval_results.parquet"

            # Batch 1: no random_baseline field at all (regression case for
            # the diagonal-concat dtype crash between an all-null column and
            # a later real Float64 column).
            row_a = flatten_result(run_info_a, _result(dataset="HumanEva"))
            upsert_results_table([row_a], path)

            # Batch 2: a different checkpoint, this time with random_baseline.
            row_b = flatten_result(
                run_info_b,
                _result(dataset="HumanEva", random_baseline=_result(f1_macro=0.2, walk_f1_macro=0.2)),
            )
            upsert_results_table([row_b], path)

            table = pl.read_parquet(path)
            self.assertEqual(table.height, 2)

            # Re-upsert an update for the same (checkpoint, dataset, ...) key.
            row_b_updated = flatten_result(
                run_info_b,
                _result(dataset="HumanEva", f1_macro=0.99, random_baseline=_result(f1_macro=0.2, walk_f1_macro=0.2)),
            )
            upsert_results_table([row_b_updated], path)

            table = pl.read_parquet(path)
            self.assertEqual(table.height, 2)  # overwrite, not a new row
            updated_row = table.filter(pl.col("checkpoint") == run_info_b["checkpoint"])
            self.assertAlmostEqual(updated_row["f1_macro"].item(), 0.99)


if __name__ == "__main__":
    unittest.main()
