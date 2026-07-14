"""Flattens eval CLI results into rows of a single, queryable cross-experiment
parquet table (runs/eval_results.parquet by default), so "what hyperparameters
gave the best BABEL f1" is one polars query instead of grepping hparams.yaml
and hand-parsing JSON files per checkpoint, as this whole session's sweep was
tracked. `--out`'s full-detail JSON (folds/dmu/attentive/pooling_comparison)
is untouched -- this table only ever holds the flattened subset that's
actually been compared across checkpoints.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl

# Nested config paths -- checkpoints from different points in this project's
# history have different config shapes (e.g. pre-masking-section runs), so
# every lookup goes through _dig and tolerates a missing key at any level.
_CONFIG_FIELDS: list[tuple[str, list[str]]] = [
    ("encoder_depth", ["architecture", "encoder", "depth"]),
    ("encoder_dropout", ["architecture", "encoder", "dropout"]),
    ("predictor_depth", ["architecture", "predictor", "depth"]),
    ("predictor_dropout", ["architecture", "predictor", "dropout"]),
    ("embed_dim", ["architecture", "embed_dim"]),
    ("segment_size", ["architecture", "segment_size"]),
    ("channels", ["architecture", "channels"]),
    ("ema_momentum", ["optim", "ema_momentum"]),
    ("learning_rate", ["optim", "learning_rate"]),
    ("weight_decay", ["optim", "weight_decay"]),
    ("tau_predict", ["optim", "tau_predict"]),
    ("tau_target", ["optim", "tau_target"]),
    ("center_momentum", ["optim", "center_momentum"]),
    ("loss", ["optim", "loss"]),
    ("mamp_target_fraction", ["masking", "mamp_target_fraction"]),
    ("mamp_temperature", ["masking", "mamp_temperature"]),
    ("train_window_size", ["data", "window_size"]),
    ("train_stride", ["data", "stride"]),
    ("train_min_valid_frames", ["data", "min_valid_frames"]),
]

# run_linear_probe's own return keys (see probes.py) -- pulled once for the
# trained encoder, and again (prefixed random_) for --random-baseline's
# untrained-encoder rerun, so both share exactly the same field list.
_CORE_METRIC_FIELDS: list[str] = [
    "n_classes", "n_folds", "n_walks_total", "chance_baseline",
    "majority_baseline_mean", "walk_majority_baseline_mean",
    "accuracy_mean", "accuracy_std", "balanced_accuracy_mean", "balanced_accuracy_std", "f1_macro",
    "knn_accuracy_mean", "knn_accuracy_std", "knn_balanced_accuracy_mean", "knn_balanced_accuracy_std", "knn_f1_macro",
    "mlp_accuracy_mean", "mlp_accuracy_std", "mlp_balanced_accuracy_mean", "mlp_balanced_accuracy_std", "mlp_f1_macro",
    "walk_accuracy_mean", "walk_accuracy_std", "walk_f1_macro",
    "walk_knn_accuracy_mean", "walk_knn_accuracy_std", "walk_knn_f1_macro",
    "walk_mlp_accuracy_mean", "walk_mlp_accuracy_std", "walk_mlp_f1_macro",
]

_DEDUP_KEY = ["checkpoint", "dataset", "root", "eval_window_size", "eval_stride", "eval_min_valid_frames"]

_STR_COLS = {
    "checkpoint", "dataset", "label_kind", "root", "split", "detail_json",
    "eval_run_at", "run_group", "loss",
}
_INT_COLS = {
    "epoch", "global_step", "eval_window_size", "eval_stride", "eval_min_valid_frames",
    "version", "n_groups", "encoder_depth", "predictor_depth", "embed_dim", "segment_size", "channels",
    "n_classes", "n_folds", "n_walks_total", "n_windows", "n_subjects",
}
_BOOL_COLS = {"has_dmu", "has_attentive", "has_pooling_comparison"}
# Every remaining column (metrics, dropout/lr/wd/momentum/etc., delta_*) is a float.


def _dig(d: dict | None, path: list[str]):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _parse_version(checkpoint: str) -> tuple[int | None, str | None]:
    """.../runs/v2/version_3/checkpoints/best-....ckpt -> (3, "v2").

    Duplicated from train.py's _resolve_resume (not shared): that version is
    allowed to raise on a malformed path (validating user-supplied config,
    failure should be loud); this one runs over potentially-relocated
    checkpoint paths during backfill and must swallow failures into None
    instead of aborting the whole ingestion.
    """
    try:
        parents = Path(checkpoint).parents
        version_dir = parents[1].name
        run_group = parents[2].name
        version = int(version_dir.removeprefix("version_"))
        return version, run_group
    except (ValueError, IndexError):
        return None, None


def flatten_result(run_info: dict, result: dict, *, detail_json: str | None = None) -> dict:
    """`result` must carry root/split/eval_window_size/eval_stride/eval_min_valid_frames
    (evaluate_dataset in cli.py populates these directly). Older JSONs that predate
    those fields (see scripts/backfill_eval_results.py) need them filled in with a
    fallback -- e.g. the checkpoint's own config.data.* -- before calling this.
    """
    checkpoint = run_info["checkpoint"]
    version, run_group = _parse_version(checkpoint)
    config = run_info.get("config", {})

    row: dict = {
        "checkpoint": checkpoint,
        "epoch": run_info.get("epoch"),
        "global_step": run_info.get("global_step"),
        "version": version,
        "run_group": run_group,
        "dataset": result["dataset"],
        "label_kind": result.get("label_kind"),
        "root": result.get("root"),
        "split": result.get("split"),
        "eval_window_size": result.get("eval_window_size"),
        "eval_stride": result.get("eval_stride"),
        "eval_min_valid_frames": result.get("eval_min_valid_frames"),
        "detail_json": detail_json,
        "eval_run_at": datetime.datetime.now().isoformat(),
        "n_windows": result.get("n_windows"),
        "n_subjects": result.get("n_subjects"),
        "n_groups": len(g) if isinstance(g := _dig(config, ["training", "groups"]), dict) else None,
        "has_dmu": "dmu" in result,
        "has_attentive": "attentive" in result,
        "has_pooling_comparison": "pooling_comparison" in result,
    }

    for name, path in _CONFIG_FIELDS:
        row[name] = _dig(config, path)

    for name in _CORE_METRIC_FIELDS:
        row[name] = result.get(name)

    random_baseline = result.get("random_baseline")
    if random_baseline is not None:
        for name in _CORE_METRIC_FIELDS:
            row[f"random_{name}"] = random_baseline.get(name)
        row["delta_f1_macro"] = result.get("f1_macro") - random_baseline.get("f1_macro")
        row["delta_walk_f1_macro"] = result.get("walk_f1_macro") - random_baseline.get("walk_f1_macro")
    else:
        for name in _CORE_METRIC_FIELDS:
            row[f"random_{name}"] = None
        row["delta_f1_macro"] = None
        row["delta_walk_f1_macro"] = None

    return row


def _schema_for(rows: list[dict]) -> dict[str, pl.PolarsDataType]:
    """Explicit dtype per column, keyed off the actual row contents -- an
    all-None column would otherwise infer as polars Null, which then can't
    concat against a later batch where that column holds real floats.
    """
    keys = set()
    for row in rows:
        keys.update(row.keys())

    schema: dict[str, pl.PolarsDataType] = {}
    for key in keys:
        if key in _STR_COLS:
            schema[key] = pl.Utf8
        elif key in _INT_COLS:
            schema[key] = pl.Int64
        elif key in _BOOL_COLS:
            schema[key] = pl.Boolean
        else:
            schema[key] = pl.Float64
    return schema


def upsert_results_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    schema = _schema_for(rows)
    new = pl.DataFrame(rows, schema=schema)

    if path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, new], how="diagonal_relaxed")
    else:
        combined = new

    combined = (
        combined
        .unique(subset=_DEDUP_KEY, keep="last", maintain_order=True)
        .sort(["run_group", "version", "epoch", "dataset"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(path)
