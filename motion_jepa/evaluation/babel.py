"""BABEL (papers/BABEL.pdf) frame-level action labels, read directly from the
raw AMASS CSVs already used for pretraining -- no separate label manifest
like CARE-PD's, and no re-preprocessing. Reuses motion_jepa.preprocess's own
load_sample_from_csv/resample_to_hz so a window's label lines up frame-exact
with the SAME processed kinematics already sitting in data/processed/motion
(verified directly: resample_to_hz's output frame count matches samples.
parquet's num_frames exactly for every checked trial).

Cheap-path eval only (see CARE-PD-REPORT.md): ACCAD/MoSh/SFU are still
config/datasets.yaml's pretrain_train, so the encoder has already seen this
motion, unlabeled, during pretraining -- this probes already-seen-but-
unlabeled data, not a fully held-out set. See config/datasets_babel.yaml for
the split analysis and the "clean" (re-preprocess) alternative.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import polars as pl

from motion_jepa.preprocess import load_sample_from_csv, resample_to_hz

# "transition"/"t pose"/"a pose" are BABEL's own calibration/connective
# segments, not a describable action -- excluded the same way CARE-PD's
# severity==3 is excluded from parse_care_pd_label, before filter_labeled_rows
# ever sees them.
_JUNK_ACTIONS = {"transition", "t pose", "a pose", ""}


@lru_cache(maxsize=256)
def _resampled_action_array(raw_root: str, dataset: str, subject: str, trial: str, hz: float) -> tuple[str, ...]:
    """One trial's per-frame primary action tag, resampled to `hz`. Cached
    per trial: windows.parquet's stride=50 means many overlapping windows
    share the same trial, and re-reading + re-resampling the raw CSV per
    window would redo the same work dozens of times.
    """
    path = Path(raw_root) / dataset / subject / trial
    sample = resample_to_hz(load_sample_from_csv(path), hz)
    return tuple(sample.extra["action"].to_list())


def babel_action_label(
    *, raw_root: Path | str, dataset: str, subject: str, trial: str, start: int, end: int, hz: float = 100.0,
) -> str | None:
    """Majority-vote primary action tag (first '|'-segment of BABEL's
    composite label, junk excluded) over one window's frame range. None if
    every frame in range is junk or past the trial's resampled length.
    """
    actions = _resampled_action_array(str(raw_root), dataset, subject, trial, hz)
    # Raw CSV empty fields read as null (polars), not "" -- e.g. frames
    # outside any BABEL-annotated segment.
    primary = ((a or "").split("|", 1)[0].strip().lower() for a in actions[start:min(end, len(actions))])
    counts = Counter(a for a in primary if a not in _JUNK_ACTIONS)
    return counts.most_common(1)[0][0] if counts else None


def attach_babel_labels(rows: pl.DataFrame, *, raw_root: Path | str, hz: float = 100.0) -> pl.DataFrame:
    """Adds eval_label/eval_label_kind columns -- same contract as
    evaluation.data.add_eval_labels, so filter_labeled_rows/run_linear_probe
    downstream need no changes.
    """
    required = {"dataset", "subject", "trial", "start", "end"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Rows missing columns required for BABEL labels: {sorted(missing)}")

    labels = [
        babel_action_label(
            raw_root=raw_root, dataset=row["dataset"], subject=row["subject"], trial=row["trial"],
            start=int(row["start"]), end=int(row["end"]), hz=hz,
        )
        for row in rows.select(["dataset", "subject", "trial", "start", "end"]).iter_rows(named=True)
    ]
    return rows.with_columns(
        pl.Series("eval_label", labels, dtype=pl.Utf8),
        pl.Series("eval_label_kind", ["babel_action"] * len(labels), dtype=pl.Utf8),
    )
