"""Linear-probe evaluation of a trained Motion-JEPA encoder.

Extracts a frozen embedding per window from the teacher encoder and fits a
linear classifier (logistic regression) on top of it, to measure how much
downstream-task-relevant information (action label) the representation
carries -- the standard linear-probe protocol for self-supervised encoders.

Evaluated on the current `validation` split (see config/datasets_babel.yaml):
HumanEva (action, parsed from the trial filename by
`motion_jepa.evaluation.data.parse_eval_label`) and the pooled pseudo-dataset
`BABEL` (ACCAD+MoSh+SFU concatenated into one combined LOSO evaluation --
see `_DATASET_GROUPS`/`load_pooled_window_table` -- with action labels
parsed from BABEL frame-level tags in the raw AMASS CSVs and collapsed to a
coarse taxonomy, see `motion_jepa.evaluation.babel`). ACCAD/MoSh/SFU
individually have too few subjects (16/16/7) for a reliable instrument on
their own; pooling gives ~39 subjects/folds to one probe instead. SOMA/
DanceDB were dropped from this default set entirely: too few subjects (2-3),
and DanceDB's label (emotion) is a different semantic task than the others
(see CARE-PD-REPORT.md and the conversation that led here). This is the
suite used to score model checkpoints after training.

Cross-validation is leave-one-subject-out (grouped by subject, never by
window), since:
  - adjacent/overlapping windows from the same trial are highly correlated,
    so a random window-level split would leak and overstate accuracy.
  - these datasets have few subjects, so held-out-subject generalization is
    both the meaningful question ("does the embedding encode this label for
    a person it never saw") and the only split that doesn't waste the
    little data available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as t

from motion_jepa.architecture.model import MotionJEPA
from motion_jepa.evaluation.data import (
    WindowRowsDataset,
    filter_labeled_rows,
    load_care_pd_labels,
    load_fixed_folds,
    load_pooled_window_table,
)
from motion_jepa.evaluation.encoder import compute_embeddings, compute_token_embeddings, load_encoder, read_checkpoint_provenance
from motion_jepa.evaluation.probes import print_report, run_attentive_probe, run_dmu_probe, run_linear_probe
from motion_jepa.evaluation.results_table import flatten_result, upsert_results_table

DEFAULT_DATASETS = ["HumanEva", "BABEL"]

# These have no filename-encoded label of their own (see parse_eval_label) --
# routed through BABEL frame-level labels (motion_jepa.evaluation.babel)
# instead, automatically, regardless of what --babel-raw-root defaults to.
# HumanEva keeps filename-based parsing.
_BABEL_DATASETS = {"ACCAD", "MoSh", "SFU"}

# Pseudo-dataset names that pool several real datasets into one combined
# LOSO evaluation instead of reporting them separately. ACCAD (16 subj) /
# MoSh (16) / SFU (7) individually produce noisy, sometimes-degenerate LOSO
# folds (n_test as low as 2) -- pooling gives ~39 subjects/folds to one
# probe. --datasets ACCAD (etc.) still works standalone, unpooled --
# additive via _DATASET_GROUPS.get(name, [name]).
_DATASET_GROUPS: dict[str, list[str]] = {"BABEL": sorted(_BABEL_DATASETS)}

# ================================================================================================================
# CLI
# ================================================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear-probe evaluation of a Motion-JEPA checkpoint on held-out labeled datasets."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", default=Path("data/processed/motion"), type=Path)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset names.")
    parser.add_argument(
        "--eval-window-size", default=None, type=int,
        help="Window size for eval windows (default: the checkpoint's own config.data.window_size).",
    )
    parser.add_argument(
        "--eval-stride", default=None, type=int,
        help=(
            "Stride between eval windows (default: the checkpoint's own config.data.stride, i.e. "
            "non-overlapping). Windows are enumerated on the fly from samples.parquet, not a "
            "precomputed table -- different eval roots have historically used different strides "
            "(e.g. CARE-PD's store was built dense at stride=50 for walk-level majority-vote "
            "robustness), so pass this explicitly for CARE-PD datasets instead of relying on the "
            "default."
        ),
    )
    parser.add_argument(
        "--eval-min-valid-frames", default=None, type=int,
        help="Minimum real frames for a short trial to still get a padded window (default: the checkpoint's own config.data.min_valid_frames).",
    )
    parser.add_argument(
        "--min-label-subjects",
        default=2,
        type=int,
        help="Drop labels observed in fewer than this many distinct subjects (can't be held-out-subject evaluated).",
    )
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-windows", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--probe-c", default=1.0, type=float, help="Inverse L2 regularization strength.")
    parser.add_argument("--probe-max-iter", default=5000, type=int)
    parser.add_argument("--probe-knn-k", default=5, type=int, help="Number of neighbors for the k-NN probe.")
    parser.add_argument(
        "--probe-mlp-hidden", default=64, type=int,
        help="Hidden layer width for the one-hidden-layer MLP probe.",
    )
    parser.add_argument("--probe-mlp-max-iter", default=500, type=int)
    parser.add_argument(
        "--compare-pooling",
        action="store_true",
        help=(
            "Also probe 'per_group' (masked mean over time only, keeping each "
            "anatomical group separate), 'per_segment' (masked mean over "
            "groups only, keeping each time segment separate), and 'max' "
            "(masked element-wise max instead of mean, same width) embeddings "
            "alongside the default fully-pooled ('mean') embedding, to check "
            "whether mean-pooling is washing out localized or peak signal."
        ),
    )
    parser.add_argument(
        "--attentive-probe",
        action="store_true",
        help=(
            "Also probe with a learned-query attention pool (AttentiveProbeHead) trained "
            "per fold on the encoder's unpooled per-token output, instead of a fixed mean/"
            "max pool -- tests whether pre-pooling is throwing away signal (see CARE-PD-REPORT.md)."
        ),
    )
    parser.add_argument("--attentive-epochs", default=100, type=int)
    parser.add_argument("--attentive-lr", default=1e-3, type=float)
    parser.add_argument("--attentive-weight-decay", default=1e-2, type=float)
    parser.add_argument("--attentive-heads", default=4, type=int)
    parser.add_argument(
        "--dmu-probe",
        action="store_true",
        help=(
            "Also compute a deviation-from-mean-unimpaired score (papers/GaitEncoder.pdf): a "
            "per-window diagonal-Mahalanobis distance from a reference class's mean/variance "
            "(refit per LOSO fold), correlated against the ordinal severity label -- only "
            "meaningful for CARE-PD-* datasets (needs --care-pd-labels)."
        ),
    )
    parser.add_argument(
        "--dmu-reference-label", default=None, type=str,
        help="Reference ('unimpaired') class for the DMU score. Default: min label per fold.",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help=(
            "Also run the identical probe on a freshly-constructed, untrained encoder of the "
            "same architecture (same --seed), reported as a delta vs. the trained encoder. Some "
            "datasets/labels turn out to be decodable from raw kinematic statistics alone, even "
            "through an untrained transformer -- a delta near zero means this dataset can't "
            "currently distinguish trained from random representations (see CARE-PD-REPORT.md)."
        ),
    )
    parser.add_argument("--out", default=None, type=Path, help="Optional path to write full results as JSON.")
    parser.add_argument(
        "--results-table", default=Path("runs/eval_results.parquet"), type=Path,
        help=(
            "Parquet file every invocation upserts a flattened row into (one row per "
            "checkpoint x dataset x eval-window-settings), for cross-checkpoint queries "
            "(hyperparameters + core metrics + random-baseline delta). See "
            "motion_jepa.evaluation.results_table. Unaffected by --out, which still writes "
            "the full-detail JSON (folds/dmu/attentive/pooling_comparison) separately."
        ),
    )
    parser.add_argument(
        "--no-results-table", action="store_true", help="Skip updating --results-table for this invocation.",
    )
    parser.add_argument(
        "--care-pd-labels", default=Path("data/raw/care_pd/carepd_mds_updrs_gait_severity.csv"), type=Path,
        help="CARE-PD's filename;score MDS-UPDRS-gait table. Only read if a --datasets entry starts with 'CARE-PD-'.",
    )
    parser.add_argument(
        "--care-pd-fold-file", default=None, type=Path,
        help=(
            "CARE-PD fold pickle (e.g. folds/UPDRS_Datasets/3DGait_43fold_participants.pkl) -- "
            "reproduces the paper's own fold assignment instead of our LeaveOneGroupOut. "
            "Applies to every CARE-PD dataset in --datasets, so evaluate one CARE-PD cohort "
            "per invocation when using this (different cohorts have different fold files)."
        ),
    )
    parser.add_argument(
        "--babel-raw-root", default=Path("data/raw/amass"), type=Path,
        help=(
            "Root of raw AMASS CSVs, used to attach BABEL frame-level action labels for "
            "datasets in _BABEL_DATASETS (ACCAD/MoSh/SFU, or the pooled 'BABEL' pseudo-dataset "
            "that combines all three -- see _DATASET_GROUPS) that have no filename-encoded label "
            "of their own (see config/datasets_babel.yaml, motion_jepa.evaluation.babel). "
            "Applied automatically only to those datasets -- HumanEva and any other "
            "--datasets entry still use filename/CARE-PD label parsing regardless of this flag."
        ),
    )
    return parser.parse_args()


def resolve_device(device: str) -> t.device:
    if device == "auto":
        return t.device("cuda" if t.cuda.is_available() else "cpu")
    if device == "cuda" and not t.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return t.device(device)

# ================================================================================================================
# ORCHESTRATION
# ================================================================================================================

def evaluate_dataset(
    *,
    dataset_name: str,
    root: Path,
    split: str,
    config,
    encoder: t.nn.Module,
    device: t.device,
    batch_size: int,
    max_windows: int | None,
    eval_window_size: int | None = None,
    eval_stride: int | None = None,
    eval_min_valid_frames: int | None = None,
    min_label_subjects: int,
    seed: int,
    probe_c: float,
    probe_max_iter: int,
    probe_knn_k: int,
    probe_mlp_hidden: int,
    probe_mlp_max_iter: int,
    compare_pooling: bool = False,
    attentive_probe: bool = False,
    attentive_epochs: int = 100,
    attentive_lr: float = 1e-3,
    attentive_weight_decay: float = 1e-2,
    attentive_heads: int = 4,
    dmu_probe: bool = False,
    dmu_reference_label: str | None = None,
    care_pd_labels: dict[str, int] | None = None,
    care_pd_fold_file: Path | None = None,
    babel_raw_root: Path | None = None,
    random_encoder: t.nn.Module | None = None,
) -> dict | None:
    group_members = _DATASET_GROUPS.get(dataset_name, [dataset_name])
    resolved_window_size = eval_window_size or config.data.window_size
    resolved_stride = eval_stride or config.data.stride
    resolved_min_valid_frames = eval_min_valid_frames or config.data.min_valid_frames
    rows = load_pooled_window_table(
        root=root, split=split, datasets=group_members, max_windows=max_windows, seed=seed,
        window_size=resolved_window_size,
        stride=resolved_stride,
        min_valid_frames=resolved_min_valid_frames,
        care_pd_labels=care_pd_labels, babel_raw_root=babel_raw_root,
    )
    rows = filter_labeled_rows(rows, min_label_subjects=min_label_subjects)

    if rows.height == 0:
        print(f"[{dataset_name}] no usable labeled windows after filtering, skipping.")
        return None

    label_kind = rows["eval_label_kind"][0]
    n_labels = rows["eval_label"].n_unique()
    n_subjects = rows["subject"].n_unique()

    if n_labels < 2 or n_subjects < 2:
        print(f"[{dataset_name}] not enough labels/subjects to probe "
              f"(n_labels={n_labels}, n_subjects={n_subjects}), skipping.")
        return None

    window_dataset = WindowRowsDataset(
        root=root,
        rows=rows.to_dicts(),
        window_size=config.data.window_size,
        segment_size=config.architecture.segment_size,
    )
    segment_count = config.data.window_size // config.architecture.segment_size
    group_count = len(config.training.groups)
    channels = config.architecture.channels
    labels = rows["eval_label"].to_numpy()
    groups = rows["subject"].to_numpy()
    walk_ids = rows["suid"].to_numpy()

    fold_indices = load_fixed_folds(care_pd_fold_file, groups) if care_pd_fold_file is not None else None

    embeddings = compute_embeddings(
        encoder=encoder,
        dataset=window_dataset,
        batch_size=batch_size,
        device=device,
        segment_count=segment_count,
        group_count=group_count,
        channels=channels,
        pooling="mean",
    )

    result = run_linear_probe(
        embeddings=embeddings,
        labels=labels,
        groups=groups,
        walk_ids=walk_ids,
        probe_c=probe_c,
        probe_max_iter=probe_max_iter,
        probe_knn_k=probe_knn_k,
        probe_mlp_hidden=probe_mlp_hidden,
        probe_mlp_max_iter=probe_mlp_max_iter,
        fold_indices=fold_indices,
    )
    result.update({
        "dataset": dataset_name,
        "label_kind": label_kind,
        "n_windows": rows.height,
        "n_subjects": n_subjects,
        "root": str(root),
        "split": split,
        "eval_window_size": resolved_window_size,
        "eval_stride": resolved_stride,
        "eval_min_valid_frames": resolved_min_valid_frames,
    })

    if compare_pooling:
        # per_group/per_segment are group_count/segment_count-times wider
        # than "mean" (each group's or each segment's own slice, concatenated
        # instead of averaged together) with the same handful of training
        # windows per fold -- a much higher ratio of free parameters to
        # samples than the "mean" probe, so treat a win there as suggestive,
        # not conclusive, unless the margin is large. "max" stays the same
        # width as "mean", so it carries no such extra overfitting risk.
        result["pooling_comparison"] = {}
        for alt_pooling in ("per_group", "per_segment", "max"):
            alt_embeddings = compute_embeddings(
                encoder=encoder,
                dataset=window_dataset,
                batch_size=batch_size,
                device=device,
                segment_count=segment_count,
                group_count=group_count,
                channels=channels,
                pooling=alt_pooling,
            )
            result["pooling_comparison"][alt_pooling] = run_linear_probe(
                embeddings=alt_embeddings,
                labels=labels,
                groups=groups,
                walk_ids=walk_ids,
                probe_c=probe_c,
                probe_max_iter=probe_max_iter,
                probe_knn_k=probe_knn_k,
                probe_mlp_hidden=probe_mlp_hidden,
                probe_mlp_max_iter=probe_mlp_max_iter,
                fold_indices=fold_indices,
            )

    if attentive_probe:
        tokens, valid_mask = compute_token_embeddings(
            encoder=encoder, dataset=window_dataset, batch_size=batch_size, device=device,
            segment_count=segment_count, group_count=group_count, channels=channels,
        )
        result["attentive"] = run_attentive_probe(
            tokens=tokens,
            valid_mask=valid_mask,
            labels=labels,
            groups=groups,
            walk_ids=walk_ids,
            n_heads=attentive_heads,
            epochs=attentive_epochs,
            lr=attentive_lr,
            weight_decay=attentive_weight_decay,
            device=device,
            fold_indices=fold_indices,
        )

    if dmu_probe and care_pd_labels is not None:
        result["dmu"] = run_dmu_probe(
            embeddings=embeddings, labels=labels, groups=groups, walk_ids=walk_ids,
            reference_label=dmu_reference_label, fold_indices=fold_indices,
        )

    if random_encoder is not None:
        random_embeddings = compute_embeddings(
            encoder=random_encoder,
            dataset=window_dataset,
            batch_size=batch_size,
            device=device,
            segment_count=segment_count,
            group_count=group_count,
            channels=channels,
            pooling="mean",
        )
        result["random_baseline"] = run_linear_probe(
            embeddings=random_embeddings,
            labels=labels,
            groups=groups,
            walk_ids=walk_ids,
            probe_c=probe_c,
            probe_max_iter=probe_max_iter,
            probe_knn_k=probe_knn_k,
            probe_mlp_hidden=probe_mlp_hidden,
            probe_mlp_max_iter=probe_mlp_max_iter,
            fold_indices=fold_indices,
        )

    return result

# ================================================================================================================
# MAIN
# ================================================================================================================

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    encoder, config = load_encoder(checkpoint=args.checkpoint, device=device)
    run_info = read_checkpoint_provenance(args.checkpoint, config)
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    care_pd_labels = None
    if any(name.startswith("CARE-PD-") for name in dataset_names):
        care_pd_labels = load_care_pd_labels(args.care_pd_labels)

    random_encoder = None
    if args.random_baseline:
        t.manual_seed(args.seed)
        random_encoder = MotionJEPA(config).teacher_encoder.to(device)
        random_encoder.eval()

    results = []
    for dataset_name in dataset_names:
        group_members = _DATASET_GROUPS.get(dataset_name, [dataset_name])
        result = evaluate_dataset(
            dataset_name=dataset_name,
            root=args.root,
            split=args.split,
            config=config,
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
            max_windows=args.max_windows,
            eval_window_size=args.eval_window_size,
            eval_stride=args.eval_stride,
            eval_min_valid_frames=args.eval_min_valid_frames,
            min_label_subjects=args.min_label_subjects,
            seed=args.seed,
            probe_c=args.probe_c,
            probe_max_iter=args.probe_max_iter,
            probe_knn_k=args.probe_knn_k,
            probe_mlp_hidden=args.probe_mlp_hidden,
            probe_mlp_max_iter=args.probe_mlp_max_iter,
            compare_pooling=args.compare_pooling,
            attentive_probe=args.attentive_probe,
            attentive_epochs=args.attentive_epochs,
            attentive_lr=args.attentive_lr,
            attentive_weight_decay=args.attentive_weight_decay,
            attentive_heads=args.attentive_heads,
            dmu_probe=args.dmu_probe,
            dmu_reference_label=args.dmu_reference_label,
            care_pd_labels=care_pd_labels,
            care_pd_fold_file=args.care_pd_fold_file,
            babel_raw_root=args.babel_raw_root if all(m in _BABEL_DATASETS for m in group_members) else None,
            random_encoder=random_encoder,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise SystemExit("No dataset produced a usable linear-probe result.")

    print_report(results, run_info)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({"run": run_info, "results": results}, f, indent=2)
        print(f"\nwrote: {args.out}")

    if not args.no_results_table:
        detail_json = str(args.out) if args.out is not None else None
        rows = [flatten_result(run_info, r, detail_json=detail_json) for r in results]
        upsert_results_table(rows, args.results_table)
        print(f"updated: {args.results_table} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
