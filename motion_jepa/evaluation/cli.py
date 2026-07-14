"""Linear-probe evaluation of a trained Motion-JEPA encoder.

Extracts a frozen embedding per window from the teacher encoder and fits a
linear classifier (logistic regression) on top of it, to measure how much
downstream-task-relevant information (action/emotion) the representation
carries -- the standard linear-probe protocol for self-supervised encoders.

Evaluated on the three held-out datasets (see config/datasets.yaml's
`validation` section): SOMA/HumanEva (action label) and DanceDB (emotion
label), each parsed from the trial filename by
`motion_jepa.evaluation.data.parse_eval_label`.

Cross-validation is leave-one-subject-out (grouped by subject, never by
window), since:
  - adjacent/overlapping windows from the same trial are highly correlated,
    so a random window-level split would leak and overstate accuracy.
  - these datasets have very few subjects (2-5), so held-out-subject
    generalization is both the meaningful question ("does the embedding
    encode this label for a person it never saw") and the only split that
    doesn't waste the little data available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as t

from motion_jepa.evaluation.data import (
    WindowRowsDataset,
    filter_labeled_rows,
    load_care_pd_labels,
    load_fixed_folds,
    load_window_table,
)
from motion_jepa.evaluation.encoder import compute_embeddings, compute_token_embeddings, load_encoder, read_checkpoint_provenance
from motion_jepa.evaluation.probes import print_report, run_attentive_probe, run_linear_probe

DEFAULT_DATASETS = ["SOMA", "HumanEva", "DanceDB"]

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
    parser.add_argument("--out", default=None, type=Path, help="Optional path to write full results as JSON.")
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
        "--babel-raw-root", default=None, type=Path,
        help=(
            "If set, attach BABEL frame-level action labels read from raw AMASS CSVs under "
            "this root (e.g. data/raw/amass) instead of filename/CARE-PD label parsing -- for "
            "datasets like ACCAD/MoSh/SFU with no filename-encoded label of their own (see "
            "config/datasets_babel.yaml). Use with --split train, since these datasets are "
            "still config/datasets.yaml's pretrain_train -- see CARE-PD-REPORT.md's caveat on "
            "evaluating labels for motion the encoder has already seen unlabeled."
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
    care_pd_labels: dict[str, int] | None = None,
    care_pd_fold_file: Path | None = None,
    babel_raw_root: Path | None = None,
) -> dict | None:
    rows = load_window_table(
        root=root, split=split, dataset=dataset_name, max_windows=max_windows, seed=seed,
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
            segment_count=segment_count, group_count=group_count,
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

    results = []
    for dataset_name in dataset_names:
        result = evaluate_dataset(
            dataset_name=dataset_name,
            root=args.root,
            split=args.split,
            config=config,
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
            max_windows=args.max_windows,
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
            care_pd_labels=care_pd_labels,
            care_pd_fold_file=args.care_pd_fold_file,
            babel_raw_root=args.babel_raw_root,
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


if __name__ == "__main__":
    main()
