"""Linear-probe evaluation of a trained Motion-JEPA encoder.

Extracts a frozen embedding per window from the teacher encoder and fits a
linear classifier (logistic regression) on top of it, to measure how much
downstream-task-relevant information (action/emotion) the representation
carries -- the standard linear-probe protocol for self-supervised encoders.

Evaluated on the three held-out datasets (see config/datasets.yaml's
`validation` section): SOMA/HumanEva (action label) and DanceDB (emotion
label), each parsed from the trial filename by `motion_jepa.eval.parse_eval_label`.

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
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch as t
from omegaconf import OmegaConf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from motion_jepa.eval import WindowRowsDataset, compute_embeddings, load_encoder, load_window_table

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
    parser.add_argument("--out", default=None, type=Path, help="Optional path to write full results as JSON.")
    return parser.parse_args()


def resolve_device(device: str) -> t.device:
    if device == "auto":
        return t.device("cuda" if t.cuda.is_available() else "cpu")
    if device == "cuda" and not t.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return t.device(device)


def read_checkpoint_provenance(checkpoint: Path, config) -> dict:
    """Epoch/step plus the full training config, for tracing a result back
    to exactly which run and which hyperparameters produced it.

    Read directly off the raw checkpoint dict rather than the loaded
    LightningModule, since `current_epoch`/`global_step` on a module loaded
    outside of a `Trainer` aren't reliably populated.
    """
    raw = t.load(str(checkpoint), map_location="cpu", weights_only=False)
    return {
        "checkpoint": str(checkpoint),
        "epoch": raw.get("epoch"),
        "global_step": raw.get("global_step"),
        "config": OmegaConf.to_container(config, resolve=True),
    }

# ================================================================================================================
# LABEL FILTERING
# ================================================================================================================

def filter_labeled_rows(rows: pl.DataFrame, *, min_label_subjects: int) -> pl.DataFrame:
    """Drop unlabeled rows and any label seen in too few distinct subjects.

    A label with fewer distinct subjects than `min_label_subjects` can never
    appear in both the train and held-out side of a leave-subject-out fold,
    so it's unevaluable noise (e.g. DanceDB's "Haniotikos"/"Mix" -- filename
    parsing artifacts from trials that don't follow the usual naming
    convention, each contributed by a single one-off subject).
    """
    rows = rows.filter(pl.col("eval_label").is_not_null())

    label_subject_counts = (
        rows.group_by("eval_label")
        .agg(pl.col("subject").n_unique().alias("n_subjects"))
    )
    valid_labels = label_subject_counts.filter(
        pl.col("n_subjects") >= min_label_subjects
    )["eval_label"].to_list()

    return rows.filter(pl.col("eval_label").is_in(valid_labels))

# ================================================================================================================
# LINEAR PROBE
# ================================================================================================================

def run_linear_probe(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    probe_c: float,
    probe_max_iter: int,
    probe_knn_k: int,
) -> dict:
    """Leave-one-subject-out linear + k-NN probes. Returns per-fold and aggregate metrics.

    The k-NN probe is a non-parametric complement to the logistic-regression
    linear probe: it has no training/regularization hyperparameters, so it
    isolates whether the embedding space's raw neighborhood structure already
    separates the label, independent of whether a linear boundary exists.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    splitter = LeaveOneGroupOut()
    folds = []

    for train_idx, test_idx in splitter.split(embeddings, y, groups=groups):
        y_train, y_test = y[train_idx], y[test_idx]

        # A fold is only meaningful if the held-out subject's labels were
        # also seen during training (otherwise the probe can't possibly
        # predict them) and if training has more than one class to learn.
        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler().fit(embeddings[train_idx])
        x_train = scaler.transform(embeddings[train_idx])
        x_test = scaler.transform(embeddings[test_idx])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf = LogisticRegression(C=probe_c, max_iter=probe_max_iter)
            clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        # k can't exceed the smallest class's training count.
        k = min(probe_knn_k, np.bincount(y_train).min())
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred_knn = knn.predict(x_test)

        majority_class = np.bincount(y_train).argmax()
        majority_pred = np.full_like(y_test, majority_class)

        # A held-out subject's true labels are usually a subset of the full
        # training vocabulary (few subjects, imbalanced fine-grained classes),
        # so the classifier will legitimately predict labels this particular
        # subject never has -- expected given the small-sample regime here,
        # not an error. balanced_accuracy_score's per-true-class recall is
        # unaffected by it; silence the informational warning rather than
        # let it drown out the actionable ConvergenceWarning above.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            balanced_acc_knn = balanced_accuracy_score(y_test, y_pred_knn)

        folds.append({
            "held_out_group": int(groups[test_idx[0]]) if np.issubdtype(groups.dtype, np.integer) else str(groups[test_idx[0]]),
            "n_test": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_acc),
            "knn_accuracy": float(accuracy_score(y_test, y_pred_knn)),
            "knn_balanced_accuracy": float(balanced_acc_knn),
            "knn_k": int(k),
            "majority_baseline_accuracy": float(accuracy_score(y_test, majority_pred)),
        })

    if not folds:
        raise ValueError("No valid leave-one-subject-out folds (need >=2 subjects with overlapping labels).")

    n_classes = len(label_encoder.classes_)
    accuracies = np.array([f["accuracy"] for f in folds])
    balanced_accuracies = np.array([f["balanced_accuracy"] for f in folds])
    knn_accuracies = np.array([f["knn_accuracy"] for f in folds])
    knn_balanced_accuracies = np.array([f["knn_balanced_accuracy"] for f in folds])
    majority_baselines = np.array([f["majority_baseline_accuracy"] for f in folds])

    return {
        "n_classes": n_classes,
        "classes": label_encoder.classes_.tolist(),
        "n_folds": len(folds),
        "chance_baseline": 1.0 / n_classes,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std()),
        "balanced_accuracy_mean": float(balanced_accuracies.mean()),
        "balanced_accuracy_std": float(balanced_accuracies.std()),
        "knn_accuracy_mean": float(knn_accuracies.mean()),
        "knn_accuracy_std": float(knn_accuracies.std()),
        "knn_balanced_accuracy_mean": float(knn_balanced_accuracies.mean()),
        "knn_balanced_accuracy_std": float(knn_balanced_accuracies.std()),
        "majority_baseline_mean": float(majority_baselines.mean()),
        "folds": folds,
    }

# ================================================================================================================
# MAIN
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
) -> dict | None:
    rows = load_window_table(
        root=root, split=split, dataset=dataset_name, max_windows=max_windows, seed=seed,
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
    embeddings = compute_embeddings(
        encoder=encoder,
        dataset=window_dataset,
        batch_size=batch_size,
        device=device,
        segment_count=config.data.window_size // config.architecture.segment_size,
        group_count=len(config.training.groups),
    )

    result = run_linear_probe(
        embeddings=embeddings,
        labels=rows["eval_label"].to_numpy(),
        groups=rows["subject"].to_numpy(),
        probe_c=probe_c,
        probe_max_iter=probe_max_iter,
        probe_knn_k=probe_knn_k,
    )
    result.update({
        "dataset": dataset_name,
        "label_kind": label_kind,
        "n_windows": rows.height,
        "n_subjects": n_subjects,
    })
    return result


def print_report(results: list[dict], run_info: dict) -> None:
    print()
    print("Linear-probe results (leave-one-subject-out)")
    print(f"checkpoint: {run_info['checkpoint']} (epoch={run_info['epoch']}, step={run_info['global_step']})")
    print("=" * 78)
    header = (
        f"{'dataset':<12} {'label':<8} {'windows':>8} {'classes':>8} {'folds':>6} "
        f"{'lin.acc':>8} {'lin.bal':>8} {'knn.acc':>8} {'knn.bal':>8} {'chance':>8} {'majority':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['dataset']:<12} {r['label_kind']:<8} {r['n_windows']:>8} {r['n_classes']:>8} "
            f"{r['n_folds']:>6} {r['accuracy_mean']:>7.1%} {r['balanced_accuracy_mean']:>7.1%} "
            f"{r['knn_accuracy_mean']:>7.1%} {r['knn_balanced_accuracy_mean']:>7.1%} "
            f"{r['chance_baseline']:>7.1%} {r['majority_baseline_mean']:>8.1%}"
        )
    print("=" * 78)
    print(
        "lin.acc/lin.bal are held-out-subject logistic-regression accuracy; knn.acc/knn.bal are\n"
        "the non-parametric k-NN probe (k up to --probe-knn-k, clipped to the smallest training\n"
        "class); chance = 1/n_classes; majority = predicting the training fold's most common label.\n"
        "n_folds == n_subjects (leave-one-subject-out) -- with only a handful of subjects "
        "per dataset, treat these numbers as noisy point estimates, not precise scores."
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    encoder, config = load_encoder(checkpoint=args.checkpoint, device=device)
    run_info = read_checkpoint_provenance(args.checkpoint, config)
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

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
