"""Leave-one-subject-out probes (logistic regression + k-NN + MLP) and report printing."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ================================================================================================================
# PROBES
# ================================================================================================================

def run_linear_probe(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    probe_c: float,
    probe_max_iter: int,
    probe_knn_k: int,
    probe_mlp_hidden: int,
    probe_mlp_max_iter: int,
) -> dict:
    """Leave-one-subject-out linear + k-NN + MLP probes. Returns per-fold and aggregate metrics.

    The k-NN probe is a non-parametric complement to the logistic-regression
    linear probe: it has no training/regularization hyperparameters, so it
    isolates whether the embedding space's raw neighborhood structure already
    separates the label, independent of whether a linear boundary exists.

    The one-hidden-layer MLP probe checks the opposite direction: if it beats
    the linear probe by a wide margin, the label information is present in
    the embedding but not linearly separable -- the linear-probe number would
    then be understating what the encoder actually captured.
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            mlp = MLPClassifier(
                hidden_layer_sizes=(probe_mlp_hidden,),
                max_iter=probe_mlp_max_iter,
                early_stopping=True,
                random_state=0,
            )
            mlp.fit(x_train, y_train)
        y_pred_mlp = mlp.predict(x_test)

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
            balanced_acc_mlp = balanced_accuracy_score(y_test, y_pred_mlp)

        folds.append({
            "held_out_group": int(groups[test_idx[0]]) if np.issubdtype(groups.dtype, np.integer) else str(groups[test_idx[0]]),
            "n_test": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_acc),
            "knn_accuracy": float(accuracy_score(y_test, y_pred_knn)),
            "knn_balanced_accuracy": float(balanced_acc_knn),
            "knn_k": int(k),
            "mlp_accuracy": float(accuracy_score(y_test, y_pred_mlp)),
            "mlp_balanced_accuracy": float(balanced_acc_mlp),
            "majority_baseline_accuracy": float(accuracy_score(y_test, majority_pred)),
        })

    if not folds:
        raise ValueError("No valid leave-one-subject-out folds (need >=2 subjects with overlapping labels).")

    n_classes = len(label_encoder.classes_)
    accuracies = np.array([f["accuracy"] for f in folds])
    balanced_accuracies = np.array([f["balanced_accuracy"] for f in folds])
    knn_accuracies = np.array([f["knn_accuracy"] for f in folds])
    knn_balanced_accuracies = np.array([f["knn_balanced_accuracy"] for f in folds])
    mlp_accuracies = np.array([f["mlp_accuracy"] for f in folds])
    mlp_balanced_accuracies = np.array([f["mlp_balanced_accuracy"] for f in folds])
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
        "mlp_accuracy_mean": float(mlp_accuracies.mean()),
        "mlp_accuracy_std": float(mlp_accuracies.std()),
        "mlp_balanced_accuracy_mean": float(mlp_balanced_accuracies.mean()),
        "mlp_balanced_accuracy_std": float(mlp_balanced_accuracies.std()),
        "majority_baseline_mean": float(majority_baselines.mean()),
        "folds": folds,
    }

# ================================================================================================================
# REPORTING
# ================================================================================================================

def print_report(results: list[dict], run_info: dict) -> None:
    print()
    print("Linear-probe results (leave-one-subject-out)")
    print(f"checkpoint: {run_info['checkpoint']} (epoch={run_info['epoch']}, step={run_info['global_step']})")
    print("=" * 78)
    header = (
        f"{'dataset':<12} {'label':<8} {'windows':>8} {'classes':>8} {'folds':>6} "
        f"{'lin.acc':>8} {'lin.bal':>8} {'knn.acc':>8} {'knn.bal':>8} {'mlp.acc':>8} {'mlp.bal':>8} "
        f"{'chance':>8} {'majority':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['dataset']:<12} {r['label_kind']:<8} {r['n_windows']:>8} {r['n_classes']:>8} "
            f"{r['n_folds']:>6} {r['accuracy_mean']:>7.1%} {r['balanced_accuracy_mean']:>7.1%} "
            f"{r['knn_accuracy_mean']:>7.1%} {r['knn_balanced_accuracy_mean']:>7.1%} "
            f"{r['mlp_accuracy_mean']:>7.1%} {r['mlp_balanced_accuracy_mean']:>7.1%} "
            f"{r['chance_baseline']:>7.1%} {r['majority_baseline_mean']:>8.1%}"
        )
    print("=" * 78)
    print(
        "lin.* is held-out-subject logistic-regression accuracy; knn.* is the non-parametric\n"
        "k-NN probe (k up to --probe-knn-k, clipped to the smallest training class); mlp.* is a\n"
        "one-hidden-layer MLP probe (--probe-mlp-hidden units) -- a large mlp > lin gap means the\n"
        "label is present but not linearly separable. chance = 1/n_classes; majority = predicting\n"
        "the training fold's most common label.\n"
        "n_folds == n_subjects (leave-one-subject-out) -- with only a handful of subjects "
        "per dataset, treat these numbers as noisy point estimates, not precise scores."
    )

    pooling_results = [r for r in results if "pooling_comparison" in r]
    if pooling_results:
        alt_poolings = ["per_group", "per_segment", "max"]
        print()
        print("Pooling comparison: full mean-pool vs. alternatives (lin.acc, delta vs. mean)")
        print("=" * 78)
        col_width = 22
        header = f"{'dataset':<12} {'mean':>8} " + " ".join(f"{p:>{col_width}}" for p in alt_poolings)
        print(header)
        print("-" * len(header))
        for r in pooling_results:
            cells = []
            for p in alt_poolings:
                alt = r["pooling_comparison"][p]
                delta = alt["accuracy_mean"] - r["accuracy_mean"]
                cells.append(f"{alt['accuracy_mean']:>6.1%} ({delta:>+5.1%})".rjust(col_width))
            print(f"{r['dataset']:<12} {r['accuracy_mean']:>7.1%} " + " ".join(cells))
        print("=" * 78)
        print(
            "per-group keeps each anatomical group's own time-pooled embedding separate\n"
            "(tests whether pooling washes out signal localized to a few joints); per-segment\n"
            "keeps each time segment's own across-body embedding separate (tests whether\n"
            "pooling washes out signal localized to a part of the window, e.g. the motion's\n"
            "onset). Both cost a group_count- or segment_count-times wider feature vector for\n"
            "the same handful of training windows per fold, so treat a small positive delta\n"
            "there as inconclusive and only a large, consistent one as real. max takes an\n"
            "element-wise max over valid tokens instead of an average -- same width as mean,\n"
            "so no added overfitting risk, but privileges peak/salient tokens over the\n"
            "sustained average rather than detecting a rectified 'feature present' signal\n"
            "the way max-pooling does over ReLU activations in the GNN literature."
        )
