"""Leave-one-subject-out probes (logistic regression + k-NN + MLP) and report printing."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
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
    walk_ids: np.ndarray,
    probe_c: float,
    probe_max_iter: int,
    probe_knn_k: int,
    probe_mlp_hidden: int,
    probe_mlp_max_iter: int,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Linear + k-NN + MLP probes over caller-chosen folds. Returns per-fold and aggregate metrics.

    `fold_indices`: if None (default), folds are leave-one-*group*-out over
    `groups` (the original behavior -- subject for SOMA/HumanEva/DanceDB).
    If given, iterates those (train_idx, test_idx) pairs instead -- e.g.
    CARE-PD's own published fold assignment (see
    motion_jepa.evaluation.data.load_fixed_folds) instead of recomputing one.

    `walk_ids`: one id per row (e.g. window-table `suid`) identifying which
    single walk/trial each window came from. Overlapping windows (stride <
    window_size) from the same walk are highly correlated, so scoring every
    window as an independent prediction differs from -- and inflates the
    apparent sample count relative to -- the walk-level protocol the CARE-PD
    paper uses (per-clip predict, then majority-vote to one score per walk
    before computing F1). See `walk_f1_macro` etc. below for that metric.

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

    if fold_indices is None:
        fold_indices = list(LeaveOneGroupOut().split(embeddings, y, groups=groups))

    folds = []
    # Pooled across all folds -- CARE-PD's UPDRS-gait label is per-walk, and
    # many LOSO folds hold out a subject whose walks happen to share one
    # class, capping per-fold macro-F1 at 1/3 even on a perfect prediction.
    # Concatenating predictions before scoring once (standard cross-val-predict
    # aggregation) avoids that degenerate-fold ceiling and matches the paper's
    # own F1_0-2.
    y_test_all, y_pred_all, y_pred_knn_all, y_pred_mlp_all = [], [], [], []
    # Walk-level (majority-voted) pooled predictions -- one entry per walk
    # per fold, matching the paper's per-clip-predict-then-majority-vote
    # protocol instead of scoring every overlapping window independently.
    walk_y_test_all, walk_y_pred_all, walk_y_pred_knn_all, walk_y_pred_mlp_all = [], [], [], []

    for train_idx, test_idx in fold_indices:
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
            # balanced_accuracy_score has no `labels` param, so a fold whose
            # held-out subject has only one label (common with small LOSO
            # folds) makes it build a 1x1 confusion matrix and warn -- same
            # benign small-sample cause as the filter above.
            warnings.filterwarnings("ignore", message="A single label was found")
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            balanced_acc_knn = balanced_accuracy_score(y_test, y_pred_knn)
            balanced_acc_mlp = balanced_accuracy_score(y_test, y_pred_mlp)

        y_test_all.append(y_test)
        y_pred_all.append(y_pred)
        y_pred_knn_all.append(y_pred_knn)
        y_pred_mlp_all.append(y_pred_mlp)

        # Majority-vote each test walk's windows down to one prediction (and
        # one true label -- constant within a walk by construction) before
        # pooling, matching the paper's per-walk scoring.
        test_walks = walk_ids[test_idx]
        unique_walks, walk_inverse = np.unique(test_walks, return_inverse=True)
        walk_true = np.empty(len(unique_walks), dtype=y_test.dtype)
        walk_pred = np.empty(len(unique_walks), dtype=y_test.dtype)
        walk_pred_knn = np.empty(len(unique_walks), dtype=y_test.dtype)
        walk_pred_mlp = np.empty(len(unique_walks), dtype=y_test.dtype)
        for i in range(len(unique_walks)):
            mask = walk_inverse == i
            walk_true[i] = y_test[mask][0]
            walk_pred[i] = np.bincount(y_pred[mask]).argmax()
            walk_pred_knn[i] = np.bincount(y_pred_knn[mask]).argmax()
            walk_pred_mlp[i] = np.bincount(y_pred_mlp[mask]).argmax()

        walk_y_test_all.append(walk_true)
        walk_y_pred_all.append(walk_pred)
        walk_y_pred_knn_all.append(walk_pred_knn)
        walk_y_pred_mlp_all.append(walk_pred_mlp)

        folds.append({
            "held_out_group": int(groups[test_idx[0]]) if np.issubdtype(groups.dtype, np.integer) else str(groups[test_idx[0]]),
            "n_test": int(len(test_idx)),
            "n_walks": int(len(unique_walks)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_acc),
            "knn_accuracy": float(accuracy_score(y_test, y_pred_knn)),
            "knn_balanced_accuracy": float(balanced_acc_knn),
            "knn_k": int(k),
            "mlp_accuracy": float(accuracy_score(y_test, y_pred_mlp)),
            "mlp_balanced_accuracy": float(balanced_acc_mlp),
            "majority_baseline_accuracy": float(accuracy_score(y_test, majority_pred)),
            "walk_accuracy": float(accuracy_score(walk_true, walk_pred)),
            "walk_knn_accuracy": float(accuracy_score(walk_true, walk_pred_knn)),
            "walk_mlp_accuracy": float(accuracy_score(walk_true, walk_pred_mlp)),
            "walk_majority_baseline_accuracy": float(accuracy_score(walk_true, np.full_like(walk_true, majority_class))),
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
    walk_accuracies = np.array([f["walk_accuracy"] for f in folds])
    walk_knn_accuracies = np.array([f["walk_knn_accuracy"] for f in folds])
    walk_mlp_accuracies = np.array([f["walk_mlp_accuracy"] for f in folds])
    walk_majority_baselines = np.array([f["walk_majority_baseline_accuracy"] for f in folds])

    pooled_labels = np.arange(n_classes)
    y_test_all = np.concatenate(y_test_all)
    f1_macro = f1_score(y_test_all, np.concatenate(y_pred_all), labels=pooled_labels, average="macro", zero_division=0)
    f1_macro_knn = f1_score(y_test_all, np.concatenate(y_pred_knn_all), labels=pooled_labels, average="macro", zero_division=0)
    f1_macro_mlp = f1_score(y_test_all, np.concatenate(y_pred_mlp_all), labels=pooled_labels, average="macro", zero_division=0)

    walk_y_test_all = np.concatenate(walk_y_test_all)
    walk_f1_macro = f1_score(walk_y_test_all, np.concatenate(walk_y_pred_all), labels=pooled_labels, average="macro", zero_division=0)
    walk_f1_macro_knn = f1_score(walk_y_test_all, np.concatenate(walk_y_pred_knn_all), labels=pooled_labels, average="macro", zero_division=0)
    walk_f1_macro_mlp = f1_score(walk_y_test_all, np.concatenate(walk_y_pred_mlp_all), labels=pooled_labels, average="macro", zero_division=0)

    return {
        "n_classes": n_classes,
        "classes": label_encoder.classes_.tolist(),
        "n_folds": len(folds),
        "chance_baseline": 1.0 / n_classes,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std()),
        "balanced_accuracy_mean": float(balanced_accuracies.mean()),
        "balanced_accuracy_std": float(balanced_accuracies.std()),
        "f1_macro": float(f1_macro),
        "knn_accuracy_mean": float(knn_accuracies.mean()),
        "knn_accuracy_std": float(knn_accuracies.std()),
        "knn_balanced_accuracy_mean": float(knn_balanced_accuracies.mean()),
        "knn_balanced_accuracy_std": float(knn_balanced_accuracies.std()),
        "knn_f1_macro": float(f1_macro_knn),
        "mlp_accuracy_mean": float(mlp_accuracies.mean()),
        "mlp_accuracy_std": float(mlp_accuracies.std()),
        "mlp_balanced_accuracy_mean": float(mlp_balanced_accuracies.mean()),
        "mlp_balanced_accuracy_std": float(mlp_balanced_accuracies.std()),
        "mlp_f1_macro": float(f1_macro_mlp),
        "majority_baseline_mean": float(majority_baselines.mean()),
        "n_walks_total": int(sum(f["n_walks"] for f in folds)),
        "walk_accuracy_mean": float(walk_accuracies.mean()),
        "walk_accuracy_std": float(walk_accuracies.std()),
        "walk_f1_macro": float(walk_f1_macro),
        "walk_knn_accuracy_mean": float(walk_knn_accuracies.mean()),
        "walk_knn_accuracy_std": float(walk_knn_accuracies.std()),
        "walk_knn_f1_macro": float(walk_f1_macro_knn),
        "walk_mlp_accuracy_mean": float(walk_mlp_accuracies.mean()),
        "walk_mlp_accuracy_std": float(walk_mlp_accuracies.std()),
        "walk_mlp_f1_macro": float(walk_f1_macro_mlp),
        "walk_majority_baseline_mean": float(walk_majority_baselines.mean()),
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
        f"{'lin.acc':>8} {'lin.bal':>8} {'lin.f1':>8} {'knn.acc':>8} {'knn.f1':>8} {'mlp.acc':>8} {'mlp.f1':>8} "
        f"{'chance':>8} {'majority':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['dataset']:<12} {r['label_kind']:<8} {r['n_windows']:>8} {r['n_classes']:>8} "
            f"{r['n_folds']:>6} {r['accuracy_mean']:>7.1%} {r['balanced_accuracy_mean']:>7.1%} "
            f"{r['f1_macro']:>7.1%} "
            f"{r['knn_accuracy_mean']:>7.1%} {r['knn_f1_macro']:>7.1%} "
            f"{r['mlp_accuracy_mean']:>7.1%} {r['mlp_f1_macro']:>7.1%} "
            f"{r['chance_baseline']:>7.1%} {r['majority_baseline_mean']:>8.1%}"
        )
    print("=" * 78)
    print(
        "lin.* is held-out-subject logistic-regression accuracy; knn.* is the non-parametric\n"
        "k-NN probe (k up to --probe-knn-k, clipped to the smallest training class); mlp.* is a\n"
        "one-hidden-layer MLP probe (--probe-mlp-hidden units) -- a large mlp > lin gap means the\n"
        "label is present but not linearly separable. *.f1 is macro-F1 pooled across all folds'\n"
        "predictions (CARE-PD paper's own metric) -- per-fold macro-F1 would be capped at 1/3 on\n"
        "folds whose held-out subject's labels happen to fall in one class, even for a perfect\n"
        "prediction. chance = 1/n_classes;\n"
        "majority = predicting the training fold's most common label.\n"
        "n_folds == n_subjects (leave-one-subject-out) -- with only a handful of subjects "
        "per dataset, treat these numbers as noisy point estimates, not precise scores."
    )

    print()
    print("Walk-level (majority-vote) results -- matches CARE-PD paper's per-clip-predict,")
    print("then majority-vote-to-one-score-per-walk protocol, instead of scoring every")
    print("overlapping window independently.")
    print("=" * 78)
    walk_header = (
        f"{'dataset':<12} {'walks':>6} {'lin.acc':>8} {'lin.f1':>8} {'knn.acc':>8} {'knn.f1':>8} "
        f"{'mlp.acc':>8} {'mlp.f1':>8} {'majority':>9}"
    )
    print(walk_header)
    print("-" * len(walk_header))
    for r in results:
        print(
            f"{r['dataset']:<12} {r['n_walks_total']:>6} "
            f"{r['walk_accuracy_mean']:>7.1%} {r['walk_f1_macro']:>7.1%} "
            f"{r['walk_knn_accuracy_mean']:>7.1%} {r['walk_knn_f1_macro']:>7.1%} "
            f"{r['walk_mlp_accuracy_mean']:>7.1%} {r['walk_mlp_f1_macro']:>7.1%} "
            f"{r['walk_majority_baseline_mean']:>8.1%}"
        )
    print("=" * 78)

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
