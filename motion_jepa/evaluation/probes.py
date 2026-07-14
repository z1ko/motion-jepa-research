"""Leave-one-subject-out probes (logistic regression + k-NN + MLP) and report printing."""

from __future__ import annotations

import warnings

import numpy as np
import torch as t
import torch.nn as nn
from scipy.stats import spearmanr
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
# DMU (DEVIATION FROM MEAN UNIMPAIRED)
# ================================================================================================================

def run_dmu_probe(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    walk_ids: np.ndarray,
    reference_label: str | None = None,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Deviation-from-mean-unimpaired score (papers/GaitEncoder.pdf Sec 4.2): a
    per-window diagonal Mahalanobis distance from a reference ("unimpaired")
    class's mean/variance, computed fresh per LOSO fold from the training
    split only. Unlike run_linear_probe, this needs no classifier at all --
    it's a continuous, unsupervised-at-scoring-time distance, tested here for
    whether it correlates with an ordinal severity label (`labels`, e.g.
    CARE-PD's "0"/"1"/"2" MDS-UPDRS-gait score) where a hard LOSO classifier
    might not (see CARE-PD-REPORT.md).

    Two deliberate deviations from the paper: (1) diagonal (per-feature)
    variance instead of full covariance -- our embed_dim (256) is much wider
    than their 16-dim latent, and a fold's reference class is too small to
    estimate a full covariance from; (2) the reference mean/variance is
    refit per LOSO fold (train split only) instead of one fixed pre-curated
    cohort, since we don't have a cohort disjoint from the labeled data.

    Correlation is computed once, pooled across all folds' held-out (dmu,
    label) pairs, not per-fold: CARE-PD-REPORT.md established severity is
    near-constant per subject, so a single held-out subject's labels carry
    ~no variance to correlate against.
    """
    if fold_indices is None:
        fold_indices = list(LeaveOneGroupOut().split(embeddings, labels, groups=groups))

    dmu_all, label_all = [], []
    walk_dmu_all, walk_label_all = [], []
    n_folds = 0

    for train_idx, test_idx in fold_indices:
        labels_train = labels[train_idx]
        ref_label = reference_label if reference_label is not None else min(labels_train)
        ref_mask = labels_train == ref_label
        if ref_mask.sum() < 5:
            continue

        mu = embeddings[train_idx][ref_mask].mean(axis=0)
        var = embeddings[train_idx][ref_mask].var(axis=0) + 1e-6
        dmu = np.sqrt(((embeddings[test_idx] - mu) ** 2 / var).sum(axis=1))

        dmu_all.append(dmu)
        label_all.append(labels[test_idx])
        n_folds += 1

        test_walks = walk_ids[test_idx]
        unique_walks, walk_inverse = np.unique(test_walks, return_inverse=True)
        for i in range(len(unique_walks)):
            mask = walk_inverse == i
            walk_dmu_all.append(dmu[mask].mean())
            walk_label_all.append(labels[test_idx][mask][0])

    if n_folds == 0:
        raise ValueError("No valid LOSO folds had >=5 reference-class training windows.")

    dmu_all = np.concatenate(dmu_all)
    label_all = np.concatenate(label_all).astype(int)
    walk_dmu_all = np.array(walk_dmu_all)
    walk_label_all = np.array(walk_label_all).astype(int)

    spearman_r, spearman_p = spearmanr(dmu_all, label_all)
    walk_spearman_r, walk_spearman_p = spearmanr(walk_dmu_all, walk_label_all)

    per_class = {}
    for cls in sorted(set(label_all.tolist())):
        mask = label_all == cls
        per_class[str(cls)] = {
            "mean": float(dmu_all[mask].mean()),
            "std": float(dmu_all[mask].std()),
            "n": int(mask.sum()),
        }

    return {
        "n_folds": n_folds,
        "reference_label": str(reference_label) if reference_label is not None else "auto (min per fold)",
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "walk_spearman_r": float(walk_spearman_r),
        "walk_spearman_p": float(walk_spearman_p),
        "per_class": per_class,
    }

# ================================================================================================================
# ATTENTIVE PROBE
# ================================================================================================================

class AttentiveProbeHead(nn.Module):
    """A learned query cross-attends over every token before the linear classifier,
    instead of a fixed mean/max pool -- tests whether pre-pooling (compute_embeddings)
    is throwing away signal the encoder actually has (see CARE-PD-REPORT.md). Trained
    fresh per LOSO fold, same as the linear/kNN/MLP probes below, so it never sees the
    held-out subject's labels.
    """

    def __init__(self, embed_dim: int, n_classes: int, n_heads: int):
        super().__init__()
        self.query = nn.Parameter(t.randn(1, 1, embed_dim) * embed_dim**-0.5)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, tokens: t.Tensor, key_padding_mask: t.Tensor) -> t.Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.classifier(self.norm(pooled.squeeze(1)))


def run_attentive_probe(
    *,
    tokens: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    walk_ids: np.ndarray,
    n_heads: int = 4,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: t.device | None = None,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Same leave-one-subject-out protocol as run_linear_probe, but the probe is an
    AttentiveProbeHead trained by gradient descent on the encoder's raw per-token
    output (`tokens`, `valid_mask` from encoder.compute_token_embeddings) instead of
    a scikit-learn classifier on a pre-pooled vector.
    """
    device = device or t.device("cuda" if t.cuda.is_available() else "cpu")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)

    if fold_indices is None:
        fold_indices = list(LeaveOneGroupOut().split(tokens, y, groups=groups))

    tokens_t = t.as_tensor(tokens, dtype=t.float32)
    key_padding_mask_all = ~t.as_tensor(valid_mask, dtype=t.bool)  # True = ignore, matches nn.MultiheadAttention convention
    y_t = t.as_tensor(y, dtype=t.long)

    accuracies, walk_accuracies = [], []
    y_test_all, y_pred_all, walk_y_test_all, walk_y_pred_all = [], [], [], []

    for train_idx, test_idx in fold_indices:
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue

        x_train = tokens_t[train_idx].to(device)
        mask_train = key_padding_mask_all[train_idx].to(device)
        y_train_t = y_t[train_idx].to(device)
        x_test = tokens_t[test_idx].to(device)
        mask_test = key_padding_mask_all[test_idx].to(device)

        head = AttentiveProbeHead(embed_dim=tokens.shape[-1], n_classes=n_classes, n_heads=n_heads).to(device)
        optimizer = t.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

        head.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = head(x_train, mask_train)
            loss = nn.functional.cross_entropy(logits, y_train_t)
            loss.backward()
            optimizer.step()

        head.eval()
        with t.no_grad():
            y_pred = head(x_test, mask_test).argmax(dim=-1).cpu().numpy()
        y_test = y[test_idx]

        accuracies.append(accuracy_score(y_test, y_pred))
        y_test_all.append(y_test)
        y_pred_all.append(y_pred)

        test_walks = walk_ids[test_idx]
        unique_walks, walk_inverse = np.unique(test_walks, return_inverse=True)
        walk_true = np.array([y_test[walk_inverse == i][0] for i in range(len(unique_walks))])
        walk_pred = np.array([np.bincount(y_pred[walk_inverse == i]).argmax() for i in range(len(unique_walks))])
        walk_accuracies.append(accuracy_score(walk_true, walk_pred))
        walk_y_test_all.append(walk_true)
        walk_y_pred_all.append(walk_pred)

    if not y_test_all:
        raise ValueError("No valid leave-one-subject-out folds (need >=2 subjects with overlapping labels).")

    pooled_labels = np.arange(n_classes)
    return {
        "n_folds": len(y_test_all),
        "accuracy_mean": float(np.mean(accuracies)),
        "f1_macro": float(f1_score(np.concatenate(y_test_all), np.concatenate(y_pred_all), labels=pooled_labels, average="macro", zero_division=0)),
        "walk_accuracy_mean": float(np.mean(walk_accuracies)),
        "walk_f1_macro": float(f1_score(np.concatenate(walk_y_test_all), np.concatenate(walk_y_pred_all), labels=pooled_labels, average="macro", zero_division=0)),
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

    attentive_results = [r for r in results if "attentive" in r]
    if attentive_results:
        print()
        print("Attentive probe: learned-query attention pool vs. fixed mean-pool (walk-level)")
        print("=" * 78)
        header = f"{'dataset':<12} {'mean.walk_f1':>13} {'attn.walk_f1':>13} {'delta':>8}"
        print(header)
        print("-" * len(header))
        for r in attentive_results:
            a = r["attentive"]
            delta = a["walk_f1_macro"] - r["walk_f1_macro"]
            print(f"{r['dataset']:<12} {r['walk_f1_macro']:>12.1%} {a['walk_f1_macro']:>12.1%} {delta:>+7.1%}")
        print("=" * 78)
        print(
            "attn.* trains a learned query to cross-attend over every token (no fixed\n"
            "pooling rule) before a linear classifier, fresh per LOSO fold -- tests whether\n"
            "mean-pooling specifically was discarding signal the encoder has. A delta near\n"
            "zero means pooling wasn't the bottleneck; a large positive delta means it was."
        )

    dmu_results = [r for r in results if "dmu" in r]
    if dmu_results:
        print()
        print("DMU (deviation from mean unimpaired, papers/GaitEncoder.pdf) vs. ordinal severity label")
        print("=" * 78)
        header = f"{'dataset':<12} {'ref':>10} {'folds':>6} {'spearman r (p)':>20} {'walk r (p)':>20}"
        print(header)
        print("-" * len(header))
        for r in dmu_results:
            d = r["dmu"]
            ref = d["reference_label"] if d["reference_label"] != "auto (min per fold)" else "auto"
            print(
                f"{r['dataset']:<12} {ref:>10} {d['n_folds']:>6} "
                f"{d['spearman_r']:>7.3f} ({d['spearman_p']:.3f}) "
                f"{d['walk_spearman_r']:>10.3f} ({d['walk_spearman_p']:.3f})"
            )
        print("-" * len(header))
        for r in dmu_results:
            classes = ", ".join(
                f"{cls}: {c['mean']:.2f}+-{c['std']:.2f} (n={c['n']})"
                for cls, c in r["dmu"]["per_class"].items()
            )
            print(f"{r['dataset']:<12} per-class dmu mean+-std: {classes}")
        print("=" * 78)
        print(
            "dmu is a per-window diagonal-Mahalanobis distance from the reference class's\n"
            "mean/variance (refit per LOSO fold, train split only) -- not a classifier, so\n"
            "there's no accuracy/chance baseline. spearman_r is pooled across all folds'\n"
            "held-out (dmu, label) pairs, not computed per-fold: severity is near-constant\n"
            "per subject (see CARE-PD-REPORT.md), so a single held-out subject's labels\n"
            "carry ~no variance to correlate against on their own. Positive r means higher\n"
            "distance from the reference class tracks higher severity, as intended."
        )
