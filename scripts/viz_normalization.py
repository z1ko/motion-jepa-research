
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import numpy as np

from motion_jepa.dataset import MotionZarrStore, _path_of_samples_index
from motion_jepa.preprocess import CHANNELS, JOINTS

# ┌──────────────────────┬─────────┬───────────┬─────────────┬───┬────────────┬───────────┬─────────────┬───────────────────────┐
# │ joint                ┆ channel ┆ mean      ┆ std         ┆ … ┆ p1         ┆ p99       ┆ p99.9       ┆ std_over_robust_sigma │
# │ ---                  ┆ ---     ┆ ---       ┆ ---         ┆   ┆ ---        ┆ ---       ┆ ---         ┆ ---                   │
# │ str                  ┆ str     ┆ f64       ┆ f64         ┆   ┆ f64        ┆ f64       ┆ f64         ┆ f64                   │
# ╞══════════════════════╪═════════╪═══════════╪═════════════╪═══╪════════════╪═══════════╪═════════════╪═══════════════════════╡
# │ pelvis_rotation      ┆ tau     ┆ -1.52729  ┆ 1568.07959  ┆ … ┆ -6.517816  ┆ 5.541761  ┆ 1490.512141 ┆ 8804.902883           │
# │ hip_adduction_r      ┆ tau     ┆ -5.108039 ┆ 1456.360352 ┆ … ┆ -13.086959 ┆ -0.101837 ┆ 645.737139  ┆ 634.733352            │
# │ pelvis_tilt          ┆ tau     ┆ 1.854938  ┆ 1416.411133 ┆ … ┆ -5.385261  ┆ 5.279772  ┆ 1875.166634 ┆ 10706.385524          │
# │ pelvis_ty            ┆ tau     ┆ -0.454495 ┆ 1081.276367 ┆ … ┆ -15.042297 ┆ 13.72496  ┆ 4611.183334 ┆ 6075.786796           │
# │ pelvis_tx            ┆ tau     ┆ -2.684009 ┆ 1047.389771 ┆ … ┆ -15.749229 ┆ 16.31806  ┆ 4675.365062 ┆ 5731.305604           │
# │ …                    ┆ …       ┆ …         ┆ …           ┆ … ┆ …          ┆ …         ┆ …           ┆ …                     │
# │ scapula_upward_rot_r ┆ tau     ┆ 0.241332  ┆ 22.357506   ┆ … ┆ -0.663382  ┆ 1.006603  ┆ 168.231818  ┆ 730.618088            │
# │ scapula_elevation_l  ┆ tau     ┆ -0.075593 ┆ 21.778679   ┆ … ┆ -0.794271  ┆ 0.770581  ┆ 121.22567   ┆ 737.983127            │
# │ shoulder_r_z         ┆ tau     ┆ -0.010187 ┆ 21.55834    ┆ … ┆ -0.756964  ┆ 0.772975  ┆ 165.580313  ┆ 495.064002            │
# │ shoulder_l_z         ┆ tau     ┆ 0.056091  ┆ 21.069241   ┆ … ┆ -0.744174  ┆ 0.74658   ┆ 151.214664  ┆ 507.326812            │
# │ scapula_elevation_r  ┆ tau     ┆ -0.080381 ┆ 20.700201   ┆ … ┆ -0.967525  ┆ 0.779436  ┆ 125.789747  ┆ 664.262133            │
# └──────────────────────┴─────────┴───────────┴─────────────┴───┴────────────┴───────────┴─────────────┴───────────────────────┘

# ┌──────────────────────┬─────────┬───────────┬──────────┬───┬───────────┬──────────┬──────────┬───────────────────────┐
# │ joint                ┆ channel ┆ mean      ┆ std      ┆ … ┆ p1        ┆ p99      ┆ p99.9    ┆ std_over_robust_sigma │
# │ ---                  ┆ ---     ┆ ---       ┆ ---      ┆   ┆ ---       ┆ ---      ┆ ---      ┆ ---                   │
# │ str                  ┆ str     ┆ f64       ┆ f64      ┆   ┆ f64       ┆ f64      ┆ f64      ┆ f64                   │
# ╞══════════════════════╪═════════╪═══════════╪══════════╪═══╪═══════════╪══════════╪══════════╪═══════════════════════╡
# │ pelvis_tilt          ┆ pos     ┆ -0.003535 ┆ 1.847046 ┆ … ┆ -3.092816 ┆ 3.090358 ┆ 3.136627 ┆ 0.777434              │
# │ pelvis_rotation      ┆ pos     ┆ -0.709561 ┆ 1.658132 ┆ … ┆ -3.03888  ┆ 3.033323 ┆ 3.131039 ┆ 0.924747              │
# │ pelvis_ty            ┆ pos     ┆ 0.179123  ┆ 0.678886 ┆ … ┆ -1.588255 ┆ 1.946658 ┆ 2.729171 ┆ 1.694022              │
# │ pelvis_tx            ┆ pos     ┆ 0.068798  ┆ 0.62306  ┆ … ┆ -1.643919 ┆ 2.183824 ┆ 3.161944 ┆ 1.924317              │
# │ elbow_flexion_r      ┆ pos     ┆ 0.884292  ┆ 0.549015 ┆ … ┆ 0.008766  ┆ 2.268825 ┆ 2.273203 ┆ 1.020157              │
# │ …                    ┆ …       ┆ …         ┆ …        ┆ … ┆ …         ┆ …        ┆ …        ┆ …                     │
# │ scapula_upward_rot_r ┆ pos     ┆ 0.077491  ┆ 0.237614 ┆ … ┆ -0.3      ┆ 0.8638   ┆ 1.000106 ┆ 1.25352               │
# │ hip_rotation_l       ┆ pos     ┆ -0.071612 ┆ 0.237036 ┆ … ┆ -0.698132 ┆ 0.616534 ┆ 0.698293 ┆ 1.173329              │
# │ scapula_upward_rot_l ┆ pos     ┆ 0.083344  ┆ 0.233443 ┆ … ┆ -0.3      ┆ 0.914541 ┆ 1.000364 ┆ 1.324961              │
# │ ankle_angle_r        ┆ pos     ┆ 0.185372  ┆ 0.218224 ┆ … ┆ -0.698028 ┆ 0.524846 ┆ 0.537511 ┆ 1.267084              │
# │ ankle_angle_l        ┆ pos     ┆ 0.212377  ┆ 0.209134 ┆ … ┆ -0.697307 ┆ 0.525726 ┆ 0.539447 ┆ 1.234906              │
# └──────────────────────┴─────────┴───────────┴──────────┴───┴───────────┴──────────┴──────────┴───────────────────────┘

def visualize_normalization_stats(
    stats,
    *,
    joints: list[str],
    channels: list[str],
    title_prefix: str = "Kinematics",
    save_dir: Path | str | None = None,
) -> None:
    
    mean = np.asarray(stats["mean"])
    std  = np.asarray(stats["std"])

    if mean.shape != (len(joints), len(channels)):
        raise ValueError(
            f"Expected mean shape {(len(joints), len(channels))}, got {mean.shape}"
        )

    if std.shape != (len(joints), len(channels)):
        raise ValueError(
            f"Expected std shape {(len(joints), len(channels))}, got {std.shape}"
        )

    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    def plot_heatmap(values: np.ndarray, name: str, cmap: str = "viridis") -> None:
        fig, ax = plt.subplots(figsize=(8, 14))

        im = ax.imshow(values, aspect="auto", cmap=cmap)

        ax.set_title(f"{title_prefix}: {name}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Joint")

        ax.set_xticks(np.arange(len(channels)))
        ax.set_xticklabels(channels)

        ax.set_yticks(np.arange(len(joints)))
        ax.set_yticklabels(joints)

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(save_dir / f"{name.lower().replace(' ', '_')}.png", dpi=200)

        plt.show()

    plot_heatmap(mean, "Mean")
    plot_heatmap(std, "Std")
    plot_heatmap(np.log10(std + 1e-8), "Log10 Std")

def normalization_stats_report(
    stats,
    *,
    joints: list[str],
    channels: list[str],
    low_std_threshold: float = 1e-3,
    top_k: int = 20,
) -> None:
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])

    rows = []

    for j, joint in enumerate(joints):
        for c, channel in enumerate(channels):
            rows.append(
                {
                    "joint": joint,
                    "channel": channel,
                    "mean": float(mean[j, c]),
                    "std": float(std[j, c]),
                    "abs_mean": float(abs(mean[j, c])),
                }
            )

    rows_sorted_by_std = sorted(rows, key=lambda r: r["std"])
    rows_sorted_by_abs_mean = sorted(rows, key=lambda r: r["abs_mean"], reverse=True)

    print("\nLowest-std dimensions")
    print("---------------------")
    for row in rows_sorted_by_std[:top_k]:
        marker = "  <-- low variance" if row["std"] < low_std_threshold else ""
        print(
            f"{row['joint']:28s} {row['channel']:>4s} "
            f"mean={row['mean']: .5f} std={row['std']: .5f}{marker}"
        )

    print("\nLargest absolute means")
    print("----------------------")
    for row in rows_sorted_by_abs_mean[:top_k]:
        print(
            f"{row['joint']:28s} {row['channel']:>4s} "
            f"mean={row['mean']: .5f} std={row['std']: .5f}"
        )

def robust_stats_report(
    root,
    *,
    split: str = "train",
    channel: str = "tau",
    top_k: int = 30,
):
    root = Path(root)
    samples = pl.read_parquet(_path_of_samples_index(root))
    samples = samples.filter(pl.col("split") == split)

    store = MotionZarrStore(root)
    c = CHANNELS.index(channel)

    rows = []

    for j, joint in enumerate(JOINTS):
        values = []

        for row in samples.iter_rows(named=True):
            suid = str(row["suid"])
            x = np.asarray(store.get_kinematics(suid)[:, j, c], dtype=np.float32)
            values.append(x)

        v = np.concatenate(values)

        p01, p1, p25, p50, p75, p99, p999 = np.percentile(
            v,
            [0.1, 1, 25, 50, 75, 99, 99.9],
        )

        std = float(v.std())
        iqr = float(p75 - p25)
        robust_sigma = iqr / 1.349 if iqr > 0 else 0.0

        rows.append(
            {
                "joint": joint,
                "channel": channel,
                "mean": float(v.mean()),
                "std": std,
                "median": float(p50),
                "iqr": iqr,
                "robust_sigma": robust_sigma,
                "p0.1": float(p01),
                "p1": float(p1),
                "p99": float(p99),
                "p99.9": float(p999),
                "std_over_robust_sigma": std / max(robust_sigma, 1e-8),
            }
        )

    df = pl.DataFrame(rows).sort("std", descending=True)
    print(df.head(top_k))
    return df

stats = np.load("data/processed/motion/normalization_stats.npz")
normalization_stats_report(stats, joints=JOINTS, channels=CHANNELS)
#robust_stats_report(
#    root="data/processed/motion",
#    channel="acc"
#)
visualize_normalization_stats(
    stats,
    joints=JOINTS,
    channels=CHANNELS,
    save_dir="statistics"
)