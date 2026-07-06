from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from motion_jepa.dataset import MotionZarrStore, _path_of_samples_index
from motion_jepa.preprocess import CHANNELS, JOINTS

BODY_LEVEL_JOINTS = [
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "lumbar_bending",
    "lumbar_extension",
    "lumbar_twist",
    "thorax_bending",
    "thorax_extension",
    "thorax_twist",
]


def collect_tau_values(
    root: Path | str,
    *,
    joints: list[str],
    split: str = "train",
    max_values_per_joint: int | None = 200_000,
    seed: int = 13,
) -> dict[str, np.ndarray]:
    root = Path(root)

    samples = pl.read_parquet(_path_of_samples_index(root))
    samples = samples.filter(pl.col("split") == split)

    store = MotionZarrStore(root)
    tau_idx = CHANNELS.index("tau")

    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {joint: [] for joint in joints}

    for row in samples.iter_rows(named=True):
        suid = str(row["suid"])
        x = np.asarray(store.get_kinematics(suid)[:, :, tau_idx], dtype=np.float32)

        for joint in joints:
            j = JOINTS.index(joint)
            out[joint].append(x[:, j])

    result: dict[str, np.ndarray] = {}

    for joint, chunks in out.items():
        values = np.concatenate(chunks)

        if max_values_per_joint is not None and values.size > max_values_per_joint:
            idx = rng.choice(values.size, size=max_values_per_joint, replace=False)
            values = values[idx]

        result[joint] = values

    return result

def plot_tau_histograms(
    tau_values: dict[str, np.ndarray],
    *,
    bins: int = 200,
    clip_percentile: float | None = 99.5,
    save_dir: Path | str | None = None,
) -> None:
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for joint, values in tau_values.items():
        values = np.asarray(values)
        values = values[np.isfinite(values)]

        if clip_percentile is not None:
            lo, hi = np.percentile(
                values,
                [100.0 - clip_percentile, clip_percentile],
            )
            values_to_plot = values[(values >= lo) & (values <= hi)]
            title_suffix = f"clipped to p{100 - clip_percentile:.1f}–p{clip_percentile:.1f}"
        else:
            values_to_plot = values
            title_suffix = "full range"

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values_to_plot, bins=bins)
        ax.set_title(f"{joint} tau distribution ({title_suffix})")
        ax.set_xlabel("tau")
        ax.set_ylabel("count")
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(save_dir / f"tau_hist_{joint}.png", dpi=200)

        plt.show()

def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))

def plot_tau_raw_vs_log(
    tau_values: dict[str, np.ndarray],
    *,
    bins: int = 200,
    raw_clip_percentile: float = 99.5,
    save_dir: Path | str | None = None,
) -> None:
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for joint, values in tau_values.items():
        values = np.asarray(values)
        values = values[np.isfinite(values)]

        lo, hi = np.percentile(
            values,
            [100.0 - raw_clip_percentile, raw_clip_percentile],
        )
        raw_clipped = values[(values >= lo) & (values <= hi)]

        log_values = signed_log1p(values)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(raw_clipped, bins=bins)
        ax.set_title(f"{joint} raw tau, clipped p{100 - raw_clip_percentile:.1f}–p{raw_clip_percentile:.1f}")
        ax.set_xlabel("raw tau")
        ax.set_ylabel("count")
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(save_dir / f"tau_raw_{joint}.png", dpi=200)

        plt.show()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(log_values, bins=bins)
        ax.set_title(f"{joint} signed log1p tau")
        ax.set_xlabel("sign(tau) * log1p(abs(tau))")
        ax.set_ylabel("count")
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(save_dir / f"tau_log1p_{joint}.png", dpi=200)

        plt.show()

def plot_tau_abs_tail(
    tau_values: dict[str, np.ndarray],
    *,
    save_dir: Path | str | None = None,
) -> None:
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for joint, values in tau_values.items():
        values = np.asarray(values)
        values = values[np.isfinite(values)]

        abs_values = np.sort(np.abs(values))
        survival = 1.0 - np.arange(1, len(abs_values) + 1) / len(abs_values)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(abs_values, survival)
        ax.set_yscale("log")
        ax.set_title(f"{joint} |tau| tail")
        ax.set_xlabel("|tau|")
        ax.set_ylabel("P(|tau| > x)")
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(save_dir / f"tau_abs_tail_{joint}.png", dpi=200)

        plt.show()

tau_values = collect_tau_values(
    "data/processed/motion",
    joints=BODY_LEVEL_JOINTS,
    split="train",
)

plot_tau_histograms(
    tau_values,
    clip_percentile=99.5,
    save_dir="data/processed/motion/tau_plots/histogram",
)

plot_tau_raw_vs_log(
    tau_values,
    save_dir="data/processed/motion/tau_plots/raw_vs_log",
)

plot_tau_abs_tail(
    tau_values,
    save_dir="data/processed/motion/tau_plots/abs_tail",
)