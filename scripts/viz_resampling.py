from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


# Adjust these imports to your project layout.
from motion_jepa.preprocess import (
    load_sample_from_csv,
    resample_to_hz,
    JOINTS,
    CHANNELS,
)


ANGLE_JOINT_INDICES = [
    JOINTS.index(joint)
    for joint in JOINTS
    if joint not in {"pelvis_tx", "pelvis_ty", "pelvis_tz"}
]


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def signed_log1p_tau(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    tau_idx = CHANNELS.index("tau")
    x[:, :, tau_idx] = np.sign(x[:, :, tau_idx]) * np.log1p(
        np.abs(x[:, :, tau_idx])
    )
    return x


def training_space_kinematics(x: np.ndarray) -> np.ndarray:
    """
    Transform kinematics into the space actually consumed by the model
    before z-score normalization.

    Currently:
      - signed_log1p on tau
      - pos/vel/acc unchanged
    """
    return signed_log1p_tau(x)


def interpolate_back_to_original_time(
    *,
    original_time: np.ndarray,
    resampled_time: np.ndarray,
    resampled_kinematics: np.ndarray,
) -> np.ndarray:
    """
    Interpolate resampled kinematics back to the original timestamps.

    original_time: [T0]
    resampled_time: [T1]
    resampled_kinematics: [T1, D, C]

    Returns:
        reconstructed: [T0, D, C]
    """
    original_time = np.asarray(original_time, dtype=np.float64)
    resampled_time = np.asarray(resampled_time, dtype=np.float64)
    x = np.asarray(resampled_kinematics, dtype=np.float32)

    if x.ndim != 3:
        raise ValueError(f"Expected resampled_kinematics shape [T, D, C], got {x.shape}")

    t0 = len(original_time)
    d = x.shape[1]
    c = x.shape[2]

    reconstructed = np.empty((t0, d, c), dtype=np.float32)

    for j in range(d):
        for k in range(c):
            reconstructed[:, j, k] = np.interp(
                original_time,
                resampled_time,
                x[:, j, k],
            )

    return reconstructed


def kinematics_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """
    Error tensor [T, D, C].

    Angular position dimensions use wrapped angular error.
    Everything else uses ordinary subtraction.
    """
    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)

    err = reconstructed - original

    pos_idx = CHANNELS.index("pos")
    err[:, ANGLE_JOINT_INDICES, pos_idx] = wrap_to_pi(
        reconstructed[:, ANGLE_JOINT_INDICES, pos_idx]
        - original[:, ANGLE_JOINT_INDICES, pos_idx]
    )

    return err


def summarize_error_by_channel(err: np.ndarray, prefix: str) -> dict[str, float]:
    """
    Summarize error over time and joints, separately for each channel.
    """
    mae = np.mean(np.abs(err), axis=(0, 1))
    rmse = np.sqrt(np.mean(np.square(err), axis=(0, 1)))
    max_abs = np.max(np.abs(err), axis=(0, 1))
    p99_abs = np.percentile(np.abs(err), 99.0, axis=(0, 1))
    p999_abs = np.percentile(np.abs(err), 99.9, axis=(0, 1))

    row: dict[str, float] = {}

    for i, channel in enumerate(CHANNELS):
        row[f"{prefix}_{channel}_mae"] = float(mae[i])
        row[f"{prefix}_{channel}_rmse"] = float(rmse[i])
        row[f"{prefix}_{channel}_p99_abs"] = float(p99_abs[i])
        row[f"{prefix}_{channel}_p999_abs"] = float(p999_abs[i])
        row[f"{prefix}_{channel}_max_abs"] = float(max_abs[i])

    return row


def worst_error_feature(err: np.ndarray) -> dict[str, Any]:
    """
    Return the time/joint/channel with largest absolute error.
    """
    abs_err = np.abs(err)
    t, j, c = np.unravel_index(np.argmax(abs_err), abs_err.shape)

    return {
        "worst_frame": int(t),
        "worst_joint": JOINTS[j],
        "worst_channel": CHANNELS[c],
        "worst_error": float(err[t, j, c]),
        "worst_abs_error": float(abs_err[t, j, c]),
    }


def analyze_one_file(
    path: Path,
    *,
    hz: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Analyze one file.

    Returns:
        row: summary metrics
        payload: arrays and samples useful for plotting
    """
    original_sample = load_sample_from_csv(path)
    resampled_sample = resample_to_hz(original_sample, hz=hz)

    original_time = original_sample.extra["time"].to_numpy()
    resampled_time = resampled_sample.extra["time"].to_numpy()

    reconstructed = interpolate_back_to_original_time(
        original_time=original_time,
        resampled_time=resampled_time,
        resampled_kinematics=resampled_sample.kinematics,
    )

    raw_err = kinematics_error(
        original_sample.kinematics,
        reconstructed,
    )

    original_train = training_space_kinematics(original_sample.kinematics)
    reconstructed_train = training_space_kinematics(reconstructed)

    train_err = kinematics_error(
        original_train,
        reconstructed_train,
    )

    row: dict[str, Any] = {
        "path": str(path),
        "original_frames": int(original_sample.kinematics.shape[0]),
        "resampled_frames": int(resampled_sample.kinematics.shape[0]),
        "duration_s": float(original_time[-1] - original_time[0]),
        "hz": float(hz),
    }

    row.update(summarize_error_by_channel(raw_err, prefix="raw"))
    row.update(summarize_error_by_channel(train_err, prefix="train"))

    raw_worst = worst_error_feature(raw_err)
    train_worst = worst_error_feature(train_err)

    for key, value in raw_worst.items():
        row[f"raw_{key}"] = value

    for key, value in train_worst.items():
        row[f"train_{key}"] = value

    payload = {
        "path": path,
        "original_sample": original_sample,
        "resampled_sample": resampled_sample,
        "reconstructed": reconstructed,
        "raw_err": raw_err,
        "train_err": train_err,
        "original_time": original_time,
        "resampled_time": resampled_time,
    }

    return row, payload


def plot_original_resampled_reconstructed(
    payload: dict[str, Any],
    *,
    joint: str,
    channel: str,
    output_path: Path,
    training_space: bool = False,
    context_frames: int | None = None,
) -> None:
    original_sample = payload["original_sample"]
    resampled_sample = payload["resampled_sample"]
    reconstructed = payload["reconstructed"]

    original_time = payload["original_time"]
    resampled_time = payload["resampled_time"]

    j = JOINTS.index(joint)
    c = CHANNELS.index(channel)

    original_x = original_sample.kinematics
    resampled_x = resampled_sample.kinematics
    reconstructed_x = reconstructed

    if training_space:
        original_x = training_space_kinematics(original_x)
        resampled_x = training_space_kinematics(resampled_x)
        reconstructed_x = training_space_kinematics(reconstructed_x)

    y_original = original_x[:, j, c]
    y_resampled = resampled_x[:, j, c]
    y_reconstructed = reconstructed_x[:, j, c]

    if context_frames is not None:
        err = np.abs(y_reconstructed - y_original)
        center = int(np.argmax(err))

        lo = max(0, center - context_frames)
        hi = min(len(original_time), center + context_frames + 1)

        t_min = original_time[lo]
        t_max = original_time[hi - 1]

        original_mask = (original_time >= t_min) & (original_time <= t_max)
        resampled_mask = (resampled_time >= t_min) & (resampled_time <= t_max)
    else:
        original_mask = np.ones_like(original_time, dtype=bool)
        resampled_mask = np.ones_like(resampled_time, dtype=bool)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        original_time[original_mask],
        y_original[original_mask],
        label="original",
        linewidth=1.2,
    )
    ax.plot(
        resampled_time[resampled_mask],
        y_resampled[resampled_mask],
        label="resampled",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.plot(
        original_time[original_mask],
        y_reconstructed[original_mask],
        label="resampled back to original time",
        linewidth=1.0,
        alpha=0.8,
    )

    space = "training-space" if training_space else "raw-space"
    ax.set_title(f"{space}: {joint} {channel}\n{payload['path']}")
    ax.set_xlabel("time")
    ax.set_ylabel(f"{joint} {channel}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_error_trace(
    payload: dict[str, Any],
    *,
    joint: str,
    channel: str,
    output_path: Path,
    training_space: bool = False,
    context_frames: int | None = None,
) -> None:
    original_sample = payload["original_sample"]
    reconstructed = payload["reconstructed"]
    original_time = payload["original_time"]

    j = JOINTS.index(joint)
    c = CHANNELS.index(channel)

    original_x = original_sample.kinematics
    reconstructed_x = reconstructed

    if training_space:
        original_x = training_space_kinematics(original_x)
        reconstructed_x = training_space_kinematics(reconstructed_x)

    err = kinematics_error(original_x, reconstructed_x)[:, j, c]
    abs_err = np.abs(err)

    if context_frames is not None:
        center = int(np.argmax(abs_err))
        lo = max(0, center - context_frames)
        hi = min(len(original_time), center + context_frames + 1)
    else:
        lo = 0
        hi = len(original_time)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(original_time[lo:hi], err[lo:hi], linewidth=1.0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    space = "training-space" if training_space else "raw-space"
    ax.set_title(f"{space} error: {joint} {channel}\n{payload['path']}")
    ax.set_xlabel("time")
    ax.set_ylabel("reconstructed - original")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compare_rates_for_file(
    path: Path,
    *,
    rates: list[float],
) -> pl.DataFrame:
    rows = []

    for hz in rates:
        row, _ = analyze_one_file(path, hz=hz)
        rows.append(row)

    return pl.DataFrame(rows)


def analyze_many_files(
    files: list[Path],
    *,
    hz: float,
    max_files: int | None,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []

    if max_files is not None:
        files = files[:max_files]

    total = len(files)

    for i, path in enumerate(files, start=1):
        try:
            row, _ = analyze_one_file(path, hz=hz)
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "error": repr(exc),
                }
            )

        if i % 25 == 0 or i == total:
            print(f"[analyze] {i}/{total}")

    return pl.DataFrame(rows)


def save_summary_tables(
    df: pl.DataFrame,
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df.write_parquet(output_dir / "resample_roundtrip_report.parquet")
    df.write_csv(output_dir / "resample_roundtrip_report.csv")

    print("\nSummary:")
    print(df.describe())

    interesting_cols = [
        "path",
        "raw_tau_rmse",
        "raw_tau_max_abs",
        "train_tau_rmse",
        "train_tau_max_abs",
        "raw_acc_rmse",
        "raw_acc_max_abs",
        "train_acc_rmse",
        "train_acc_max_abs",
        "raw_worst_joint",
        "raw_worst_channel",
        "raw_worst_abs_error",
        "train_worst_joint",
        "train_worst_channel",
        "train_worst_abs_error",
    ]

    existing = [c for c in interesting_cols if c in df.columns]

    for metric in [
        "raw_tau_rmse",
        "raw_tau_max_abs",
        "train_tau_rmse",
        "train_tau_max_abs",
        "raw_acc_rmse",
        "train_acc_rmse",
        "train_worst_abs_error",
    ]:
        if metric not in df.columns:
            continue

        top = df.sort(metric, descending=True).select(existing).head(50)
        top.write_csv(output_dir / f"top_by_{metric}.csv")

        print(f"\nTop by {metric}:")
        print(top.head(10))


def make_worst_case_plots(
    df: pl.DataFrame,
    *,
    output_dir: Path,
    hz: float,
    metric: str,
    top_k: int,
    context_frames: int,
) -> None:
    if metric not in df.columns:
        print(f"Metric {metric!r} not found; skipping plots.")
        return

    plot_dir = output_dir / f"plots_top_by_{metric}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    worst = df.filter(pl.col("error").is_null()) if "error" in df.columns else df
    worst = worst.sort(metric, descending=True).head(top_k)

    for rank, row in enumerate(worst.iter_rows(named=True), start=1):
        path = Path(str(row["path"]))

        print(f"[plot] rank={rank}, metric={metric}, path={path}")

        _, payload = analyze_one_file(path, hz=hz)

        raw_joint = str(row.get("raw_worst_joint", "pelvis_rotation"))
        raw_channel = str(row.get("raw_worst_channel", "tau"))

        train_joint = str(row.get("train_worst_joint", raw_joint))
        train_channel = str(row.get("train_worst_channel", raw_channel))

        prefix = f"rank_{rank:02d}"

        plot_original_resampled_reconstructed(
            payload,
            joint=raw_joint,
            channel=raw_channel,
            output_path=plot_dir / f"{prefix}_raw_signal_{raw_joint}_{raw_channel}.png",
            training_space=False,
            context_frames=context_frames,
        )

        plot_error_trace(
            payload,
            joint=raw_joint,
            channel=raw_channel,
            output_path=plot_dir / f"{prefix}_raw_error_{raw_joint}_{raw_channel}.png",
            training_space=False,
            context_frames=context_frames,
        )

        plot_original_resampled_reconstructed(
            payload,
            joint=train_joint,
            channel=train_channel,
            output_path=plot_dir / f"{prefix}_training_signal_{train_joint}_{train_channel}.png",
            training_space=True,
            context_frames=context_frames,
        )

        plot_error_trace(
            payload,
            joint=train_joint,
            channel=train_channel,
            output_path=plot_dir / f"{prefix}_training_error_{train_joint}_{train_channel}.png",
            training_space=True,
            context_frames=context_frames,
        )


def rate_comparison_for_worst_files(
    df: pl.DataFrame,
    *,
    output_dir: Path,
    metric: str,
    rates: list[float],
    top_k: int,
) -> None:
    if metric not in df.columns:
        print(f"Metric {metric!r} not found; skipping rate comparison.")
        return

    rows = (
        df
        .sort(metric, descending=True)
        .head(top_k)
        .iter_rows(named=True)
    )

    out: list[pl.DataFrame] = []

    for row in rows:
        path = Path(str(row["path"]))
        print(f"[rates] {path}")

        rate_df = compare_rates_for_file(path, rates=rates)
        rate_df = rate_df.with_columns([
            pl.lit(str(path)).alias("source_path"),
            pl.lit(metric).alias("selected_by_metric"),
        ])
        out.append(rate_df)

    if out:
        result = pl.concat(out, how="vertical")
        result.write_csv(output_dir / f"rate_comparison_top_by_{metric}.csv")
        result.write_parquet(output_dir / f"rate_comparison_top_by_{metric}.parquet")

        print("\nRate comparison:")
        print(result.select([
            "source_path",
            "hz",
            "raw_tau_rmse",
            "train_tau_rmse",
            "raw_acc_rmse",
            "train_acc_rmse",
            "raw_tau_max_abs",
            "train_tau_max_abs",
        ]))


def parse_rates(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-glob",
        type=str,
        required=True,
        help="Glob for raw CSV files, e.g. '../motion-jepa/data/raw/**/*.csv'",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=60.0,
        help="Target resampling rate.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=200,
        help="Maximum number of files to analyze. Use -1 for all files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resample_analysis"),
    )
    parser.add_argument(
        "--plot-metric",
        type=str,
        default="train_tau_rmse",
        help=(
            "Metric used to select worst files for plots. "
            "Examples: raw_tau_rmse, train_tau_rmse, raw_tau_max_abs, "
            "train_tau_max_abs, train_worst_abs_error."
        ),
    )
    parser.add_argument(
        "--top-k-plots",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=300,
        help="Frames around the worst error to show in plots.",
    )
    parser.add_argument(
        "--compare-rates",
        type=str,
        default="60,120,240",
        help="Comma-separated Hz values for worst-file comparison.",
    )
    parser.add_argument(
        "--top-k-rates",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    import glob

    files = sorted(Path(p) for p in glob.glob(args.raw_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {args.raw_glob}")

    max_files = None if args.max_files < 0 else args.max_files

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found files: {len(files)}")
    print(f"Analyzing: {len(files) if max_files is None else min(max_files, len(files))}")
    print(f"Target hz: {args.hz}")
    print(f"Output dir: {args.output_dir}")

    df = analyze_many_files(
        files,
        hz=args.hz,
        max_files=max_files,
    )

    save_summary_tables(
        df,
        output_dir=args.output_dir,
    )

    make_worst_case_plots(
        df,
        output_dir=args.output_dir,
        hz=args.hz,
        metric=args.plot_metric,
        top_k=args.top_k_plots,
        context_frames=args.context_frames,
    )

    rates = parse_rates(args.compare_rates)

    rate_comparison_for_worst_files(
        df,
        output_dir=args.output_dir,
        metric=args.plot_metric,
        rates=rates,
        top_k=args.top_k_rates,
    )

    print("\nDone.")
    print(f"Report directory: {args.output_dir}")


if __name__ == "__main__":
    main()