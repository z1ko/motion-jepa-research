from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def estimate_hz_from_time(time: np.ndarray) -> dict[str, float]:
    time = np.asarray(time, dtype=np.float64)

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if dt.size == 0:
        raise ValueError("No valid positive time differences found.")

    hz = 1.0 / np.median(dt)

    return {
        "estimated_hz": float(hz),
        "median_dt": float(np.median(dt)),
        "mean_dt": float(np.mean(dt)),
        "std_dt": float(np.std(dt)),
        "min_dt": float(np.min(dt)),
        "max_dt": float(np.max(dt)),
        "num_frames": int(time.size),
        "duration_s": float(time[-1] - time[0]),
        "dt_jitter_ratio": float(np.std(dt) / (np.median(dt) + 1e-12)),
    }


def read_time_column(path: Path, time_col: str = "time") -> np.ndarray:
    df = pl.read_csv(
        path,
        columns=[time_col],
    )

    if time_col not in df.columns:
        raise ValueError(f"Missing time column {time_col!r}")

    return df[time_col].to_numpy()


def analyze_csv_files(
    raw_glob: str,
    *,
    time_col: str = "time",
    max_files: int | None = None,
) -> pl.DataFrame:
    files = sorted(Path(p) for p in glob.glob(raw_glob, recursive=True))

    if max_files is not None:
        files = files[:max_files]

    rows = []

    for i, path in enumerate(files, start=1):
        try:
            time = read_time_column(path, time_col=time_col)
            row = estimate_hz_from_time(time)
            row["path"] = str(path)
            row["error"] = None
        except Exception as exc:
            row = {
                "path": str(path),
                "estimated_hz": None,
                "median_dt": None,
                "mean_dt": None,
                "std_dt": None,
                "min_dt": None,
                "max_dt": None,
                "num_frames": None,
                "duration_s": None,
                "dt_jitter_ratio": None,
                "error": repr(exc),
            }

        rows.append(row)

        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} files")

    return pl.DataFrame(rows)


def save_hz_histogram(df: pl.DataFrame, out_dir: Path) -> None:
    valid = df.filter(pl.col("estimated_hz").is_not_null())
    hz = valid["estimated_hz"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(hz, bins=80)
    ax.set_title("Estimated CSV sampling-rate distribution")
    ax.set_xlabel("estimated Hz")
    ax.set_ylabel("number of files")
    fig.tight_layout()
    fig.savefig(out_dir / "hz_distribution_histogram.png", dpi=200)
    plt.close(fig)


def save_hz_rounded_barplot(df: pl.DataFrame, out_dir: Path) -> None:
    valid = df.filter(pl.col("estimated_hz").is_not_null())

    counts = (
        valid
        .with_columns(pl.col("estimated_hz").round(0).cast(pl.Int64).alias("hz_rounded"))
        .group_by("hz_rounded")
        .len()
        .sort("hz_rounded")
    )

    x = counts["hz_rounded"].to_numpy()
    y = counts["len"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, y)
    ax.set_title("Estimated CSV sampling rates, rounded to nearest Hz")
    ax.set_xlabel("rounded estimated Hz")
    ax.set_ylabel("number of files")
    fig.tight_layout()
    fig.savefig(out_dir / "hz_distribution_rounded_barplot.png", dpi=200)
    plt.close(fig)

    counts.write_csv(out_dir / "hz_rounded_counts.csv")


def save_duration_vs_hz_scatter(df: pl.DataFrame, out_dir: Path) -> None:
    valid = df.filter(
        pl.col("estimated_hz").is_not_null()
        & pl.col("duration_s").is_not_null()
    )

    hz = valid["estimated_hz"].to_numpy()
    duration = valid["duration_s"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(hz, duration, s=8, alpha=0.5)
    ax.set_title("Duration vs estimated Hz")
    ax.set_xlabel("estimated Hz")
    ax.set_ylabel("duration [s]")
    fig.tight_layout()
    fig.savefig(out_dir / "duration_vs_hz.png", dpi=200)
    plt.close(fig)


def save_jitter_plot(df: pl.DataFrame, out_dir: Path) -> None:
    valid = df.filter(
        pl.col("estimated_hz").is_not_null()
        & pl.col("dt_jitter_ratio").is_not_null()
    )

    hz = valid["estimated_hz"].to_numpy()
    jitter = valid["dt_jitter_ratio"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(hz, jitter, s=8, alpha=0.5)
    ax.set_title("Timestamp jitter ratio vs estimated Hz")
    ax.set_xlabel("estimated Hz")
    ax.set_ylabel("std(dt) / median(dt)")
    fig.tight_layout()
    fig.savefig(out_dir / "jitter_vs_hz.png", dpi=200)
    plt.close(fig)


def print_summary(df: pl.DataFrame) -> None:
    valid = df.filter(pl.col("estimated_hz").is_not_null())
    errors = df.filter(pl.col("estimated_hz").is_null())

    print("\nSummary")
    print("-------")
    print(f"total files: {df.height}")
    print(f"valid files: {valid.height}")
    print(f"error files: {errors.height}")

    if valid.height > 0:
        print("\nEstimated Hz describe:")
        print(valid.select("estimated_hz").describe())

        print("\nRounded Hz counts:")
        counts = (
            valid
            .with_columns(pl.col("estimated_hz").round(0).cast(pl.Int64).alias("hz_rounded"))
            .group_by("hz_rounded")
            .len()
            .sort("hz_rounded")
        )
        print(counts)

        print("\nLargest timestamp jitter:")
        print(
            valid
            .sort("dt_jitter_ratio", descending=True)
            .select([
                "path",
                "estimated_hz",
                "median_dt",
                "std_dt",
                "dt_jitter_ratio",
                "num_frames",
                "duration_s",
            ])
            .head(20)
        )

    if errors.height > 0:
        print("\nFiles with errors:")
        print(errors.select(["path", "error"]).head(20))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-glob",
        type=str,
        required=False,
        default="/home/fziche/nas/MAEVE/HUMAN_MODEL/CARE-PD_torque/*.csv",
        help="Glob for CSV files, e.g. '../motion-jepa/data/raw/**/*.csv'",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="time",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("statistics/hz_analysis"),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Use -1 for all files.",
    )

    args = parser.parse_args()

    max_files = None if args.max_files < 0 else args.max_files
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = analyze_csv_files(
        args.raw_glob,
        time_col=args.time_col,
        max_files=max_files,
    )

    df.write_csv(args.out_dir / "csv_hz_report.csv")
    df.write_parquet(args.out_dir / "csv_hz_report.parquet")

    print_summary(df)

    save_hz_histogram(df, args.out_dir)
    save_hz_rounded_barplot(df, args.out_dir)
    save_duration_vs_hz_scatter(df, args.out_dir)
    save_jitter_plot(df, args.out_dir)

    print("\nSaved:")
    print(f"  {args.out_dir / 'csv_hz_report.csv'}")
    print(f"  {args.out_dir / 'csv_hz_report.parquet'}")
    print(f"  {args.out_dir / 'hz_distribution_histogram.png'}")
    print(f"  {args.out_dir / 'hz_distribution_rounded_barplot.png'}")
    print(f"  {args.out_dir / 'hz_rounded_counts.csv'}")
    print(f"  {args.out_dir / 'duration_vs_hz.png'}")
    print(f"  {args.out_dir / 'jitter_vs_hz.png'}")


if __name__ == "__main__":
    main()