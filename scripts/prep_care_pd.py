
import argparse
from pathlib import Path

from motion_jepa.preprocess import assign_dataset_splits, create_raw_motion_dataset

# Only these sub-cohorts have both OpenSim-style CSVs AND an MDS-UPDRS-gait
# label (see papers/CarePD.pdf Table 1). DNE/E-LC/KUL-DT-T have CSVs now too
# but no gait-severity label in carepd_mds_updrs_gait_severity.csv (FoG-only
# cohorts) -- excluded for the same reason DNE always was. T-SDU-PD still has
# no CSVs at all.
_COHORTS = ("3DGait", "BMCLab", "PD-GaM")


def build_tree(*, samples_dir: Path, tree_dir: Path) -> None:
    """Symlink CARE-PD's flat `{cohort}_canonical__{subject}__{trial}.csv`
    files into the `<dataset>/<subject>/<trial>.csv` layout
    `motion_jepa.preprocess.load_sample_from_csv` expects (path.parts[-3:]).

    The symlink keeps the FULL original filename (not just the trial
    segment) -- `trial` ends up being the whole original stem, which is
    exactly the join key `carepd_mds_updrs_gait_severity.csv`'s `filename`
    column uses (see load_care_pd_labels/parse_care_pd_label).
    """
    tree_dir.mkdir(parents=True, exist_ok=True)

    counts = {cohort: 0 for cohort in _COHORTS}
    for csv_path in sorted(samples_dir.glob("*.csv")):
        parts = csv_path.name.split("__")
        if len(parts) != 3:
            continue
        cohort = parts[0].removesuffix("_canonical")
        if cohort not in _COHORTS:
            continue
        subject = parts[1]

        dataset_dir = tree_dir / f"CARE-PD-{cohort}" / subject
        dataset_dir.mkdir(parents=True, exist_ok=True)
        link = dataset_dir / csv_path.name
        if not link.exists():
            link.symlink_to(csv_path.resolve())
        counts[cohort] += 1

    for cohort, n in counts.items():
        print(f"[tree] CARE-PD-{cohort}: {n} files linked")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CARE-PD (3DGait+BMCLab) into an isolated eval-only store.")
    parser.add_argument("--samples-dir", type=Path, default=Path("data/raw/care_pd/samples"))
    parser.add_argument("--tree-dir", type=Path, default=Path("data/raw/care_pd_tree"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/care-pd"))
    parser.add_argument("--pretrain-root", type=Path, default=Path("data/processed/motion"))
    parser.add_argument("--datasets-config", type=Path, default=Path("config/datasets_care_pd.yaml"))
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument(
        "--window-size", type=int, default=400,
        help="Not used here (windows are enumerated at eval time, not prep time) -- "
             "informational only, pass the same value to evaluate's --eval-window-size.",
    )
    parser.add_argument(
        "--stride", type=int, default=50,
        help="Not used here -- pass to evaluate's --eval-stride (CARE-PD's historical "
             "dense-overlap value; the main pretrain corpus instead used stride==window_size).",
    )
    parser.add_argument(
        "--min-valid-frames", type=int, default=200,
        help="Not used here -- pass to evaluate's --eval-min-valid-frames.",
    )
    args = parser.parse_args()

    build_tree(samples_dir=args.samples_dir, tree_dir=args.tree_dir)

    create_raw_motion_dataset(
        raw_glob=str(args.tree_dir / "**" / "*.csv"),
        output_root=args.output_root,
        hz=args.hz,
        overwrite=True,
        # CARE-PD is ~30Hz native (measured), well below the AMASS-corpus
        # quality floor -- bypassed here only, resample_to_hz still upsamples
        # it to the same `hz` as everything else so window_size means the
        # same real-world duration across the whole eval suite.
        min_original_hz=0,
    )

    assign_dataset_splits(root=args.output_root, datasets_config=args.datasets_config)

    # Reuse the pretrain corpus's normalization -- never compute CARE-PD's
    # own. Relative symlink so it stays in sync if the pretrain stats are
    # ever regenerated.
    norm_link = args.output_root / "normalization_stats.npz"
    norm_target = Path("..") / args.pretrain_root.name / "normalization_stats.npz"
    if norm_link.exists() or norm_link.is_symlink():
        norm_link.unlink()
    norm_link.symlink_to(norm_target)
    print(f"[norm] {norm_link} -> {norm_target}")
    print(
        f"[eval] windows are enumerated at eval time now -- pass --eval-window-size "
        f"{args.window_size} --eval-stride {args.stride} --eval-min-valid-frames "
        f"{args.min_valid_frames} to `evaluate` for this store."
    )


if __name__ == "__main__":
    main()
