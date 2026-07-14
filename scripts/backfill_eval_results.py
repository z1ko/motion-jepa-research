"""One-off ingestion of every existing eval CLI JSON output into the unified
runs/eval_results.parquet table (see motion_jepa.evaluation.results_table).

Globs runs/**/*.json (not runs/**/evaluations/*.json -- the older
runs/v1/version_{16,17}/evaluation_v*.json files live directly under the
version directory, predating the evaluations/ subdirectory convention).
Since the dedup key is read from each JSON's own run.checkpoint field (not
the filename), this naturally self-heals the known naming bugs accumulated
this session (a misnamed 0069.json actually holding epoch 99's results, and
byte-identical/near-identical _random-suffixed duplicates) with no special-
casing beyond sorting by mtime so "most recently written" (a deliberate,
meaningful tiebreak) wins duplicate keys, not glob/alphabetical order.
"""

from __future__ import annotations

import json
from pathlib import Path

from motion_jepa.evaluation.results_table import flatten_result, upsert_results_table

_DEFAULT_ROOT = "data/processed/motion"
_CARE_PD_ROOT = "data/processed/care-pd"


def _resolved_eval_window_params(result: dict, config: dict) -> dict:
    """Old JSONs predate result['eval_window_size']/['eval_stride']/
    ['eval_min_valid_frames']/['root']/['split'] (added alongside the
    results-table feature) -- fill them in with the same defaults the CLI
    itself used before --eval-* overrides existed: the checkpoint's own
    config.data.* and root/split inferred from the dataset name.
    """
    if "eval_window_size" in result:
        return result

    dataset = result.get("dataset", "")
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    result = dict(result)
    result.setdefault("root", _CARE_PD_ROOT if dataset.startswith("CARE-PD-") else _DEFAULT_ROOT)
    result.setdefault("split", "eval")
    result.setdefault("eval_window_size", data_cfg.get("window_size"))
    result.setdefault("eval_stride", data_cfg.get("stride"))
    result.setdefault("eval_min_valid_frames", data_cfg.get("min_valid_frames"))
    return result


def main() -> None:
    files = sorted(Path("runs").glob("**/*.json"), key=lambda p: p.stat().st_mtime)
    print(f"found {len(files)} eval JSON file(s)")

    rows = []
    for path in files:
        try:
            data = json.loads(path.read_text())
            run_info = data["run"]
            config = run_info.get("config", {})
            for result in data["results"]:
                result = _resolved_eval_window_params(result, config)
                rows.append(flatten_result(run_info, result, detail_json=str(path)))
        except Exception as e:
            print(f"skipping {path}: {e}")

    upsert_results_table(rows, Path("runs/eval_results.parquet"))
    print(f"wrote {len(rows)} row(s) to runs/eval_results.parquet")


if __name__ == "__main__":
    main()
