#!/usr/bin/env python3
"""Run all experiment run-configs for a single seed and write a per-seed CSV.

This script is intended to be called from the repository root. It imports the
experiment machinery and calls the internal _run_single to compute results for
every run configuration that matches the requested seed.

Example:
  python scripts/run_single_seed.py --config experiments/configs/gmm_em.yaml --seed 1 --out-dir experiments/results
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Make imports robust: if this script is launched from outside the repo root,
# ensure the repository root is on sys.path so `import src.*` works.
# The script lives in <repo>/scripts/, so repo root is two parents up from this file
# (parents[1]).
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.config as config_mod
import src.experiment as experiment_mod


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/gmm_em.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    args = parser.parse_args(argv)

    cfg = config_mod.load_config(args.config)

    # Collect results for all run-configs with this seed
    rows = []
    for run_cfg in config_mod.iter_run_configs(cfg):
        if int(run_cfg.seed) != int(args.seed):
            continue
        # use the internal runner to compute a dict row
        row = experiment_mod._run_single(cfg, run_cfg)
        rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tmp = out_dir / f"gmm_em_results_seed_{args.seed}.csv.tmp"
    out_final = out_dir / f"gmm_em_results_seed_{args.seed}.csv"

    if not rows:
        print(f"No run-configs found for seed={args.seed} (check config). Nothing written.")
        return 0

    df = pd.DataFrame(rows)
    # Write atomically: write to tmp then rename
    df.to_csv(out_tmp, index=False)
    os.replace(out_tmp, out_final)
    print(f"Wrote {len(df)} rows to {out_final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
