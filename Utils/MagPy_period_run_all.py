#!/usr/bin/env python3
"""Run MagPy-period validation end-to-end: prepare → train → score → CCMC JSON.

Training is long (50 Optuna-tuned splits). Use ``--skip-train`` if models exist,
or ``--n-splits 2`` for a smoke test.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(_SCRIPT_DIR / script), *extra]
    print("\n" + "=" * 72)
    print("[magpy] RUN:", " ".join(cmd))
    print("=" * 72, flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--n-splits", type=int, default=50)
    ap.add_argument("--optuna-n-trials", type=int, default=20)
    ap.add_argument("--max-models", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stride-hours", type=float, default=1.0)
    ap.add_argument("--all-clear-threshold", type=float, default=0.1)
    args = ap.parse_args()

    if not args.skip_prepare:
        _run("MagPy_period_prepare.py", [])
    if not args.skip_train:
        _run(
            "MagPy_period_train.py",
            ["--n-splits", str(args.n_splits), "--optuna-n-trials", str(args.optuna_n_trials)],
        )
    if not args.skip_score:
        _run(
            "MagPy_period_score.py",
            ["--max-models", str(args.max_models), "--device", str(args.device)],
        )
    if not args.skip_export:
        _run(
            "MagPy_period_export_ccmc_json.py",
            [
                "--stride-hours", str(args.stride_hours),
                "--all-clear-threshold", str(args.all_clear_threshold),
            ],
        )
    print("\n[magpy] all requested stages finished.")


if __name__ == "__main__":
    main()
