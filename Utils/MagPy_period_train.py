#!/usr/bin/env python3
"""Retrain SEPNET-PRISM-R MTM-LSTM ensemble with MagPy train cutoff.

Wraps ``SEP-Prediction-Operational/.../train_save_mtm_lstm_fulltrain_sharp_flare.py``
using the pre-2023-07 filtered 24h feature table so MagPy-period rows are **not**
in the training set.

Requires prepare step first::

  python Utils/MagPy_period_prepare.py
  python Utils/MagPy_period_train.py --n-splits 50
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P, S  # noqa: E402

_DEFAULT_TRAIN_SCRIPT = (
    Path("~/pphd-files/SEP-Prediction-Operational/Real_time_Pycode")
    .expanduser()
    / "train_save_mtm_lstm_fulltrain_sharp_flare.py"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(P("magpy_validation_dir")))
    ap.add_argument(
        "--train-script",
        default=str(_DEFAULT_TRAIN_SCRIPT),
        help="Operational fulltrain sharp+flare trainer",
    )
    ap.add_argument(
        "--base-dir",
        default=str(Path("~/pphd-files/SEP-Prediction-Operational").expanduser()),
    )
    ap.add_argument("--n-splits", type=int, default=50)
    ap.add_argument("--optuna-n-trials", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--val-mode", default="random", choices=("random", "calendar"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    train_csv = root / "features" / "train_pre202307_24h.csv"
    out_dir = root / "models"
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Missing {train_csv}. Run: python Utils/MagPy_period_prepare.py"
        )
    train_script = Path(args.train_script).expanduser().resolve()
    if not train_script.is_file():
        raise FileNotFoundError(f"Train script not found: {train_script}")

    # fulltrain resolves paths as (base / train_csv); absolute train_csv/out_dir are fine.
    cmd = [
        str(args.python),
        str(train_script),
        "--base-dir", str(Path(args.base_dir).expanduser().resolve()),
        "--train-csv", str(train_csv),
        "--out-dir", str(out_dir),
        "--n-splits", str(int(args.n_splits)),
        "--optuna-n-trials", str(int(args.optuna_n_trials)),
        "--val-frac", str(float(args.val_frac)),
        "--val-mode", str(args.val_mode),
        "--calibrate-cls-temperature",
    ]
    log_path = log_dir / "train_fulltrain_pre202307.log"
    print("[train] CMD:", " ".join(cmd))
    print("[train] log:", log_path)
    if args.dry_run:
        return

    env = os.environ.copy()
    # Ensure Operational Pycode / this Utils are importable if needed.
    pycode = S("pycode_dir", "").strip()
    extras = [str(_SCRIPT_DIR)]
    if pycode:
        extras.insert(0, str(Path(pycode).expanduser().resolve()))
    extras.append(str(train_script.parent))
    env["PYTHONPATH"] = os.pathsep.join(extras + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    with log_path.open("w") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise SystemExit(
            f"Training failed (exit {proc.returncode}). See {log_path}"
        )
    print(f"[train] done → {out_dir}")


if __name__ == "__main__":
    main()
