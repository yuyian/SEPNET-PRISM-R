#!/usr/bin/env python3
"""Continuous MagPy-period scoring (2023-07-01 → 2024-12-31).

Scores the cutoff-retrained ensemble on the padded 1h feature slice, then
clips outputs to the strict MagPy window and writes ensemble + uncertainty.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P  # noqa: E402
from Step2_score_train_mtm_lstm import main as score_main  # noqa: E402


TEST_START = "2023-07-01 00:00:00+00:00"
TEST_END = "2024-12-31 23:59:59+00:00"


def _parse_ts(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


def _clip_and_uncertainty(root: Path, test_start: pd.Timestamp, test_end: pd.Timestamp) -> dict:
    pred_dir = root / "predictions"
    concat = pred_dir / "mtm_lstm_fulltrain_train1h_predictions.csv"
    ens = pred_dir / "mtm_lstm_fulltrain_train1h_ensemble.csv"
    if not ens.is_file():
        raise FileNotFoundError(ens)

    print("[score] clipping ensemble to MagPy window + computing cross-model uncertainty ...")
    e = pd.read_csv(ens, low_memory=False)
    e["window_begin"] = _parse_ts(e["window_begin"])
    e["window_end"] = _parse_ts(e["window_end"])
    m = (e["window_begin"] >= test_start) & (e["window_begin"] <= test_end)
    e = e.loc[m].sort_values("window_begin").reset_index(drop=True)

    # Uncertainty = std of cls_prob across split models (when concat available).
    if concat.is_file():
        # Stream only needed columns for MagPy window
        usecols = ["window_begin", "window_end", "mtm_lstm_cls_prob", "split_seed", "model_file"]
        parts = []
        for ch in pd.read_csv(concat, usecols=lambda c: c in usecols, chunksize=200_000, low_memory=False):
            ch["window_begin"] = _parse_ts(ch["window_begin"])
            ch["window_end"] = _parse_ts(ch["window_end"])
            mm = (ch["window_begin"] >= test_start) & (ch["window_begin"] <= test_end)
            if mm.any():
                parts.append(ch.loc[mm])
        if parts:
            raw = pd.concat(parts, ignore_index=True)
            raw["mtm_lstm_cls_prob"] = pd.to_numeric(raw["mtm_lstm_cls_prob"], errors="coerce")
            g = (
                raw.groupby(["window_begin", "window_end"], as_index=False)["mtm_lstm_cls_prob"]
                .agg(cls_prob_std="std", cls_prob_mean="mean", n_models="count")
            )
            e = e.merge(g, on=["window_begin", "window_end"], how="left")
        else:
            e["cls_prob_std"] = np.nan
            e["cls_prob_mean"] = e.get("mtm_lstm_cls_prob")
            e["n_models"] = np.nan
    else:
        e["cls_prob_std"] = np.nan
        e["cls_prob_mean"] = e.get("mtm_lstm_cls_prob")
        e["n_models"] = np.nan

    out = pred_dir / "magpy_period_ensemble_continuous.csv"
    e.to_csv(out, index=False)
    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_rows": int(len(e)),
        "window_begin_min": str(e["window_begin"].min()) if len(e) else None,
        "window_begin_max": str(e["window_begin"].max()) if len(e) else None,
        "output": str(out),
        "cls_prob_mean": float(pd.to_numeric(e["mtm_lstm_cls_prob"], errors="coerce").mean())
        if len(e) else None,
        "n_true_sep": int(pd.to_numeric(e.get("mtm_lstm_cls_true"), errors="coerce").fillna(0).eq(1).sum())
        if "mtm_lstm_cls_true" in e.columns else None,
    }
    (pred_dir / "magpy_period_score_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[score] wrote {out} rows={len(e):,}")
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(P("magpy_validation_dir")))
    ap.add_argument("--max-models", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--test-end", default=TEST_END)
    ap.add_argument("--skip-score", action="store_true",
                    help="Only clip/uncertainty from existing prediction CSVs")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    models = root / "models"
    feat = root / "features" / "test_202307_202412_1h_padded.csv"
    out_dir = root / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_score:
        if not feat.is_file():
            raise FileNotFoundError(f"Missing {feat}; run MagPy_period_prepare.py")
        model_files = sorted(models.glob("mtm_lstm_fulltrain_split_*.pt"))
        if not model_files:
            raise FileNotFoundError(
                f"No models in {models}. Run MagPy_period_train.py first."
            )
        print(f"[score] {len(model_files)} models | input={feat}")
        score_argv = [
            "--input-csv", str(feat),
            "--models-dir", str(models),
            "--out-dir", str(out_dir),
            "--max-models", str(int(args.max_models)),
            "--device", str(args.device),
        ]
        score_main(score_argv)

    _clip_and_uncertainty(
        root,
        pd.Timestamp(args.test_start, tz="UTC"),
        pd.Timestamp(args.test_end, tz="UTC"),
    )


if __name__ == "__main__":
    main()
