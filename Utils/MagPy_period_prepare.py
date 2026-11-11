#!/usr/bin/env python3
"""Slice feature tables for MagPy-period validation.

Train:  window_end < 2023-07-01  (through 2023-06-30)
Test:   2023-07-01 ≤ window_begin ≤ 2024-12-31
        (+ lag pad before test start for MTM sequence history)

Writes under ``Data/MagPy_Period_Validation/features/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P  # noqa: E402

TRAIN_END_EXCLUSIVE = "2023-07-01 00:00:00+00:00"  # train through 2023-06-30
TEST_START = "2023-07-01 00:00:00+00:00"
TEST_END = "2024-12-31 23:59:59+00:00"
DEFAULT_LAG_PAD_HOURS = 72.0


def _parse_ts(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


def _slice_csv(
    src: Path,
    *,
    out: Path,
    mode: str,
    train_end_excl: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    lag_pad_hours: float,
) -> dict:
    print(f"[prepare] reading {src}")
    df = pd.read_csv(src, low_memory=False)
    for c in ("window_begin", "window_end"):
        if c not in df.columns:
            raise KeyError(f"{src} missing {c}")
        df[c] = _parse_ts(df[c])
    df = df[df["window_begin"].notna() & df["window_end"].notna()].copy()

    if mode == "train":
        # Train on windows verified before MagPy test start.
        m = df["window_end"] < train_end_excl
        label = "train_end_excl"
    elif mode == "test":
        pad0 = test_start - pd.Timedelta(hours=float(lag_pad_hours))
        # Keep lag history + MagPy test window (filter SHARP later at score time).
        m = (df["window_begin"] >= pad0) & (df["window_begin"] <= test_end)
        label = "test_with_lag_pad"
    else:
        raise ValueError(mode)

    sub = df.loc[m].sort_values("window_begin").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    meta = {
        "mode": mode,
        "label": label,
        "source": str(src),
        "output": str(out),
        "n_rows": int(len(sub)),
        "n_sharp": int(pd.to_numeric(sub.get("SHARP_label"), errors="coerce").fillna(0).eq(1).sum())
        if "SHARP_label" in sub.columns else None,
        "window_begin_min": str(sub["window_begin"].min()) if len(sub) else None,
        "window_begin_max": str(sub["window_begin"].max()) if len(sub) else None,
        "window_end_min": str(sub["window_end"].min()) if len(sub) else None,
        "window_end_max": str(sub["window_end"].max()) if len(sub) else None,
    }
    print(
        f"[prepare] wrote {out} rows={meta['n_rows']:,} "
        f"SHARP={meta['n_sharp']}  {meta['window_begin_min']} → {meta['window_begin_max']}"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(P("magpy_validation_dir")))
    ap.add_argument(
        "--train-csv-24h",
        default=str(P("feature_dataset_csv").parent / "rolling_combinded_seq_24hours.csv"),
        help="24h rolling table used by fulltrain",
    )
    ap.add_argument(
        "--test-csv-1h",
        default=str(P("feature_dataset_csv")),
        help="1h rolling table used for continuous scoring",
    )
    ap.add_argument("--train-end-exclusive", default=TRAIN_END_EXCLUSIVE)
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--test-end", default=TEST_END)
    ap.add_argument("--lag-pad-hours", type=float, default=DEFAULT_LAG_PAD_HOURS)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    feat = root / "features"
    feat.mkdir(parents=True, exist_ok=True)

    t_end = pd.Timestamp(args.train_end_exclusive, tz="UTC")
    t0 = pd.Timestamp(args.test_start, tz="UTC")
    t1 = pd.Timestamp(args.test_end, tz="UTC")

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "train_end_exclusive": str(t_end),
        "test_start": str(t0),
        "test_end": str(t1),
        "lag_pad_hours": float(args.lag_pad_hours),
        "note": "Train cutoff is window_end < 2023-07-01 (through 2023-06-30).",
    }
    meta["train_24h"] = _slice_csv(
        Path(args.train_csv_24h),
        out=feat / "train_pre202307_24h.csv",
        mode="train",
        train_end_excl=t_end,
        test_start=t0,
        test_end=t1,
        lag_pad_hours=args.lag_pad_hours,
    )
    meta["test_1h"] = _slice_csv(
        Path(args.test_csv_1h),
        out=feat / "test_202307_202412_1h_padded.csv",
        mode="test",
        train_end_excl=t_end,
        test_start=t0,
        test_end=t1,
        lag_pad_hours=args.lag_pad_hours,
    )
    # Strict MagPy window index (no pad) for reporting
    test_strict = pd.read_csv(feat / "test_202307_202412_1h_padded.csv", low_memory=False)
    test_strict["window_begin"] = _parse_ts(test_strict["window_begin"])
    m = (test_strict["window_begin"] >= t0) & (test_strict["window_begin"] <= t1)
    strict = test_strict.loc[m].reset_index(drop=True)
    strict_path = feat / "test_202307_202412_1h.csv"
    strict.to_csv(strict_path, index=False)
    meta["test_1h_strict"] = {
        "output": str(strict_path),
        "n_rows": int(len(strict)),
        "n_sharp": int(pd.to_numeric(strict.get("SHARP_label"), errors="coerce").fillna(0).eq(1).sum())
        if "SHARP_label" in strict.columns else None,
    }
    print(f"[prepare] wrote strict test window {strict_path} rows={len(strict):,}")

    meta_path = feat / "prepare_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[prepare] wrote {meta_path}")


if __name__ == "__main__":
    main()
