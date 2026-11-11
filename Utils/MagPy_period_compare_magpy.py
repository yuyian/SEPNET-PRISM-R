#!/usr/bin/env python3
"""Compare MagPy Scoreboard vs SEPNET-PRISM-R on the MagPy validation window.

Uses the same truth labels (``mtm_lstm_cls_true`` / Future OSEP on our scored
hours) and the same probability thresholds. MagPy forecasts are matched by
``prediction_window.start_time`` ↔ our ``window_begin`` (forecast issue / start
of the 24 h prediction window), rounded to the hour.
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

from hsmm_analysis import calc_cls_metrics  # noqa: E402
from paths_config import P  # noqa: E402

TEST_START = pd.Timestamp("2023-07-01", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")


def _parse_ts(s) -> pd.Timestamp:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _load_magpy_jsons(json_root: Path) -> pd.DataFrame:
    files = sorted(json_root.rglob("MagPy-HMI-SHARP-Vector*.json"))
    if not files:
        raise FileNotFoundError(
            f"No MagPy JSONs under {json_root}. "
            "Run MagPy_period_fetch_magpy_scoreboard.py first."
        )
    rows = []
    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        sub = obj.get("sep_forecast_submission") or obj
        issue = sub.get("issue_time")
        forecasts = sub.get("forecasts") or []
        for fc in forecasts:
            ec = fc.get("energy_channel") or {}
            if float(ec.get("min", -1)) != 10:
                continue
            if str(fc.get("species", "proton")).lower() != "proton":
                continue
            if str(fc.get("location", "earth")).lower() != "earth":
                continue
            pw = fc.get("prediction_window") or {}
            probs = fc.get("probabilities") or []
            if not probs:
                continue
            pr = probs[0]
            ac = fc.get("all_clear") or {}
            rows.append(
                {
                    "file": fp.name,
                    "issue_time": issue,
                    "pred_start": pw.get("start_time"),
                    "pred_end": pw.get("end_time"),
                    "magpy_prob": pr.get("probability_value"),
                    "magpy_unc": pr.get("uncertainty"),
                    "magpy_all_clear": ac.get("all_clear_boolean"),
                    "magpy_prob_thr_native": ac.get("probability_threshold"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Parsed 0 MagPy forecasts")
    for c in ("issue_time", "pred_start", "pred_end"):
        df[c] = df[c].map(_parse_ts)
    df["magpy_prob"] = pd.to_numeric(df["magpy_prob"], errors="coerce")
    df["magpy_unc"] = pd.to_numeric(df["magpy_unc"], errors="coerce")
    df["magpy_prob_thr_native"] = pd.to_numeric(df["magpy_prob_thr_native"], errors="coerce")
    # Keep forecasts whose 24 h window starts inside MagPy test window.
    m = (df["pred_start"] >= TEST_START) & (df["pred_start"] <= TEST_END)
    df = df.loc[m].copy()
    df["pred_start_hour"] = df["pred_start"].dt.floor("h")
    # One forecast per hour: prefer latest issue_time
    df = (
        df.sort_values(["pred_start_hour", "issue_time"])
        .groupby("pred_start_hour", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return df


def _metrics_row(name: str, model: str, thr: float, y: pd.Series, p: pd.Series) -> dict:
    ok = y.notna() & p.notna()
    yy = y[ok].astype(int)
    pp = p[ok].astype(float)
    pred = (pp >= thr).astype(int)
    m = calc_cls_metrics(yy, pred, prob=pp)
    return {
        "model": model,
        "name": name,
        "threshold": float(thr),
        "n": int(len(yy)),
        "n_true": int(yy.sum()),
        "n_pred": int(pred.sum()),
        "ACC": float(m.get("Acc", m.get("ACC", m.get("accuracy", np.nan)))),
        "HSS": float(m["HSS"]),
        "TSS": float(m["TSS"]),
        "POD": float(m["POD"]),
        "FAR": float(m["FAR"]),
        "TP": float(m["TP"]),
        "FP": float(m["FP"]),
        "TN": float(m["TN"]),
        "FN": float(m["FN"]),
        "prob_mean": float(pp.mean()) if len(pp) else np.nan,
        "prob_mean_true1": float(pp[yy == 1].mean()) if (yy == 1).any() else np.nan,
        "prob_mean_true0": float(pp[yy == 0].mean()) if (yy == 0).any() else np.nan,
    }


def _daily_agg(df: pd.DataFrame, prob_col: str, truth_col: str = "y") -> pd.DataFrame:
    d = df.copy()
    d["day"] = d["window_begin"].dt.floor("D")
    return (
        d.groupby("day", as_index=False)
        .agg(
            y=(truth_col, lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max() >= 1)),
            p=(prob_col, lambda s: float(pd.to_numeric(s, errors="coerce").max())),
        )
        .rename(columns={"p": prob_col})
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(P("magpy_validation_dir")))
    ap.add_argument(
        "--ours",
        default=None,
        help="Our MagPy-period ensemble CSV (default: predictions/magpy_period_ensemble_continuous.csv)",
    )
    ap.add_argument(
        "--magpy-json-dir",
        default=None,
        help="Downloaded MagPy JSON root (default: magpy_scoreboard_json/2.X)",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    ours_path = Path(args.ours) if args.ours else (root / "predictions" / "magpy_period_ensemble_continuous.csv")
    magpy_dir = Path(args.magpy_json_dir) if args.magpy_json_dir else (root / "magpy_scoreboard_json" / "2.X")
    out_dir = root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] loading MagPy JSONs from {magpy_dir} ...", flush=True)
    magpy = _load_magpy_jsons(magpy_dir)
    print(f"[compare] MagPy unique hours in window: {len(magpy):,}", flush=True)

    ours = pd.read_csv(ours_path, low_memory=False)
    ours["window_begin"] = ours["window_begin"].map(_parse_ts)
    ours["window_end"] = ours["window_end"].map(_parse_ts)
    ours["ours_prob"] = pd.to_numeric(ours["mtm_lstm_cls_prob"], errors="coerce")
    ours["y"] = pd.to_numeric(ours["mtm_lstm_cls_true"], errors="coerce")
    ours_thr = float(pd.to_numeric(ours["mtm_lstm_threshold_used"], errors="coerce").median())
    magpy_native_thr = float(magpy["magpy_prob_thr_native"].dropna().median() or 0.19)

    # Align on forecast issue / prediction-window start hour.
    # Operational scoring treats window_begin as issue time and window_end as
    # verification time; MagPy prediction_window.start_time is magnetogram
    # last_data_time (start of its 24 h SEP probability window).
    ours["key"] = ours["window_begin"].dt.floor("h")
    merged = ours.merge(
        magpy.rename(columns={"pred_start_hour": "key"})[
            ["key", "magpy_prob", "magpy_all_clear", "magpy_prob_thr_native", "issue_time", "file"]
        ],
        on="key",
        how="inner",
    )
    print(
        f"[compare] intersection hours: {len(merged):,} "
        f"(ours={len(ours):,} magpy={len(magpy):,})",
        flush=True,
    )
    if merged.empty:
        raise RuntimeError("No overlapping hours between MagPy and our ensemble")

    series_path = out_dir / "magpy_vs_ours_aligned_hourly.csv"
    merged[
        [
            "window_begin",
            "window_end",
            "y",
            "ours_prob",
            "magpy_prob",
            "magpy_all_clear",
            "issue_time",
            "file",
        ]
    ].to_csv(series_path, index=False)

    thresholds = [
        ("thr_0.19_MagPy_native", magpy_native_thr),
        ("thr_0.50", 0.50),
        ("thr_ours_model", ours_thr),
    ]

    rows: list[dict] = []
    for name, thr in thresholds:
        rows.append(_metrics_row(f"hourly_{name}", "SEPNET-PRISM-R", thr, merged["y"], merged["ours_prob"]))
        rows.append(_metrics_row(f"hourly_{name}", "MagPy", thr, merged["y"], merged["magpy_prob"]))

    # MagPy native all_clear boolean (their operational rule)
    ac = merged["magpy_all_clear"]
    if ac.notna().any():
        # all_clear True → predict no SEP; False → predict SEP
        pred = (~ac.astype("boolean").fillna(True)).astype(int)
        m = calc_cls_metrics(merged["y"].astype(int), pred, prob=merged["magpy_prob"])
        rows.append(
            {
                "model": "MagPy",
                "name": "hourly_native_all_clear_flag",
                "threshold": magpy_native_thr,
                "n": int(len(merged)),
                "n_true": int(merged["y"].sum()),
                "n_pred": int(pred.sum()),
                "ACC": float(m.get("Acc", m.get("ACC", np.nan))),
                "HSS": float(m["HSS"]),
                "TSS": float(m["TSS"]),
                "POD": float(m["POD"]),
                "FAR": float(m["FAR"]),
                "TP": float(m["TP"]),
                "FP": float(m["FP"]),
                "TN": float(m["TN"]),
                "FN": float(m["FN"]),
                "prob_mean": float(merged["magpy_prob"].mean()),
                "prob_mean_true1": float(merged.loc[merged["y"] == 1, "magpy_prob"].mean()),
                "prob_mean_true0": float(merged.loc[merged["y"] == 0, "magpy_prob"].mean()),
            }
        )

    # Daily (max prob / any true)
    for model, col in (("SEPNET-PRISM-R", "ours_prob"), ("MagPy", "magpy_prob")):
        daily = _daily_agg(merged, col)
        for name, thr in thresholds:
            rows.append(
                _metrics_row(f"daily_{name}", model, thr, daily["y"], daily[col])
            )

    out = pd.DataFrame(rows)
    metrics_path = out_dir / "magpy_vs_ours_same_labels_thresholds.csv"
    out.to_csv(metrics_path, index=False)

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "alignment": "MagPy prediction_window.start_time hour == our window_begin",
        "truth": "mtm_lstm_cls_true (Future OSEP on our MagPy-period scored hours)",
        "n_intersection_hourly": int(len(merged)),
        "ours_model_thr_median": ours_thr,
        "magpy_native_prob_thr": magpy_native_thr,
        "series_csv": str(series_path),
        "metrics_csv": str(metrics_path),
        "magpy_json_dir": str(magpy_dir),
        "ours_csv": str(ours_path),
    }
    (out_dir / "magpy_vs_ours_meta.json").write_text(json.dumps(meta, indent=2))

    # Compact print: side-by-side HSS/TSS at shared thresholds
    print("\n=== Same-label comparison (intersection hours) ===")
    show = out[out["name"].str.startswith(("hourly_thr_", "daily_thr_"))].copy()
    piv = show.pivot_table(index=["name", "threshold"], columns="model", values=["HSS", "TSS", "POD", "FAR", "ACC"])
    print(piv.round(4).to_string())
    print(f"\n[compare] wrote {metrics_path}")
    print(f"[compare] wrote {series_path}")


if __name__ == "__main__":
    main()
