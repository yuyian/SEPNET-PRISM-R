#!/usr/bin/env python3
"""Apply HSMM post-analysis to MagPy-period cutoff ensemble forecasts.

Uses the operational ``hsmm_params.json`` (same smoother as the postanalysis
website / NRT pipeline): merge proton & XRS covariates from the MagPy-period
feature table, then write ``hsmm_prob`` / ``hsmm_warn`` and skill metrics
alongside the raw LSTM ensemble.
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

from hsmm_analysis import (  # noqa: E402
    add_hsmm_columns,
    calc_cls_metrics,
    load_params,
)
from paths_config import P  # noqa: E402


def _parse_ts(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


def _metrics_row(name: str, model: str, thr: float, y, p, pred=None) -> dict:
    yy = pd.to_numeric(pd.Series(y), errors="coerce")
    pp = pd.to_numeric(pd.Series(p), errors="coerce")
    ok = yy.notna() & pp.notna()
    yy = yy[ok].astype(int)
    pp = pp[ok].astype(float)
    if pred is None:
        pred_bin = (pp >= float(thr)).astype(int)
    else:
        pred_bin = pd.to_numeric(pd.Series(pred), errors="coerce")[ok].fillna(0).astype(int)
    m = calc_cls_metrics(yy, pred_bin, prob=pp)
    return {
        "model": model,
        "name": name,
        "threshold": float(thr),
        "n": int(len(yy)),
        "n_true": int(yy.sum()),
        "n_pred": int(pred_bin.sum()),
        "ACC": float(m["ACC"]),
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


def _daily(df: pd.DataFrame, prob_col: str, warn_col: str | None = None) -> pd.DataFrame:
    d = df.copy()
    d["day"] = d["window_begin"].dt.floor("D")
    agg = {
        "y": ("mtm_lstm_cls_true", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max() >= 1)),
        "p": (prob_col, lambda s: float(pd.to_numeric(s, errors="coerce").max())),
    }
    if warn_col and warn_col in d.columns:
        agg["w"] = (warn_col, lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max() >= 1))
    g = d.groupby("day", as_index=False).agg(**{k: v for k, v in [
        ("y", agg["y"]),
        (prob_col, agg["p"]),
        *( [("warn", agg["w"])] if "w" in agg else [] ),
    ]})
    return g


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(P("magpy_validation_dir")))
    ap.add_argument("--hsmm-params", default=str(P("hsmm_params_json")))
    ap.add_argument(
        "--ensemble",
        default=None,
        help="Default: predictions/magpy_period_ensemble_continuous.csv",
    )
    ap.add_argument(
        "--features",
        default=None,
        help="Default: features/test_202307_202412_1h.csv (for proton/XRS covariates)",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    ens_path = Path(args.ensemble) if args.ensemble else (
        root / "predictions" / "magpy_period_ensemble_continuous.csv"
    )
    feat_path = Path(args.features) if args.features else (
        root / "features" / "test_202307_202412_1h.csv"
    )
    params_path = Path(args.hsmm_params).expanduser().resolve()
    out_dir = root / "predictions"
    metrics_dir = root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hsmm] loading ensemble {ens_path}", flush=True)
    ens = pd.read_csv(ens_path, low_memory=False)
    ens["window_begin"] = _parse_ts(ens["window_begin"])
    ens["window_end"] = _parse_ts(ens["window_end"])

    need = {"window_begin", "ProtonFlux_max", "XRSB_max"}
    print(f"[hsmm] loading covariates from {feat_path}", flush=True)
    feat = pd.read_csv(
        feat_path,
        usecols=lambda c: c in need,
        low_memory=False,
    )
    feat["window_begin"] = _parse_ts(feat["window_begin"])
    feat = feat.drop_duplicates("window_begin", keep="last")
    ens = ens.merge(feat, on="window_begin", how="left")
    ens["ProtonFlux_max_pfu_1h_before_end"] = pd.to_numeric(ens["ProtonFlux_max"], errors="coerce")
    ens["XRSB_max_1h_before_end"] = pd.to_numeric(ens["XRSB_max"], errors="coerce")

    n_miss_pf = int(ens["ProtonFlux_max_pfu_1h_before_end"].isna().sum())
    n_miss_xrs = int(ens["XRSB_max_1h_before_end"].isna().sum())
    print(f"[hsmm] covariate missing: proton={n_miss_pf} xrs={n_miss_xrs} / {len(ens)}", flush=True)

    params, cfg = load_params(params_path)
    print(
        f"[hsmm] params={params_path.name} w={params['best_w']} thr={params['threshold']}",
        flush=True,
    )
    scored = add_hsmm_columns(
        ens,
        params["fit"],
        cfg,
        float(params["best_w"]),
        float(params["threshold"]),
    )
    scored["window_begin"] = _parse_ts(scored["window_begin"])
    scored["window_end"] = _parse_ts(scored["window_end"])

    out_csv = out_dir / "magpy_period_ensemble_hsmm.csv"
    scored.to_csv(out_csv, index=False)
    print(f"[hsmm] wrote {out_csv} rows={len(scored):,}", flush=True)

    y = scored["mtm_lstm_cls_true"]
    lstm_thr = float(pd.to_numeric(scored["mtm_lstm_threshold_used"], errors="coerce").median())
    hsmm_thr = float(params["threshold"])
    rows = [
        _metrics_row("hourly_lstm_model_thr", "LSTM", lstm_thr, y, scored["mtm_lstm_cls_prob"]),
        _metrics_row("hourly_lstm_thr_0.50", "LSTM", 0.50, y, scored["mtm_lstm_cls_prob"]),
        _metrics_row(
            "hourly_hsmm_warn",
            "LSTM+HSMM",
            hsmm_thr,
            y,
            scored["hsmm_prob"],
            pred=scored["hsmm_warn"],
        ),
        _metrics_row("hourly_hsmm_thr_0.50", "LSTM+HSMM", 0.50, y, scored["hsmm_prob"]),
    ]

    # Daily: max prob / any warn
    scored_day = scored.copy()
    scored_day["day"] = scored_day["window_begin"].dt.floor("D")
    daily = scored_day.groupby("day", as_index=False).agg(
        y=("mtm_lstm_cls_true", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max() >= 1)),
        lstm_p=("mtm_lstm_cls_prob", lambda s: float(pd.to_numeric(s, errors="coerce").max())),
        hsmm_p=("hsmm_prob", lambda s: float(pd.to_numeric(s, errors="coerce").max())),
        hsmm_w=("hsmm_warn", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max() >= 1)),
    )
    rows.extend(
        [
            _metrics_row("daily_lstm_thr_0.50", "LSTM", 0.50, daily["y"], daily["lstm_p"]),
            _metrics_row("daily_lstm_model_thr", "LSTM", lstm_thr, daily["y"], daily["lstm_p"]),
            _metrics_row(
                "daily_hsmm_warn",
                "LSTM+HSMM",
                hsmm_thr,
                daily["y"],
                daily["hsmm_p"],
                pred=daily["hsmm_w"],
            ),
            _metrics_row("daily_hsmm_thr_0.50", "LSTM+HSMM", 0.50, daily["y"], daily["hsmm_p"]),
        ]
    )

    metrics = pd.DataFrame(rows)
    metrics_path = metrics_dir / "magpy_period_metrics_lstm_vs_hsmm.csv"
    metrics.to_csv(metrics_path, index=False)

    # Side-by-side with MagPy if aligned series exists
    aligned = metrics_dir / "magpy_vs_ours_aligned_hourly.csv"
    if aligned.is_file():
        a = pd.read_csv(aligned, low_memory=False)
        a["window_begin"] = _parse_ts(a["window_begin"])
        merge_cols = ["window_begin", "hsmm_prob", "hsmm_warn"]
        m3 = a.merge(
            scored[merge_cols],
            on="window_begin",
            how="inner",
        )
        extra = [
            _metrics_row("hourly_thr_0.50", "LSTM", 0.50, m3["y"], m3["ours_prob"]),
            _metrics_row(
                "hourly_hsmm_warn",
                "LSTM+HSMM",
                hsmm_thr,
                m3["y"],
                m3["hsmm_prob"],
                pred=m3["hsmm_warn"],
            ),
            _metrics_row("hourly_thr_0.19_MagPy_native", "MagPy", 0.19, m3["y"], m3["magpy_prob"]),
        ]
        # daily three-way
        for model, pcol, thr, pred_col in (
            ("LSTM", "ours_prob", 0.50, None),
            ("LSTM+HSMM", "hsmm_prob", hsmm_thr, "hsmm_warn"),
            ("MagPy", "magpy_prob", 0.19, None),
        ):
            d = m3.copy()
            d["day"] = d["window_begin"].dt.floor("D")
            ag = {"y": ("y", "max"), "p": (pcol, "max")}
            if pred_col:
                ag["w"] = (pred_col, "max")
            g = d.groupby("day", as_index=False).agg(**{
                "y": ag["y"],
                "p": ag["p"],
                **({"w": ag["w"]} if pred_col else {}),
            })
            name = f"daily_{'hsmm_warn' if pred_col else f'thr_{thr}'}"
            if pred_col:
                extra.append(_metrics_row(name, model, thr, g["y"], g["p"], pred=g["w"]))
            else:
                extra.append(_metrics_row(name, model, thr, g["y"], g["p"]))
        three = pd.DataFrame(extra)
        three_path = metrics_dir / "magpy_vs_lstm_vs_hsmm.csv"
        three.to_csv(three_path, index=False)
        print(f"[hsmm] wrote {three_path}", flush=True)
    else:
        three_path = None

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "note": (
            "HSMM post-analysis on MagPy-period cutoff ensemble using operational "
            "hsmm_params.json (fit on production fulltrain train set, not re-fit on cutoff)."
        ),
        "hsmm_params": str(params_path),
        "best_w": float(params["best_w"]),
        "hsmm_threshold": hsmm_thr,
        "lstm_model_thr_median": lstm_thr,
        "n_hourly": int(len(scored)),
        "ensemble_hsmm_csv": str(out_csv),
        "metrics_csv": str(metrics_path),
        "three_way_csv": str(three_path) if three_path else None,
    }
    (metrics_dir / "magpy_period_hsmm_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== LSTM vs HSMM (MagPy period) ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if three_path:
        print("\n=== vs MagPy (intersection) ===")
        print(pd.read_csv(three_path).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
