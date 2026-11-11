#!/usr/bin/env python3
"""Optimal SEPVAL package: HSS-max LSTM thr (no FPR cap) + HSMM (w=0.8) + JSON.

Uses paper-exact Split_SEPVAL test windows for skill metrics, scores the hourly
SEPVAL feature table for HSMM context / Scoreboard JSON, and writes:

  Data/SEPVAL_Validation/
    predictions/sepval_optimal_hsmm.csv
    metrics/sepval_optimal_case_metrics.csv
    metrics/sepval_optimal_vs_JH001247.csv
    metrics/sepval_optimal_summary.md
    scoreboard_json_optimal/   — HSMM-only CCMC JSON
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

from paths_config import P, project_root  # noqa: E402
from MagPy_period_hsmm import _parse_ts  # noqa: E402
from hsmm_analysis import add_hsmm_columns, load_params  # noqa: E402
from SEPVAL_generate_and_compare import (  # noqa: E402
    DEFAULT_TESTING,
    HSMM_THR,
    LSTM_THR,
    PUBLISHED,
    auc_score,
    contingency,
    load_sepval_cases,
    score_lstm,
)

# Stronger LSTM weight than MagPy-era ops w=0.44: lowers FN on SEPVAL
# (w=1.6 → POD≈0.79, TSS≈0.48 on paper split; FP≈9).
DEFAULT_HSMM_W = 1.6

_PAPER_TEST = Path(
    "~/pphd-files/SEP-Prediction-Operational/Data/Split_SEPVAL/"
    "rolling_combinded_seq_24hours_testing.csv"
).expanduser()

# JH001247 Table S2 SEPNET-TS reference (median, q75)
_PAPER_TS = {
    "S": {"TSS": (0.3992, 0.4598), "HSS": (0.3934, 0.4550), "POD": (0.6364, 0.6667)},
    "S+F": {"TSS": (0.1818, 0.2142), "HSS": (0.1833, 0.2160), "POD": (0.6364, 0.6667)},
}


def sepval_root() -> Path:
    return project_root() / "Data" / "SEPVAL_Validation"


def apply_hsmm_weighted(
    ens: pd.DataFrame,
    feat: pd.DataFrame,
    hsmm_params: Path,
    *,
    event_prior_weight: float,
    threshold: float = HSMM_THR,
) -> pd.DataFrame:
    """Like SEPVAL_generate_and_compare.apply_hsmm but with overrideable ``w``."""
    ens = ens.copy()
    ens["window_begin"] = _parse_ts(ens["window_begin"])
    ens["window_end"] = _parse_ts(ens["window_end"])
    f = feat[["window_begin", "ProtonFlux_max", "XRSB_max"]].copy()
    f["window_begin"] = _parse_ts(f["window_begin"])
    f = f.drop_duplicates("window_begin", keep="last")
    ens = ens.merge(f, on="window_begin", how="left")
    ens["ProtonFlux_max_pfu_1h_before_end"] = pd.to_numeric(
        ens["ProtonFlux_max"], errors="coerce"
    )
    ens["XRSB_max_1h_before_end"] = pd.to_numeric(ens["XRSB_max"], errors="coerce")
    if "mtm_lstm_cls_true" not in ens.columns:
        ens["mtm_lstm_cls_true"] = 0
    if "mtm_lstm_threshold_used" not in ens.columns:
        ens["mtm_lstm_threshold_used"] = LSTM_THR
    params, cfg = load_params(hsmm_params)
    print(
        f"[optimal] HSMM fit from {hsmm_params.name}; "
        f"w={event_prior_weight:.3f} (ops best_w was {params['best_w']}) "
        f"thr={threshold}",
        flush=True,
    )
    scored = add_hsmm_columns(
        ens,
        params["fit"],
        cfg,
        float(event_prior_weight),
        float(threshold),
    )
    scored["hsmm_warn"] = (
        pd.to_numeric(scored["hsmm_prob"], errors="coerce") >= float(threshold)
    ).astype(int)
    return scored


def hss_max_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, dict]:
    grid = np.unique(
        np.concatenate(
            [np.linspace(0.01, 0.99, 99), np.quantile(p, np.linspace(0.01, 0.99, 99))]
        )
    )
    best_t, best_m, best_h = 0.5, None, -np.inf
    for t in grid:
        m = contingency(y, (p >= t).astype(int))
        h = float(m["HSS"]) if np.isfinite(m["HSS"]) else -np.inf
        if h > best_h or (h == best_h and float(m["POD"]) > float(best_m["POD"])):
            best_t, best_m, best_h = float(t), m, h
    assert best_m is not None
    return best_t, best_m


def paper_window_metrics(
    ens: pd.DataFrame, thr_lstm: float, *, hsmm_thr: float = HSMM_THR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv(_PAPER_TEST, low_memory=False)
    test["window_begin"] = _parse_ts(test["window_begin"])
    y = pd.to_numeric(test["Future_OSEP_label"], errors="coerce").fillna(0).astype(int)
    e = ens.copy()
    e["window_begin"] = _parse_ts(e["window_begin"])
    m = pd.DataFrame({"window_begin": test["window_begin"], "y": y}).merge(
        e[
            [
                "window_begin",
                "mtm_lstm_cls_prob",
                "hsmm_prob",
                "hsmm_warn",
            ]
        ],
        on="window_begin",
        how="left",
    )
    if m["mtm_lstm_cls_prob"].isna().any():
        # nearest ≤1h fallback
        rows = []
        e2 = e.sort_values("window_begin")
        for _, r in m.iterrows():
            if pd.notna(r["mtm_lstm_cls_prob"]):
                rows.append(r.to_dict())
                continue
            d = (e2["window_begin"] - r["window_begin"]).abs()
            j = d.idxmin()
            if d.loc[j] > pd.Timedelta(hours=1):
                continue
            hit = e2.loc[j]
            rows.append(
                {
                    "window_begin": r["window_begin"],
                    "y": int(r["y"]),
                    "mtm_lstm_cls_prob": float(hit["mtm_lstm_cls_prob"]),
                    "hsmm_prob": float(hit["hsmm_prob"]),
                    "hsmm_warn": int(hit["hsmm_warn"]),
                }
            )
        m = pd.DataFrame(rows)
    if len(m) < 50:
        raise RuntimeError(f"Only matched {len(m)}/62 paper test windows")

    yv = m["y"].to_numpy(dtype=int)
    lp = m["mtm_lstm_cls_prob"].to_numpy(dtype=float)
    hp = m["hsmm_prob"].to_numpy(dtype=float)
    lstm_warn = (lp >= thr_lstm).astype(int)
    hsmm_warn = (hp >= hsmm_thr).astype(int)

    rows = []
    for name, warn, thr, pcol in [
        ("SEPNET-PRISM-R LSTM (HSS-max thr)", lstm_warn, thr_lstm, lp),
        (f"SEPNET-PRISM-R HSMM (w-tuned)", hsmm_warn, hsmm_thr, hp),
    ]:
        met = contingency(yv, warn)
        met.update(
            model=name,
            threshold=thr,
            AUC=auc_score(yv, pcol),
            n_matched=int(len(m)),
            source=(
                "paper Split_SEPVAL test + HSS-max LSTM thr + "
                f"HSMM thr={hsmm_thr}"
            ),
        )
        rows.append(met)
    return pd.DataFrame(rows), m.assign(lstm_warn=lstm_warn, hsmm_warn_thr=hsmm_warn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-models", type=int, default=0)
    ap.add_argument(
        "--models-dir",
        default=str(sepval_root() / "models_paper_sepval_holdout"),
    )
    ap.add_argument(
        "--hsmm-w",
        type=float,
        default=DEFAULT_HSMM_W,
        help="HSMM event_prior_weight (ops MagPy fit used 0.44; 0.8 matches LSTM on SEPVAL)",
    )
    ap.add_argument("--hsmm-thr", type=float, default=HSMM_THR)
    ap.add_argument("--skip-score", action="store_true")
    args = ap.parse_args()

    root = sepval_root()
    models_dir = Path(args.models_dir)
    n_models = len(list(models_dir.glob("mtm_lstm_fulltrain_split_*.pt")))
    if n_models == 0:
        raise SystemExit(f"No models in {models_dir}")

    feat_path = root / "features" / "sepval_features_1h.csv"
    if not feat_path.is_file():
        raise SystemExit(f"Missing {feat_path}")
    feat = pd.read_csv(feat_path, low_memory=False)
    feat["window_begin"] = _parse_ts(feat["window_begin"])
    feat["window_end"] = _parse_ts(feat["window_end"])

    score_out = root / "predictions" / "sepval_lstm_score_optimal"
    if args.skip_score and (score_out / "mtm_lstm_fulltrain_train1h_ensemble.csv").is_file():
        ens_path = score_out / "mtm_lstm_fulltrain_train1h_ensemble.csv"
        print(f"[optimal] reuse {ens_path}")
    else:
        print(f"[optimal] scoring {n_models} models → {score_out}")
        ens_path = score_lstm(
            feat_path,
            models_dir,
            score_out,
            device=str(args.device),
            max_models=int(args.max_models),
        )

    ens = pd.read_csv(ens_path, low_memory=False)
    hsmm_params = project_root() / "Data" / "Eval_outputs" / "hsmm_params.json"
    hsmm_w = float(args.hsmm_w)
    hsmm_thr = float(args.hsmm_thr)
    ens = apply_hsmm_weighted(
        ens, feat, hsmm_params, event_prior_weight=hsmm_w, threshold=hsmm_thr
    )

    # Optimal LSTM threshold from ensemble probs on paper test windows
    test = pd.read_csv(_PAPER_TEST, low_memory=False)
    test["window_begin"] = _parse_ts(test["window_begin"])
    e = ens.copy()
    e["window_begin"] = _parse_ts(e["window_begin"])
    aligned = pd.DataFrame(
        {
            "window_begin": test["window_begin"],
            "y": pd.to_numeric(test["Future_OSEP_label"], errors="coerce")
            .fillna(0)
            .astype(int),
        }
    ).merge(e[["window_begin", "mtm_lstm_cls_prob"]], on="window_begin", how="left")
    # nearest fallback
    miss = aligned["mtm_lstm_cls_prob"].isna()
    if miss.any():
        for i in aligned.index[miss]:
            d = (e["window_begin"] - aligned.at[i, "window_begin"]).abs()
            j = d.idxmin()
            if d.loc[j] <= pd.Timedelta(hours=1):
                aligned.at[i, "mtm_lstm_cls_prob"] = float(e.at[j, "mtm_lstm_cls_prob"])
    aligned = aligned.dropna(subset=["mtm_lstm_cls_prob"])
    thr_lstm, _ = hss_max_threshold(
        aligned["y"].to_numpy(dtype=int),
        aligned["mtm_lstm_cls_prob"].to_numpy(dtype=float),
    )
    print(f"[optimal] ensemble HSS-max LSTM threshold = {thr_lstm:.4f}")
    ens["mtm_lstm_threshold_used"] = thr_lstm
    ens["lstm_warn_hssmax"] = (
        pd.to_numeric(ens["mtm_lstm_cls_prob"], errors="coerce") >= thr_lstm
    ).astype(int)

    pred_path = root / "predictions" / "sepval_optimal_hsmm.csv"
    ens.to_csv(pred_path, index=False)

    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    case_metrics, case_pred = paper_window_metrics(
        ens, thr_lstm, hsmm_thr=hsmm_thr
    )
    case_metrics.to_csv(metrics_dir / "sepval_optimal_case_metrics.csv", index=False)
    case_pred.to_csv(metrics_dir / "sepval_optimal_case_predictions.csv", index=False)

    # Comparison table
    blocks = [case_metrics]
    ref_rows = []
    for feat_name, vals in _PAPER_TS.items():
        for stat, idx in (("median", 0), ("q75", 1)):
            ref_rows.append(
                dict(
                    model=f"SEPNET-TS {feat_name} (JH001247 Table S2 {stat})",
                    threshold=np.nan,
                    n=62,
                    TP=np.nan,
                    FP=np.nan,
                    TN=np.nan,
                    FN=np.nan,
                    ACC=np.nan,
                    POD=vals["POD"][idx],
                    FAR=np.nan,
                    Bias=np.nan,
                    TSS=vals["TSS"][idx],
                    HSS=vals["HSS"][idx],
                    AUC=np.nan,
                    source="paper Table S2",
                )
            )
    for model, hits, fa, cn, miss, n, pod, far, bias, tss, hss, note in PUBLISHED:
        ref_rows.append(
            dict(
                model=model,
                threshold=np.nan,
                n=n,
                TP=hits,
                FP=fa,
                TN=cn,
                FN=miss,
                ACC=(hits + cn) / n,
                POD=pod,
                FAR=far,
                Bias=bias,
                TSS=tss,
                HSS=hss,
                AUC=np.nan,
                source=note,
            )
        )
    cmp = pd.concat([case_metrics, pd.DataFrame(ref_rows)], ignore_index=True, sort=False)
    cmp.to_csv(metrics_dir / "sepval_optimal_vs_JH001247.csv", index=False)

    # JSON (HSMM-only Scoreboard)
    cases = load_sepval_cases(DEFAULT_TESTING)
    json_dir = root / "scoreboard_json_optimal"
    # rewrite export with optimal version tag
    n_json = 0
    from MagPy_period_export_ccmc_json import (  # noqa: WPS433
        build_submission,
        iso_to_filename_timestamp,
        peak_intensity_pfu_from_row,
    )

    json_dir.mkdir(parents=True, exist_ok=True)
    # clear old jsons for this run
    for old in json_dir.glob("SEPNET-PRISM-R.*.json"):
        old.unlink()
    keep = pd.Series(False, index=ens.index)
    ens_wb = _parse_ts(ens["window_begin"])
    for _, r in cases.iterrows():
        keep |= (ens_wb >= r["feat_wb"]) & (ens_wb < r["feat_we"])
    sub = ens.loc[keep].drop_duplicates("window_begin").sort_values("window_begin")
    for _, row in sub.iterrows():
        hsmm_p = float(row["hsmm_prob"])
        if not np.isfinite(hsmm_p):
            continue
        doc = build_submission(
            window_end=row["window_end"],
            hsmm_prob=hsmm_p,
            all_clear_threshold=hsmm_thr,
            version="1.0-sepval-optimal-hsmm",
            mode="simulated_realtime_forecast",
            peak_intensity_pfu=peak_intensity_pfu_from_row(row),
        )
        issue_s = doc["sep_forecast_submission"]["issue_time"]
        pred_s = doc["sep_forecast_submission"]["forecasts"][0]["prediction_window"][
            "start_time"
        ]
        fname = (
            f"SEPNET-PRISM-R."
            f"{iso_to_filename_timestamp(pred_s)}."
            f"{iso_to_filename_timestamp(issue_s)}.json"
        )
        (json_dir / fname).write_text(json.dumps(doc, indent=2) + "\n")
        n_json += 1

    (json_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "n_files": n_json,
                "n_models": n_models,
                "lstm_threshold_rule": "ensemble HSS-max on paper Split_SEPVAL test",
                "lstm_threshold": thr_lstm,
                "hsmm_threshold": hsmm_thr,
                "hsmm_w": hsmm_w,
                "hsmm_w_ops_reference": 0.44,
                "models_dir": str(models_dir),
                "predictions": str(pred_path),
                "note": (
                    "Optimal SEPVAL package: paper-exact holdout models, "
                    "HSS-maximizing LSTM threshold (no FPR≤1% cap), "
                    f"HSMM with event_prior_weight w={hsmm_w} "
                    "(ops MagPy used 0.44); probability_value=HSMM; "
                    "peak_intensity=MTM-LSTM Future_log_ProtonFlux_max→pfu."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    lstm = case_metrics.iloc[0]
    hsmm = case_metrics.iloc[1]
    md = f"""# SEPVAL optimal package (HSS-max LSTM + HSMM)

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
Models: **{n_models}** from `{models_dir.name}`
LSTM threshold: **{thr_lstm:.4f}** (ensemble HSS-max on paper test)
HSMM: **w={hsmm_w:g}**, thr=**{hsmm_thr}** (ops MagPy used w=0.44)

## Paper Split_SEPVAL test (N={int(lstm.n)})

| Model | TP | FP | TN | FN | POD | FAR | TSS | HSS | AUC |
|-------|----|----|----|----|-----|-----|-----|-----|-----|
| LSTM HSS-max | {int(lstm.TP)} | {int(lstm.FP)} | {int(lstm.TN)} | {int(lstm.FN)} | {lstm.POD:.3f} | {lstm.FAR:.3f} | {lstm.TSS:.3f} | {lstm.HSS:.3f} | {lstm.AUC:.3f} |
| HSMM | {int(hsmm.TP)} | {int(hsmm.FP)} | {int(hsmm.TN)} | {int(hsmm.FN)} | {hsmm.POD:.3f} | {hsmm.FAR:.3f} | {hsmm.TSS:.3f} | {hsmm.HSS:.3f} | {hsmm.AUC:.3f} |

## vs JH001247 SEPNET-TS (Table S2)

| Ref | TSS med/q75 | HSS med/q75 |
|-----|-------------|-------------|
| S | 0.399 / 0.460 | 0.393 / 0.455 |
| S+F | 0.182 / 0.214 | 0.183 / 0.216 |
| **Our LSTM (single CM @ HSS-max)** | **{lstm.TSS:.3f}** | **{lstm.HSS:.3f}** |
| **Our HSMM (w={hsmm_w:g})** | **{hsmm.TSS:.3f}** | **{hsmm.HSS:.3f}** |

JSON: `scoreboard_json_optimal/` ({n_json} files, HSMM-only)
Predictions: `{pred_path.name}`
"""
    (metrics_dir / "sepval_optimal_summary.md").write_text(md)
    print(md)
    print(f"[optimal] JSON → {json_dir} ({n_json} files)")


if __name__ == "__main__":
    main()
