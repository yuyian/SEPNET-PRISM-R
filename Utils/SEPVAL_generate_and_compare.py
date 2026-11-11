#!/usr/bin/env python3
"""Generate SEPVAL Phase-III Scoreboard forecasts + criteria vs published models.

Uses MagPy-cutoff MTM-LSTM ensemble (train window_end < 2023-07-01) + operational
HSMM params. Case list = CCMC SEP Validation Challenge Phase III (via
``rolling_combinded_testing.csv``: 33 SEP + 29 non-events).

Outputs under ``Data/SEPVAL_Validation/``:
  * features/sepval_features_1h.csv          — hourly rows covering SEPVAL periods
  * predictions/sepval_ensemble_hsmm.csv     — LSTM + HSMM scores
  * scoreboard_json/                        — CCMC JSON (HSMM-only)
  * metrics/sepval_case_metrics.csv          — one decision per case (N≈62)
  * metrics/sepval_vs_published_models.csv   — comparison table
  * metrics/sepval_summary.md
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
from MagPy_period_export_ccmc_json import (  # noqa: E402
    build_submission,
    iso_to_filename_timestamp,
    peak_intensity_pfu_from_row,
)
from MagPy_period_hsmm import _parse_ts  # noqa: E402
from hsmm_analysis import add_hsmm_columns, load_params  # noqa: E402
from Step1_fetch_aggregate_eval_nrt import aggregate_predictions_across_splits  # noqa: E402

DEFAULT_TESTING = Path(
    "/Users/yuyian/pphd-files/ML-CMESEP/SEP prediction-20251104/rolling_combinded_testing.csv"
)
HSMM_THR = 0.1
LSTM_THR = 0.5

# Published SEPVAL >10 MeV / 10 pfu all-clear (SHINE 2024 SEPVAL poster / SPHINX)
PUBLISHED = [
    # model, hits, FA, CN, miss, n, POD, FAR, bias, TSS, HSS, note
    ("MagPy", 3, 2, 27, 29, 62, 0.12, 0.33, 0.18, 0.05, 0.05,
     "SEPVAL poster SPHINX all-clear; magnetogram-only"),
    ("Lavasa", 18, 5, 24, 6, 53, 0.75, 0.22, 0.96, 0.58, 0.58, "SEPVAL poster"),
    ("PPS SFS-Update", 27, 23, 7, 2, 59, 0.93, 0.46, 1.7, 0.16, 0.16, "SEPVAL poster"),
    ("SAWS-ASPECS flare+CME", 26, 21, 9, 3, 59, 0.90, 0.45, 1.62, 0.20, 0.19, "SEPVAL poster"),
]


def contingency(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    n = tp + fp + tn + fn
    pod = tp / (tp + fn) if (tp + fn) else np.nan
    far = fp / (tp + fp) if (tp + fp) else np.nan
    acc = (tp + tn) / n if n else np.nan
    # TSS / HSS
    tss = pod - (fp / (fp + tn) if (fp + tn) else np.nan)
    pc = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n) if n else np.nan
    hss = ((tp + tn) / n - pc) / (1 - pc) if n and pc != 1 else np.nan
    bias = (tp + fp) / (tp + fn) if (tp + fn) else np.nan
    return dict(
        n=n, TP=tp, FP=fp, TN=tn, FN=fn, ACC=acc, POD=pod, FAR=far,
        Bias=bias, TSS=tss, HSS=hss,
    )


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann–Whitney
    wins = 0.0
    for a in pos:
        wins += float((a > neg).sum()) + 0.5 * float((a == neg).sum())
    return wins / (len(pos) * len(neg))


def load_sepval_cases(testing_csv: Path) -> pd.DataFrame:
    t = pd.read_csv(testing_csv, low_memory=False)
    t["window_begin"] = _parse_ts(t["window_begin"])
    t["window_end"] = _parse_ts(t["window_end"])
    t["y"] = pd.to_numeric(t["future_Operational_SEP_label"], errors="coerce").fillna(0).astype(int)
    t = t.dropna(subset=["window_begin"]).sort_values("window_begin").reset_index(drop=True)
    t["case_id"] = [f"SEPVAL_{i:03d}" for i in range(len(t))]
    # Hour-aligned feature key (SEPVAL windows are :05; our features are :00)
    t["feat_wb"] = t["window_begin"].dt.floor("h")
    t["feat_we"] = t["feat_wb"] + pd.Timedelta(hours=24)
    return t[["case_id", "window_begin", "window_end", "feat_wb", "feat_we", "y"]]


def extract_feature_hours(
    feature_csv: Path,
    cases: pd.DataFrame,
    *,
    pad_hours: int = 48,
) -> pd.DataFrame:
    """All hourly feature rows covering SEPVAL case periods (±pad for HSMM context)."""
    intervals = []
    for _, r in cases.iterrows():
        intervals.append((r["feat_wb"] - pd.Timedelta(hours=pad_hours),
                          r["feat_we"] + pd.Timedelta(hours=pad_hours)))
    # merge overlapping
    intervals.sort()
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))

    parts = []
    for ch in pd.read_csv(feature_csv, chunksize=200_000, low_memory=False):
        ch["window_begin"] = _parse_ts(ch["window_begin"])
        ch["window_end"] = _parse_ts(ch["window_end"])
        m = pd.Series(False, index=ch.index)
        for a, b in merged:
            m |= (ch["window_begin"] >= a) & (ch["window_begin"] <= b)
        if m.any():
            parts.append(ch.loc[m])
    if not parts:
        raise RuntimeError("No feature rows found for SEPVAL periods")
    feat = pd.concat(parts, ignore_index=True)
    feat = feat.drop_duplicates(["window_begin", "window_end"], keep="last")
    feat = feat.sort_values("window_begin").reset_index(drop=True)
    return feat


def score_lstm(feat_csv: Path, models_dir: Path, out_dir: Path, *, device: str, max_models: int) -> Path:
    from Step2_score_train_mtm_lstm import main as score_main

    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--input-csv", str(feat_csv),
        "--models-dir", str(models_dir),
        "--out-dir", str(out_dir),
        "--device", device,
        "--no-keep-by-model",
    ]
    if max_models and max_models > 0:
        argv += ["--max-models", str(max_models)]
    score_main(argv)
    ens = out_dir / "mtm_lstm_fulltrain_train1h_ensemble.csv"
    if not ens.is_file():
        # aggregate from concat if needed
        concat = out_dir / "mtm_lstm_fulltrain_train1h_predictions.csv"
        ens_df = aggregate_predictions_across_splits(pd.read_csv(concat, low_memory=False))
        ens_df.to_csv(ens, index=False)
    return ens


def apply_hsmm(ens: pd.DataFrame, feat: pd.DataFrame, hsmm_params: Path) -> pd.DataFrame:
    ens = ens.copy()
    ens["window_begin"] = _parse_ts(ens["window_begin"])
    ens["window_end"] = _parse_ts(ens["window_end"])
    f = feat[["window_begin", "ProtonFlux_max", "XRSB_max"]].copy()
    f["window_begin"] = _parse_ts(f["window_begin"])
    f = f.drop_duplicates("window_begin", keep="last")
    ens = ens.merge(f, on="window_begin", how="left")
    ens["ProtonFlux_max_pfu_1h_before_end"] = pd.to_numeric(ens["ProtonFlux_max"], errors="coerce")
    ens["XRSB_max_1h_before_end"] = pd.to_numeric(ens["XRSB_max"], errors="coerce")
    if "mtm_lstm_cls_true" not in ens.columns:
        ens["mtm_lstm_cls_true"] = 0
    if "mtm_lstm_threshold_used" not in ens.columns:
        ens["mtm_lstm_threshold_used"] = LSTM_THR
    params, cfg = load_params(hsmm_params)
    print(
        f"[sepval] HSMM params w={params['best_w']} thr={params['threshold']}",
        flush=True,
    )
    scored = add_hsmm_columns(
        ens,
        params["fit"],
        cfg,
        float(params["best_w"]),
        float(params["threshold"]),
    )
    # Keep Scoreboard all_clear at 0.1 even if params threshold differs slightly.
    scored["hsmm_warn"] = (pd.to_numeric(scored["hsmm_prob"], errors="coerce") >= HSMM_THR).astype(int)
    return scored


def export_json(ens: pd.DataFrame, cases: pd.DataFrame, out_dir: Path) -> int:
    """Export JSON for case hours + all hours inside each case [feat_wb, feat_we)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    case_hours = set(cases["feat_wb"])
    # also all hours in padded? For Scoreboard SEPVAL set: hours with feat_wb in any case window
    keep = ens["window_begin"].isin(case_hours)
    # continuous within each 24h case feature window
    for _, r in cases.iterrows():
        keep |= (ens["window_begin"] >= r["feat_wb"]) & (ens["window_begin"] < r["feat_we"])
    sub = ens.loc[keep].drop_duplicates("window_begin").sort_values("window_begin")
    n = 0
    for _, row in sub.iterrows():
        hsmm_p = float(row["hsmm_prob"])
        if not np.isfinite(hsmm_p):
            continue
        # window_end drives last_data_time cadences in build_submission
        doc = build_submission(
            window_end=row["window_end"],
            hsmm_prob=hsmm_p,
            all_clear_threshold=HSMM_THR,
            version="1.0-sepval-magpy-cutoff-hsmm",
            mode="simulated_realtime_forecast",
            peak_intensity_pfu=peak_intensity_pfu_from_row(row),
        )
        issue_s = doc["sep_forecast_submission"]["issue_time"]
        pred_s = doc["sep_forecast_submission"]["forecasts"][0]["prediction_window"]["start_time"]
        fname = (
            f"SEPNET-PRISM-R."
            f"{iso_to_filename_timestamp(pred_s)}."
            f"{iso_to_filename_timestamp(issue_s)}.json"
        )
        (out_dir / fname).write_text(json.dumps(doc, indent=2) + "\n")
        n += 1
    return n


def case_level_table(cases: pd.DataFrame, ens: pd.DataFrame) -> pd.DataFrame:
    e = ens.copy()
    e["window_begin"] = _parse_ts(e["window_begin"])
    rows = []
    for _, c in cases.iterrows():
        m = e[e["window_begin"] == c["feat_wb"]]
        if m.empty:
            # nearest within 1h
            d = (e["window_begin"] - c["feat_wb"]).abs()
            j = d.idxmin()
            if d.loc[j] > pd.Timedelta(hours=1):
                rows.append({**c.to_dict(), "matched": False})
                continue
            m = e.loc[[j]]
        r = m.iloc[0]
        lstm_p = float(r["mtm_lstm_cls_prob"])
        hsmm_p = float(r["hsmm_prob"])
        rows.append({
            "case_id": c["case_id"],
            "case_window_begin": c["window_begin"],
            "feat_window_begin": r["window_begin"],
            "feat_window_end": r["window_end"],
            "y": int(c["y"]),
            "lstm_prob": lstm_p,
            "hsmm_prob": hsmm_p,
            "lstm_warn": int(lstm_p >= LSTM_THR),
            "hsmm_warn": int(hsmm_p >= HSMM_THR),
            "matched": True,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(project_root() / "Data" / "SEPVAL_Validation"))
    ap.add_argument("--testing-csv", default=str(DEFAULT_TESTING))
    ap.add_argument("--feature-csv", default=str(P("feature_dataset_csv")))
    ap.add_argument(
        "--models-dir",
        default=str(P("magpy_validation_dir") / "models"),
    )
    ap.add_argument(
        "--hsmm-params",
        default=str(project_root() / "Data" / "Eval_outputs" / "hsmm_params.json"),
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-models", type=int, default=0)
    ap.add_argument("--skip-score", action="store_true",
                    help="Reuse existing predictions/sepval_lstm_score/")
    ap.add_argument("--pad-hours", type=int, default=48)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    feat_dir = root / "features"
    pred_dir = root / "predictions"
    metrics_dir = root / "metrics"
    json_dir = root / "scoreboard_json"
    for d in (feat_dir, pred_dir, metrics_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("[sepval] loading case list …")
    cases = load_sepval_cases(Path(args.testing_csv))
    print(f"[sepval] cases={len(cases)} SEP={int((cases.y==1).sum())} non={int((cases.y==0).sum())}")
    cases.to_csv(feat_dir / "sepval_cases.csv", index=False)

    feat_path = feat_dir / "sepval_features_1h.csv"
    if not feat_path.is_file():
        print("[sepval] extracting feature hours from archive …")
        feat = extract_feature_hours(Path(args.feature_csv), cases, pad_hours=int(args.pad_hours))
        feat.to_csv(feat_path, index=False)
        print(f"[sepval] wrote {len(feat):,} feature rows → {feat_path}")
    else:
        feat = pd.read_csv(feat_path, low_memory=False)
        feat["window_begin"] = _parse_ts(feat["window_begin"])
        feat["window_end"] = _parse_ts(feat["window_end"])
        print(f"[sepval] reused features n={len(feat):,}")

    score_out = pred_dir / "sepval_lstm_score"
    ens_path = score_out / "mtm_lstm_fulltrain_train1h_ensemble.csv"
    if args.skip_score and ens_path.is_file():
        print(f"[sepval] skip-score; using {ens_path}")
    else:
        print("[sepval] scoring MagPy-cutoff LSTM ensemble …")
        ens_path = score_lstm(
            feat_path,
            Path(args.models_dir),
            score_out,
            device=args.device,
            max_models=int(args.max_models),
        )

    ens = pd.read_csv(ens_path, low_memory=False)
    print("[sepval] applying HSMM …")
    hsmm_params = Path(args.hsmm_params)
    if not hsmm_params.is_file():
        # fallback
        alt = P("magpy_validation_dir").parent / "Eval_outputs" / "hsmm_params.json"
        hsmm_params = alt if alt.is_file() else hsmm_params
    ens = apply_hsmm(ens, feat, hsmm_params)
    hsmm_out = pred_dir / "sepval_ensemble_hsmm.csv"
    ens.to_csv(hsmm_out, index=False)
    print(f"[sepval] wrote {hsmm_out}")

    print("[sepval] case-level metrics …")
    case_tbl = case_level_table(cases, ens)
    case_tbl.to_csv(metrics_dir / "sepval_case_predictions.csv", index=False)
    ok = case_tbl[case_tbl["matched"] == True]  # noqa: E712
    rows = []
    for name, thr_col, thr in [
        ("SEPNET-PRISM-R LSTM", "lstm_warn", LSTM_THR),
        ("SEPNET-PRISM-R HSMM", "hsmm_warn", HSMM_THR),
    ]:
        m = contingency(ok["y"].to_numpy(), ok[thr_col].to_numpy())
        pcol = "lstm_prob" if "LSTM" in name else "hsmm_prob"
        m.update(
            model=name,
            threshold=thr,
            AUC=auc_score(ok["y"].to_numpy(), ok[pcol].to_numpy()),
            source="this run (MagPy-cutoff ensemble + operational HSMM)",
        )
        rows.append(m)
    our = pd.DataFrame(rows)
    our.to_csv(metrics_dir / "sepval_case_metrics.csv", index=False)

    # Published comparison
    pub_rows = []
    for model, hits, fa, cn, miss, n, pod, far, bias, tss, hss, note in PUBLISHED:
        pub_rows.append(dict(
            model=model, threshold=np.nan, n=n, TP=hits, FP=fa, TN=cn, FN=miss,
            ACC=(hits + cn) / n if n else np.nan, POD=pod, FAR=far, Bias=bias,
            TSS=tss, HSS=hss, AUC=np.nan, source=note,
        ))
    # Classic ML from SEPVAL-Othergeneralmodels (SHARP+Flare+CME RF/XGB etc.)
    classic = [
        ("LR (SHARP+Flare+CME)", 9, 8, 21, 24, 62, 0.2727, 0.4706, -0.0031, -0.0030, 0.6144),
        ("SVM (SHARP+Flare+CME)", 9, 7, 22, 24, 62, 0.2727, 0.4375, 0.0313, 0.0303, 0.5653),
        ("RF (SHARP+Flare+CME)", 16, 6, 23, 17, 62, 0.4848, 0.2727, 0.2780, 0.2717, 0.6850),
        ("XGB (SHARP+Flare+CME)", 16, 9, 20, 17, 62, 0.4848, 0.3600, 0.1745, 0.1716, 0.6646),
    ]
    # Fix classic: from confusion [[TN FP],[FN TP]] in the txt
    # LM: [[21 8],[24 9]] → TN=21 FP=8 FN=24 TP=9
    classic_fixed = [
        ("LR (SHARP+Flare+CME)", 9, 8, 21, 24, 0.6144),
        ("SVM (SHARP+Flare+CME)", 9, 7, 22, 24, 0.5653),
        ("RF (SHARP+Flare+CME)", 16, 6, 23, 17, 0.6850),
        ("XGB (SHARP+Flare+CME)", 16, 9, 20, 17, 0.6646),
    ]
    for model, tp, fp, tn, fn, auc in classic_fixed:
        m = contingency(
            np.array([1] * (tp + fn) + [0] * (tn + fp)),
            np.array([1] * tp + [0] * fn + [0] * tn + [1] * fp),
        )
        # recompute properly from counts
        m = dict(n=tp+fp+tn+fn, TP=tp, FP=fp, TN=tn, FN=fn)
        m["POD"] = tp / (tp + fn) if (tp + fn) else np.nan
        m["FAR"] = fp / (tp + fp) if (tp + fp) else np.nan
        m["ACC"] = (tp + tn) / m["n"]
        m["Bias"] = (tp + fp) / (tp + fn) if (tp + fn) else np.nan
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        m["TSS"] = m["POD"] - fpr
        pc = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (m["n"] ** 2)
        m["HSS"] = (m["ACC"] - pc) / (1 - pc) if pc != 1 else np.nan
        m.update(model=model, threshold=np.nan, AUC=auc,
                 source="ML-CMESEP SEPVAL-Othergeneralmodels.txt")
        pub_rows.append(m)

    cmp = pd.concat([our, pd.DataFrame(pub_rows)], ignore_index=True)
    col_order = ["model", "threshold", "n", "TP", "FP", "TN", "FN", "ACC", "POD", "FAR",
                 "Bias", "TSS", "HSS", "AUC", "source"]
    cmp = cmp[[c for c in col_order if c in cmp.columns]]
    cmp.to_csv(metrics_dir / "sepval_vs_published_models.csv", index=False)

    print("[sepval] exporting Scoreboard JSON …")
    n_json = export_json(ens, cases, json_dir)
    (json_dir / "export_manifest.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_files": n_json,
        "n_cases": len(cases),
        "mode": "simulated_realtime_forecast",
        "probability": "hsmm_prob",
        "all_clear_threshold": HSMM_THR,
        "models": "MagPy-cutoff 50-split MTM-LSTM + operational hsmm_params.json",
        "note": "SEPVAL Phase III case windows (hour-floored). Train cutoff <2023-07-01; "
                "SEPVAL cases ≤2023-03 so MagPy-period holdout is respected, but pre-2023 "
                "SEPVAL hours may overlap MagPy-cutoff training data (not leave-SEPVAL-out).",
    }, indent=2) + "\n")

    # Markdown summary
    hsmm_row = our[our["model"].str.contains("HSMM")].iloc[0]
    magpy = next(r for r in pub_rows if r["model"] == "MagPy")
    md = f"""# SEPVAL Phase III — SEPNET-PRISM-R results

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Setup

- Cases: **{len(ok)}** (SEP={int((ok.y==1).sum())}, non={int((ok.y==0).sum())}) from Phase III list
- Model: MagPy-cutoff MTM-LSTM ensemble (train `window_end < 2023-07-01`) + operational HSMM
- Decision: one forecast per case at hour-floored `window_begin`
- HSMM all-clear threshold: **{HSMM_THR}** (warn if `hsmm_prob ≥ {HSMM_THR}`)

## Our criteria (case-level)

| Model | N | TP | FP | TN | FN | POD | FAR | TSS | HSS | AUC |
|-------|---|----|----|----|----|-----|-----|-----|-----|-----|
| HSMM | {int(hsmm_row.n)} | {int(hsmm_row.TP)} | {int(hsmm_row.FP)} | {int(hsmm_row.TN)} | {int(hsmm_row.FN)} | {hsmm_row.POD:.3f} | {hsmm_row.FAR:.3f} | {hsmm_row.TSS:.3f} | {hsmm_row.HSS:.3f} | {hsmm_row.AUC:.3f} |
| LSTM thr=0.5 | {int(our.iloc[0].n)} | {int(our.iloc[0].TP)} | {int(our.iloc[0].FP)} | {int(our.iloc[0].TN)} | {int(our.iloc[0].FN)} | {our.iloc[0].POD:.3f} | {our.iloc[0].FAR:.3f} | {our.iloc[0].TSS:.3f} | {our.iloc[0].HSS:.3f} | {our.iloc[0].AUC:.3f} |

## vs MagPy (SEPVAL poster SPHINX all-clear)

| | HSMM (ours) | MagPy (published) |
|--|-------------|-------------------|
| Hits / Misses | {int(hsmm_row.TP)} / {int(hsmm_row.FN)} | {magpy['TP']} / {magpy['FN']} |
| FA / CN | {int(hsmm_row.FP)} / {int(hsmm_row.TN)} | {magpy['FP']} / {magpy['TN']} |
| POD | {hsmm_row.POD:.2f} | {magpy['POD']:.2f} |
| FAR | {hsmm_row.FAR:.2f} | {magpy['FAR']:.2f} |
| TSS | **{hsmm_row.TSS:.2f}** | {magpy['TSS']:.2f} |
| HSS | **{hsmm_row.HSS:.2f}** | {magpy['HSS']:.2f} |

Full table: `metrics/sepval_vs_published_models.csv`  
Scoreboard JSON: `scoreboard_json/` ({n_json} files)

**Caveat:** MagPy-cutoff training includes pre-2023-07 data that can overlap SEPVAL case hours (not a strict leave-SEPVAL-out retrain). Official SPHINX matching may differ from this one-decision-per-case protocol.
"""
    (metrics_dir / "sepval_summary.md").write_text(md)
    print(md)
    print(f"[sepval] done → {root}")


if __name__ == "__main__":
    main()
