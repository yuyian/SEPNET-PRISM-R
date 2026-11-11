#!/usr/bin/env python3
"""HSMM smoother (PREDICT-only) for the SEPNET-PRISM full-train predictions.

Loads the PRE-TRAINED parameters from ``hsmm_params.json`` and applies the forward-backward
smoother to a Data_test file (e.g. ``nrt_Data_test_1h.csv``), producing ``hsmm_prob`` and
``hsmm_warn`` columns. No fitting or tuning happens here -- ``predict_hsmm`` is the entry point
imported by ``Step1_fetch_aggregate_eval_nrt.py``.

Covariates (per utils.R): standardized logit(LSTM prob), log1p(proton flux), log1p(XRS-B),
and optionally log1p(CME speed).

The training/tuning code that produced ``hsmm_params.json`` was removed once the parameters were
saved (the operational pipeline no longer needs ``Data_train.csv``); recover it from version
control if a re-fit is ever required.

Usage:
    python Utils/hsmm_analysis.py --test-csv Data/nrt_Data_test_1h.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P  # noqa: E402  (central path config -> Utils/paths.json)


# ---------------------------------------------------------------------------
# Config (defaults copied from hsmm_analysis.R)
# ---------------------------------------------------------------------------
@dataclass
class HsmmConfig:
    feature_file: str = "sharp_flare"
    prob_col: str = "mtm_lstm_cls_prob"
    proton_col: str = "ProtonFlux_max_pfu_1h_before_end"
    xrs_col: str = "XRSB_max_1h_before_end"
    cme_col: Optional[str] = None  # HSMM_CME_COL <- NULL
    truth_col: str = "mtm_lstm_cls_true"
    time_col: str = "window_end"
    max_duration: int = 72

    tune_objective: str = "tp_fp_target"
    weight_grid: tuple = (0.44, 0.46, 0.48, 0.50, 0.52)  # seq(0.44, 0.52, 0.02)
    tp_target_test: float = 150.0
    fp_target_test: float = 200.0
    train_tp_scale: float = 0.015
    train_fp_scale: float = 0.072
    tp_weight: float = 4.0
    fp_weight: float = 1.0
    fp_cap_frac: float = 0.90
    tp_margin: float = 80.0
    w_target: float = 0.46


# ---------------------------------------------------------------------------
# Low-level helpers (ported from utils.R)
# ---------------------------------------------------------------------------
def safe_logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def apply_numeric_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    cols = scaler["cols"]
    x = df[cols].astype(np.float64)
    center = pd.Series(scaler["center"])[cols]
    scale = pd.Series(scaler["scale"])[cols]
    return (x - center) / scale


def hsmm_covariate_frame(df: pd.DataFrame, cfg: HsmmConfig) -> pd.DataFrame:
    """.hsmm_covariate_frame: logit(prob), log1p(proton), log1p(xrs), [log1p(cme)]."""
    out = pd.DataFrame(index=df.index)
    out["x_prob"] = safe_logit(df[cfg.prob_col])
    out["x_pf"] = np.log1p(np.maximum(pd.to_numeric(df[cfg.proton_col], errors="coerce"), 0.0))
    out["x_xrs"] = np.log1p(np.maximum(pd.to_numeric(df[cfg.xrs_col], errors="coerce"), 0.0))
    if cfg.cme_col is not None:
        out["x_cme"] = np.log1p(np.maximum(pd.to_numeric(df[cfg.cme_col], errors="coerce"), 0.0))
    return out


def _scaled_cov_cols(cfg: HsmmConfig) -> list[str]:
    cols = ["x_prob_s", "x_pf_s", "x_xrs_s"]
    if cfg.cme_col is not None:
        cols.append("x_cme_s")
    return cols


def _eta(newd: pd.DataFrame, coef: dict, feature_cols: list[str]) -> np.ndarray:
    """Linear predictor intercept + sum coef*col (missing coef -> 0)."""
    eta = np.full(len(newd), float(coef.get("(Intercept)", 0.0)))
    for c in feature_cols:
        b = float(coef.get(c, 0.0))
        if b != 0.0:
            eta = eta + b * newd[c].to_numpy(dtype=np.float64)
    return eta


def _plogis(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# HSMM forward-backward smoothing
# ---------------------------------------------------------------------------
def predict_hsmm_smoothed_prob(
    df: pd.DataFrame, fit_obj: dict, cfg: HsmmConfig, event_prior_weight: float = 0.0
) -> np.ndarray:
    """Port of predict_hsmm_smoothed_prob (assumes df already sorted by time_col)."""
    n = len(df)
    if n == 0:
        return np.array([], dtype=np.float64)

    raw = hsmm_covariate_frame(df, cfg)
    z = apply_numeric_scaler(raw, fit_obj["scaler"])
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled_cols = [c + "_s" for c in raw.columns]
    z.columns = scaled_cols
    cov_cols = _scaled_cov_cols(cfg)

    r = _plogis(_eta(z, fit_obj["emit_coef"], cov_cols))
    r = np.clip(r, 1e-6, 1 - 1e-6)
    w = float(event_prior_weight) if np.isfinite(event_prior_weight) else 0.0
    p0, p1 = fit_obj["prior"]
    e_q = (1 - r) / (p0 ** w)
    e_ev = r / (p1 ** w)

    a = _plogis(_eta(z, fit_obj["entry_coef"], cov_cols))
    a = np.clip(a, 1e-6, 1 - 1e-6)

    exit_coef = fit_obj["exit_coef"]
    eta_cov = _eta(z, exit_coef, cov_cols)  # excludes dur term
    b_dur = float(exit_coef.get("dur_s", 0.0))
    D = int(fit_obj["D"])
    dc = float(fit_obj["dur_scaler"]["center"]["dur_prev"])
    ds = float(fit_obj["dur_scaler"]["scale"]["dur_prev"])
    if not np.isfinite(ds) or ds <= 0:
        ds = 1.0
    dur_term = b_dur * ((np.arange(1, D + 1, dtype=np.float64) - dc) / ds)

    def clamp(v):
        return np.clip(v, 1e-6, 1 - 1e-6)

    alpha_q = np.zeros(n)
    alpha_e = np.zeros((n, D))
    aq = p0 * e_q[0] * 0.99
    ae = np.zeros(D)
    ae[0] = p1 * e_ev[0] * 0.01
    s1 = aq + ae.sum()
    if s1 <= 0 or not np.isfinite(s1):
        aq, ae, s1 = 0.5, np.full(D, 0.5 / D), 1.0
    alpha_q[0] = aq / s1
    alpha_e[0, :] = ae / s1

    for t in range(1, n):
        h = clamp(_plogis(eta_cov[t] + dur_term))
        prev_q = alpha_q[t - 1]
        prev_e = alpha_e[t - 1, :]
        cont = prev_e * (1 - h)
        new_e = np.zeros(D)
        if D >= 2:
            new_e[1:D] = cont[0:D - 1]
        new_e[D - 1] += cont[D - 1]
        new_e[0] += prev_q * a[t]
        new_q = prev_q * (1 - a[t]) + np.sum(prev_e * h)
        new_q *= e_q[t]
        new_e *= e_ev[t]
        s = new_q + new_e.sum()
        if s <= 0 or not np.isfinite(s):
            new_q, new_e, s = 0.5, np.full(D, 0.5 / D), 1.0
        alpha_q[t] = new_q / s
        alpha_e[t, :] = new_e / s

    beta_q = np.zeros(n)
    beta_e = np.zeros((n, D))
    beta_q[n - 1] = 1.0
    beta_e[n - 1, :] = 1.0
    for t in range(n - 2, -1, -1):
        tt = t + 1
        h = clamp(_plogis(eta_cov[tt] + dur_term))
        a1 = a[tt]
        bq_next = beta_q[tt]
        be_next = beta_e[tt, :]
        bq = (1 - a1) * e_q[tt] * bq_next + a1 * e_ev[tt] * be_next[0]
        cont_beta = np.zeros(D)
        if D >= 2:
            cont_beta[0:D - 1] = be_next[1:D]
        cont_beta[D - 1] = be_next[D - 1]
        be = (1 - h) * e_ev[tt] * cont_beta + h * e_q[tt] * bq_next
        s = bq + be.sum()
        if s <= 0 or not np.isfinite(s):
            bq, be, s = 0.5, np.full(D, 0.5 / D), 1.0
        beta_q[t] = bq / s
        beta_e[t, :] = be / s

    gq = alpha_q * beta_q
    ge = np.sum(alpha_e * beta_e, axis=1)
    denom = gq + ge
    denom[(denom <= 0) | (~np.isfinite(denom))] = 1.0
    return np.clip(ge / denom, 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------------------
# Metrics (used for optional reporting when a truth column is present)
# ---------------------------------------------------------------------------
def calc_cls_metrics(truth, pred, prob=None, positive=1) -> dict:
    t = pd.to_numeric(pd.Series(truth), errors="coerce").to_numpy()
    pr = pd.to_numeric(pd.Series(pred), errors="coerce").to_numpy()
    pos = float(positive)
    TP = float(np.sum((pr == pos) & (t == pos)))
    TN = float(np.sum((pr != pos) & (t != pos)))
    FP = float(np.sum((pr == pos) & (t != pos)))
    FN = float(np.sum((pr != pos) & (t == pos)))

    def sd(num, den):
        return np.nan if den == 0 else num / den

    acc = sd(TP + TN, TP + TN + FP + FN)
    pod = sd(TP, TP + FN)
    precision = sd(TP, TP + FP)
    f1 = np.nan if (np.isnan(precision) or np.isnan(pod) or (precision + pod) == 0) \
        else 2 * precision * pod / (precision + pod)
    fpr = sd(FP, FP + TN)
    far = sd(FP, TP + FP)
    specificity = sd(TN, TN + FP)
    tss = np.nan if (np.isnan(pod) or np.isnan(specificity)) else pod + specificity - 1
    hss_den = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    hss = np.nan if hss_den == 0 else 2 * (TP * TN - FP * FN) / hss_den

    auc_val = np.nan
    if prob is not None:
        p = pd.to_numeric(pd.Series(prob), errors="coerce").to_numpy()
        yb = (t == pos).astype(float)
        ok = np.isfinite(p) & np.isfinite(yb)
        if ok.sum() > 0 and len(np.unique(yb[ok])) == 2 and len(np.unique(p[ok])) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                auc_val = float(roc_auc_score(yb[ok], p[ok]))
            except Exception:
                auc_val = np.nan
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "ACC": acc, "AUC": auc_val,
            "FPR": fpr, "F1": f1, "POD": pod, "FAR": far, "HSS": hss, "TSS": tss}


def add_hsmm_columns(df: pd.DataFrame, fit_obj: dict, cfg: HsmmConfig,
                     event_prior_weight: float, threshold: float) -> pd.DataFrame:
    out = df.sort_values(cfg.time_col, kind="stable").reset_index(drop=True)
    out["hsmm_prob"] = predict_hsmm_smoothed_prob(out, fit_obj, cfg, event_prior_weight)
    out["hsmm_warn"] = (out["hsmm_prob"] >= threshold).astype(int)
    prob = pd.to_numeric(out[cfg.prob_col], errors="coerce").to_numpy()
    th = pd.to_numeric(out["mtm_lstm_threshold_used"], errors="coerce").to_numpy()
    out["mtm_lstm_cls_pred"] = (np.isfinite(prob) & (prob >= th)).astype(int)
    return out


# ---------------------------------------------------------------------------
# PREDICT entry points
# ---------------------------------------------------------------------------
def load_params(params_path: Path) -> tuple[dict, HsmmConfig]:
    params = json.loads(Path(params_path).read_text())
    cfg = HsmmConfig(**params["config"])
    return params, cfg


def predict_hsmm(test_csv: Path, params_path: Path, out_csv: Path) -> pd.DataFrame:
    params, cfg = load_params(params_path)
    fit_obj = params["fit"]
    best_w = float(params["best_w"])
    threshold = float(params["threshold"])

    test_csv = Path(test_csv)
    df = pd.read_csv(test_csv, low_memory=False)
    print(f"[hsmm-predict] {test_csv} rows={len(df):,} | w={best_w:.2f} threshold={threshold:.6f}")

    scored = add_hsmm_columns(df, fit_obj, cfg, best_w, threshold)

    if cfg.truth_col in scored.columns and pd.to_numeric(scored[cfg.truth_col], errors="coerce").notna().any():
        truth = scored[cfg.truth_col]
        if len(pd.unique(pd.to_numeric(truth, errors="coerce").dropna())) == 2:
            m = calc_cls_metrics(truth, scored["hsmm_warn"], scored["hsmm_prob"], positive=1)
            print(f"[hsmm-predict] Test HSMM: TP={m['TP']:.0f} FP={m['FP']:.0f} POD={m['POD']:.3f} "
                  f"FAR={m['FAR']:.3f} TSS={m['TSS']:.3f} AUC={m['AUC']:.3f}")
            b = calc_cls_metrics(truth, scored["mtm_lstm_cls_pred"], scored[cfg.prob_col], positive=1)
            print(f"[hsmm-predict] Test LSTM:  TP={b['TP']:.0f} FP={b['FP']:.0f} POD={b['POD']:.3f} "
                  f"FAR={b['FAR']:.3f} TSS={b['TSS']:.3f}")

    n_warn = int(scored["hsmm_warn"].sum())
    print(f"[hsmm-predict] hsmm_warn=1 in {n_warn}/{len(scored)} windows "
          f"(hsmm_prob range {scored['hsmm_prob'].min():.4f}..{scored['hsmm_prob'].max():.4f})")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_csv, index=False)
    print(f"[hsmm-predict] wrote {out_csv}")
    return scored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    # All path defaults come from Utils/paths.json via paths_config (P).
    p = argparse.ArgumentParser(
        description="HSMM smoother (predict-only): apply saved hsmm_params.json to a Data_test file."
    )
    p.add_argument("--test-csv", default=str(P("data_test_csv")))
    p.add_argument("--params", default=str(P("hsmm_params_json")))
    p.add_argument("--out-csv", default=str(P("hsmm_out_csv")))
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    predict_hsmm(Path(args.test_csv), Path(args.params), Path(args.out_csv))


if __name__ == "__main__":
    main()
