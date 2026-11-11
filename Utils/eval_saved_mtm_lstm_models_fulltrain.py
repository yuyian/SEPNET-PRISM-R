#!/usr/bin/env python3
"""Scoring helpers for the FULL-TRAIN MTM-LSTM ensemble.

Imported by Step1_fetch_aggregate_eval_nrt.py; provides:
  * score_full_file() -- build per-file MTM sequences and predict across a saved split model
    (classification prob + horizon-aware regression with optional online de-bias, plus XGB/RF
    tree teacher meta-features when the artifact carries them),
  * filter_rows()     -- apply a model's row filter and parse window timestamps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Optional extra module dir (e.g. where multitask_model lives) from Utils/paths.json.
try:
    from paths_config import S as _S

    _pycode = _S("pycode_dir", "").strip()
    if _pycode:
        _pycode = str(Path(_pycode).expanduser().resolve())
        if _pycode not in sys.path:
            sys.path.insert(0, _pycode)
except Exception:
    pass

from multitask_model import (
    MultiTaskModel_HybridLSTMTabular,
    build_mtm_sequences,
    evaluate_model,
    predictions_to_dataframe,
)

# PyTorch 2.6+ defaults torch.load(..., weights_only=True). Saved artifacts here are full
# pickles (model + sklearn scaler + optional tree teachers), so we disable that restriction.
torch.serialization.add_safe_globals([MultiTaskModel_HybridLSTMTabular])


def mm01(a):
    return np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)


def tabular_finite(a):
    arr = np.asarray(a, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def parse_ts_mixed(s: pd.Series) -> pd.Series:
    """Parse mixed timestamp formats (e.g., YYYY-MM-DD and YYYY-MM-DD HH:MM:SS)."""
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


def add_tree_meta_features(art, X_raw: pd.DataFrame, scaler, X_scaled: pd.DataFrame) -> None:
    """Append XGB/RF teacher meta-probability columns in place (no-op when unused).

    Teacher input matches training exactly: tabular_finite(mm01(scaler.transform(X))).
    Requires the teacher models to be present in the artifact (see
    inject_tree_teachers_fulltrain.py if they are missing).
    """
    if not bool(art.get("tree_use_xgb_meta_feature", False)):
        return
    backend_used = str(art.get("tree_backend_used", "xgb"))
    xgb_teacher_model = art.get("tree_xgb_teacher_model")
    rf_teacher_model = art.get("tree_rf_teacher_model")
    need_xgb = backend_used in ("both", "xgb")
    need_rf = backend_used in ("both", "rf")
    if (need_xgb and xgb_teacher_model is None) or (need_rf and rf_teacher_model is None):
        raise RuntimeError(
            "Model uses tree meta-features but the teacher model(s) are not saved in the "
            "artifact. Run inject_tree_teachers_fulltrain.py first to repair the .pt files."
        )
    teacher_in = tabular_finite(mm01(scaler.transform(X_raw)))
    if need_xgb:
        X_scaled["xgb_meta_prob"] = xgb_teacher_model.predict_proba(teacher_in)[:, 1].astype(np.float64)
    if need_rf:
        X_scaled["rf_meta_prob"] = rf_teacher_model.predict_proba(teacher_in)[:, 1].astype(np.float64)


def score_full_file(
    df_filtered: pd.DataFrame,
    *,
    art: dict,
    model,
    selector,
    scaler,
    predictors,
    y_reg_cols,
    y_cls_col,
    seq_len: int,
    lag_h: float,
    lag_tol: float,
    wcol_begin: str,
    wcol_end: str,
    device,
    threshold: float,
    reg_online_debias: bool = True,
    reg_online_window_days: float = 30.0,
    reg_online_min_samples: int = 3,
) -> pd.DataFrame:
    """Build sequences over the entire file (history == targets == all rows) and predict."""
    X = df_filtered[predictors]
    y_reg = df_filtered[y_reg_cols]
    y_cls = df_filtered[[y_cls_col]]

    X_scaled = pd.DataFrame(mm01(scaler.transform(X)), index=X.index, columns=X.columns)
    add_tree_meta_features(art, X, scaler, X_scaled)

    order = df_filtered.loc[X.index, wcol_begin].sort_values().index
    ts = pd.to_datetime(df_filtered.loc[X.index, wcol_begin], utc=True)
    seq_kw = dict(time_series=ts, lag_hours=lag_h, lag_tolerance_hours=lag_tol, pad_missing="nan")
    hist = set(X.index)

    X_np, y_reg_np, y_cls_np, target_ix = build_mtm_sequences(
        order, X_scaled, y_reg, y_cls, list(X.index), hist, seq_len,
        return_target_ix=True, stride=1, **seq_kw,
    )
    X_t = torch.tensor(X_np, dtype=torch.float32).cpu()
    y_reg_t = torch.tensor(y_reg_np, dtype=torch.float32).cpu()
    y_cls_t = torch.tensor(y_cls_np, dtype=torch.float32).cpu()

    # Per-row forecast issue time (window_begin) and verification time (window_end), aligned to
    # the predicted sequences, so the online de-bias only uses windows verified before each forecast.
    reg_obs_time = reg_verify_time = None
    if reg_online_debias:
        reg_obs_time = pd.to_datetime(df_filtered.loc[target_ix, wcol_begin], utc=True, errors="coerce").to_numpy()
        reg_verify_time = pd.to_datetime(df_filtered.loc[target_ix, wcol_end], utc=True, errors="coerce").to_numpy()

    raw = evaluate_model(
        model, selector, X_t, y_reg_t, y_cls_t, device,
        threshold=threshold, resultsOutput=True, reg_var_names=y_reg_cols,
        reg_online_debias=reg_online_debias,
        reg_obs_time=reg_obs_time,
        reg_verify_time=reg_verify_time,
        reg_online_window_days=reg_online_window_days,
        reg_online_min_samples=reg_online_min_samples,
    )
    df_out = predictions_to_dataframe(
        raw, target_ix, reg_var_names=y_reg_cols, model_prefix="mtm_lstm", threshold_used=threshold
    )
    df_out["mtm_lstm_cls_pred_uncal"] = (df_out["mtm_lstm_cls_prob"] >= 0.5).astype(int)
    meta = df_filtered.loc[df_out.index, [wcol_begin, wcol_end]].copy()
    df_out = df_out.join(meta, how="left")
    return df_out


def filter_rows(df: pd.DataFrame, row_filter: dict, wcol_begin: str, wcol_end: str) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for c, v in row_filter.items():
        if c in df.columns:
            mask &= df[c] == v
    out = df.loc[mask].copy()
    out[wcol_begin] = parse_ts_mixed(out[wcol_begin])
    out[wcol_end] = parse_ts_mixed(out[wcol_end])
    out = out[out[wcol_end].notna()].copy()
    return out
