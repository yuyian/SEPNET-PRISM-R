#!/usr/bin/env python3
"""SHAP feature-importance analysis for the FULL-TRAIN MTM-LSTM ensemble.

Explains the estimated SEP-event probability (mtm_lstm_cls_prob) in terms of the
input features used by train_save_mtm_lstm_fulltrain.py (SHARP_* and Flare_*
predictors, minus the AR/excluded columns). Tree-teacher meta channels
(xgb_meta_prob / rf_meta_prob) are still passed to the model when required for
inference, but are excluded from SHAP importance outputs.

For each saved split model we:
  * rebuild the exact model input (artifact scaler -> mm01 -> tree meta features ->
    MTM sequences), identical to eval_saved_mtm_lstm_models_fulltrain.py,
  * wrap the model so it returns the calibrated classification probability
    p = sigmoid(logits / T),
  * run shap.GradientExplainer (background sample vs. eval sample) to get per-feature,
    per-timestep SHAP values, and
  * reduce to mean |SHAP| per feature (importance) and mean signed SHAP (direction).

Outputs (in --out-dir, default ``Data/Eval_outputs/SHAP`` from paths.json):
  * shap_feature_importance_ensemble.csv       (global mean across split models, ranked)
  * shap_feature_importance_by_model.csv       (global, per split model)
  * shap_feature_importance_by_forecast_time.csv  (per window_begin, ensemble mean |SHAP|)
  * shap_feature_importance_by_forecast_time_by_model.csv  (per window_begin, per model)
  * shap_top_features_by_forecast_time.csv     (top-K features at each forecast time)
  * shap_summary_bar.png                       (global top-K bar chart)

All paths (models dir, model glob, input CSV, output dir) default to the values in
``Utils/paths.json`` resolved through ``paths_config``; override any of them via CLI flags.
The default input file is the NRT 1h rolling CSV (the source of
mtm_lstm_fulltrain_nrt1h_predictions.csv); use --input-csv to explain a different scored set.

Default run (no CLI flags needed)::

  python Utils/Step2_SHAP_postanalysis.py

explains every forecast time in the NRT 1h CSV across all saved split models.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Avoid matplotlib cache warnings when ~/.matplotlib is not writable (e.g. sandbox/CI).
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P, S  # noqa: E402  (central path config -> Utils/paths.json)

# Optional extra module dir (e.g. where multitask_model lives) from Utils/paths.json.
try:
    _pycode = S("pycode_dir", "").strip()
    if _pycode:
        _pycode = str(Path(_pycode).expanduser().resolve())
        if _pycode not in sys.path:
            sys.path.insert(0, _pycode)
except Exception:
    pass

from multitask_model import (  # noqa: E402
    MultiTaskModel_HybridLSTMTabular,
    build_mtm_sequences,
    _select_transform_features,
)

torch.serialization.add_safe_globals([MultiTaskModel_HybridLSTMTabular])

# Meta channels required by some saved models, but not reported in SHAP rankings.
_SHAP_EXCLUDE_FEATURES = frozenset({"xgb_meta_prob", "rf_meta_prob"})


def mm01(a):
    return np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)


def tabular_finite(a):
    arr = np.asarray(a, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def parse_ts_mixed(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


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


def build_feature_frame(art: dict, df_filtered: pd.DataFrame, scaler, predictors):
    """Replicate the training/eval feature construction; returns (X_scaled_df, feature_names)."""
    X = df_filtered[predictors]
    X_scaled = pd.DataFrame(mm01(scaler.transform(X)), index=X.index, columns=X.columns)

    feature_names = list(predictors)
    if bool(art.get("tree_use_xgb_meta_feature", False)):
        backend_used = str(art.get("tree_backend_used", "xgb"))
        xgb_teacher = art.get("tree_xgb_teacher_model")
        rf_teacher = art.get("tree_rf_teacher_model")
        need_xgb = backend_used in ("both", "xgb")
        need_rf = backend_used in ("both", "rf")
        if (need_xgb and xgb_teacher is None) or (need_rf and rf_teacher is None):
            raise RuntimeError(
                "Model uses tree meta-features but teacher model(s) missing from artifact. "
                "Run inject_tree_teachers_fulltrain.py to repair the .pt files first."
            )
        teacher_in = tabular_finite(mm01(scaler.transform(X)))
        if need_xgb:
            X_scaled["xgb_meta_prob"] = xgb_teacher.predict_proba(teacher_in)[:, 1].astype(np.float64)
            feature_names.append("xgb_meta_prob")
        if need_rf:
            X_scaled["rf_meta_prob"] = rf_teacher.predict_proba(teacher_in)[:, 1].astype(np.float64)
            feature_names.append("rf_meta_prob")
    return X_scaled, feature_names


class ClsProbWrapper(nn.Module):
    """Maps model input -> calibrated SEP-event probability p = sigmoid(logits / T)."""

    def __init__(self, model: nn.Module, temperature: float):
        super().__init__()
        self.model = model
        self.T = float(max(temperature, 1e-6))

    def forward(self, x):
        out = self.model(x)
        logits = out[1]
        logits = logits.reshape(x.shape[0], -1)[:, :1]
        return torch.sigmoid(logits / self.T)


def _filter_sequences_by_time(
    target_ix: list,
    df_filtered: pd.DataFrame,
    wcol_begin: str,
    wcol_end: str,
    time_start,
    time_end,
) -> np.ndarray:
    """Boolean mask over sequence index 0..N-1 for forecast times in [start, end]."""
    n = len(target_ix)
    if time_start is None and time_end is None:
        return np.ones(n, dtype=bool)
    # Use positional arrays — do not align on df index (train CSV has sparse indices).
    wb = pd.to_datetime(df_filtered.loc[target_ix, wcol_begin], utc=True)
    wb_vals = wb.to_numpy(dtype="datetime64[ns]")
    mask = np.ones(n, dtype=bool)
    if time_start is not None:
        t0 = np.datetime64(pd.to_datetime(time_start, utc=True).to_datetime64())
        mask &= wb_vals >= t0
    if time_end is not None:
        t1 = np.datetime64(pd.to_datetime(time_end, utc=True).to_datetime64())
        mask &= wb_vals <= t1
    return mask


def _subsample_eligible(
    eligible: np.ndarray,
    target_ix: list,
    df_filtered: pd.DataFrame,
    wcol_begin: str,
    *,
    stride_hours: float | None,
    max_forecast_times: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Thin eligible sequence indices for large training-set SHAP-over-time runs."""
    out = np.asarray(eligible, dtype=int)
    if out.size == 0:
        return out

    if stride_hours is not None and float(stride_hours) > 0:
        wb = pd.to_datetime(df_filtered.loc[[target_ix[int(i)] for i in out], wcol_begin], utc=True)
        # Keep the first forecast time in each stride bucket (UTC epoch hours).
        epoch_s = wb.astype("int64") // 10**9
        bucket = (epoch_s.to_numpy(dtype=np.int64) // int(float(stride_hours) * 3600))
        keep_pos = []
        seen = set()
        for pos, b in enumerate(bucket):
            if int(b) in seen:
                continue
            seen.add(int(b))
            keep_pos.append(pos)
        out = out[np.asarray(keep_pos, dtype=int)]

    if max_forecast_times is not None and int(max_forecast_times) > 0 and out.size > int(max_forecast_times):
        # Evenly spaced over time (not random) so over-time plots stay representative.
        idx = np.linspace(0, out.size - 1, int(max_forecast_times))
        out = out[np.unique(np.round(idx).astype(int))]

    return out


def shap_for_model(
    art, df_filtered, *, predictors, y_reg_cols, y_cls_col, seq_len, lag_h, lag_tol,
    wcol_begin, wcol_end, device, background_size, eval_size, rng,
    forecast_time_start=None, forecast_time_end=None, all_forecast_times=False,
    stride_hours=None, max_forecast_times=None,
):
    model = art["model"].to(device).eval()
    selector = art["selector"]
    scaler = art["scaler"]
    temperature = float(getattr(model, "_cls_temperature", art.get("cls_temperature", 1.0)))

    X_scaled, feature_names = build_feature_frame(art, df_filtered, scaler, predictors)
    y_reg = df_filtered[y_reg_cols]
    y_cls = df_filtered[[y_cls_col]]

    order = df_filtered.loc[X_scaled.index, wcol_begin].sort_values().index
    ts = pd.to_datetime(df_filtered.loc[X_scaled.index, wcol_begin], utc=True)
    seq_kw = dict(time_series=ts, lag_hours=lag_h, lag_tolerance_hours=lag_tol, pad_missing="nan")

    X_np, _, _, target_ix = build_mtm_sequences(
        order, X_scaled, y_reg, y_cls, list(X_scaled.index), set(X_scaled.index), seq_len,
        return_target_ix=True, stride=1, **seq_kw,
    )
    target_ix = list(target_ix)
    X_sel = _select_transform_features(np.asarray(X_np), selector)  # (N, L, F_sel)
    feat_names_sel = feature_names
    if hasattr(selector, "get_support"):
        try:
            support = np.asarray(selector.get_support())
            feat_names_sel = [f for f, keep in zip(feature_names, support) if keep]
        except Exception:
            pass

    n = X_sel.shape[0]
    if n == 0:
        raise RuntimeError("No sequences built for SHAP (empty filtered frame).")

    time_mask = _filter_sequences_by_time(
        target_ix, df_filtered, wcol_begin, wcol_end,
        forecast_time_start, forecast_time_end,
    )
    eligible = np.flatnonzero(time_mask)
    if eligible.size == 0:
        raise RuntimeError("No sequences in the requested forecast-time window.")

    # Background can use the full eligible pool; eval set may be thinned for train-scale runs.
    bg_pool = eligible
    bg_k = min(int(background_size), bg_pool.size)
    bg_idx = rng.choice(bg_pool, size=bg_k, replace=False)

    if all_forecast_times or eval_size is None or int(eval_size) <= 0:
        ev_idx = _subsample_eligible(
            eligible, target_ix, df_filtered, wcol_begin,
            stride_hours=stride_hours, max_forecast_times=max_forecast_times, rng=rng,
        )
    else:
        ev_k = min(int(eval_size), eligible.size)
        ev_idx = rng.choice(eligible, size=ev_k, replace=False)

    bg = torch.tensor(X_sel[bg_idx], dtype=torch.float32, device=device)
    ev = torch.tensor(X_sel[ev_idx], dtype=torch.float32, device=device)

    import shap

    wrapper = ClsProbWrapper(model, temperature).to(device).eval()
    explainer = shap.GradientExplainer(wrapper, bg)
    sv = explainer.shap_values(ev)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 4:  # (N, L, F, n_outputs)
        sv = sv[..., 0]

    # Per forecast time: aggregate SHAP over sequence lags (LSTM output uses full sequence).
    abs_per_sample = np.mean(np.abs(sv), axis=1)   # (N_ev, F_sel)
    signed_per_sample = np.mean(sv, axis=1)

    keep = np.array(
        [nm not in _SHAP_EXCLUDE_FEATURES for nm in feat_names_sel], dtype=bool
    )
    feat_names_sel = [nm for nm, k in zip(feat_names_sel, keep) if k]
    abs_per_sample = abs_per_sample[:, keep]
    signed_per_sample = signed_per_sample[:, keep]

    abs_per_feat = np.mean(abs_per_sample, axis=0)
    signed_per_feat = np.mean(signed_per_sample, axis=0)

    ev_target_ix = [target_ix[int(i)] for i in ev_idx]
    meta = pd.DataFrame({
        "row_index": ev_target_ix,
        "window_begin": df_filtered.loc[ev_target_ix, wcol_begin].to_numpy(),
        "window_end": df_filtered.loc[ev_target_ix, wcol_end].to_numpy(),
    })
    meta["window_begin"] = parse_ts_mixed(meta["window_begin"])
    meta["window_end"] = parse_ts_mixed(meta["window_end"])

    per_time_rows = []
    for j in range(abs_per_sample.shape[0]):
        wb = meta.iloc[j]["window_begin"]
        we = meta.iloc[j]["window_end"]
        for f_i, nm in enumerate(feat_names_sel):
            per_time_rows.append({
                "row_index": int(meta.iloc[j]["row_index"]),
                "window_begin": wb,
                "window_end": we,
                "feature": nm,
                "abs_shap": float(abs_per_sample[j, f_i]),
                "signed_shap": float(signed_per_sample[j, f_i]),
            })

    return feat_names_sel, abs_per_feat, signed_per_feat, int(len(ev_idx)), per_time_rows


def _load_feature_csv(
    path: Path,
    *,
    wcol_begin: str = "window_begin",
    time_start=None,
    time_end=None,
    pad_hours: float = 72.0,
) -> pd.DataFrame:
    """Load a rolling feature CSV; optionally restrict to [start−pad, end+pad]."""
    path = Path(path).expanduser().resolve()
    if time_start is None and time_end is None:
        print(f"[shap] reading full table {path}")
        return pd.read_csv(path, low_memory=False)

    t0 = pd.to_datetime(time_start, utc=True) - pd.Timedelta(hours=float(pad_hours)) if time_start is not None else None
    t1 = pd.to_datetime(time_end, utc=True) + pd.Timedelta(hours=float(pad_hours)) if time_end is not None else None
    print(f"[shap] chunk-loading {path} for window "
          f"{t0} → {t1} (pad={pad_hours}h around requested forecast times)")
    chunks: list[pd.DataFrame] = []
    for ch in pd.read_csv(path, chunksize=80_000, low_memory=False):
        if wcol_begin not in ch.columns:
            raise KeyError(f"{path} missing {wcol_begin}")
        wb = parse_ts_mixed(ch[wcol_begin])
        m = wb.notna()
        if t0 is not None:
            m &= wb >= t0
        if t1 is not None:
            m &= wb <= t1
        if m.any():
            chunks.append(ch.loc[m])
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    print(f"[shap] loaded {len(out):,} rows in padded window")
    return out


def main():
    ap = argparse.ArgumentParser(description="SHAP feature importance for full-train MTM-LSTM SEP probability.")
    # Path defaults come from Utils/paths.json via paths_config (P/S).
    ap.add_argument("--models-dir", default=str(P("models_dir")))
    ap.add_argument("--model-glob", default=S("model_glob"))
    ap.add_argument("--input-csv", default=str(P("nrt_rolling_csv")),
                    help="CSV scored for SHAP (default = NRT 1h, source of mtm_lstm_fulltrain_nrt1h_predictions.csv).")
    ap.add_argument("--out-dir", default=str(P("shap_outputs_dir")),
                    help="Output dir for SHAP CSVs/plot (default: Data/Eval_outputs/SHAP from paths.json)")
    ap.add_argument("--background-size", type=int, default=100)
    ap.add_argument("--eval-size", type=int, default=0,
                    help="Random forecast-time subsample when --no-all-forecast-times (default 0 = unused).")
    ap.add_argument("--all-forecast-times", action=argparse.BooleanOptionalAction, default=True,
                    help="Explain every forecast time in the input CSV (default: on). "
                         "Use --no-all-forecast-times --eval-size N for a random subsample.")
    ap.add_argument("--forecast-time-start", default=None,
                    help="UTC inclusive start for window_begin filter, e.g. 2025-09-10 00:00:00+00:00")
    ap.add_argument("--forecast-time-end", default=None,
                    help="UTC inclusive end for window_begin filter.")
    ap.add_argument("--stride-hours", type=float, default=0,
                    help="When explaining all times, keep one forecast per N-hour bucket (0 = no stride). "
                         "Useful for training-set SHAP-over-time.")
    ap.add_argument("--max-forecast-times", type=int, default=0,
                    help="Cap number of explained forecast times after stride (0 = unlimited). "
                         "Evenly spaced over the eligible window.")
    ap.add_argument("--top-k-per-time", type=int, default=10,
                    help="Top features kept per forecast time in shap_top_features_by_forecast_time.csv.")
    ap.add_argument("--max-models", type=int, default=0, help="0 = all matching split models.")
    ap.add_argument("--top-k", type=int, default=25, help="Top features shown in the bar chart.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="SHAP gradient backprop is most stable on CPU for LSTM models.")
    args = ap.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_time_start = (
        pd.to_datetime(args.forecast_time_start, utc=True)
        if args.forecast_time_start else None
    )
    forecast_time_end = (
        pd.to_datetime(args.forecast_time_end, utc=True)
        if args.forecast_time_end else None
    )
    all_forecast_times = bool(args.all_forecast_times)

    model_files = sorted(models_dir.glob(args.model_glob))
    if not model_files:
        raise FileNotFoundError(f"No model files matching {args.model_glob} in {models_dir}")
    if args.max_models and args.max_models > 0:
        model_files = model_files[: int(args.max_models)]

    # Peek first artifact for window column + lag pad so we can time-slice large train CSVs.
    art0 = torch.load(model_files[0], map_location="cpu", weights_only=False)
    wcol0 = str(art0["wcol_begin"])
    pad_h = float(art0.get("seq_lag_hours", 24.0)) * max(int(art0.get("seq_len", 2)), 1) + float(
        art0.get("seq_lag_tolerance_hours", 3.0)
    ) + 24.0
    df_in = _load_feature_csv(
        Path(args.input_csv),
        wcol_begin=wcol0,
        time_start=forecast_time_start,
        time_end=forecast_time_end,
        pad_hours=pad_h,
    )
    if df_in.empty:
        raise RuntimeError("Input feature CSV is empty after optional time-window load.")
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    rng = np.random.default_rng(int(args.seed))

    by_model_rows = []
    by_time_by_model_rows = []
    feat_abs_acc: dict[str, list[float]] = {}
    feat_signed_acc: dict[str, list[float]] = {}

    for mi, mf in enumerate(model_files):
        art = torch.load(mf, map_location="cpu", weights_only=False)
        predictors = list(art["predictors"])
        y_reg_cols = list(art["y_reg_cols"])
        y_cls_col = str(art["y_cls_col"])
        row_filter = dict(art["row_filter"])
        seq_len = int(art["seq_len"])
        lag_h = float(art["seq_lag_hours"])
        lag_tol = float(art["seq_lag_tolerance_hours"])
        wcol_begin = str(art["wcol_begin"])
        wcol_end = str(art["wcol_end"])

        f_in = filter_rows(df_in, row_filter, wcol_begin, wcol_end)
        names, abs_imp, signed_imp, n_eval, per_time_rows = shap_for_model(
            art, f_in, predictors=predictors, y_reg_cols=y_reg_cols, y_cls_col=y_cls_col,
            seq_len=seq_len, lag_h=lag_h, lag_tol=lag_tol, wcol_begin=wcol_begin,
            wcol_end=wcol_end, device=device, background_size=args.background_size,
            eval_size=args.eval_size, rng=rng,
            forecast_time_start=forecast_time_start, forecast_time_end=forecast_time_end,
            all_forecast_times=all_forecast_times,
            stride_hours=(float(args.stride_hours) if float(args.stride_hours) > 0 else None),
            max_forecast_times=(int(args.max_forecast_times) if int(args.max_forecast_times) > 0 else None),
        )
        for nm, a, s in zip(names, abs_imp, signed_imp):
            by_model_rows.append({
                "model_file": mf.name, "split_seed": art.get("split_seed"),
                "feature": nm, "mean_abs_shap": float(a), "mean_signed_shap": float(s),
            })
            feat_abs_acc.setdefault(nm, []).append(float(a))
            feat_signed_acc.setdefault(nm, []).append(float(s))
        for row in per_time_rows:
            by_time_by_model_rows.append({
                "model_file": mf.name,
                "split_seed": art.get("split_seed"),
                **row,
            })
        print(f"[{mi + 1}/{len(model_files)}] {mf.name} | n_forecast_times={n_eval} | features={len(names)}")

    by_model = pd.DataFrame(by_model_rows)
    ens = (
        pd.DataFrame({
            "feature": list(feat_abs_acc.keys()),
            "mean_abs_shap": [float(np.mean(v)) for v in feat_abs_acc.values()],
            "std_abs_shap_across_models": [float(np.std(v)) for v in feat_abs_acc.values()],
            "mean_signed_shap": [float(np.mean(feat_signed_acc[k])) for k in feat_abs_acc],
            "n_models": [len(v) for v in feat_abs_acc.values()],
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ens.insert(0, "rank", np.arange(1, len(ens) + 1))

    p_ens = out_dir / "shap_feature_importance_ensemble.csv"
    p_by = out_dir / "shap_feature_importance_by_model.csv"
    ens.to_csv(p_ens, index=False)
    by_model.to_csv(p_by, index=False)
    print(f"Wrote {p_ens} ({len(ens)} features)")
    print(f"Wrote {p_by} ({len(by_model)} rows)")

    # --- Per forecast time (window_begin) -----------------------------------
    by_time_by_model = pd.DataFrame(by_time_by_model_rows)
    if not by_time_by_model.empty:
        grp = by_time_by_model.groupby(
            ["window_begin", "window_end", "row_index", "feature"], as_index=False
        ).agg(
            mean_abs_shap=("abs_shap", "mean"),
            std_abs_shap_across_models=("abs_shap", "std"),
            mean_signed_shap=("signed_shap", "mean"),
            n_models=("model_file", "nunique"),
        )
        grp = grp.sort_values(
            ["window_begin", "mean_abs_shap"], ascending=[True, False]
        ).reset_index(drop=True)
        grp["rank_within_time"] = grp.groupby("window_begin").cumcount() + 1

        p_time = out_dir / "shap_feature_importance_by_forecast_time.csv"
        grp.to_csv(p_time, index=False)
        print(f"Wrote {p_time} ({len(grp)} rows; {grp['window_begin'].nunique()} forecast times)")

        p_time_by = out_dir / "shap_feature_importance_by_forecast_time_by_model.csv"
        by_time_by_model.to_csv(p_time_by, index=False)
        print(f"Wrote {p_time_by} ({len(by_time_by_model)} rows)")

        top_k = int(args.top_k_per_time)
        top_time = grp[grp["rank_within_time"] <= top_k].copy()
        p_top_time = out_dir / "shap_top_features_by_forecast_time.csv"
        top_time.to_csv(p_top_time, index=False)
        print(f"Wrote {p_top_time} ({len(top_time)} rows; top {top_k} per forecast time)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = ens.head(int(args.top_k)).iloc[::-1]
        fig, axx = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
        colors = ["#c0392b" if s >= 0 else "#2471a3" for s in top["mean_signed_shap"]]
        axx.barh(top["feature"], top["mean_abs_shap"], color=colors)
        axx.set_xlabel("Mean |SHAP|  (contribution to SEP-event probability)")
        axx.set_title(f"Top {len(top)} feature importance — full-train MTM-LSTM ensemble\n"
                      f"(red = pushes prob up on average, blue = down)")
        fig.tight_layout()
        p_png = out_dir / "shap_summary_bar.png"
        fig.savefig(p_png, dpi=150)
        plt.close(fig)
        print(f"Wrote {p_png}")
    except Exception as e:  # plotting is optional
        print(f"[warn] skipped bar chart: {e}")

    print("\nTop 15 features by mean |SHAP|:")
    print(ens.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
