#!/usr/bin/env python3
"""Plot SEPNET-PRISM-R probability/warning with observed proton flux + XRSB.

Python port of ProbsPFXRSB.R, adapted for this repo:

* SEPNET-O is dropped (no SEPNET-O result file exists here) -- only the
  SEPNET-PRISM-R probability line and warning markers are drawn.
* Instead of a separate bottom panel, the observed proton-flux and XRSB curves are
  merged into the SAME figure as the probability/warning, drawn against a secondary
  (right) y-axis.

Left axis  : Probability [0, 1.1]
    - SEPNET-PRISM-R probability line (default: mtm_lstm_cls_prob),
    - SEPNET-PRISM-R warning markers (hsmm_warn == 1) near the top.
Right axis : shared scale for
    - log10(Proton Flux)  [ProtonFlux_max_pfu_1h_before_end, >0],
    - XRSB x 1e4          [XRSB_max_1h_before_end * 1e4],
    with reference lines at 1 (SEP 10 pfu threshold, solid) and 0.1 (dashed).

A second figure (port of PFXRSBPred.R) is also produced: three stacked panels for the
future-max regression targets (Proton Flux, XRSB, CME speed), each showing observed
(red) vs predicted (black) log10 values on the left axis and the SEPNET-PRISM-R
classification probability on the right axis. Written to <figures_dir>/PFXRSBPred.png.

Two SHAP figures are also produced (ports of SHAP_Importance.R and
SHAP_importance_summarized.R), using the per-forecast-time SHAP CSV written by
Step2_SHAP_postanalysis.py:
  * shap_importance_over_time.png -- stacked panels of SEP probability, All SHARP /
    All Flare mass, and the top-N base features over time.
  * shap_importance_summary.png   -- mean |SHAP| importance bars beside a signed-SHAP
    beeswarm colored by feature value.
These are skipped automatically (with a note) if the SHAP CSV is absent, or with --skip-shap.

All paths default to Utils/paths.json via paths_config; input is the HSMM-smoothed
Data_test CSV; outputs land in <figures_dir> (and side-car CSVs in <shap_outputs_dir>).

Usage:
    python Utils/Step3_postanalysis_plots.py
    python Utils/Step3_postanalysis_plots.py --prob-col hsmm_prob
    python Utils/Step3_postanalysis_plots.py --no-smooth-xrs-cme --shap-agg-by-day
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P  # noqa: E402  (central path config -> Utils/paths.json)

# Colors mirroring ProbsPFXRSB.R
_COL_PROB = "black"
_COL_WARN = "red"
_COL_FLUX = "#1046ff"
_COL_XRSB = "#f39c12"
_COL_REF = "#8B2323"  # R "brown4"


def _parse_ts(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


def make_plot(
    df: pd.DataFrame,
    out_png: Path,
    *,
    prob_col: str = "mtm_lstm_cls_prob",
    warn_col: str = "hsmm_warn",
    time_col: str = "window_end",
    proton_col: str = "ProtonFlux_max_pfu_1h_before_end",
    xrs_col: str = "XRSB_max_1h_before_end",
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    df = df.copy()
    df[time_col] = _parse_ts(df[time_col])
    df = df[df[time_col].notna()].sort_values(time_col).reset_index(drop=True)

    prob = pd.to_numeric(df[prob_col], errors="coerce")
    t = df[time_col]

    # Right-axis series: log10(proton flux) and XRSB * 1e4.
    pf = pd.to_numeric(df[proton_col], errors="coerce")
    flux_log10 = np.where(pf > 0, np.log10(pf.where(pf > 0)), np.nan)
    xrsb_1e4 = pd.to_numeric(df[xrs_col], errors="coerce") * 1e4

    fig, ax = plt.subplots(figsize=(12, 5))
    axr = ax.twinx()

    # ---- Left axis: probability + warnings ----------------------------------
    ax.plot(t, prob, color=_COL_PROB, lw=1.2, alpha=0.85, zorder=3,
            label="SEPNET-PRISM-R Probability")

    warn = pd.to_numeric(df.get(warn_col), errors="coerce") if warn_col in df.columns else None
    if warn is not None and (warn == 1).any():
        wt = t[warn == 1]
        ax.scatter(wt, np.full(len(wt), 1.08), marker="x", color=_COL_WARN,
                   s=28, linewidths=1.0, zorder=4, label="SEPNET-PRISM-R Warning")

    ax.set_ylim(-0.02, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Probability")
    ax.set_xlabel("Forecast Start Time (UTC)", labelpad=6)

    # ---- Right axis: proton flux + XRSB (shared scale) -----------------------
    both = np.concatenate([
        flux_log10[np.isfinite(flux_log10)],
        xrsb_1e4.to_numpy()[np.isfinite(xrsb_1e4.to_numpy())],
    ]) if np.isfinite(flux_log10).any() or np.isfinite(xrsb_1e4.to_numpy()).any() else np.array([])

    axr.plot(t, flux_log10, color=_COL_FLUX, lw=1.0, alpha=0.85, zorder=2,
             label=r"$\log_{10}(\mathrm{Proton\ Flux})$")
    axr.plot(t, xrsb_1e4, color=_COL_XRSB, lw=1.0, alpha=0.9, zorder=2,
             label=r"$\mathrm{XRSB}\times 10^{4}$")
    axr.axhline(1.0, color=_COL_REF, lw=0.5, alpha=0.8, ls="-", zorder=1)   # 10 pfu SEP line
    axr.axhline(0.1, color=_COL_REF, lw=0.5, alpha=0.8, ls="--", zorder=1)

    if both.size:
        rmin, rmax = float(np.min(both)), float(np.max(both))
        # Center the right axis on y = 1 (the 10 pfu SEP reference line).
        half = max(1.0 - rmin, rmax - 1.0)
        if half <= 0:
            half = 0.5
        pad = 0.05 * (2 * half)
        axr.set_ylim(1.0 - half - pad, 1.0 + half + pad)
    axr.set_ylabel(r"$\log_{10}(\mathrm{Proton\ Flux})\ /\ \mathrm{XRSB}\times 10^{4}$",
                   color=_COL_REF)
    axr.tick_params(axis="y", colors=_COL_REF)
    axr.spines["right"].set_color(_COL_REF)

    # ---- X axis formatting ---------------------------------------------------
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(90)
        lbl.set_ha("center")
        lbl.set_fontsize(7)

    # ---- Combined legend (both axes) -----------------------------------------
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.42),
              ncol=5, frameon=False, fontsize=8.5, columnspacing=1.2, handlelength=1.8)

    ax.spines["top"].set_visible(False)
    axr.spines["top"].set_visible(False)
    ax.set_title("SEPNET-PRISM-R probability & warning with observed proton flux / XRSB",
                 fontsize=11)

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Second figure: future regression predictions vs observed (port of PFXRSBPred.R)
# ---------------------------------------------------------------------------
# Regression color scheme (mirrors PFXRSBPred.R).
_COL_OBS = "red"
_COL_PRED = "black"
_COL_PROB_REG = "#8c8c8c"   # R "gray55"
_COL_RAXIS = "#666666"      # R "gray40"
_COL_HLINE = "0.7"          # R "gray70"

# Rate-limit hour-to-hour changes for XRSB and CME predictions (not proton flux),
# matching PFXRSBPred.R PRED_MAX_STEP_DEV.
_PRED_MAX_STEP_DEV = {"xrs": 0.05, "cme": 0.05}


def rate_limit_step_changes(x: np.ndarray, max_dev: float) -> np.ndarray:
    """Move at most ``max_dev`` toward the raw value each step (port of R helper)."""
    out = np.asarray(x, dtype=np.float64).copy()
    if out.size < 2 or not np.isfinite(max_dev) or max_dev <= 0:
        return out
    for i in range(1, out.size):
        if not np.isfinite(x[i]) or not np.isfinite(out[i - 1]):
            out[i] = x[i]
            continue
        d = x[i] - out[i - 1]
        out[i] = out[i - 1] + np.sign(d) * max_dev if abs(d) > max_dev else x[i]
    return out


def _panel_limits(y_obs: np.ndarray, y_pred: np.ndarray, pad_frac: float = 0.08):
    vals = np.concatenate([np.asarray(y_obs, float), np.asarray(y_pred, float)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(pad_frac * (hi - lo), 0.1)
    return (lo - pad, hi + pad)


def _reg_panel(ax, t, y_obs, y_pred, prob, *, y_lim, hline_y, ylab_left, show_legend):
    """One regression panel: observed/predicted on left axis, probability on right [0,1]."""
    import matplotlib.pyplot as plt  # noqa: F401  (ensures backend set by caller)

    ax.set_ylim(*y_lim)
    if hline_y is not None and y_lim[0] <= hline_y <= y_lim[1]:
        ax.axhline(hline_y, color=_COL_HLINE, lw=0.5, ls="--", zorder=1)

    ax.plot(t, y_obs, color=_COL_OBS, lw=0.7, alpha=0.9, zorder=3, label="Observed")
    ax.plot(t, y_pred, color=_COL_PRED, lw=0.7, alpha=0.9, zorder=3, label="Predicted")
    ax.set_ylabel(ylab_left, fontsize=9)

    axr = ax.twinx()
    axr.plot(t, prob, color=_COL_PROB_REG, lw=0.7, alpha=0.75, zorder=2,
             label="SEPNET-PRISM-R Probability")
    axr.set_ylim(0.0, 1.0)
    axr.set_yticks(np.arange(0, 1.01, 0.2))
    axr.set_ylabel("SEPNET-PRISM-R Probability", color=_COL_RAXIS, fontsize=9)
    axr.tick_params(axis="y", colors=_COL_RAXIS)
    axr.spines["right"].set_color(_COL_RAXIS)
    ax.spines["top"].set_visible(False)
    axr.spines["top"].set_visible(False)

    if show_legend:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.55),
                  ncol=3, frameon=False, fontsize=8.5)
    return axr


def make_reg_plot(
    df: pd.DataFrame,
    out_png: Path,
    *,
    prob_col: str = "mtm_lstm_cls_prob",
    time_col: str = "window_end",
    smooth_xrs_cme: bool = True,
) -> Path:
    """Port of PFXRSBPred.R: 3 stacked panels (PF, XRSB, CME) obs vs pred + probability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    cols = {
        "pf": ("mtm_lstm_true_Future_log_ProtonFlux_max", "mtm_lstm_pred_Future_log_ProtonFlux_max"),
        "xrs": ("mtm_lstm_true_Future_log_XRSB_max", "mtm_lstm_pred_Future_log_XRSB_max"),
        "cme": ("mtm_lstm_true_Future_log_DONKICME_speed_km_s_max",
                "mtm_lstm_pred_Future_log_DONKICME_speed_km_s_max"),
    }

    df = df.copy()
    df[time_col] = _parse_ts(df[time_col])
    need = [prob_col] + [c for pair in cols.values() for c in pair]
    df = df[df[time_col].notna()].sort_values(time_col).reset_index(drop=True)
    df = df.dropna(subset=[c for c in need if c in df.columns]).reset_index(drop=True)

    t = df[time_col]
    prob = pd.to_numeric(df[prob_col], errors="coerce").to_numpy()

    obs_pf = pd.to_numeric(df[cols["pf"][0]], errors="coerce").to_numpy()
    pred_pf = pd.to_numeric(df[cols["pf"][1]], errors="coerce").to_numpy()
    obs_xrs = pd.to_numeric(df[cols["xrs"][0]], errors="coerce").to_numpy()
    pred_xrs = pd.to_numeric(df[cols["xrs"][1]], errors="coerce").to_numpy()
    obs_cme = pd.to_numeric(df[cols["cme"][0]], errors="coerce").to_numpy()
    pred_cme = pd.to_numeric(df[cols["cme"][1]], errors="coerce").to_numpy()

    if smooth_xrs_cme:
        pred_xrs = rate_limit_step_changes(pred_xrs, _PRED_MAX_STEP_DEV["xrs"])
        pred_cme = rate_limit_step_changes(pred_cme, _PRED_MAX_STEP_DEV["cme"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    _reg_panel(axes[0], t, obs_pf, pred_pf, prob,
               y_lim=_panel_limits(obs_pf, pred_pf), hline_y=1.0,
               ylab_left=r"Future Max $\log_{10}$(Proton Flux)", show_legend=False)
    _reg_panel(axes[1], t, obs_xrs, pred_xrs, prob,
               y_lim=_panel_limits(obs_xrs, pred_xrs), hline_y=-5.0,
               ylab_left=r"Future Max $\log_{10}$(XRSB)", show_legend=False)
    _reg_panel(axes[2], t, obs_cme, pred_cme, prob,
               y_lim=_panel_limits(obs_cme, pred_cme), hline_y=3.0,
               ylab_left=r"Future Max $\log_{10}$(CME Speed)", show_legend=True)

    axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lbl in axes[2].get_xticklabels():
        lbl.set_rotation(90)
        lbl.set_ha("center")
        lbl.set_fontsize(7)
    axes[2].set_xlabel("Forecast Start Time (UTC)")

    fig.suptitle("SEPNET-PRISM-R future regression predictions vs observed", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# SHAP figures (ports of SHAP_Importance.R and SHAP_importance_summarized.R)
# ---------------------------------------------------------------------------
_META_FEATURES = ("rf_meta_prob", "xgb_meta_prob")

# Blue -> red feature-value gradient (no white midtone), mirroring the R beeswarm.
_SHAP_VALUE_COLORS = ["#08306b", "#2171b5", "#6baed6", "#fc9272", "#ef3b2c", "#a50f15"]
_SHAP_VALUE_STOPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Qualitative palette for individual top features (avoids gray / group blue & orange).
_INDIV_PALETTE = [
    "#E41A1C", "#1B9E77", "#6A3D9A", "#E7298A", "#A6761D",
    "#E6AB02", "#66A61E", "#1F78B4", "#B2DF8A", "#7570B3",
    "#FF7F00", "#CAB2D6",
]
_COL_ALL_SHARP = "#0072B2"
_COL_ALL_FLARE = "#D55E00"


def _base_feature(feature: str) -> str:
    return re.sub(r"_(min|max|avg)$", "", str(feature))


def _feature_group(base: str) -> str:
    if base.startswith("Flare_"):
        return "Flare"
    if base.startswith("SHARP_"):
        return "SHARP"
    return "Other"


def _filter_time(df: pd.DataFrame, col: str, start, end) -> pd.DataFrame:
    if start is not None:
        df = df[df[col] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        df = df[df[col] <= pd.to_datetime(end, utc=True)]
    return df


def make_shap_over_time_plot(
    shap_bytime_csv: Path,
    pred_csv: Path,
    out_png: Path,
    out_csv: Path,
    *,
    top_n_base: int = 5,
    exclude_meta: bool = True,
    agg_by_day: bool = False,
    time_start=None,
    time_end=None,
    outlier_q: float = 0.99,
) -> Path:
    """Port of SHAP_Importance.R: stacked panels of SHAP contribution + SEP prob over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    shap = pd.read_csv(shap_bytime_csv, low_memory=False)
    shap["window_begin"] = _parse_ts(shap["window_begin"])
    shap = _filter_time(shap, "window_begin", time_start, time_end)
    if exclude_meta:
        shap = shap[~shap["feature"].isin(_META_FEATURES)]

    pred = pd.read_csv(pred_csv, low_memory=False)
    pred["window_begin"] = _parse_ts(pred["window_begin"])
    pred = _filter_time(pred, "window_begin", time_start, time_end)
    prob_ens = (pred.groupby("window_begin")["mtm_lstm_cls_prob"]
                .median().rename("prob").reset_index())

    shap["base_feature"] = shap["feature"].map(_base_feature)
    shap["feature_group"] = shap["base_feature"].map(_feature_group)
    shap = shap[shap["feature_group"].isin(["Flare", "SHARP"])]

    base_win = (shap.groupby(["window_begin", "base_feature", "feature_group"])["mean_abs_shap"]
                .sum().rename("shap_mass").reset_index())
    group_win = (base_win.groupby(["window_begin", "feature_group"])["shap_mass"]
                 .sum().reset_index())
    group_win["base_feature"] = np.where(group_win["feature_group"] == "Flare",
                                         "All Flare", "All SHARP")

    if agg_by_day:
        for d in (base_win, group_win):
            d["time_key"] = d["window_begin"].dt.floor("D")
        base_t = (base_win.groupby(["time_key", "base_feature", "feature_group"])["shap_mass"]
                  .mean().reset_index())
        group_t = (group_win.groupby(["time_key", "base_feature"])["shap_mass"]
                   .mean().reset_index())
        prob_ens["time_key"] = prob_ens["window_begin"].dt.floor("D")
        prob_t = prob_ens.groupby("time_key")["prob"].median().reset_index()
    else:
        base_t = base_win.rename(columns={"window_begin": "time_key"})
        group_t = group_win.rename(columns={"window_begin": "time_key"})
        prob_t = prob_ens.rename(columns={"window_begin": "time_key"})

    rank = (base_t.groupby("base_feature")["shap_mass"].mean()
            .sort_values(ascending=False))
    top_base = list(rank.index[:top_n_base])

    grp = group_t[["time_key", "base_feature", "shap_mass"]].copy()
    grp["is_group"] = True
    ind = base_t[base_t["base_feature"].isin(top_base)][
        ["time_key", "base_feature", "shap_mass"]].copy()
    ind["is_group"] = False
    series_t = pd.concat([grp, ind], ignore_index=True).sort_values(["base_feature", "time_key"])

    # Drop pathological All-SHARP outlier days from the line (kept in the CSV).
    sharp_obs = series_t[(series_t["base_feature"] == "All SHARP") & series_t["shap_mass"].notna()]
    outlier_days = set()
    if len(sharp_obs) > 1:
        thr = float(np.quantile(sharp_obs["shap_mass"], outlier_q))
        outlier_days = set(sharp_obs.loc[sharp_obs["shap_mass"] > thr, "time_key"])
    series_plot = series_t[~series_t["time_key"].isin(outlier_days)]
    prob_plot = prob_t[~prob_t["time_key"].isin(outlier_days)].sort_values("time_key")

    indiv_cols = {f: _INDIV_PALETTE[i % len(_INDIV_PALETTE)] for i, f in enumerate(top_base)}

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    # Panel 1: SEP probability
    axes[0].plot(prob_plot["time_key"], prob_plot["prob"], color="0.2", lw=0.8,
                 label="SEP probability")
    axes[0].set_ylabel("SEP event\nprobability", fontsize=9)
    axes[0].set_ylim(-0.02, 1.02)

    # Panel 2: All SHARP / All Flare
    for name, col in (("All SHARP", _COL_ALL_SHARP), ("All Flare", _COL_ALL_FLARE)):
        s = series_plot[series_plot["base_feature"] == name]
        if len(s):
            axes[1].plot(s["time_key"], s["shap_mass"], color=col, lw=0.8, label=name)
    axes[1].set_ylabel("All SHARP /\nAll Flare", fontsize=9)

    # Panel 3: top individual base features
    for f in top_base:
        s = series_plot[series_plot["base_feature"] == f]
        if len(s):
            axes[2].plot(s["time_key"], s["shap_mass"], color=indiv_cols[f], lw=0.8, label=f)
    axes[2].set_ylabel(f"Top {len(top_base)} features", fontsize=9)

    for ax in axes:
        ax.grid(True, color="0.92", lw=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if agg_by_day:
        axes[2].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    else:
        axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=6))
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lbl in axes[2].get_xticklabels():
        lbl.set_rotation(90)
        lbl.set_ha("center")
        lbl.set_fontsize(7)
    axes[2].set_xlabel("Forecast Start Time (UTC)")

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("SHAP contribution and SEP predicted probability over time", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    (series_t.merge(prob_t, on="time_key", how="left")
     .to_csv(out_csv, index=False))
    return out_png


def _sina_offsets(x: np.ndarray, width: float = 0.4, bins: int = 24, seed: int = 0) -> np.ndarray:
    """Density-scaled vertical jitter to approximate ggforce::geom_sina."""
    x = np.asarray(x, dtype=np.float64)
    off = np.zeros_like(x)
    finite = np.isfinite(x)
    xf = x[finite]
    if xf.size == 0:
        return off
    if xf.max() > xf.min():
        hist, edges = np.histogram(xf, bins=bins)
        maxh = hist.max() if hist.max() > 0 else 1
        idx = np.clip(np.digitize(xf, edges[1:-1]), 0, bins - 1)
        dens = hist[idx] / maxh
    else:
        dens = np.ones_like(xf)
    rng = np.random.default_rng(seed)
    off[finite] = (rng.random(xf.size) * 2 - 1) * width * dens
    return off


def make_shap_summary_plot(
    shap_bytime_csv: Path,
    input_csv: Path,
    out_png: Path,
    out_csv: Path,
    out_group_csv: Path,
    *,
    top_n_base: int = 20,
    max_points: int = 900,
    exclude_meta: bool = True,
    seed: int = 42,
    keep_shap_q: float = 0.85,
    keep_val_q: float = 0.90,
    time_start=None,
    time_end=None,
) -> Path:
    """Port of SHAP_importance_summarized.R: importance bars + signed-SHAP beeswarm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    shap = pd.read_csv(shap_bytime_csv, low_memory=False)
    shap["window_begin"] = _parse_ts(shap["window_begin"])
    shap["window_end"] = _parse_ts(shap["window_end"])
    shap = _filter_time(shap, "window_begin", time_start, time_end)
    if exclude_meta:
        shap = shap[~shap["feature"].isin(_META_FEATURES)]

    shap["base_feature"] = shap["feature"].map(_base_feature)
    shap["feature_group"] = shap["base_feature"].map(_feature_group)

    keys = ["window_begin", "window_end", "row_index", "base_feature", "feature_group"]
    agg = (shap.groupby(keys)
           .agg(signed_shap=("mean_signed_shap", "sum"), abs_shap=("mean_abs_shap", "sum"))
           .reset_index())

    grp_src = agg[agg["feature_group"].isin(["Flare", "SHARP"])]
    group_agg = (grp_src.groupby(["window_begin", "window_end", "row_index", "feature_group"])
                 .agg(signed_shap=("signed_shap", "sum"), abs_shap=("abs_shap", "sum"))
                 .reset_index())
    group_agg["base_feature"] = np.where(group_agg["feature_group"] == "Flare",
                                         "All Flare features", "All SHARP features")
    plot_dt = pd.concat([agg, group_agg[keys[:3] + ["base_feature", "feature_group",
                                                    "signed_shap", "abs_shap"]]],
                        ignore_index=True)

    rank = (plot_dt.assign(abs_signed=plot_dt["signed_shap"].abs())
            .groupby("base_feature")
            .agg(mean_abs_signed=("abs_signed", "mean"),
                 mean_abs_mass=("abs_shap", "mean"))
            .reset_index())
    groups = ["All Flare features", "All SHARP features"]
    base_rank = rank[~rank["base_feature"].isin(groups)].sort_values("mean_abs_signed",
                                                                     ascending=False)
    keep_base = list(base_rank["base_feature"][:top_n_base])
    keep_feat = groups + keep_base
    plot_dt = plot_dt[plot_dt["base_feature"].isin(keep_feat)].copy()

    group_by_mass = list(rank[rank["base_feature"].isin(groups)]
                         .sort_values("mean_abs_mass", ascending=False)["base_feature"])
    order_top = group_by_mass + keep_base  # index 0 == top of the chart

    # ---- feature values for color (rank within feature -> [0,1]) -------------
    feat_value = pd.Series(np.nan, index=plot_dt.index)
    input_csv = Path(input_csv)
    if input_csv.is_file():
        raw = pd.read_csv(input_csv, low_memory=False)
        if "window_begin" in raw.columns:
            raw["window_begin"] = _parse_ts(raw["window_begin"])

        def _col_for(bf):
            for c in (f"{bf}_avg", f"{bf}_max", f"{bf}_min", bf):
                if c in raw.columns:
                    return c
            return None

        raw_idx = raw.set_index("window_begin") if "window_begin" in raw.columns else None
        for bf in keep_base:
            vc = _col_for(bf)
            if vc is None or raw_idx is None:
                continue
            idx = plot_dt.index[plot_dt["base_feature"] == bf]
            if not len(idx):
                continue
            keysb = plot_dt.loc[idx, "window_begin"]
            vals = pd.to_numeric(raw_idx[vc].reindex(keysb).to_numpy(), errors="coerce")
            r = pd.Series(vals).rank(method="average")
            feat_value.loc[idx] = ((r - 1) / max(r.max() - 1, 1)).to_numpy()
        for gname in groups:
            prefix = "Flare_" if "Flare" in gname else "SHARP_"
            members = [b for b in agg["base_feature"].unique() if str(b).startswith(prefix)]
            idx = plot_dt.index[plot_dt["base_feature"] == gname]
            if not len(idx) or raw_idx is None:
                continue
            keysb = plot_dt.loc[idx, "window_begin"]
            mat = []
            for bf in members:
                vc = _col_for(bf)
                if vc is None:
                    continue
                vals = pd.to_numeric(raw_idx[vc].reindex(keysb).to_numpy(), errors="coerce")
                r = pd.Series(vals).rank(method="average")
                mat.append(((r - 1) / max(r.max() - 1, 1)).to_numpy())
            if mat:
                feat_value.loc[idx] = np.nanmean(np.vstack(mat), axis=0)
    else:
        for bf, sub in plot_dt.groupby("base_feature"):
            r = sub["signed_shap"].abs().rank(method="average")
            feat_value.loc[sub.index] = ((r - 1) / max(r.max() - 1, 1)).to_numpy()
    plot_dt["feature_value"] = feat_value

    # ---- subsample beeswarm: keep extremes, fill with random core ------------
    rng = np.random.default_rng(seed)
    parts = []
    for bf, sub in plot_dt.groupby("base_feature"):
        if len(sub) <= max_points:
            parts.append(sub)
            continue
        abs_s = sub["signed_shap"].abs().to_numpy()
        thr_s = np.quantile(abs_s, keep_shap_q)
        fv = sub["feature_value"].to_numpy()
        with np.errstate(invalid="ignore"):
            thr_lo = np.nanquantile(fv, 1 - keep_val_q)
            thr_hi = np.nanquantile(fv, keep_val_q)
        keep_mask = (abs_s >= thr_s) | (np.isfinite(fv) & ((fv <= thr_lo) | (fv >= thr_hi)))
        keep_ix = np.flatnonzero(keep_mask)
        rest = np.flatnonzero(~keep_mask)
        n_fill = max(0, min(len(rest), max_points - len(keep_ix)))
        if n_fill > 0:
            keep_ix = np.concatenate([keep_ix, rng.choice(rest, n_fill, replace=False)])
        parts.append(sub.iloc[np.sort(keep_ix)])
    plot_sub = pd.concat(parts, ignore_index=True)

    # ---- summary bar values --------------------------------------------------
    def _bar_fill(bf):
        if bf == "All Flare features":
            return _COL_ALL_FLARE
        if bf == "All SHARP features":
            return _COL_ALL_SHARP
        if bf.startswith("Flare_"):
            return "#E69F00"
        if bf.startswith("SHARP_"):
            return "#56B4E9"
        return "gray"

    summ = (plot_dt.assign(abs_signed=plot_dt["signed_shap"].abs())
            .groupby("base_feature")
            .agg(mean_abs_net=("abs_signed", "mean"),
                 mean_abs_mass=("abs_shap", "mean"),
                 mean_signed=("signed_shap", "mean"))
            .reset_index())
    summ["is_group"] = summ["base_feature"].isin(groups)
    summ["mean_abs"] = np.where(summ["is_group"], summ["mean_abs_mass"], summ["mean_abs_net"])

    # y positions: index 0 in order_top -> top of chart.
    ypos = {f: len(order_top) - 1 - i for i, f in enumerate(order_top)}
    summ = summ[summ["base_feature"].isin(order_top)].copy()
    summ["y"] = summ["base_feature"].map(ypos)
    summ = summ.sort_values("y")

    # ---- figure --------------------------------------------------------------
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "shapval", list(zip(_SHAP_VALUE_STOPS, _SHAP_VALUE_COLORS)))
    fig, (ax_imp, ax_bee) = plt.subplots(
        1, 2, figsize=(11, max(4, 0.32 * len(order_top))),
        gridspec_kw={"width_ratios": [0.95, 1.35]})

    ax_imp.barh(summ["y"], summ["mean_abs"],
                color=[_bar_fill(b) for b in summ["base_feature"]], height=0.72)
    for _, row in summ.iterrows():
        ax_imp.text(row["mean_abs"], row["y"], f" {row['mean_abs']:.4f}",
                    va="center", ha="left", fontsize=7)
    ax_imp.set_yticks(list(ypos.values()))
    ax_imp.set_yticklabels(list(ypos.keys()), fontsize=8.5)
    ax_imp.set_xlabel("Mean |SHAP|")
    ax_imp.set_title("Feature importance", fontsize=10, loc="left", fontweight="bold")
    ax_imp.margins(x=0.18)
    ax_imp.grid(True, axis="x", color="0.9", lw=0.3)
    ax_imp.spines["top"].set_visible(False)
    ax_imp.spines["right"].set_visible(False)

    ax_bee.axvline(0.0, color="0.45", lw=0.4)
    plot_sub = plot_sub[plot_sub["base_feature"].isin(order_top)].copy()
    plot_sub["y0"] = plot_sub["base_feature"].map(ypos)
    # Draw mid-valued points first so saturated low/high colors land on top.
    plot_sub = plot_sub.assign(_ord=(plot_sub["feature_value"] - 0.5).abs()).sort_values(
        "_ord", na_position="first")
    sc = None
    for bf, sub in plot_sub.groupby("base_feature"):
        y0 = ypos[bf]
        yy = y0 + _sina_offsets(sub["signed_shap"].to_numpy(), width=0.39, seed=seed)
        sc = ax_bee.scatter(sub["signed_shap"], yy, c=sub["feature_value"], cmap=cmap,
                            vmin=0, vmax=1, s=6, alpha=0.85, linewidths=0,
                            edgecolors="none")
    ax_bee.set_yticks(list(ypos.values()))
    ax_bee.set_yticklabels([])
    ax_bee.set_ylim(-0.7, len(order_top) - 0.3)
    ax_imp.set_ylim(-0.7, len(order_top) - 0.3)
    ax_bee.set_xlabel("SHAP value (impact on predicted SEP event probability)")
    ax_bee.set_title("SHAP value distribution", fontsize=10, loc="left", fontweight="bold")
    ax_bee.grid(True, axis="y", color="0.9", lw=0.35)
    ax_bee.spines["top"].set_visible(False)
    ax_bee.spines["right"].set_visible(False)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax_bee, fraction=0.03, pad=0.02, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(["Low", "Mid", "High"])
        cbar.set_label("Feature value", fontsize=9)

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    # ---- side-car CSVs -------------------------------------------------------
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    summ.sort_values("mean_abs", ascending=False).to_csv(out_csv, index=False)
    group_compare = (group_agg.assign(abs_signed=group_agg["signed_shap"].abs())
                     .groupby(["feature_group", "base_feature"])
                     .agg(total_mass=("abs_shap", "mean"),
                          net_effect=("abs_signed", "mean"),
                          mean_signed_net=("signed_shap", "mean"))
                     .reset_index())
    group_compare.to_csv(out_group_csv, index=False)
    return out_png


def _build_arg_parser() -> argparse.ArgumentParser:
    # Path defaults come from Utils/paths.json via paths_config (P).
    p = argparse.ArgumentParser(
        description="Plot SEPNET-PRISM-R probability/warning + observed proton flux/XRSB (dual axis)."
    )
    p.add_argument("--input-csv", default=str(P("hsmm_out_csv")),
                   help="HSMM-smoothed Data_test CSV (default from paths.json)")
    p.add_argument("--out-png", default=str(P("figures_dir") / "ProbsPFXRSB_merged.png"),
                   help="Output PNG path (default: <figures_dir>/ProbsPFXRSB_merged.png)")
    p.add_argument("--pred-out-png", default=str(P("figures_dir") / "PFXRSBPred.png"),
                   help="Output PNG for the regression obs-vs-pred figure (PFXRSBPred.R port)")
    p.add_argument("--prob-col", default="mtm_lstm_cls_prob",
                   help="Probability column for the left-axis line (e.g. hsmm_prob)")
    p.add_argument("--warn-col", default="hsmm_warn",
                   help="Binary warning column drawn as markers")
    p.add_argument("--smooth-xrs-cme", action=argparse.BooleanOptionalAction, default=True,
                   help="Rate-limit XRSB/CME predicted steps in the regression figure (default: on)")
    # SHAP figures (ports of SHAP_Importance.R / SHAP_importance_summarized.R)
    p.add_argument("--shap-bytime-csv",
                   default=str(P("shap_outputs_dir") / "shap_feature_importance_by_forecast_time.csv"),
                   help="Per-forecast-time SHAP CSV from Step2 (default from paths.json)")
    p.add_argument("--pred-csv", default=str(P("nrt_predictions_csv")),
                   help="Per-split predictions CSV (for ensemble-median SEP probability)")
    p.add_argument("--feature-csv", default=str(P("nrt_rolling_csv")),
                   help="Rolling feature table used to color the beeswarm by feature value")
    p.add_argument("--shap-time-out-png",
                   default=str(P("figures_dir") / "shap_importance_over_time.png"))
    p.add_argument("--shap-summary-out-png",
                   default=str(P("figures_dir") / "shap_importance_summary.png"))
    p.add_argument("--shap-agg-by-day", action=argparse.BooleanOptionalAction, default=False,
                   help="Aggregate the SHAP-over-time figure by day (default: hourly)")
    p.add_argument("--skip-shap", action="store_true",
                   help="Skip the two SHAP figures (only make the prob/regression plots)")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    in_csv = Path(args.input_csv)
    if not in_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")
    df = pd.read_csv(in_csv, low_memory=False)
    out = make_plot(df, Path(args.out_png), prob_col=args.prob_col, warn_col=args.warn_col)
    n_warn = int(pd.to_numeric(df.get(args.warn_col), errors="coerce").fillna(0).sum()) \
        if args.warn_col in df.columns else 0
    print(f"[plot] {in_csv} rows={len(df):,} warnings={n_warn}")
    print(f"[plot] wrote {out}")

    out_reg = make_reg_plot(df, Path(args.pred_out_png), prob_col=args.prob_col,
                            smooth_xrs_cme=bool(args.smooth_xrs_cme))
    print(f"[plot] wrote {out_reg}")

    if not args.skip_shap:
        shap_csv = Path(args.shap_bytime_csv)
        if not shap_csv.is_file():
            print(f"[plot] SHAP figures skipped: {shap_csv} not found "
                  f"(run Utils/Step2_SHAP_postanalysis.py first)")
        else:
            shap_dir = P("shap_outputs_dir")
            out_t = make_shap_over_time_plot(
                shap_csv, Path(args.pred_csv),
                Path(args.shap_time_out_png), shap_dir / "shap_feature_contrib_over_time.csv",
                agg_by_day=bool(args.shap_agg_by_day),
            )
            print(f"[plot] wrote {out_t}")
            out_s = make_shap_summary_plot(
                shap_csv, Path(args.feature_csv),
                Path(args.shap_summary_out_png),
                shap_dir / "shap_summary_beeswarm_base_features.csv",
                shap_dir / "shap_group_total_vs_net.csv",
            )
            print(f"[plot] wrote {out_s}")


if __name__ == "__main__":
    main()
