#!/usr/bin/env python3
"""End-to-end near-real-time (NRT) pipeline in one Python script.

This chains together three pieces that are currently split across languages/files:

1. FETCH   -- pull fresh solar datasets (SHARP, GOES flares, DONKI CMEs, GOES >10 MeV
              proton flux, GOES XRS-B) using the fetchers in ``fetch_solar_data.py``.
2. AGGREGATE -- a faithful Python port of ``Data_Aggregation.R``: build the hourly
              24h rolling-window feature table and write
              ``Data/nrt_rolling_combinded_seq_1hours.csv``.
3. EVALUATE -- score that NRT table with the already-trained full-train MTM-LSTM
              ensemble (NO training happens here), reusing the scoring logic in
              ``eval_saved_mtm_lstm_models_fulltrain.py``. Writes per-split predictions to
              ``mtm_lstm_fulltrain_nrt1h_predictions.csv``.
4. POST-ANALYSIS -- a Python port of the R "Data_test" builder (utils.R): collapse the
              per-split predictions to an ensemble median per window and attach the raw
              max proton flux / XRS-B / CME speed in the hour before each forecast time.
              Writes ``nrt_Data_test_1h.csv``.
5. HSMM      -- apply the PRE-TRAINED HSMM smoother (hsmm_analysis.py) to the
              Data_test table, producing ``hsmm_prob`` and ``hsmm_warn``. Uses the saved
              ``hsmm_params.json`` only; no retraining and no Data_train file needed.

Run (defaults are workspace-relative):

    python Utils/fetch_aggregate_eval_nrt.py --start 2025-09-10 --end now

Useful flags:
    --skip-fetch      reuse the CSVs already in --data-dir
    --skip-aggregate  reuse an existing --nrt-csv
    --products        which products to fetch (default: all)
    --no-append       overwrite raw product CSVs instead of append/dedupe
    --raw-keep-days   after append, drop raw rows older than start - N days (default 30)

Notes:
    * Fetch **appends** into ``Data/Data_nrt/*.csv`` by default (merge + dedupe), then
      prunes any raw rows older than ``fetch_start - 1 month``.
    * ``nrt_rolling_combinded_seq_1hours.csv`` is rebuilt only for windows with
      ``window_begin`` in the fetch ``[start, end]`` interval (not the full raw archive).
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from fetch_solar_data import fetch_all_solar_data, parse_cli_datetime  # noqa: E402
from paths_config import P, S  # noqa: E402  (central path config -> Utils/paths.json)

# ---------------------------------------------------------------------------
# Constants mirroring Data_Aggregation.R
# ---------------------------------------------------------------------------
# SHARP meta columns excluded from the aggregated feature set (setdiff in the R script).
SHARP_META_COLS = {
    "T_OBS", "T_REC", "CADENCE", "HARPNUM", "GWILL", "WAVELENTH",
    "QUALITY", "NOAA_AR", "T_REC_posix",
}
FEATURES_FLARE = ("Duration", "RisenTime", "log_Strength")
FEATURES_DONKICME = ("lon_deg", "halfAngle_deg", "speed_km_s")

# Empty-window fill values (copied verbatim from Data_Aggregation.R).
FLARE_LOG_STRENGTH_EMPTY = -7.221849
XRS_EMPTY = 4.180751e-10
PF_EMPTY = 0.01576289

_FLARE_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
_FLARE_RE = re.compile(r"^([ABCMX])([0-9.]*)$")


# ---------------------------------------------------------------------------
# AGGREGATION helpers (Python port of Data_Aggregation.R)
# ---------------------------------------------------------------------------
def _safe_stats(values: np.ndarray) -> tuple[float, float, float]:
    """R ``safe_stats``: min/max/avg ignoring NaN; NaN triple when empty/all-NaN."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(arr.min()), float(arr.max()), float(arr.mean()))


def _label_to_flux(labels: pd.Series) -> pd.Series:
    """R ``label_to_Flux``: GOES class string -> W/m^2 peak flux."""
    def one(x) -> float:
        if pd.isna(x):
            return np.nan
        # Mirror R exactly: gsub("[^ABCMXSF0-9]", "", gsub(",", ".", x)) then toupper.
        # NOTE: the R char class has no ".", so the decimal point is stripped too
        # (e.g. "M1.5" -> "M15"). We reproduce that here so feature scaling stays
        # identical to the training aggregation.
        s = "".join(ch for ch in str(x).replace(",", ".").upper() if ch in "ABCMXSF0123456789")
        if s == "SF":
            s = "C"
        # R: regexec("^([ABCMX])([0-9.]*)$"); the dot was already stripped above.
        m = _FLARE_RE.match(s)
        if m is None:
            return np.nan
        cls, mag_str = m.group(1), m.group(2)
        base = _FLARE_BASE.get(cls)
        if base is None:
            return np.nan
        if mag_str == "":
            mag = 1.0
        else:
            try:
                mag = float(mag_str)
            except ValueError:
                mag = 1.0
            if not np.isfinite(mag) or mag <= 0:
                mag = 1.0
        return mag * base

    return labels.map(one)


def _to_ns_naive(series: pd.Series) -> np.ndarray:
    """tz-aware UTC Series -> naive ``datetime64[ns]`` array (NaT preserved)."""
    return series.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")


def _parse_utc(series: pd.Series) -> pd.Series:
    """Robust UTC timestamp parse (handles ``+00:00``, ``Z``, plain formats)."""
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, utc=True, errors="coerce")


def _flare_feature_extraction(flare: pd.DataFrame) -> pd.DataFrame:
    """R ``Flare_feature_extraction``: Duration, RisenTime, Strength, log_Strength."""
    start = _parse_utc(flare["start_time"])
    peak = _parse_utc(flare["peak_time"])
    end = _parse_utc(flare["end_time"])

    # Peak after end -> swap peak and end.
    swap = peak > end
    if swap.any():
        tmp = peak[swap].copy()
        peak.loc[swap] = end[swap]
        end.loc[swap] = tmp

    # Peak before start -> set peak to midnight of the end date (legacy rule).
    fix = peak < start
    if fix.any():
        peak.loc[fix] = end[fix].dt.floor("D")

    bad = (peak < start) | (peak > end)
    if bad.any():
        print(f"Flares with inconsistent peak time after repair: {int(bad.sum())}")

    out = flare.copy()
    out["start_time_posix"] = start
    out["peak_time_posix"] = peak
    out["end_time_posix"] = end
    out["Duration"] = (end - start).dt.total_seconds() / 60.0
    out["RisenTime"] = (peak - start).dt.total_seconds() / 60.0
    out["Strength"] = _label_to_flux(flare["label"])
    out["log_Strength"] = np.log10(out["Strength"])
    return out


def _clean_sharp(sharp: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce SHARP feature columns to numeric and apply R's NA rules."""
    sharp = sharp.copy()
    sharp["T_REC_posix"] = _parse_utc(sharp["T_REC"])
    features = [c for c in sharp.columns if c not in SHARP_META_COLS]
    for feat in features:
        sharp[feat] = pd.to_numeric(sharp[feat], errors="coerce")

    def zero_to_nan(col: str) -> None:
        if col in sharp.columns:
            sharp.loc[sharp[col] == 0, col] = np.nan

    def nonfinite_to_nan(col: str) -> None:
        if col in sharp.columns:
            sharp.loc[~np.isfinite(sharp[col]), col] = np.nan

    zero_to_nan("USFLUX")
    if "TOTUSJZ" in sharp.columns:
        sharp.loc[(sharp["TOTUSJZ"] == 0) | (~np.isfinite(sharp["TOTUSJZ"])), "TOTUSJZ"] = np.nan
    nonfinite_to_nan("MEANJZH")
    if "SAVNCPP" in sharp.columns:
        sharp.loc[(sharp["SAVNCPP"] == 0) | (~np.isfinite(sharp["SAVNCPP"])), "SAVNCPP"] = np.nan
    zero_to_nan("TOTPOT")
    if "USFLUXL" in sharp.columns:
        sharp.loc[sharp["USFLUXL"] <= 0, "USFLUXL"] = np.nan
    zero_to_nan("CMASKL")
    return sharp, features


def build_nrt_rolling(
    read_dir: Path,
    out_csv: Path,
    *,
    window_begin_start=None,
    window_begin_end=None,
) -> pd.DataFrame:
    """Port of ``Data_Aggregation.R``: hourly 24h rolling-window feature table.

    If ``window_begin_start`` / ``window_begin_end`` are given (UTC datetimes),
    only windows whose ``window_begin`` falls in that closed interval are written
    (i.e. the newly fetched analysis range). Raw product CSVs may still hold a
    longer lookback archive for feature construction inside each 24h window.
    """
    read_dir = Path(read_dir)

    sharp = pd.read_csv(read_dir / "sharp_nrt.csv", low_memory=False)
    sharp, features_sharp = _clean_sharp(sharp)

    flare = pd.read_csv(read_dir / "flare_nrt.csv")
    flare = _flare_feature_extraction(flare)

    donki = pd.read_csv(read_dir / "donki_cme_nrt.csv")
    donki["start_time_posix"] = _parse_utc(donki["startTime"])

    flux = pd.read_csv(read_dir / "GOESHAPI_ProtonFlux_nrt.csv")
    flux["time_posix"] = _parse_utc(flux["time"])
    flux.loc[flux["ProtonFlux"] < 0, "ProtonFlux"] = np.nan

    xrs = pd.read_csv(read_dir / "XRS_nrt.csv")
    xrs["time_posix"] = _parse_utc(xrs["time"])

    # ---- Time index arrays (naive UTC ns) for fast windowing ----
    t_sharp = _to_ns_naive(sharp["T_REC_posix"])
    t_flare = _to_ns_naive(flare["peak_time_posix"])
    t_donki = _to_ns_naive(donki["start_time_posix"])
    t_flux = _to_ns_naive(flux["time_posix"])
    t_xrs = _to_ns_naive(xrs["time_posix"])

    flux_valid = flux["time_posix"].dropna()
    if flux_valid.empty:
        raise RuntimeError("Proton-flux file has no valid timestamps; cannot build windows.")
    data_start = flux_valid.min()
    data_end = flux_valid.max()

    # Analysis windows: prefer the fetch [start, end]; fall back to full flux span.
    win_lo = data_start
    win_hi = data_end
    if window_begin_start is not None:
        wb = pd.Timestamp(window_begin_start)
        if wb.tzinfo is None:
            wb = wb.tz_localize("UTC")
        else:
            wb = wb.tz_convert("UTC")
        win_lo = max(win_lo, wb)
    if window_begin_end is not None:
        we = pd.Timestamp(window_begin_end)
        if we.tzinfo is None:
            we = we.tz_localize("UTC")
        else:
            we = we.tz_convert("UTC")
        win_hi = min(win_hi, we)
    if win_lo > win_hi:
        raise RuntimeError(
            f"Empty rolling window range: begin_start={win_lo} > begin_end={win_hi} "
            f"(flux span {data_start} -> {data_end})"
        )

    window_start = pd.date_range(start=win_lo, end=win_hi, freq="1h", tz="UTC")
    print(f"[aggregate] {len(window_start):,} hourly windows: {win_lo} -> {win_hi} "
          f"(raw flux archive {data_start} -> {data_end})")

    day = np.timedelta64(24, "h")
    records: list[dict] = []

    for w_start in window_start:
        start = np.datetime64(w_start.tz_convert("UTC").tz_localize(None), "ns")
        end = start + day
        fut_start = end
        fut_end = fut_start + day

        m_sharp = (t_sharp >= start) & (t_sharp < end)
        m_flare = (t_flare >= start) & (t_flare < end)
        m_fut_flare = (t_flare >= fut_start) & (t_flare < fut_end)
        m_donki = (t_donki >= start) & (t_donki < end)
        m_fut_donki = (t_donki >= fut_start) & (t_donki < fut_end)
        m_flux = (t_flux >= start) & (t_flux < end)
        m_fut_flux = (t_flux >= fut_start) & (t_flux < fut_end)
        m_xrs = (t_xrs >= start) & (t_xrs < end)
        m_fut_xrs = (t_xrs >= fut_start) & (t_xrs < fut_end)

        sharp_w = sharp.loc[m_sharp]
        flare_w = flare.loc[m_flare]
        donki_w = donki.loc[m_donki]
        flux_w = flux.loc[m_flux]
        xrs_w = xrs.loc[m_xrs]

        rec: dict = {
            "window_begin": w_start.strftime("%Y-%m-%d %H:%M:%S"),
            "window_end": (w_start + pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
            "OSEP_label": 0,
            "GSEP_label": 0,
        }

        # ---- SHARP block ----
        if len(sharp_w) > 0:
            rec["SHARP_label"] = 1
            rec["SHARP_From_SMARP_label"] = 0
            for ci in features_sharp:
                mn, mx, av = _safe_stats(sharp_w[ci].to_numpy())
                rec[f"SHARP_{ci}_min"], rec[f"SHARP_{ci}_max"], rec[f"SHARP_{ci}_avg"] = mn, mx, av
        else:
            rec["SHARP_label"] = 0
            rec["SHARP_From_SMARP_label"] = 0
            for ci in features_sharp:
                rec[f"SHARP_{ci}_min"] = rec[f"SHARP_{ci}_max"] = rec[f"SHARP_{ci}_avg"] = np.nan

        # ---- SHARP restricted to flaring active regions ----
        flare_ar = pd.unique(flare_w["ar_noaanum"]) if "ar_noaanum" in flare_w.columns else np.array([])
        if len(sharp_w) > 0 and "NOAA_AR" in sharp_w.columns:
            sharp_ar = sharp_w.loc[sharp_w["NOAA_AR"].isin(flare_ar)]
        else:
            sharp_ar = sharp_w.iloc[0:0]
        if len(sharp_ar) > 0:
            rec["SHARP_AR_label"] = 1
            rec["SHARP_AR_From_SMARP_label"] = 0
            for ci in features_sharp:
                mn, mx, av = _safe_stats(sharp_ar[ci].to_numpy())
                rec[f"SHARP_AR_{ci}_min"], rec[f"SHARP_AR_{ci}_max"], rec[f"SHARP_AR_{ci}_avg"] = mn, mx, av
        else:
            rec["SHARP_AR_label"] = 0
            rec["SHARP_AR_From_SMARP_label"] = 0
            for ci in features_sharp:
                rec[f"SHARP_AR_{ci}_min"] = rec[f"SHARP_AR_{ci}_max"] = rec[f"SHARP_AR_{ci}_avg"] = np.nan

        # ---- Flare block ----
        if len(flare_w) > 0:
            rec["Flare_label"] = 1
            rec["Flare_num"] = int(len(flare_w))
            for ci in FEATURES_FLARE:
                mn, mx, av = _safe_stats(flare_w[ci].to_numpy())
                rec[f"Flare_{ci}_min"], rec[f"Flare_{ci}_max"], rec[f"Flare_{ci}_avg"] = mn, mx, av
        else:
            rec["Flare_label"] = 0
            rec["Flare_num"] = 0
            rec["Flare_Duration_min"] = rec["Flare_Duration_max"] = rec["Flare_Duration_avg"] = 0
            rec["Flare_RisenTime_min"] = rec["Flare_RisenTime_max"] = rec["Flare_RisenTime_avg"] = 0
            rec["Flare_log_Strength_min"] = FLARE_LOG_STRENGTH_EMPTY
            rec["Flare_log_Strength_max"] = FLARE_LOG_STRENGTH_EMPTY
            rec["Flare_log_Strength_avg"] = FLARE_LOG_STRENGTH_EMPTY

        # ---- DONKI CME block ----
        if len(donki_w) > 0:
            rec["DONKICME_label"] = 1
            rec["DONKICME_num"] = int(len(donki_w))
            rec["DONKICME_From_CDAW_label"] = 0
            for ci in FEATURES_DONKICME:
                mn, mx, av = _safe_stats(donki_w[ci].to_numpy()) if ci in donki_w.columns else (np.nan,) * 3
                rec[f"DONKICME_{ci}_min"], rec[f"DONKICME_{ci}_max"], rec[f"DONKICME_{ci}_avg"] = mn, mx, av
        else:
            rec["DONKICME_label"] = 0
            rec["DONKICME_num"] = 0
            rec["DONKICME_From_CDAW_label"] = 0
            for ci in FEATURES_DONKICME:
                rec[f"DONKICME_{ci}_min"] = rec[f"DONKICME_{ci}_max"] = rec[f"DONKICME_{ci}_avg"] = 0

        # ---- Proton flux block ----
        if len(flux_w) > 0:
            rec["ProtonFlux_label"] = 1
            mn, mx, av = _safe_stats(flux_w["ProtonFlux"].to_numpy())
            rec["ProtonFlux_min"], rec["ProtonFlux_max"], rec["ProtonFlux_avg"] = mn, mx, av
        else:
            rec["ProtonFlux_label"] = 0
            rec["ProtonFlux_min"] = rec["ProtonFlux_max"] = rec["ProtonFlux_avg"] = 0

        # ---- XRS-B block ----
        if len(xrs_w) > 0:
            rec["XRS_label"] = 1
            mn, mx, av = _safe_stats(xrs_w["xrsb"].to_numpy())
            rec["XRSB_min"], rec["XRSB_max"], rec["XRSB_avg"] = mn, mx, av
        else:
            rec["XRS_label"] = 0
            rec["XRSB_min"] = rec["XRSB_max"] = rec["XRSB_avg"] = 0

        # ---- Future (next 24h) targets ----
        rec["Future_OSEP_label"] = 0
        rec["Future_GSEP_label"] = 0
        rec["Future_Flare_num"] = int(m_fut_flare.sum())

        fut_donki = donki.loc[m_fut_donki]
        if len(fut_donki) > 0 and fut_donki["speed_km_s"].notna().any():
            spd_max = float(np.nanmax(fut_donki["speed_km_s"].to_numpy(dtype=np.float64)))
            rec["Future_DONKICME_num"] = int(len(fut_donki))
            rec["Future_DONKICME_speed_km_s_max"] = spd_max
            rec["Future_log_DONKICME_speed_km_s_max"] = np.log10(spd_max) if spd_max > 0 else 0
        else:
            rec["Future_DONKICME_num"] = int(len(fut_donki))
            rec["Future_DONKICME_speed_km_s_max"] = 0
            rec["Future_log_DONKICME_speed_km_s_max"] = 0

        fut_flux = flux.loc[m_fut_flux]
        if len(fut_flux) > 0 and fut_flux["ProtonFlux"].notna().any():
            pf_max = float(np.nanmax(fut_flux["ProtonFlux"].to_numpy(dtype=np.float64)))
            rec["Future_ProtonFlux_max"] = pf_max
            rec["Future_log_ProtonFlux_max"] = np.log10(pf_max)
        else:
            rec["Future_ProtonFlux_max"] = 0
            rec["Future_log_ProtonFlux_max"] = np.log10(PF_EMPTY)

        fut_xrs = xrs.loc[m_fut_xrs]
        if len(fut_xrs) > 0 and fut_xrs["xrsb"].notna().any():
            xrs_max = float(np.nanmax(fut_xrs["xrsb"].to_numpy(dtype=np.float64)))
            rec["Future_XRSB_max"] = xrs_max
            rec["Future_log_XRSB_max"] = np.log10(xrs_max)
        else:
            rec["Future_XRSB_max"] = 0
            rec["Future_log_XRSB_max"] = np.log10(XRS_EMPTY)

        records.append(rec)

    out = pd.DataFrame(records)

    # Interior linear interpolation of the future CME speed target (R: na.approx).
    col = "Future_log_DONKICME_speed_km_s_max"
    out.loc[out[col] == 0, col] = np.nan
    out[col] = out[col].interpolate(method="linear", limit_area="inside")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[aggregate] wrote {out_csv} rows={len(out):,} cols={out.shape[1]}")
    return out


# ---------------------------------------------------------------------------
# EVALUATION (reuse the full-train scoring logic; no training)
# ---------------------------------------------------------------------------
def evaluate_nrt(
    nrt_csv: Path,
    models_dir: Path,
    model_glob: str,
    out_dir: Path,
    *,
    reg_online_debias: bool,
    reg_online_window_days: float,
    reg_online_min_samples: int,
) -> Path:
    """Score the NRT rolling table across every saved split model and concatenate."""
    import torch  # imported lazily so fetch/aggregate work without torch installed

    from eval_saved_mtm_lstm_models_fulltrain import filter_rows, score_full_file

    models_dir = Path(models_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_files = sorted(models_dir.glob(model_glob))
    if not model_files:
        raise FileNotFoundError(f"No model files matching {model_glob} in {models_dir}")

    df_nrt = pd.read_csv(nrt_csv, low_memory=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] {len(model_files)} split models | device={device} | nrt rows={len(df_nrt):,}")

    all_nrt = []
    for mi, mf in enumerate(model_files):
        art = torch.load(mf, map_location="cpu", weights_only=False)
        model = art["model"].to(device)
        model.eval()
        common = dict(
            art=art,
            model=model,
            selector=art["selector"],
            scaler=art["scaler"],
            predictors=list(art["predictors"]),
            y_reg_cols=list(art["y_reg_cols"]),
            y_cls_col=str(art["y_cls_col"]),
            seq_len=int(art["seq_len"]),
            lag_h=float(art["seq_lag_hours"]),
            lag_tol=float(art["seq_lag_tolerance_hours"]),
            wcol_begin=str(art["wcol_begin"]),
            wcol_end=str(art["wcol_end"]),
            device=device,
            threshold=float(art["threshold"]),
            reg_online_debias=reg_online_debias,
            reg_online_window_days=reg_online_window_days,
            reg_online_min_samples=reg_online_min_samples,
        )
        f_nrt = filter_rows(df_nrt, dict(art["row_filter"]), common["wcol_begin"], common["wcol_end"])
        df_out = score_full_file(f_nrt, **common)
        df_out["model_file"] = mf.name
        df_out["split_seed"] = art.get("split_seed")
        all_nrt.append(df_out.reset_index(drop=False).rename(columns={"index": "row_index"}))
        print(f"  [{mi + 1}/{len(model_files)}] {mf.name} | n_nrt={len(df_out)}")

    out_nrt = pd.concat(all_nrt, axis=0, ignore_index=True)
    p_nrt = out_dir / "mtm_lstm_fulltrain_nrt1h_predictions.csv"
    out_nrt.to_csv(p_nrt, index=False)
    print(f"[evaluate] wrote {p_nrt} rows={len(out_nrt):,}")
    return p_nrt


# ---------------------------------------------------------------------------
# POST-ANALYSIS (Python port of the R "Data_test" builder in utils.R)
# ---------------------------------------------------------------------------
PRED_ID_COLS = ("window_begin", "window_end")
PRED_EXCLUDE_COLS = ("row_index", "model_file", "split_seed")
PRED_EXCLUDE_FROM_MEAN = ("mtm_lstm_cls_pred_uncal",)
PRED_CARRY_COLS = ("mtm_lstm_cls_true",)


def aggregate_predictions_across_splits(
    pred: pd.DataFrame, split_col: str = "split_seed"
) -> pd.DataFrame:
    """R ``aggregate_predictions_across_splits``: ensemble = median across split models.

    Grouped by (window_begin, window_end); ``mtm_lstm_cls_pred_uncal`` is dropped and
    ``mtm_lstm_cls_true`` is carried over (by window_begin) instead of averaged.
    """
    pred = pred.copy()
    carry_cols = [c for c in PRED_CARRY_COLS if c in pred.columns]
    excluded = set(PRED_ID_COLS) | set(PRED_EXCLUDE_COLS) | set(PRED_EXCLUDE_FROM_MEAN) | set(carry_cols)
    avg_cols = [c for c in pred.columns if c not in excluded]
    for c in avg_cols:
        pred[c] = pd.to_numeric(pred[c], errors="coerce")

    averaged = (
        pred.groupby(list(PRED_ID_COLS), as_index=False)[avg_cols].median(numeric_only=True)
    )

    if carry_cols:
        first_seed = pd.unique(pred[split_col])[0]
        carried = (
            pred.loc[pred[split_col] == first_seed, [PRED_ID_COLS[0], *carry_cols]]
            .drop_duplicates(subset=[PRED_ID_COLS[0]], keep="first")
        )
        averaged = averaged.merge(carried, on=PRED_ID_COLS[0], how="left")
    return averaged


def _attach_raw_max(
    summary: pd.DataFrame,
    src: pd.DataFrame,
    time_col: str,
    value_col: str,
    out_col: str,
    *,
    lag_hours: float = 1.0,
    min_points: int = 1,
) -> pd.DataFrame:
    """R ``attach_proton_flux_raw`` family: max non-negative value in
    ``[window_end - lag_hours, window_end]`` (inclusive) per summary row."""
    out = summary.copy()
    wend = _parse_utc(out["window_end"])
    end_ns = _to_ns_naive(wend)
    start_ns = _to_ns_naive(wend - pd.Timedelta(hours=lag_hours))

    t = _parse_utc(src[time_col])
    v = pd.to_numeric(src[value_col], errors="coerce")
    keep = t.notna().to_numpy()
    t_ns = _to_ns_naive(t[keep])
    v = v[keep].to_numpy(dtype=np.float64)
    order = np.argsort(t_ns)
    t_ns, v = t_ns[order], v[order]

    res = np.full(len(out), np.nan)
    for i in range(len(out)):
        s, e = start_ns[i], end_ns[i]
        if np.isnat(s) or np.isnat(e):
            continue
        lo = np.searchsorted(t_ns, s, side="left")
        hi = np.searchsorted(t_ns, e, side="right")
        seg = v[lo:hi]
        seg = seg[np.isfinite(seg) & (seg >= 0)]
        if seg.size >= min_points:
            res[i] = seg.max()
    out[out_col] = res
    return out


def build_data_test(pred_csv: Path, data_dir: Path, out_csv: Path) -> pd.DataFrame:
    """Port of the R ``Data_test`` pipeline: ensemble-median predictions plus the
    raw pre-forecast-time max proton flux / XRS-B / CME speed."""
    pred_csv, data_dir, out_csv = Path(pred_csv), Path(data_dir), Path(out_csv)
    pred = pd.read_csv(pred_csv, low_memory=False)
    data_test = aggregate_predictions_across_splits(pred)

    flux = pd.read_csv(data_dir / "GOESHAPI_ProtonFlux_nrt.csv")
    data_test = _attach_raw_max(
        data_test, flux, "time", "ProtonFlux", "ProtonFlux_max_pfu_1h_before_end", lag_hours=1
    )
    xrs = pd.read_csv(data_dir / "XRS_nrt.csv")
    data_test = _attach_raw_max(
        data_test, xrs, "time", "xrsb", "XRSB_max_1h_before_end", lag_hours=1
    )
    cme = pd.read_csv(data_dir / "donki_cme_nrt.csv")
    data_test = _attach_raw_max(
        data_test, cme, "startTime", "speed_km_s", "CME_speed_max_24h_before_end", lag_hours=24
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    data_test.to_csv(out_csv, index=False)
    print(f"[postanalysis] wrote {out_csv} rows={len(data_test):,} cols={data_test.shape[1]}")
    return data_test


# ---------------------------------------------------------------------------
# HSMM smoothing (predict-only; uses pre-trained params from hsmm_analysis)
# ---------------------------------------------------------------------------
def run_hsmm_smoothing(data_test_csv: Path, params_path: Path, out_csv: Path) -> None:
    """Apply the pre-trained HSMM smoother to the NRT Data_test (hsmm_prob + hsmm_warn).

    Training is done once via ``hsmm_analysis.py train``; this step only loads the
    saved parameters, so no Data_train file is required at run time.
    """
    data_test_csv, params_path, out_csv = Path(data_test_csv), Path(params_path), Path(out_csv)
    if not data_test_csv.is_file():
        print(f"[hsmm] skipped: Data_test not found ({data_test_csv})")
        return
    if not params_path.is_file():
        print(
            f"[hsmm] skipped: HSMM params not found ({params_path}). "
            "Train once with: python Utils/hsmm_analysis.py train"
        )
        return

    from hsmm_analysis import predict_hsmm  # lazy import (statsmodels not needed to predict)

    predict_hsmm(test_csv=data_test_csv, params_path=params_path, out_csv=out_csv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    # All path defaults come from Utils/paths.json via paths_config (P/S).
    p = argparse.ArgumentParser(
        description="Fetch NRT solar data, build the rolling-window table, and score the "
        "full-train MTM-LSTM ensemble (no training).",
    )
    p.add_argument("--start", default=None,
                   help="Start time (UTC), e.g. 2025-09-10 (default: now - 7 days)")
    p.add_argument("--end", default="now", help="End time (UTC); 'now' = current UTC time")
    p.add_argument("--data-dir", default=str(P("nrt_product_dir")),
                   help="Directory for the fetched NRT product CSVs")
    p.add_argument("--nrt-csv", default=str(P("nrt_rolling_csv")),
                   help="Output path for the aggregated rolling-window table")
    p.add_argument("--models-dir", default=str(P("models_dir")),
                   help="Directory containing the saved split .pt models")
    p.add_argument("--model-glob", default=S("model_glob"))
    p.add_argument("--out-dir", default=str(P("predictions_dir")),
                   help="Prediction output dir (default: Data/)")
    p.add_argument("--data-test-csv", default=str(P("data_test_csv")),
                   help="Output path for the ensemble-median post-analysis table (Data_test)")
    p.add_argument("--hsmm-params", default=str(P("hsmm_params_json")),
                   help="Pre-trained HSMM parameters JSON (no retraining is done here)")
    p.add_argument("--hsmm-out-csv", default=str(P("hsmm_out_csv")),
                   help="Output path for the HSMM-smoothed Data_test (hsmm_prob + hsmm_warn)")
    p.add_argument("--products", default="all",
                   help="Comma list: sharp,flare,donki_cme,proton_flux,xrsb,sunspot,all")
    p.add_argument("--no-append", action="store_true", help="Overwrite product CSVs instead of append")
    p.add_argument("--raw-keep-days", type=float, default=30.0,
                   help="After append-fetch, drop raw product rows older than "
                        "fetch-start minus this many days (default: 30 ≈ 1 month)")
    p.add_argument("--sharp-email", default="uhek@umich.edu",
                   help="Email registered with JSOC/DRMS for SHARP queries")
    p.add_argument("--skip-fetch", action="store_true", help="Reuse existing CSVs in --data-dir")
    p.add_argument("--skip-aggregate", action="store_true", help="Reuse existing --nrt-csv")
    p.add_argument("--skip-eval", action="store_true", help="Stop after aggregation")
    p.add_argument("--skip-hsmm", action="store_true", help="Skip the HSMM smoothing step")
    p.add_argument("--reg-online-debias", dest="reg_online_debias", action="store_true", default=True)
    p.add_argument("--no-reg-online-debias", dest="reg_online_debias", action="store_false")
    p.add_argument("--reg-online-window-days", type=float, default=30.0)
    p.add_argument("--reg-online-min-samples", type=int, default=3)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    data_dir = Path(args.data_dir)
    nrt_csv = Path(args.nrt_csv)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir) if args.out_dir else P("predictions_dir")

    # Resolve analysis window once (also used when --skip-fetch).
    end = parse_cli_datetime(args.end)
    start = (end - timedelta(days=7)) if args.start is None else parse_cli_datetime(args.start)

    # 1) FETCH (append/dedupe by default) + prune raw older than start - 1 month
    if not args.skip_fetch:
        products = [p.strip() for p in args.products.split(",") if p.strip()]
        print(f"[fetch] {start.isoformat()} -> {end.isoformat()} products={products}")
        fetch_all_solar_data(
            start_time=start,
            end_time=end,
            data_dir=data_dir,
            products=products,
            append=not args.no_append,
            sharp_email=args.sharp_email,
            raw_keep_days=float(args.raw_keep_days),
        )
    else:
        print("[fetch] skipped (using existing CSVs)")

    # 2) AGGREGATE — only windows with window_begin in [start, end]
    if not args.skip_aggregate:
        build_nrt_rolling(
            data_dir, nrt_csv,
            window_begin_start=start,
            window_begin_end=end,
        )
    else:
        print(f"[aggregate] skipped (using existing {nrt_csv})")

    # 3) EVALUATE (no training)
    data_test_csv = Path(args.data_test_csv)
    if not args.skip_eval:
        p_nrt = evaluate_nrt(
            nrt_csv=nrt_csv,
            models_dir=models_dir,
            model_glob=args.model_glob,
            out_dir=out_dir,
            reg_online_debias=bool(args.reg_online_debias),
            reg_online_window_days=float(args.reg_online_window_days),
            reg_online_min_samples=int(args.reg_online_min_samples),
        )
        # 4) POST-ANALYSIS: ensemble-median Data_test + raw pre-forecast maxima
        build_data_test(p_nrt, data_dir, data_test_csv)
    else:
        print("[evaluate] skipped")

    # 5) HSMM smoothing (uses the pre-trained params; NO retraining, no Data_train needed)
    if not args.skip_hsmm:
        run_hsmm_smoothing(
            data_test_csv=data_test_csv,
            params_path=Path(args.hsmm_params),
            out_csv=Path(args.hsmm_out_csv),
        )
    else:
        print("[hsmm] skipped")


if __name__ == "__main__":
    main()
