#!/usr/bin/env python3
"""Export MagPy-period HSMM forecasts to CCMC SEP Scoreboard JSON (schema 2026-10-23).

HSMM-only product (no LSTM custom fields). ``short_name`` remains ``SEPNET-PRISM-R``.

References (mirrored under ``Data/MagPy_Period_Validation/docs/``):
  * https://ccmc.gsfc.nasa.gov/static/files/SEPSB/sepscoreboard_visual_schema.v20261023.pdf
  * https://ccmc.gsfc.nasa.gov/static/files/SEPSB/sep_scoreboard_example_json_file.json
  * https://ccmc.gsfc.nasa.gov/scoreboards/sep/

One JSON file per forecast issue time::

  <short_name>.<prediction_window.start_time>.<issue_time>.json

Continuous forecasts (protons ≥10 MeV, Earth, 24 h window): HSMM probability /
all-clear, plus MTM-LSTM predicted peak proton intensity when available.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P  # noqa: E402

DEFAULT_SHORT_NAME = "SEPNET-PRISM-R"
DEFAULT_VERSION = "1.0-magpy-cutoff-hsmm"
DEFAULT_SPASE = "spase://CCMC/SimulationModel/SEPNET-PRISM-R/1.0"
SEP_THRESHOLD_PFU = 10.0
DEFAULT_ALL_CLEAR_THRESHOLD = 0.1
# Match MagPy / CCMC magnetogram product naming used in Scoreboard examples.
DEFAULT_MAG_PRODUCT = "hmi.sharp_cea_720s_nrt"
# Cadences of Processed Data archives used for MagPy-period 24 h summaries
# (verified on SEP-PRISM_Data/Processed Data/{SMHARP,GOESHAPI_ProtonFlux,Merged_XRays}.csv):
#   SHARP T_REC_posix: 12 min → last sample in [wb, we) is we - 12 min
#   Proton / Merged XRS time_posix: 5 min → last sample is we - 5 min
SHARP_CADENCE = timedelta(minutes=12)
GOES_CADENCE = timedelta(minutes=5)


def _parse_ts(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


def to_iso_z(dt: datetime | pd.Timestamp) -> str:
    """CCMC timestamps as UTC ISO-8601 with trailing Z (seconds included)."""
    t = pd.Timestamp(dt)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_to_filename_timestamp(iso_str: str) -> str:
    """Filename stamp: yyyy-mm-ddTHHMMSS (no colons)."""
    s = iso_str.strip().replace("Z", "").replace("z", "")
    if "T" in s and ":" in s:
        date_part, time_part = s.split("T", 1)
        time_part = time_part.replace(":", "")
        s = f"{date_part}T{time_part}"
    return s


def _clip01(x: float) -> float:
    p = float(x)
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


PRED_LOG_PROTON_FLUX_COL = "mtm_lstm_pred_Future_log_ProtonFlux_max"


def log10_proton_flux_to_pfu(log10_pfu: float) -> float | None:
    """Convert MTM ``Future_log_ProtonFlux_max`` (log10 pfu) → intensity in pfu."""
    try:
        x = float(log10_pfu)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    intensity = 10.0 ** x
    if not math.isfinite(intensity) or intensity < 0.0:
        return None
    return float(f"{intensity:.6g}")


def peak_intensity_pfu_from_row(row: pd.Series | dict) -> float | None:
    """Read ensemble predicted peak proton flux (pfu) from a prediction CSV row."""
    get = row.get if hasattr(row, "get") else lambda k, default=None: row[k] if k in row else default
    raw = get(PRED_LOG_PROTON_FLUX_COL, None)
    if raw is None:
        return None
    try:
        logv = float(pd.to_numeric(raw, errors="coerce"))
    except (TypeError, ValueError):
        return None
    return log10_proton_flux_to_pfu(logv)


def last_data_times_from_window_end(window_end: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """True last sample times inside ``[window_begin, window_end)`` from raw cadences.

    Aggregation (Step1 / Data_Aggregation.R) uses ``time >= wb & time < we``.
    With regular sampling that yields:
      magnetogram (SHARP 12 min) → we - 12 min
      proton / Merged XRS (5 min) → we - 5 min
    """
    we = pd.Timestamp(window_end)
    if we.tzinfo is None:
        we = we.tz_localize("UTC")
    else:
        we = we.tz_convert("UTC")
    return {
        "magnetogram": we - SHARP_CADENCE,
        "proton": we - GOES_CADENCE,
        "xrs": we - GOES_CADENCE,
    }


def build_submission(
    *,
    window_end: pd.Timestamp,
    hsmm_prob: float,
    all_clear_threshold: float = DEFAULT_ALL_CLEAR_THRESHOLD,
    short_name: str = DEFAULT_SHORT_NAME,
    version: str = DEFAULT_VERSION,
    spase_id: str = DEFAULT_SPASE,
    mode: str = "simulated_realtime_forecast",
    sep_threshold_pfu: float = SEP_THRESHOLD_PFU,
    mag_product: str = DEFAULT_MAG_PRODUCT,
    include_model_version: bool = True,
    include_particle_xrs_inputs: bool = True,
    peak_intensity_pfu: float | None = None,
    peak_intensity_uncertainty_pfu: float | None = None,
    peak_intensity_time: str | None = None,
) -> dict:
    """Build one ``sep_forecast_submission`` object (HSMM prob + optional peak flux).

    Timing from raw/processed input features (not the 24 h summary clock):
      * Features are 24 h summaries over half-open ``[window_begin, window_end)``.
      * Magnetogram ``last_data_time`` = last SHARP ``T_REC`` = ``window_end - 12 min``.
      * Proton / XRS ``last_data_time`` = last 5 min sample = ``window_end - 5 min``.
      * ``prediction_window.start_time`` / simulated ``issue_time`` = latest input
        (``window_end - 5 min``), per SPHINX: forecast only after last last_data_time.
      * ``prediction_window.end_time`` = start + 24 h.

    ``peak_intensity`` (when provided) is the MTM-LSTM ensemble prediction of
    future-window max ≥10 MeV proton flux (pfu), converted from
    ``mtm_lstm_pred_Future_log_ProtonFlux_max`` (log10 pfu). Peak *time* is not
    predicted by the current model, so ``time`` is omitted unless passed explicitly.
    """
    lasts = last_data_times_from_window_end(window_end)
    mag_last_s = to_iso_z(lasts["magnetogram"])
    proton_last_s = to_iso_z(lasts["proton"])
    # Latest ingested modality drives prediction start / simulated issue_time.
    pred_start = max(lasts.values())
    issue_s = to_iso_z(pred_start)
    end_s = to_iso_z(pred_start + timedelta(hours=24))
    primary_p = _clip01(float(hsmm_prob))
    all_clear = primary_p < float(all_clear_threshold)
    native_id = f"{pd.Timestamp(pred_start).strftime('%Y%m%d_%H%M%S')}_10MeV_sepnet_prism_r"

    model: dict = {
        "short_name": short_name,
        "spase_id": spase_id,
    }
    if include_model_version and version:
        model["version"] = version

    inputs: list[dict] = [
        {
            "magnetogram": {
                "observatory": "SDO",
                "instrument": "HMI",
                "products": [
                    {
                        "product": mag_product,
                        "last_data_time": mag_last_s,
                    }
                ],
            }
        }
    ]

    triggers: list[dict] = []
    if include_particle_xrs_inputs:
        # GOES proton (HSMM). Merged XRSB uses the same 5 min cadence / last_data_time
        # (field notes); no separate continuous-XRS Scoreboard block.
        triggers.append(
            {
                "particle_intensity": {
                    "observatory": "GOES",
                    "instrument": "EPS",
                    "last_data_time": proton_last_s,
                }
            }
        )

    forecast: dict = {
        "energy_channel": {
            "min": 10,
            "max": -1,
            "units": "MeV",
        },
        "species": "proton",
        "location": "earth",
        "prediction_window": {
            "start_time": issue_s,
            "end_time": end_s,
        },
    }
    if peak_intensity_pfu is not None and math.isfinite(float(peak_intensity_pfu)):
        peak_block: dict = {
            "intensity": float(f"{float(peak_intensity_pfu):.6g}"),
            "units": "pfu",
        }
        if peak_intensity_time:
            peak_block["time"] = str(peak_intensity_time)
        if (
            peak_intensity_uncertainty_pfu is not None
            and math.isfinite(float(peak_intensity_uncertainty_pfu))
        ):
            peak_block["uncertainty"] = float(
                f"{float(peak_intensity_uncertainty_pfu):.6g}"
            )
        forecast["peak_intensity"] = peak_block
    forecast["probabilities"] = [
        {
            # Keep 6 decimals so quiet-time HSMM probs (e.g. 1e-6) are not 0.
            "probability_value": float(f"{primary_p:.6f}"),
            "threshold": float(sep_threshold_pfu),
            "threshold_units": "pfu",
        }
    ]
    forecast["all_clear"] = {
        "all_clear_boolean": bool(all_clear),
        "threshold": float(sep_threshold_pfu),
        "threshold_units": "pfu",
        "probability_threshold": float(all_clear_threshold),
    }

    submission: dict = {
        "model": model,
        "issue_time": issue_s,
        "mode": mode,
    }
    if triggers:
        submission["triggers"] = triggers
    submission["inputs"] = inputs
    submission["forecasts"] = [forecast]
    # Katie: source_info is a peer of forecasts (after forecasts), not a child.
    submission["source_info"] = {
        "native_id": native_id,
    }

    return {"sep_forecast_submission": submission}


def write_schema_example(out_path: Path, **kwargs) -> None:
    """Write one illustrative JSON matching our MagPy-period HSMM product."""
    # window_end hour → magnetogram last = :48, proton/issue = :55
    window_end = pd.Timestamp("2023-07-02T00:00:00Z")
    sub = build_submission(
        window_end=window_end,
        hsmm_prob=kwargs.get("hsmm_prob", 1e-6),
        all_clear_threshold=kwargs.get("all_clear_threshold", DEFAULT_ALL_CLEAR_THRESHOLD),
        short_name=kwargs.get("short_name", DEFAULT_SHORT_NAME),
        version=kwargs.get("version", DEFAULT_VERSION),
        spase_id=kwargs.get("spase_id", DEFAULT_SPASE),
        mode=kwargs.get("mode", "simulated_realtime_forecast"),
        include_particle_xrs_inputs=kwargs.get("include_particle_xrs_inputs", True),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sub, indent=2) + "\n")
    print(f"[export] wrote schema example → {out_path}")


def np_isfinite(x: float) -> bool:
    try:
        return x == x and abs(x) != float("inf")
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(P("magpy_validation_dir")))
    ap.add_argument(
        "--input-csv",
        default=None,
        help="Default: predictions/magpy_period_ensemble_hsmm.csv (needs hsmm_prob).",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--short-name", default=DEFAULT_SHORT_NAME)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--spase-id", default=DEFAULT_SPASE)
    ap.add_argument(
        "--mode",
        default="simulated_realtime_forecast",
        choices=("forecast", "historical", "nowcast", "simulated_realtime_forecast"),
        help="Use simulated_realtime_forecast for MagPy-period retrospective continuous run.",
    )
    ap.add_argument(
        "--all-clear-threshold",
        type=float,
        default=DEFAULT_ALL_CLEAR_THRESHOLD,
        help="probability_threshold for all_clear (default: 0.1).",
    )
    ap.add_argument("--mag-product", default=DEFAULT_MAG_PRODUCT)
    ap.add_argument(
        "--no-particle-xrs-inputs",
        action="store_true",
        help="Omit GOES proton / XRS trigger blocks (magnetogram-only inputs).",
    )
    ap.add_argument("--stride-hours", type=float, default=1.0,
                    help="Keep one forecast every N hours (1 = every hourly row)")
    ap.add_argument("--limit", type=int, default=0, help="Optional max files (0 = all)")
    ap.add_argument(
        "--write-schema-example-only",
        action="store_true",
        help="Only write docs/SEPNET-PRISM-R_scoreboard_schema_example.json and exit",
    )
    ap.add_argument(
        "--review-examples-only",
        action="store_true",
        help="Rewrite docs/srag_review_examples from the HSMM CSV (quiet + SEP rows) and exit",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    ac_thr = float(args.all_clear_threshold)
    include_px = not bool(args.no_particle_xrs_inputs)

    if args.write_schema_example_only:
        write_schema_example(
            docs / "SEPNET-PRISM-R_scoreboard_schema_example.json",
            short_name=args.short_name,
            version=args.version,
            spase_id=args.spase_id,
            mode=args.mode,
            all_clear_threshold=ac_thr,
            include_particle_xrs_inputs=include_px,
        )
        return

    src = Path(args.input_csv) if args.input_csv else (
        root / "predictions" / "magpy_period_ensemble_hsmm.csv"
    )
    if not src.is_file():
        raise FileNotFoundError(
            f"Missing {src}. Run MagPy_period_score.py and MagPy_period_hsmm.py first "
            f"(or pass --write-schema-example-only)."
        )

    df = pd.read_csv(src, low_memory=False)
    df["window_end"] = _parse_ts(df["window_end"])
    df["window_begin"] = _parse_ts(df["window_begin"])
    if "hsmm_prob" not in df.columns:
        raise ValueError(f"Missing hsmm_prob in {src}. Run MagPy_period_hsmm.py first.")
    df = df[df["window_end"].notna()].sort_values("window_end").reset_index(drop=True)

    def _row_submission(row: pd.Series) -> dict | None:
        hsmm_p = float(pd.to_numeric(row.get("hsmm_prob"), errors="coerce"))
        if not np_isfinite(hsmm_p):
            return None
        return build_submission(
            window_end=row["window_end"],
            hsmm_prob=hsmm_p,
            all_clear_threshold=ac_thr,
            short_name=args.short_name,
            version=args.version,
            spase_id=args.spase_id,
            mode=args.mode,
            mag_product=args.mag_product,
            include_particle_xrs_inputs=include_px,
            peak_intensity_pfu=peak_intensity_pfu_from_row(row),
        )

    if args.review_examples_only:
        review_dir = docs / "srag_review_examples"
        review_dir.mkdir(parents=True, exist_ok=True)
        # Same physical feature rows as prior review pack, keyed by window_begin.
        # JSON stamps last_data_time from raw cadences relative to window_end.
        targets = [
            ("quiet_all_clear", pd.Timestamp("2023-07-09T08:00:00Z")),
            ("sep_warning", pd.Timestamp("2023-07-17T02:00:00Z")),
        ]
        for old in review_dir.glob("SEPNET-PRISM-R.*.json"):
            old.unlink()
        manifest = []
        for tag, wb in targets:
            m = df[df["window_begin"] == wb]
            if m.empty:
                if tag.startswith("quiet"):
                    idx = df["hsmm_prob"].astype(float).idxmin()
                else:
                    jul = df[(df["window_begin"] >= "2023-07-01") & (df["window_begin"] < "2023-08-01")]
                    idx = (jul if len(jul) else df)["hsmm_prob"].astype(float).idxmax()
                row = df.loc[idx]
            else:
                row = m.iloc[0]
            sub = _row_submission(row)
            if sub is None:
                continue
            issue_s = sub["sep_forecast_submission"]["issue_time"]
            pred_s = sub["sep_forecast_submission"]["forecasts"][0]["prediction_window"]["start_time"]
            fname = (
                f"{args.short_name}."
                f"{iso_to_filename_timestamp(pred_s)}."
                f"{iso_to_filename_timestamp(issue_s)}.json"
            )
            (review_dir / fname).write_text(json.dumps(sub, indent=2) + "\n")
            hsmm_p = float(sub["sep_forecast_submission"]["forecasts"][0]["probabilities"][0]["probability_value"])
            ac = bool(sub["sep_forecast_submission"]["forecasts"][0]["all_clear"]["all_clear_boolean"])
            manifest.append(
                {
                    "file": fname,
                    "tag": tag,
                    "note": (
                        "Quiet: HSMM≈0 → all_clear=true"
                        if tag.startswith("quiet")
                        else "SEP: HSMM high → all_clear=false"
                    ),
                    "hsmm": hsmm_p,
                    "all_clear": ac,
                }
            )
            print(f"[export] review example → {review_dir / fname}")
        (review_dir / "examples_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        write_schema_example(
            docs / "SEPNET-PRISM-R_scoreboard_schema_example.json",
            short_name=args.short_name,
            version=args.version,
            spase_id=args.spase_id,
            mode=args.mode,
            all_clear_threshold=ac_thr,
            include_particle_xrs_inputs=include_px,
        )
        return

    out_dir = Path(args.out_dir) if args.out_dir else (root / "scoreboard_json")
    out_dir.mkdir(parents=True, exist_ok=True)

    if float(args.stride_hours) > 1:
        epoch_h = (df["window_end"].astype("int64") // 10**9) // int(float(args.stride_hours) * 3600)
        df = df.loc[~epoch_h.duplicated(keep="first")].reset_index(drop=True)

    if args.limit and args.limit > 0:
        df = df.iloc[: int(args.limit)].copy()

    n = 0
    for _, row in df.iterrows():
        sub = _row_submission(row)
        if sub is None:
            continue
        issue_s = sub["sep_forecast_submission"]["issue_time"]
        pred_s = sub["sep_forecast_submission"]["forecasts"][0]["prediction_window"]["start_time"]
        fname = (
            f"{args.short_name}."
            f"{iso_to_filename_timestamp(pred_s)}."
            f"{iso_to_filename_timestamp(issue_s)}.json"
        )
        (out_dir / fname).write_text(json.dumps(sub, indent=2) + "\n")
        n += 1

    write_schema_example(
        docs / "SEPNET-PRISM-R_scoreboard_schema_example.json",
        short_name=args.short_name,
        version=args.version,
        spase_id=args.spase_id,
        mode=args.mode,
        all_clear_threshold=ac_thr,
        include_particle_xrs_inputs=include_px,
    )

    manifest = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n_files": n,
        "out_dir": str(out_dir),
        "source": str(src),
        "probability_value": "hsmm_prob (HSMM only; no LSTM probability fields)",
        "peak_intensity": (
            f"10**{PRED_LOG_PROTON_FLUX_COL} (MTM-LSTM ensemble; pfu; "
            "peak time not predicted)"
        ),
        "all_clear_threshold": ac_thr,
        "short_name": args.short_name,
        "version": args.version,
        "mode": args.mode,
        "stride_hours": float(args.stride_hours),
        "inputs_note": (
            "24 h summaries over [window_begin, window_end). Raw last samples: "
            "SHARP magnetogram last_data_time = window_end - 12 min; "
            "GOES proton / Merged XRS = window_end - 5 min. "
            "issue_time / prediction_window.start = latest input (window_end - 5 min)."
        ),
        "native_id_placement": "sep_forecast_submission.source_info.native_id (peer after forecasts)",
        "schema_refs": {
            "visual_schema_pdf": "https://ccmc.gsfc.nasa.gov/static/files/SEPSB/sepscoreboard_visual_schema.v20261023.pdf",
            "example_json": "https://ccmc.gsfc.nasa.gov/static/files/SEPSB/sep_scoreboard_example_json_file.json",
            "scoreboard_home": "https://ccmc.gsfc.nasa.gov/scoreboards/sep/",
            "local_docs": str(docs),
        },
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[export] wrote {n:,} JSON files → {out_dir} "
        f"(HSMM-only; all_clear thr={ac_thr})"
    )


if __name__ == "__main__":
    main()
