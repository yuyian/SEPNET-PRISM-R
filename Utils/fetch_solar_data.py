from __future__ import annotations

import argparse
import json
import os
import ssl
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from bs4 import BeautifulSoup
try:
    import drms  # type: ignore
except Exception:  # pragma: no cover
    drms = None

try:
    from sunpy.net import Fido, attrs as a  # type: ignore
except Exception:  # pragma: no cover
    Fido = None
    a = None

UTC = timezone.utc
ISWA_HAPI_URL = "https://iswa.gsfc.nasa.gov/IswaSystemWebApp/hapi/data"
ISWA_FILL_VALUES = {-100000, -100000.0}

# Global SSL fix (safe to call multiple times)
def ensure_unverified_ssl() -> None:
    """Allow HTTPS connections without verifying SSL certificates."""
    ssl._create_default_https_context = ssl._create_unverified_context

# Shared utilities
def convert_datetime(date_str: Any) -> pd.Timestamp:
    """Parse GOES/HEK datetime formats to UTC Timestamp."""
    if pd.isna(date_str):
        return pd.NaT
    formats = ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d/%H:%M")
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt, utc=True)
        except Exception:
            continue
    return pd.NaT

def flare_strength_to_flux(goes_class: Any) -> float:
    """Convert GOES class (e.g. 'M1.5') to W/m² flux."""
    if pd.isna(goes_class):
        return np.nan
    s = str(goes_class).strip()
    if not s:
        return np.nan
    type_char = s[0].upper()
    try:
        number = float(s[1:]) or 1.0
    except Exception:
        return np.nan
    scale_map = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
    return number * scale_map.get(type_char, np.nan)

def _normalize_time_range(
    start_time: datetime | None,
    end_time: datetime | None,
    default_days_back: int,
    anchor: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Normalize start/end times with flexible defaults."""
    if anchor is None:
        anchor = datetime.now(tz=UTC)
    if start_time is not None and end_time is not None:
        return start_time, end_time
    if end_time is not None:
        return end_time - timedelta(days=default_days_back), end_time
    if start_time is not None:
        return start_time, start_time + timedelta(days=default_days_back)
    end_time = anchor
    return end_time - timedelta(days=default_days_back), end_time

def safe_float(s: str) -> float:
    """Parse CDAW floats, handling ----, >87, etc."""
    if pd.isna(s) or s in ('----', '', '-'):
        return np.nan
    s_clean = s.lstrip('><')
    try:
        return float(s_clean.replace(',', ''))
    except ValueError:
        return np.nan

def _next_month(dt: datetime) -> datetime:
    """Advance to next month."""
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1)
    return dt.replace(month=dt.month + 1)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_cli_datetime(value: str) -> datetime:
    """Parse CLI datetime; ``now`` means current UTC."""
    if value.strip().lower() in {"now", "today"}:
        return datetime.now(tz=UTC)
    ts = pd.to_datetime(value, utc=True)
    return ts.to_pydatetime()


def _parse_iswa_hapi_csv(text: str) -> pd.DataFrame:
    """Parse ISWA HAPI CSV (header lines start with ``#``)."""
    header_lines, data_lines = [], []
    for line in text.splitlines():
        if line.startswith("#"):
            header_lines.append(line[1:])
        elif line.strip():
            data_lines.append(line)
    if not data_lines:
        return pd.DataFrame()
    header_json = json.loads("\n".join(header_lines))
    colnames = [p["name"] for p in header_json["parameters"]]
    return pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=colnames)


def _clean_iswa_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(~values.isin(ISWA_FILL_VALUES))


def fetch_iswa_hapi(
    dataset_id: str,
    start_time: datetime,
    end_time: datetime,
    *,
    chunk_days: int = 30,
    timeout: int = 120,
) -> pd.DataFrame:
    """Fetch one ISWA HAPI dataset in time chunks."""
    start_time, end_time = _ensure_utc(start_time), _ensure_utc(end_time)
    frames: list[pd.DataFrame] = []
    current = start_time
    while current < end_time:
        chunk_end = min(current + timedelta(days=chunk_days), end_time)
        print(f"  ISWA {dataset_id}: {current.date()} -> {chunk_end.date()}")
        params = {
            "id": dataset_id,
            "time.min": current.strftime("%Y-%m-%dT%H:%M:%S.0Z"),
            "time.max": chunk_end.strftime("%Y-%m-%dT%H:%M:%S.0Z"),
            "format": "csv",
            "include": "header",
        }
        try:
            r = requests.get(ISWA_HAPI_URL, params=params, timeout=timeout)
            r.raise_for_status()
            chunk = _parse_iswa_hapi_csv(r.text)
            if not chunk.empty:
                frames.append(chunk)
        except Exception as exc:
            print(f"    chunk failed: {exc}")
        current = chunk_end
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    """
    Robustly parse common timestamp formats used in this repo.

    Handles:
    - ISO: 2025-09-10T23:55:00Z
    - JSOC/DRMS: 2025.09.10_00:00:00_TAI (treated as UTC for ordering/dedup)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):  # type: ignore[arg-type]
        return pd.NaT
    s = str(value).strip().strip('"')
    if not s:
        return pd.NaT
    if s.endswith("_TAI"):
        base = s[:-4]
        ts = pd.to_datetime(base, format="%Y.%m.%d_%H:%M:%S", errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(base, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        return ts.tz_localize(UTC)
    return pd.to_datetime(s, utc=True, errors="coerce")


def _read_csv_for_schema_merge(
    path: Path,
    schema_columns: list[str],
) -> tuple[pd.DataFrame, bool]:
    """
    Load an on-disk CSV for schema-aware merge.

    Older ``sharp_nrt.csv`` files may mix 34-column headers with 41+ column
    append rows; fall back to a tolerant parser instead of failing the run.

    Returns (dataframe, ragged_on_disk).
    """
    try:
        return pd.read_csv(path, low_memory=False), False
    except pd.errors.ParserError as exc:
        print(
            f"WARNING: {path.name} has ragged rows ({exc}); "
            "reading valid rows with on_bad_lines='skip'."
        )
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if df.empty:
            raise
        return df, True


def _read_csv_last_timestamp(path: Path, time_col: str) -> pd.Timestamp | None:
    """Read the last timestamp from a large CSV without loading the whole file."""
    last_line = None
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        if pos == 0:
            return None
        block = min(8192, pos)
        fh.seek(-block, os.SEEK_END)
        lines = fh.read().splitlines()
        for raw in reversed(lines):
            if raw.strip():
                last_line = raw.decode("utf-8", errors="replace")
                break
    if last_line is None or last_line.startswith(time_col):
        return None
    ts_raw = last_line.split(",", 1)[0].strip().strip('"')
    ts = _to_utc_timestamp(ts_raw)
    return None if pd.isna(ts) else ts


def append_csv(
    new_df: pd.DataFrame,
    path: str | os.PathLike[str],
    time_col: str,
    *,
    schema_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Append rows to CSV, dedupe on ``time_col``, sort chronologically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_df.empty:
        if path.is_file():
            return pd.read_csv(path)
        return new_df

    out = new_df.copy()
    # SHARP comes back with JSOC time strings (often *_TAI) — parse robustly.
    out[time_col] = out[time_col].map(_to_utc_timestamp)
    out = out.dropna(subset=[time_col]).drop_duplicates(subset=[time_col], keep="last")

    if schema_columns:
        for col in schema_columns:
            if col not in out.columns:
                out[col] = np.nan
        out = out[list(schema_columns)]

    if path.is_file():
        if schema_columns:
            ragged_on_disk = False
            try:
                existing, ragged_on_disk = _read_csv_for_schema_merge(path, schema_columns)
            except pd.errors.ParserError:
                backup = path.with_name(path.stem + ".corrupt.bak" + path.suffix)
                path.replace(backup)
                print(
                    f"WARNING: could not read {path.name}; backed up to {backup.name}. "
                    f"Writing {len(out):,} incoming rows as the new file."
                )
                existing = pd.DataFrame(columns=schema_columns)
            existing[time_col] = existing[time_col].map(_to_utc_timestamp)
            for col in schema_columns:
                if col not in existing.columns:
                    existing[col] = np.nan
            existing = existing[list(schema_columns)]
            last_ts = existing[time_col].max()
            if ragged_on_disk:
                print(
                    "Merging full incoming batch (not incremental-by-last-ts) "
                    "to repair skipped malformed rows."
                )
            elif pd.notna(last_ts):
                out = out[out[time_col] > last_ts]
            if out.empty:
                print(f"No new rows after {last_ts} -> {path}")
                return existing
            combined = (
                pd.concat([existing, out], ignore_index=True)
                .drop_duplicates(subset=[time_col], keep="last")
                .sort_values(time_col)
                .reset_index(drop=True)
            )
            if time_col == "date":
                combined[time_col] = pd.to_datetime(combined[time_col], utc=True).dt.strftime("%Y-%m-%d")
            combined.to_csv(path, index=False)
            print(f"Merged {len(out):,} new rows ({len(combined):,} total) -> {path}")
            return combined

        last_ts = _read_csv_last_timestamp(path, time_col)
        if last_ts is not None:
            out = out[out[time_col] > last_ts]
        if out.empty:
            print(f"No new rows after {last_ts} -> {path}")
            return pd.read_csv(path)
        write_header = False
        out_sorted = out.sort_values(time_col)
        # Preserve existing on-disk string formatting for GOES files.
        if time_col == "time_posix":
            out_sorted = out_sorted.assign(
                time_posix=out_sorted["time_posix"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        elif time_col == "time":
            out_sorted = out_sorted.assign(
                time=out_sorted["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        elif time_col == "date":
            out_sorted = out_sorted.assign(
                date=pd.to_datetime(out_sorted["date"], utc=True).dt.strftime("%Y-%m-%d")
            )
        out_sorted.to_csv(path, mode="a", header=write_header, index=False)
        print(f"Appended {len(out_sorted):,} rows -> {path}")
        return out_sorted

    out = out.sort_values(time_col).reset_index(drop=True)
    if time_col == "time_posix":
        out["time_posix"] = out["time_posix"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif time_col == "time":
        out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif time_col == "date":
        out["date"] = pd.to_datetime(out["date"], utc=True).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    print(f"Wrote {len(out):,} rows -> {path}")
    return out


# Raw NRT product CSVs written under data_dir (append/dedupe by default).
NRT_RAW_PRODUCTS: tuple[tuple[str, str], ...] = (
    ("sharp_nrt.csv", "T_REC"),
    ("flare_nrt.csv", "start_time"),
    ("donki_cme_nrt.csv", "startTime"),
    ("GOESHAPI_ProtonFlux_nrt.csv", "time"),
    ("XRS_nrt.csv", "time"),
    ("sunspot_nrt.csv", "date"),
)

SILSO_DAILY_URL = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
# NOAA SWPC observed daily SESC sunspot number (near-real-time; updates through today).
SWPC_SSN_URL = "https://services.swpc.noaa.gov/json/solar-cycle/swpc_observed_ssn.json"
# SILSO Estimated International Sunspot Number (updates every few minutes; current month).
SILSO_EISN_URL = "https://www.sidc.be/SILSO/DATA/EISN/EISN_current.csv"
# SWPC/SESC (~old Zürich / V1) -> SILSO V2.0: drop the historical 0.6 factor => × 1/0.6.
SWPC_TO_SILSO_V2_SCALE = 1.0 / 0.6  # ≈ 1.6667
SSN_NRT_COLUMNS: tuple[str, ...] = (
    "date", "ssn", "ssn_std", "n_obs", "is_definitive", "decimal_year", "source",
)


def prune_nrt_raw_csvs(
    data_dir: str | os.PathLike[str],
    keep_after: datetime,
) -> None:
    """Drop rows older than ``keep_after`` from every raw NRT product CSV.

    Used after append-fetch so the on-disk archive does not grow without bound.
    """
    data_dir = Path(data_dir)
    keep_after = _ensure_utc(keep_after)
    for fname, time_col in NRT_RAW_PRODUCTS:
        path = data_dir / fname
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[prune] skip {path.name}: read failed ({exc})")
            continue
        if time_col not in df.columns or df.empty:
            continue
        ts = df[time_col].map(_to_utc_timestamp)
        keep = ts >= keep_after
        n_drop = int((~keep & ts.notna()).sum())
        if n_drop == 0:
            print(f"[prune] {path.name}: keep all {len(df):,} rows (>= {keep_after.isoformat()})")
            continue
        out = df.loc[keep].copy()
        # Normalize time strings for GOES-style files (same as append_csv writes).
        out[time_col] = ts.loc[keep]
        if time_col == "time":
            out[time_col] = out[time_col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif time_col == "date":
            out[time_col] = out[time_col].dt.strftime("%Y-%m-%d")
        elif time_col == "T_REC":
            # Keep original T_REC string from the filtered rows (already in df).
            out[time_col] = df.loc[keep, time_col].to_numpy()
        else:
            # ISO-ish write for flare / DONKI timestamps.
            out[time_col] = (
                pd.to_datetime(out[time_col], utc=True, errors="coerce")
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        out.to_csv(path, index=False)
        print(
            f"[prune] {path.name}: dropped {n_drop:,} rows older than "
            f"{keep_after.isoformat()} -> {len(out):,} remain"
        )


# SHARP keywords aligned with rolling_combinded_seq SHARP_* aggregates (All_features_RData.R).
SHARP_FEATURE_KEYS: tuple[str, ...] = (
    "USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH",
    "MEANJZD", "TOTUSJZ", "MEANALP", "MEANJZH", "SAVNCPP", "MEANPOT", "TOTPOT",
    "MEANSHR", "SHRGT45", "USFLUXL", "MEANGBL",
    "NPIX", "SIZE", "AREA", "NACR", "SIZE_ACR", "AREA_ACR",
    "LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX",
    "R_VALUE", "CAR_ROT", "CMASKL",
    "LAT_FWT", "CRLT_OBS", "LON_FWT", "CRLN_OBS", "DSUN_OBS", "RSUN_OBS",
)
SHARP_META_KEYS: tuple[str, ...] = (
    "T_OBS", "T_REC", "CADENCE", "QUALITY", "HARPNUM", "NOAA_AR",
)
SHARP_QUERY_KEYS: tuple[str, ...] = SHARP_META_KEYS + SHARP_FEATURE_KEYS
SHARP_MIN_CHUNK = timedelta(hours=6)


def normalize_sharp_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every training/meta SHARP column exists with stable column order."""
    if df.empty:
        return pd.DataFrame(columns=list(SHARP_QUERY_KEYS))
    out = df.copy()
    for col in SHARP_QUERY_KEYS:
        if col not in out.columns:
            out[col] = np.nan
    extra = [c for c in out.columns if c not in SHARP_QUERY_KEYS]
    return out[list(SHARP_QUERY_KEYS) + extra]


def report_sharp_feature_coverage(df: pd.DataFrame) -> None:
    """Log whether JSOC returned all columns used by the R aggregation pipeline."""
    if df.empty:
        return
    missing_cols = [c for c in SHARP_FEATURE_KEYS if c not in df.columns]
    empty_cols = [
        c for c in SHARP_FEATURE_KEYS
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").isna().all()
    ]
    bad = sorted(set(missing_cols) | set(empty_cols))
    if bad:
        print(f"[SHARP] WARNING: training features missing or all-NaN: {bad}")
        print(f"[SHARP] Feature coverage: {len(SHARP_FEATURE_KEYS) - len(bad)}/{len(SHARP_FEATURE_KEYS)}")
    else:
        print(f"[SHARP] All {len(SHARP_FEATURE_KEYS)} training feature columns present with data.")


# SHARP fetcher
def fetch_sharp(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    default_days_back: int = 3,
    weeks_per_chunk: int = 1,
    output_csv: Optional[str] = "sharp_nrt.csv",
    email: str = "uhek@umich.edu",
) -> pd.DataFrame:
    """Fetch SHARP data (HMI definitive + NRT fallback) in week chunks."""
    if drms is None:
        raise ModuleNotFoundError(
            "Missing dependency 'drms'. Install it (e.g. `pip install drms`) to fetch SHARP."
        )
    ensure_unverified_ssl()
    SHARP_SERIES = ["hmi.sharp_cea_720s", "hmi.sharp_cea_720s_nrt"]
    SHARP_FROM = datetime(2010, 5, 1, tzinfo=UTC)

    def _is_sharp_chunk_too_large(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "out of memory for query result" in msg
            or "query failed" in msg
            or "status=6" in msg
            or "[status=6]" in msg
        )

    def _fetch_sharp_range(range_start: datetime, range_end: datetime) -> pd.DataFrame:
        client = drms.Client(email=email)
        for series in SHARP_SERIES:
            try:
                start_str = range_start.strftime("%Y.%m.%d_%H:%M")
                end_str = range_end.strftime("%Y.%m.%d_%H:%M")
                df_chunk = client.query(
                    f"{series}[][{start_str}-{end_str}]",
                    key=list(SHARP_QUERY_KEYS),
                )
                if not df_chunk.empty:
                    return normalize_sharp_dataframe(df_chunk)
            except Exception as exc:
                print(f"  {series} failed: {exc}")
                if _is_sharp_chunk_too_large(exc):
                    # Let caller decide whether to split this range into smaller chunks.
                    raise
        return pd.DataFrame()

    def _collect_sharp_frames(
        range_start: datetime,
        range_end: datetime,
        *,
        min_chunk: timedelta = SHARP_MIN_CHUNK,
    ) -> list[pd.DataFrame]:
        """Fetch a time range, recursively splitting on JSOC OOM."""
        try:
            df_chunk = _fetch_sharp_range(range_start, range_end)
            return [df_chunk] if not df_chunk.empty else []
        except Exception as exc:
            if not _is_sharp_chunk_too_large(exc) or (range_end - range_start) <= min_chunk:
                print(f"  SHARP chunk failed {range_start} -> {range_end}: {exc}")
                return []
            mid = range_start + (range_end - range_start) / 2
            print(
                f"  Chunk {range_start.date()} {range_start.time()} -> "
                f"{range_end.date()} {range_end.time()} too large; splitting..."
            )
            return _collect_sharp_frames(range_start, mid, min_chunk=min_chunk) + _collect_sharp_frames(
                mid, range_end, min_chunk=min_chunk
            )

    start_time, end_time = _normalize_time_range(start_time, end_time, default_days_back)
    current_start = max(start_time, SHARP_FROM)
    all_frames: list[pd.DataFrame] = []

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(weeks=weeks_per_chunk), end_time)
        print(f"Fetching SHARP {current_start.date()} to {chunk_end.date()}...")
        all_frames.extend(_collect_sharp_frames(current_start, chunk_end))
        current_start = chunk_end

    if not all_frames:
        raise RuntimeError("No SHARP data retrieved.")

    sharp = normalize_sharp_dataframe(
        pd.concat(all_frames, ignore_index=True).sort_values("T_REC")
    )
    report_sharp_feature_coverage(sharp)
    if output_csv:
        sharp.to_csv(output_csv, index=False)
    print(f"Retrieved {len(sharp):,} SHARP records with columns: {list(sharp.columns)}")
    return sharp

# SMARP fetcher
def fetch_smarp(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    default_days_back: int = 3,
    weeks_per_chunk: int = 1,
    output_csv: Optional[str] = "smarp_nrt.csv",
    email: str = "uhek@umich.edu",
) -> pd.DataFrame:
    """Fetch SMARP (MDI) data in week chunks (1996-2010)."""
    ensure_unverified_ssl()
    SERIES = "mdi.smarp_cea_96m"
    FROM, TO = datetime(1996, 4, 23, tzinfo=UTC), datetime(2010, 10, 27, tzinfo=UTC)
    
    start_time, end_time = _normalize_time_range(start_time, end_time, default_days_back)
    current_start = max(start_time, FROM)
    end_time = min(end_time, TO)
    
    if current_start >= end_time:
        raise RuntimeError(f"No SMARP data in {start_time}–{end_time}; valid: {FROM}–{TO}")
    
    all_frames = []
    while current_start < end_time:
        chunk_end = min(current_start + timedelta(weeks=weeks_per_chunk), end_time)
        print(f"Fetching SMARP {current_start.date()} to {chunk_end.date()}...")
        try:
            client = drms.Client(email=email)
            start_str, end_str = current_start.strftime("%Y.%m.%d_%H:%M"), chunk_end.strftime("%Y.%m.%d_%H:%M")
            df_chunk = client.query(f"{SERIES}[][{start_str}-{end_str}]", key=[
                "T_OBS", "T_REC", "CADENCE", "USFLUXL", "MEANGBL", "R_VALUE", "QUALITY",
                "TARPNUM", "NPIX", "SIZE", "AREA", "NACR", "SIZE_ACR", "AREA_ACR",
                "LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX", "NOAA_AR"
            ])
            if not df_chunk.empty:
                all_frames.append(df_chunk)
        except Exception as e:
            print(f"  Failed: {e}")
        current_start = chunk_end
    
    if not all_frames:
        raise RuntimeError("No SMARP data retrieved.")
    
    smarp = pd.concat(all_frames, ignore_index=True).sort_values("T_REC").reset_index(drop=True)
    if output_csv:
        smarp.to_csv(output_csv, index=False)
    print(f"Retrieved {len(smarp):,} SMARP records")
    return smarp

# GOES flares
def fetch_flares(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    default_days_back: int = 3,
    output_csv: Optional[str] = "flare_nrt.csv",
    min_goes_class: str = "A1.0",
    chunk_days: int = 30,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch GOES flares from HEK in time chunks (avoids huge single-query timeouts)."""
    if Fido is None or a is None:
        raise ModuleNotFoundError(
            "Missing dependency 'sunpy'. Install it (e.g. `pip install sunpy`) to fetch flares from HEK."
        )
    ensure_unverified_ssl()
    start_time, end_time = _normalize_time_range(start_time, end_time, default_days_back)
    hek_cols = [
        "event_starttime", "event_peaktime", "event_endtime",
        "fl_goescls", "ar_noaanum", "hgs_x", "hgs_y",
    ]
    out_cols = [
        "start_time", "peak_time", "end_time", "label",
        "ar_noaanum", "hgs_x_sunpy", "hgs_y_sunpy",
    ]

    all_frames: list[pd.DataFrame] = []
    current = start_time
    while current < end_time:
        chunk_end = min(current + timedelta(days=chunk_days), end_time)
        t_start = current.strftime("%Y/%m/%d")
        t_end = (chunk_end + timedelta(days=1)).strftime("%Y/%m/%d")
        print(f"Fetching flares {current.date()} -> {chunk_end.date()}...")
        chunk_df: pd.DataFrame | None = None
        for attempt in range(1, max_retries + 1):
            try:
                result = Fido.search(
                    a.Time(t_start, t_end),
                    a.hek.EventType("FL"),
                    a.hek.FL.GOESCls > min_goes_class,
                    a.hek.OBS.Observatory == "GOES",
                )
                if len(result) == 0 or len(result["hek"]) == 0:
                    chunk_df = pd.DataFrame(columns=hek_cols)
                    break
                chunk_df = result["hek"][hek_cols].to_pandas()
                break
            except OSError as exc:
                if attempt < max_retries:
                    wait_s = 2 * attempt
                    print(f"  HEK chunk failed (attempt {attempt}/{max_retries}): {exc}; retry in {wait_s}s")
                    time.sleep(wait_s)
                else:
                    print(f"  HEK chunk failed after {max_retries} attempts: {exc}")
            except Exception as exc:
                print(f"  HEK chunk failed: {exc}")
                break
        if chunk_df is not None and not chunk_df.empty:
            all_frames.append(chunk_df)
        current = chunk_end

    if not all_frames:
        flare_df = pd.DataFrame(columns=out_cols + ["fl_cls"])
    else:
        flare_df = pd.concat(all_frames, ignore_index=True)
        flare_df.columns = out_cols
        flare_df["fl_cls"] = flare_df["label"].astype(str).str[0]
        flare_df = flare_df.drop_duplicates(
            subset=["start_time", "peak_time", "end_time", "label"], keep="last"
        ).reset_index(drop=True)

    if output_csv:
        flare_df.to_csv(output_csv, index=False)
    print(f"Retrieved {len(flare_df):,} flare records")
    return flare_df

# 10 MeV flux
def fetch_10mev_flux(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    default_days_back: int = 3,
    output_csv: Optional[str] = "goes_10mev_flux_nrt.csv",
    primary_satellite: str = "primary",
) -> pd.DataFrame:
    """Fetch GOES >10 MeV proton flux from ISWA."""
    start_time, end_time = _normalize_time_range(start_time, end_time, default_days_back)
    dataset_id = {"primary": "goesp_part_flux_P5M", "secondary": "goess_part_flux_P5M"}[primary_satellite]
    raw = fetch_iswa_hapi(dataset_id, start_time, end_time)
    if raw.empty:
        flux_df = pd.DataFrame(columns=["time", "flux"])
    else:
        tmp = raw.copy()
        tmp["time"] = pd.to_datetime(tmp["Time"], utc=True, errors="coerce")
        tmp["flux"] = _clean_iswa_numeric(tmp["P10"])
        flux_df = (
            tmp[["time", "flux"]]
            .dropna(subset=["time"])
            .drop_duplicates(subset=["time"], keep="last")
            .sort_values("time")
            .reset_index(drop=True)
        )
    if output_csv and not flux_df.empty:
        flux_df.to_csv(output_csv, index=False)
    return flux_df


def fetch_proton_flux_goeshapi(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    default_days_back: int = 3,
    output_csv: Optional[str] = None,
    primary_satellite: str = "primary",
) -> pd.DataFrame:
    """
    Fetch GOES >10 MeV proton flux in ``Data/GOESHAPI_ProtonFlux.csv`` schema:
    ``time, ProtonFlux, goes`` (``goes=RT`` for ISWA real-time primary).
    """
    start_time, end_time = _normalize_time_range(start_time, end_time, default_days_back)
    dataset_id = {"primary": "goesp_part_flux_P5M", "secondary": "goess_part_flux_P5M"}[primary_satellite]
    raw = fetch_iswa_hapi(dataset_id, start_time, end_time)
    if raw.empty:
        out = pd.DataFrame(columns=["time", "ProtonFlux", "goes"])
    else:
        tmp = raw.copy()
        tmp["time"] = pd.to_datetime(tmp["Time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        tmp["ProtonFlux"] = _clean_iswa_numeric(tmp["P10"])
        tmp["goes"] = "RT"
        out = (
            tmp[["time", "ProtonFlux", "goes"]]
            .dropna(subset=["time", "ProtonFlux"])
            .drop_duplicates(subset=["time"], keep="last")
            .sort_values("time")
            .reset_index(drop=True)
        )
    if output_csv and not out.empty:
        out.to_csv(output_csv, index=False)
    print(f"Retrieved {len(out):,} GOESHAPI proton-flux rows")
    return out


def fetch_xrsb(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    default_days_back: int = 3,
    output_csv: Optional[str] = None,
    primary_satellite: str = "primary",
    cadence_minutes: int = 5,
) -> pd.DataFrame:
    """
    Fetch GOES 1-8 Å XRS-B flux (``Long_Wave``) in ``Data/XRS.csv`` schema: ``time, xrsb``.
    ISWA serves 1-minute data; resample to 5-minute timestamps to match existing files.
    """
    start_time, end_time = _normalize_time_range(start_time, end_time, default_days_back)
    dataset_id = {"primary": "goesp_xray_flux_P1M", "secondary": "goess_xray_flux_P1M"}[primary_satellite]
    raw = fetch_iswa_hapi(dataset_id, start_time, end_time, chunk_days=14)
    if raw.empty:
        out = pd.DataFrame(columns=["time", "xrsb"])
    else:
        tmp = raw.copy()
        tmp["time"] = pd.to_datetime(tmp["Time"], utc=True, errors="coerce")
        tmp["xrsb"] = _clean_iswa_numeric(tmp["Long_Wave"])
        tmp = tmp.dropna(subset=["time", "xrsb"]).sort_values("time")
        if cadence_minutes > 1:
            tmp = tmp.set_index("time")
            tmp = tmp.resample(f"{cadence_minutes}min").last().dropna(subset=["xrsb"]).reset_index()
        out = (
            tmp.assign(time=lambda d: d["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            [["time", "xrsb"]]
            .drop_duplicates(subset=["time"], keep="last")
            .reset_index(drop=True)
        )
    if output_csv and not out.empty:
        out.to_csv(output_csv, index=False)
    print(f"Retrieved {len(out):,} XRS-B rows")
    return out

# DONKI CMEs
def fetch_donki_cme(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    default_days_back: int = 3,
    output_csv: Optional[str] = "donki_cme_nrt.csv",
    output_gcs_csv: Optional[str] = "donki_gcs_cme_nrt.csv",
    api_url: str = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME",
) -> pd.DataFrame:
    """Fetch DONKI CMEs with GCS propagation to 1 AU."""
    start_time, end_time = _normalize_time_range(start_time, end_time, default_days_back)
    params = {"startDate": start_time.strftime("%Y-%m-%d"), "endDate": end_time.strftime("%Y-%m-%d")}
    
    try:
        r = requests.get(api_url, params=params, timeout=600)
        r.raise_for_status()
        cmes = r.json()
        if not cmes:
            return pd.DataFrame()
        
        records = []
        for cme in cmes:
            rec = {
                'activityID': cme.get('activityID'),
                'startTime': pd.to_datetime(cme.get('startTime'), utc=True),
                'versionId': cme.get('versionId'),
                'sourceLocation': cme.get('sourceLocation'),
                'activeRegionNum': cme.get('activeRegionNum'),
                'note': cme.get('note'),
                'catalog': cme.get('catalog'),
                'link': cme.get('link'),
            }
            analyses = cme.get('cmeAnalyses', [])
            if analyses:
                most_acc = next((a for a in analyses if a.get('isMostAccurate')), analyses[0])
                rec.update({
                    'time21_5': pd.to_datetime(most_acc.get('time21_5'), utc=True) if most_acc.get('time21_5') else pd.NaT,
                    'lat_deg': most_acc.get('latitude'),
                    'lon_deg': most_acc.get('longitude'),
                    'halfAngle_deg': most_acc.get('halfAngle'),
                    'speed_km_s': most_acc.get('speed'),
                    'gcs_type': most_acc.get('type'),
                })
            records.append(rec)
        
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df = df[(df['startTime'] >= start_time) & (df['startTime'] <= end_time)].copy()
        df['has_gcs'] = df['gcs_type'].notna() & df['lat_deg'].notna()
        gcs_df = df[df['has_gcs']].copy()
        
        if len(gcs_df) > 0:
            r0 = 21.5
            dist_remaining = (215 - r0) * u.R_sun
            speeds = gcs_df['speed_km_s'].values * u.km/u.s
            t_travel = dist_remaining / speeds
            t_travel_h = t_travel.to(u.hour).value
            gcs_df['t_travel_h'] = t_travel_h
            gcs_df['gcs_1au_time'] = gcs_df['time21_5'] + pd.to_timedelta(gcs_df['t_travel_h'], unit='h')
            gcs_df['gcs_1au_speed'] = gcs_df['speed_km_s'] * np.sqrt(r0 / 215)
            gcs_df['gcs_full_width'] = 2 * gcs_df['halfAngle_deg']
            gcs_df['gcs_earth_lon'] = gcs_df['lon_deg']
            df = df.merge(gcs_df[['activityID'] + [col for col in gcs_df if col.startswith('gcs_') or col == 't_travel_h']], 
                          on='activityID', how='left')
        
        if output_csv:
            df.to_csv(output_csv, index=False)
        if output_gcs_csv and len(gcs_df) > 0:
            gcs_df.to_csv(output_gcs_csv, index=False)
        print(f"Retrieved {len(df)} DONKI CMEs")
        return df
    except Exception as e:
        print(f"DONKI failed: {e}")
        return pd.DataFrame()

# CDAW CMEs
def fetch_cdaw_cme(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    default_days_back: int = 3,
    output_csv: Optional[str] = "cdaw_cme_nrt.csv",
    output_gcs_csv: Optional[str] = "cdaw_gcs_cme_nrt.csv",
    min_width: float = 0.0,
) -> pd.DataFrame:
    """Fetch CDAW CMEs with pseudo-GCS propagation."""
    start_time, end_time = _normalize_time_range(start_time, end_time, default_days_back)
    base_url = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2"
    all_cmes = []
    
    current = start_time.replace(day=1)
    while current <= end_time:
        year, month = current.year, current.month
        url = f"{base_url}/{year}_{month:02d}/univ{year}_{month:02d}.html"
        try:
            r = requests.get(url, timeout=60, headers={'User-Agent': 'solar-research'})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            table = soup.find('table', {'border': '1'})
            if table:
                for row in table.find_all('tr')[1:]:
                    cols = [td.text.strip() for td in row.find_all('td')]
                    if len(cols) < 11 or not cols[0]:
                        continue
                    try:
                        date_str = cols[0].replace('/', '')
                        time_str = cols[1].replace(':', '')
                        cme_time = pd.to_datetime(f"{date_str} {time_str}", format='%Y%m%d %H%M%S', utc=True)
                    except:
                        continue
                    if not (start_time <= cme_time <= end_time):
                        continue
                    width_deg = 360.0 if 'Halo' in cols[4] else safe_float(cols[4])
                    rec = {
                        'cme_time': cme_time, 'central_pa_deg': safe_float(cols[3]), 'width_deg': width_deg,
                        'halfAngle_deg': width_deg / 2, 'linear_speed_km_s': safe_float(cols[5]),
                        'speed_final_ht_km_s': safe_float(cols[6]), 'speed_20Rs_km_s': safe_float(cols[7]),
                        'accel_ms2': safe_float(cols[8]), 'mass_g': safe_float(cols[9]), 'energy_erg': safe_float(cols[10]),
                        'mpa_diagram': cols[11].strip() if len(cols) > 11 else '', 'notes': ' '.join(cols[12:]),
                    }
                    all_cmes.append(rec)
        except Exception as e:
            print(f"Failed {url}: {e}")
        current = _next_month(current)
    
    if not all_cmes:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_cmes)
    df['speed_km_s'] = df['linear_speed_km_s'].fillna(df['speed_20Rs_km_s'])
    df['has_gcs_proxy'] = (df['speed_km_s'] > 0) & (df['width_deg'] > 30)
    gcs_df = df[df['has_gcs_proxy']].copy()
    
    if len(gcs_df) > 0:
        r0 = 20.0
        dist_remaining = (215 - r0) * u.R_sun
        speeds = gcs_df['speed_km_s'].dropna().values * u.km/u.s
        if len(speeds) > 0:
            t_travel = dist_remaining / speeds
            t_travel_h = t_travel.to(u.hour).value
            gcs_df.loc[gcs_df['speed_km_s'].notna(), 't_travel_h'] = t_travel_h
            gcs_df.loc[gcs_df['speed_km_s'].notna(), 'gcs_1au_time'] = gcs_df.loc[gcs_df['speed_km_s'].notna(), 'cme_time'] + pd.to_timedelta(t_travel_h, unit='h')
            gcs_df.loc[gcs_df['speed_km_s'].notna(), 'gcs_1au_speed'] = gcs_df.loc[gcs_df['speed_km_s'].notna(), 'speed_km_s'] * np.sqrt(r0 / 215)
            gcs_df['gcs_earth_lon'] = gcs_df['central_pa_deg']
            gcs_df['gcs_earth_lat'] = 0.0
        gcs_cols = [col for col in gcs_df if 'gcs_' in col or col == 't_travel_h']
        df = df.merge(gcs_df[['cme_time'] + gcs_cols], on='cme_time', how='left', suffixes=('', '_gcs'))
    
    if min_width > 0:
        df = df[df['width_deg'] >= min_width]
    
    if output_csv:
        df.to_csv(output_csv, index=False)
    if output_gcs_csv and len(gcs_df) > 0:
        gcs_df.to_csv(output_gcs_csv, index=False)
    print(f"Retrieved {len(df)} CDAW CMEs")
    return df


def _decimal_year(ts: pd.Series) -> pd.Series:
    """UTC decimal year for a datetime Series."""
    y = ts.dt.year
    start = pd.to_datetime(y.astype(str) + "-01-01", utc=True)
    end = pd.to_datetime((y + 1).astype(str) + "-01-01", utc=True)
    return y + (ts - start).dt.total_seconds() / (end - start).dt.total_seconds()


def fetch_sunspot_number_silso(
    start_time: datetime,
    end_time: datetime,
    *,
    url: str = SILSO_DAILY_URL,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Fetch daily SILSO / SIDC V2.0 sunspot number (definitive + provisional).

    Good for long historical spans; typically lags the last ~7–10 days.
    """
    start_ts = _ensure_utc(start_time).replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = _ensure_utc(end_time).replace(hour=0, minute=0, second=0, microsecond=0)
    if start_ts > end_ts:
        raise ValueError(f"start ({start_ts.date()}) > end ({end_ts.date()})")

    print(f"[ssn/silso] downloading {url} ...")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cols = [
        "year", "month", "day", "decimal_year",
        "ssn", "ssn_std", "n_obs", "is_definitive",
    ]
    raw = pd.read_csv(
        StringIO(resp.text), sep=";", header=None, names=cols, skipinitialspace=True,
    )
    raw["date"] = pd.to_datetime(
        dict(year=raw["year"], month=raw["month"], day=raw["day"]),
        utc=True, errors="coerce",
    )
    raw["ssn"] = pd.to_numeric(raw["ssn"], errors="coerce")
    raw.loc[raw["ssn"] < 0, "ssn"] = np.nan
    raw["ssn_std"] = pd.to_numeric(raw["ssn_std"], errors="coerce")
    raw.loc[raw["ssn_std"] < 0, "ssn_std"] = np.nan
    raw["n_obs"] = pd.to_numeric(raw["n_obs"], errors="coerce")
    raw["is_definitive"] = pd.to_numeric(raw["is_definitive"], errors="coerce")
    raw["source"] = "SILSO"
    mask = (raw["date"] >= start_ts) & (raw["date"] <= end_ts)
    return (
        raw.loc[mask, list(SSN_NRT_COLUMNS)]
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_sunspot_number_swpc(
    start_time: datetime,
    end_time: datetime,
    *,
    url: str = SWPC_SSN_URL,
    timeout: float = 60.0,
    scale_to_silso_v2: bool = True,
) -> pd.DataFrame:
    """Fetch NOAA SWPC observed daily SESC sunspot number (near-real-time).

    Source: https://services.swpc.noaa.gov/json/solar-cycle/swpc_observed_ssn.json
    Typically available through the current UTC day.

    By default, values are rescaled to the SILSO V2.0 convention
    (``ssn_v2 = ssn_swpc * (1/0.6)``) so they can sit next to SILSO / EISN.
    """
    start_ts = _ensure_utc(start_time).replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = _ensure_utc(end_time).replace(hour=0, minute=0, second=0, microsecond=0)
    if start_ts > end_ts:
        raise ValueError(f"start ({start_ts.date()}) > end ({end_ts.date()})")

    print(f"[ssn/swpc] downloading {url} ...")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    raw = pd.DataFrame(resp.json())
    if raw.empty:
        return pd.DataFrame(columns=list(SSN_NRT_COLUMNS))
    raw["date"] = pd.to_datetime(raw["Obsdate"], utc=True, errors="coerce").dt.normalize()
    raw["ssn"] = pd.to_numeric(raw["swpc_ssn"], errors="coerce")
    # SWPC uses -1 / missing markers infrequently; treat negatives as missing.
    raw.loc[raw["ssn"] < 0, "ssn"] = np.nan
    if scale_to_silso_v2:
        raw["ssn"] = raw["ssn"] * SWPC_TO_SILSO_V2_SCALE
        raw["source"] = "SWPC_x1.667_SILSO_V2"
    else:
        raw["source"] = "SWPC"
    raw["ssn_std"] = np.nan
    raw["n_obs"] = np.nan
    raw["is_definitive"] = 0  # operational / near-real-time
    raw["decimal_year"] = _decimal_year(raw["date"])
    mask = (raw["date"] >= start_ts) & (raw["date"] <= end_ts)
    out = (
        raw.loc[mask, list(SSN_NRT_COLUMNS)]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if scale_to_silso_v2 and not out.empty:
        print(
            f"[ssn/swpc] scaled to SILSO V2.0 with factor "
            f"{SWPC_TO_SILSO_V2_SCALE:.4f} (= 1/0.6)"
        )
    return out


def fetch_sunspot_number_eisn(
    start_time: datetime,
    end_time: datetime,
    *,
    url: str = SILSO_EISN_URL,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch SILSO Estimated International Sunspot Number (EISN; near-real-time).

    Updates every few minutes for the current month; intended as a gap-filler
    when SWPC has not yet published the newest day.
    """
    start_ts = _ensure_utc(start_time).replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = _ensure_utc(end_time).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[ssn/eisn] downloading {url} ...")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cols = [
        "year", "month", "day", "decimal_year",
        "ssn", "ssn_std", "n_stations_used", "n_stations_avail",
    ]
    raw = pd.read_csv(
        StringIO(resp.text), header=None, names=cols, skipinitialspace=True,
    )
    raw["date"] = pd.to_datetime(
        dict(year=raw["year"], month=raw["month"], day=raw["day"]),
        utc=True, errors="coerce",
    )
    raw["ssn"] = pd.to_numeric(raw["ssn"], errors="coerce")
    raw.loc[raw["ssn"] < 0, "ssn"] = np.nan
    raw["ssn_std"] = pd.to_numeric(raw["ssn_std"], errors="coerce")
    raw.loc[raw["ssn_std"] < 0, "ssn_std"] = np.nan
    raw["n_obs"] = pd.to_numeric(raw["n_stations_used"], errors="coerce")
    raw["is_definitive"] = 0
    raw["source"] = "SILSO_EISN"
    mask = (raw["date"] >= start_ts) & (raw["date"] <= end_ts)
    return (
        raw.loc[mask, list(SSN_NRT_COLUMNS)]
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_sunspot_number(
    start_time: datetime,
    end_time: datetime,
    *,
    timeout: float = 120.0,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch near-real-time daily sunspot number for ``[start, end]`` on SILSO V2 scale.

    Priority for NRT:
      1. NOAA SWPC observed SESC SSN (daily, usually through today), automatically
         rescaled to SILSO V2.0 by ``× (1/0.6) ≈ 1.6667``
      2. SILSO EISN to fill any trailing days still missing from SWPC
         (already on the international / SILSO scale; no extra scaling)
    """
    start_ts = _ensure_utc(start_time).replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = _ensure_utc(end_time).replace(hour=0, minute=0, second=0, microsecond=0)
    if start_ts > end_ts:
        raise ValueError(f"start ({start_ts.date()}) > end ({end_ts.date()})")

    swpc = fetch_sunspot_number_swpc(start_ts, end_ts, timeout=timeout, scale_to_silso_v2=True)
    out = swpc
    # Fill trailing / missing days with SILSO EISN (near-real-time estimate).
    try:
        eisn = fetch_sunspot_number_eisn(start_ts, end_ts, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[ssn] EISN unavailable ({exc}); using SWPC only")
        eisn = pd.DataFrame(columns=list(SSN_NRT_COLUMNS))

    if not eisn.empty:
        have = set(pd.to_datetime(out["date"], utc=True).dt.normalize()) if not out.empty else set()
        eisn_dates = pd.to_datetime(eisn["date"], utc=True).dt.normalize()
        fill = eisn.loc[~eisn_dates.isin(have)].copy()
        if not fill.empty:
            out = (
                pd.concat([out, fill], ignore_index=True)
                .sort_values("date")
                .reset_index(drop=True)
            )
            print(f"[ssn] filled {len(fill):,} day(s) from SILSO EISN")

    print(
        f"[ssn] kept {len(out):,} daily rows "
        f"({start_ts.date()} → {end_ts.date()}); "
        f"missing SSN={int(out['ssn'].isna().sum()) if not out.empty else 0:,}; "
        f"sources={sorted(out['source'].dropna().unique().tolist()) if not out.empty else []}"
    )
    if output_csv:
        write = out.copy()
        write["date"] = pd.to_datetime(write["date"], utc=True).dt.strftime("%Y-%m-%d")
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        write.to_csv(output_csv, index=False)
        print(f"[ssn] wrote {output_csv}")
    return out


def fetch_all_solar_data(
    start_time: datetime,
    end_time: datetime,
    data_dir: str | os.PathLike[str],
    *,
    products: Iterable[str] = ("sharp", "flare", "donki_cme", "proton_flux", "xrsb", "sunspot"),
    append: bool = True,
    sharp_email: str = "uhek@umich.edu",
    raw_keep_days: float = 30.0,
) -> dict[str, pd.DataFrame]:
    """Fetch selected solar products and optionally append to ``data_dir`` CSVs.

    When ``append`` is True (default), new rows are merged into existing product
    CSVs (deduped). Afterwards, every raw product CSV is pruned to drop rows
    older than ``start_time - raw_keep_days`` (default: 1 month before fetch start).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    selected = {p.strip().lower() for p in products}
    if "all" in selected:
        selected = {"sharp", "flare", "donki_cme", "proton_flux", "xrsb", "sunspot"}

    start_time, end_time = _ensure_utc(start_time), _ensure_utc(end_time)
    print(f"Fetching solar data {start_time.isoformat()} -> {end_time.isoformat()}")
    results: dict[str, pd.DataFrame] = {}

    if "sharp" in selected:
        sharp_path = data_dir / "sharp_nrt.csv"
        sharp = fetch_sharp(
            start_time=start_time,
            end_time=end_time,
            output_csv=None if append else str(sharp_path),
            email=sharp_email,
        )
        results["sharp"] = (
            append_csv(sharp, sharp_path, "T_REC", schema_columns=list(SHARP_QUERY_KEYS))
            if append
            else sharp
        )

    if "flare" in selected:
        flare_path = data_dir / "flare_nrt.csv"
        flares = fetch_flares(start_time=start_time, end_time=end_time, output_csv=None)
        results["flare"] = append_csv(flares, flare_path, "start_time") if append else flares
        if not append and not flares.empty:
            flares.to_csv(flare_path, index=False)

    if "donki_cme" in selected:
        donki_path = data_dir / "donki_cme_nrt.csv"
        donki = fetch_donki_cme(
            start_time=start_time,
            end_time=end_time,
            output_csv=None,
            output_gcs_csv=None,
        )
        results["donki_cme"] = append_csv(donki, donki_path, "startTime") if append else donki
        if not append:
            donki.to_csv(donki_path, index=False)

    if "proton_flux" in selected:
        proton_path = data_dir / "GOESHAPI_ProtonFlux_nrt.csv"
        proton = fetch_proton_flux_goeshapi(start_time=start_time, end_time=end_time, output_csv=None)
        results["proton_flux"] = append_csv(proton, proton_path, "time") if append else proton
        if not append and not proton.empty:
            proton.to_csv(proton_path, index=False)

    if "xrsb" in selected:
        xrs_path = data_dir / "XRS_nrt.csv"
        xrs = fetch_xrsb(start_time=start_time, end_time=end_time, output_csv=None)
        results["xrsb"] = append_csv(xrs, xrs_path, "time") if append else xrs
        if not append and not xrs.empty:
            xrs.to_csv(xrs_path, index=False)

    if "sunspot" in selected:
        ssn_path = data_dir / "sunspot_nrt.csv"
        ssn = fetch_sunspot_number(start_time=start_time, end_time=end_time, output_csv=None)
        results["sunspot"] = (
            append_csv(ssn, ssn_path, "date", schema_columns=list(SSN_NRT_COLUMNS))
            if append
            else ssn
        )
        if not append and not ssn.empty:
            write = ssn.copy()
            write["date"] = pd.to_datetime(write["date"], utc=True).dt.strftime("%Y-%m-%d")
            write.to_csv(ssn_path, index=False)

    # Bound on-disk growth: keep only [start - raw_keep_days, ...] in raw archives.
    if raw_keep_days is not None and float(raw_keep_days) >= 0:
        keep_after = start_time - timedelta(days=float(raw_keep_days))
        print(f"[prune] raw CSVs keep_after={keep_after.isoformat()} "
              f"(fetch start - {float(raw_keep_days):g}d)")
        prune_nrt_raw_csvs(data_dir, keep_after)

    return results


def _default_data_dir() -> Path:
    # Prefer the central path config (Utils/paths.json); fall back if unavailable.
    try:
        import sys as _sys
        _here = str(Path(__file__).resolve().parent)
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from paths_config import P  # noqa: WPS433
        return P("nrt_product_dir")
    except Exception:
        # Fallback mirrors paths.json "nrt_product_dir" (repo root = parent of Utils/).
        return Path(__file__).resolve().parents[1] / "Data" / "Data_nrt"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch SHARP, flares, CMEs, GOES proton flux, XRS-B, and sunspot number.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start time (UTC), e.g. 2025-09-10 or 2025-09-10T00:00:00Z (default: end - 7 days)",
    )
    parser.add_argument(
        "--end",
        default="now",
        help="End time (UTC); use 'now' for current UTC time",
    )
    parser.add_argument(
        "--data-dir",
        default=str(_default_data_dir()),
        help="Output directory for CSV files (default: Data/Data_nrt/ from paths.json)",
    )
    parser.add_argument(
        "--products",
        default="all",
        help="Comma-separated list: sharp,flare,donki_cme,proton_flux,xrsb,sunspot,all",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite product CSVs instead of append/dedupe",
    )
    parser.add_argument(
        "--sharp-email",
        default="uhek@umich.edu",
        help="Email registered with JSOC/DRMS for SHARP queries",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    end = parse_cli_datetime(args.end)
    start = (end - timedelta(days=7)) if args.start is None else parse_cli_datetime(args.start)
    products = [p.strip() for p in args.products.split(",") if p.strip()]
    fetch_all_solar_data(
        start_time=start,
        end_time=end,
        data_dir=args.data_dir,
        products=products,
        append=not args.no_append,
        sharp_email=args.sharp_email,
    )
