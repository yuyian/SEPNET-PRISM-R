#!/usr/bin/env python3
"""Download daily sunspot numbers.

* Historical / archive (default here): SILSO SIDC V2.0
* NRT pipeline (``fetch_solar_data.fetch_all_solar_data``): NOAA SWPC near-real-time
  (+ SILSO EISN gap-fill) written to ``Data/Data_nrt/sunspot_nrt.csv``

SWPC SESC values are automatically rescaled to the SILSO V2.0 convention
(``ssn_v2 = ssn_swpc / 0.6``) so NRT and archive series share one scale.
EISN is already on the SILSO scale and is left unchanged.

Default archive span matches SEP-PRISM analyzed window:
  1986-02-03 → 2025-09-10

Usage:
    python Utils/fetch_sunspot_number.py
    python Utils/fetch_sunspot_number.py --start 1986-02-03 --end 2025-09-10
    python Utils/fetch_sunspot_number.py --source swpc --start 2026-07-01 --end now

Credit:
  SILSO: WDC-SILSO, Royal Observatory of Belgium (CC BY-NC)
  SWPC: NOAA / NWS Space Weather Prediction Center
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from fetch_solar_data import (  # noqa: E402
    fetch_sunspot_number,
    fetch_sunspot_number_silso,
    parse_cli_datetime,
)
from paths_config import P  # noqa: E402

_DEFAULT_START = "1986-02-03"
_DEFAULT_END = "2025-09-10"


def _default_out() -> Path:
    try:
        return P("sunspot_daily_csv")
    except Exception:
        return (
            Path(__file__).resolve().parents[1]
            / "SEP-PRISM_Data" / "Raw Data" / "sunspot_number_daily.csv"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default=_DEFAULT_START, help="Inclusive start date (UTC)")
    p.add_argument("--end", default=_DEFAULT_END, help="Inclusive end date (UTC)")
    p.add_argument("--out", default=str(_default_out()),
                   help="Output CSV path")
    p.add_argument(
        "--source",
        choices=("silso", "swpc"),
        default="silso",
        help="silso = long archive (default); swpc = near-real-time NOAA SWPC",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    start = parse_cli_datetime(args.start)
    end = parse_cli_datetime(args.end)
    if args.source == "swpc":
        df = fetch_sunspot_number(start, end)
    else:
        df = fetch_sunspot_number_silso(start, end)
        print(
            f"[ssn] kept {len(df):,} SILSO rows "
            f"({start.date()} → {end.date()})"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write = df.copy()
    write["date"] = pd.to_datetime(write["date"], utc=True).dt.strftime("%Y-%m-%d")
    write.to_csv(out, index=False)
    print(f"[ssn] wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
