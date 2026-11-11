#!/usr/bin/env python3
"""Download MagPy SEP Scoreboard JSONs for the MagPy validation window.

Source (ISWA data tree, MagPy 2.X continuous runs)::

  https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/model/heliosphere/sep_scoreboard/MagPy/2.X/{YYYY}/{MM}/

Default window: 2023-07-01 → 2024-12-31 (issue-month folders covering that span).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P  # noqa: E402

BASE = (
    "https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/"
    "model/heliosphere/sep_scoreboard/MagPy/2.X"
)
FILE_RE = re.compile(r'href="(MagPy-HMI-SHARP-Vector[^"]+\.json)"')
DIR_RE = re.compile(r'href="(\d{2})/"')


class _HrefCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for k, v in attrs:
            if k == "href" and v:
                self.hrefs.append(v)


def _fetch_text(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "SEPNET-PRISM-R/MagPy-compare"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _list_json_names(year: int, month: int) -> list[str]:
    url = f"{BASE}/{year:04d}/{month:02d}/"
    html = _fetch_text(url)
    return sorted(set(FILE_RE.findall(html)))


def _download_one(url: str, dest: Path, force: bool) -> tuple[str, str]:
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest.name, "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SEPNET-PRISM-R/MagPy-compare"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        tmp.replace(dest)
        return dest.name, "ok"
    except Exception as exc:  # noqa: BLE001
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return dest.name, f"err:{exc}"


def _months_in_range(start: datetime, end: datetime) -> list[tuple[int, int]]:
    y, m = start.year, start.month
    out: list[tuple[int, int]] = []
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(P("magpy_validation_dir")))
    ap.add_argument("--start", default="2023-07-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    out_dir = root / "magpy_scoreboard_json" / "2.X"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    jobs: list[tuple[str, Path]] = []
    for y, m in _months_in_range(start, end):
        print(f"[fetch] listing {y:04d}/{m:02d} ...", flush=True)
        try:
            names = _list_json_names(y, m)
        except urllib.error.HTTPError as e:
            print(f"[fetch] WARN {y:04d}/{m:02d}: {e}")
            continue
        month_dir = out_dir / f"{y:04d}" / f"{m:02d}"
        for name in names:
            url = f"{BASE}/{y:04d}/{m:02d}/{name}"
            jobs.append((url, month_dir / name))
        print(f"[fetch]   {len(names)} files", flush=True)

    print(f"[fetch] downloading {len(jobs)} files → {out_dir} (workers={args.workers})", flush=True)
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(_download_one, u, p, args.force) for u, p in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            _, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                if n_err <= 10:
                    print(f"[fetch] {status}", flush=True)
            if i % 500 == 0 or i == len(futs):
                print(
                    f"[fetch] progress {i}/{len(futs)} ok={n_ok} skip={n_skip} err={n_err}",
                    flush=True,
                )

    print(
        f"[fetch] done in {time.time()-t0:.1f}s | ok={n_ok} skip={n_skip} err={n_err} | dir={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
