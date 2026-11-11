#!/usr/bin/env python3
"""Single source of truth for filesystem paths used by the Utils scripts.

Every script under ``Utils/`` that needs a path reads it from ``Utils/paths.json``
through this module, so relocating the project only requires (at most) editing that
one JSON file -- never the Python code.

Resolution rules
----------------
* ``project_root`` in paths.json:
    - empty string / missing  -> auto-detected as the parent folder of ``Utils/``
      (i.e. the repo root), so moving the whole project needs no edits at all.
    - an absolute path        -> used as-is (handy when code and data are split).
* Every other value is a path RELATIVE to ``project_root`` unless it is already
  absolute, in which case it is used verbatim.

Usage
-----
    from paths_config import P, S
    models_dir = P("models_dir")      # -> pathlib.Path (absolute, resolved)
    glob_pat   = S("model_glob")      # -> str (non-path config value)
"""
from __future__ import annotations

import json
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent          # .../Utils
_DEFAULT_ROOT = _CONFIG_DIR.parent                      # repo root (parent of Utils/)
_PATHS_FILE = _CONFIG_DIR / "paths.json"


def _load() -> dict:
    if _PATHS_FILE.is_file():
        with _PATHS_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


_RAW = _load()


def project_root() -> Path:
    """Absolute project root (auto-detected unless overridden in paths.json)."""
    override = str(_RAW.get("project_root", "") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_ROOT


def S(key: str, default: str | None = None) -> str:
    """Return a raw (non-path) config value as a string, e.g. a glob pattern."""
    val = _RAW.get(key, default)
    if val is None:
        raise KeyError(f"paths.json is missing key '{key}' (and no default given)")
    return str(val)


def P(key: str, default: str | None = None) -> Path:
    """Return a config path as an absolute, resolved ``Path``.

    Relative entries are anchored at :func:`project_root`; absolute entries are
    returned as-is.
    """
    raw = _RAW.get(key, default)
    if raw is None:
        raise KeyError(f"paths.json is missing key '{key}' (and no default given)")
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = project_root() / p
    return p.resolve()


if __name__ == "__main__":
    print(f"project_root -> {project_root()}")
    for _k in sorted(_RAW):
        if _k.startswith("_") or _k == "project_root":
            continue
        try:
            print(f"{_k:20s} -> {P(_k)}")
        except Exception:
            print(f"{_k:20s} -> {S(_k)}  (non-path)")
