#!/usr/bin/env python3
"""Main NRT analysis entry point.

Chains the three pipeline stages end-to-end (paths from Utils/paths.json):

  1. Step1_fetch_aggregate_eval_nrt.py
       fetch → aggregate → MTM-LSTM eval → Data_test → HSMM
  2. Step2_SHAP_postanalysis.py
       SHAP importance for the NRT rolling table
  3. Step3_build_postanalysis_website.py
       interactive Postanalysis_Website (prob / regression / SHAP)

Usage:
    python Utils/Main_nrt_analysis.py
    python Utils/Main_nrt_analysis.py --start 2026-07-23 --end now
    python Utils/Main_nrt_analysis.py --skip-fetch --skip-shap
    python Utils/Main_nrt_analysis.py --max-models 3 --open

Stage skips:
    --skip-step1 / --skip-shap / --skip-website
    (Step1 also accepts its own --skip-fetch / --skip-aggregate / --skip-eval / --skip-hsmm)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P  # noqa: E402


def _run(script: Path, argv: list[str]) -> None:
    cmd = [sys.executable, str(script), *argv]
    print("\n" + "=" * 72)
    print(f"[main] RUN: {' '.join(cmd)}")
    print("=" * 72, flush=True)
    subprocess.run(cmd, check=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Step1 → Step2 SHAP → post-analysis website.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- orchestration ----
    p.add_argument("--skip-step1", action="store_true",
                   help="Skip fetch/aggregate/eval/HSMM (reuse existing Eval_outputs)")
    p.add_argument("--skip-shap", action="store_true",
                   help="Skip Step2 SHAP post-analysis")
    p.add_argument("--skip-website", action="store_true",
                   help="Skip building Postanalysis_Website")
    p.add_argument("--open", action="store_true",
                   help="Open Postanalysis_Website/index.html when done")

    # ---- Step1 ----
    g1 = p.add_argument_group("Step1 (fetch / aggregate / eval / HSMM)")
    g1.add_argument("--start", default=None,
                    help="UTC start (default in Step1: end - 7 days)")
    g1.add_argument("--end", default="now")
    g1.add_argument("--products", default="all")
    g1.add_argument("--data-dir", default=None)
    g1.add_argument("--nrt-csv", default=None)
    g1.add_argument("--models-dir", default=None)
    g1.add_argument("--model-glob", default=None)
    g1.add_argument("--out-dir", default=None,
                    help="Step1 prediction output dir (default from paths.json)")
    g1.add_argument("--data-test-csv", default=None)
    g1.add_argument("--hsmm-params", default=None)
    g1.add_argument("--hsmm-out-csv", default=None)
    g1.add_argument("--sharp-email", default=None)
    g1.add_argument("--no-append", action="store_true")
    g1.add_argument("--raw-keep-days", type=float, default=None,
                    help="Drop raw rows older than fetch-start minus N days (default 30 in Step1)")
    g1.add_argument("--skip-fetch", action="store_true")
    g1.add_argument("--skip-aggregate", action="store_true")
    g1.add_argument("--skip-eval", action="store_true")
    g1.add_argument("--skip-hsmm", action="store_true")
    g1.add_argument("--reg-online-debias", dest="reg_online_debias",
                    action=argparse.BooleanOptionalAction, default=True)
    g1.add_argument("--reg-online-window-days", type=float, default=None)
    g1.add_argument("--reg-online-min-samples", type=int, default=None)

    # ---- Step2 SHAP ----
    g2 = p.add_argument_group("Step2 (SHAP)")
    g2.add_argument("--shap-out-dir", default=None,
                    help="SHAP output dir (default: shap_outputs_dir in paths.json)")
    g2.add_argument("--shap-input-csv", default=None,
                    help="CSV for SHAP (default: nrt_rolling_csv)")
    g2.add_argument("--background-size", type=int, default=None)
    g2.add_argument("--eval-size", type=int, default=None)
    g2.add_argument("--all-forecast-times", dest="all_forecast_times",
                    action=argparse.BooleanOptionalAction, default=True)
    g2.add_argument("--forecast-time-start", default=None)
    g2.add_argument("--forecast-time-end", default=None)
    g2.add_argument("--max-models", type=int, default=None,
                    help="Limit number of split models for SHAP (0/None = all)")
    g2.add_argument("--top-k", type=int, default=None)
    g2.add_argument("--top-k-per-time", type=int, default=None)
    g2.add_argument("--seed", type=int, default=None)
    g2.add_argument("--device", default=None, choices=("cpu", "cuda"))

    # ---- Website ----
    g3 = p.add_argument_group("Post-analysis website")
    g3.add_argument("--website-dir", default=None,
                    help="Output dir (default: postanalysis_website_dir in paths.json)")
    g3.add_argument("--prob-col", default=None)
    g3.add_argument("--warn-col", default=None)
    g3.add_argument("--smooth-xrs-cme", dest="smooth_xrs_cme",
                    action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    step1 = _SCRIPT_DIR / "Step1_fetch_aggregate_eval_nrt.py"
    step2 = _SCRIPT_DIR / "Step2_SHAP_postanalysis.py"
    site = _SCRIPT_DIR / "Step3_build_postanalysis_website.py"

    # ---- 1) fetch / aggregate / eval / HSMM --------------------------------
    if not args.skip_step1:
        argv1: list[str] = []
        if args.start is not None:
            argv1 += ["--start", args.start]
        if args.end is not None:
            argv1 += ["--end", args.end]
        for name in ("products", "data_dir", "nrt_csv", "models_dir", "model_glob",
                     "out_dir", "data_test_csv", "hsmm_params", "hsmm_out_csv",
                     "sharp_email", "reg_online_window_days", "reg_online_min_samples",
                     "raw_keep_days"):
            val = getattr(args, name)
            if val is not None:
                argv1 += ["--" + name.replace("_", "-"), str(val)]
        for flag in ("no_append", "skip_fetch", "skip_aggregate", "skip_eval", "skip_hsmm"):
            if getattr(args, flag):
                argv1.append("--" + flag.replace("_", "-"))
        argv1.append("--reg-online-debias" if args.reg_online_debias else "--no-reg-online-debias")
        _run(step1, argv1)
    else:
        print("[main] Step1 skipped")

    # ---- 2) SHAP -----------------------------------------------------------
    if not args.skip_shap:
        argv2: list[str] = []
        # Reuse Step1 models-dir / model-glob when provided.
        if args.models_dir is not None:
            argv2 += ["--models-dir", args.models_dir]
        if args.model_glob is not None:
            argv2 += ["--model-glob", args.model_glob]
        if args.shap_input_csv is not None:
            argv2 += ["--input-csv", args.shap_input_csv]
        elif args.nrt_csv is not None:
            argv2 += ["--input-csv", args.nrt_csv]
        if args.shap_out_dir is not None:
            argv2 += ["--out-dir", args.shap_out_dir]
        for name in ("background_size", "eval_size", "forecast_time_start",
                     "forecast_time_end", "max_models", "top_k", "top_k_per_time",
                     "seed", "device"):
            val = getattr(args, name)
            if val is not None:
                argv2 += ["--" + name.replace("_", "-"), str(val)]
        argv2.append("--all-forecast-times" if args.all_forecast_times
                     else "--no-all-forecast-times")
        _run(step2, argv2)
    else:
        print("[main] Step2 SHAP skipped")

    # ---- 3) website --------------------------------------------------------
    website_dir = Path(args.website_dir) if args.website_dir else P("postanalysis_website_dir")
    if not args.skip_website:
        argv3: list[str] = []
        if args.hsmm_out_csv is not None:
            argv3 += ["--hsmm-csv", args.hsmm_out_csv]
        if args.nrt_csv is not None:
            argv3 += ["--feature-csv", args.nrt_csv]
        if args.shap_out_dir is not None:
            argv3 += ["--shap-bytime-csv",
                      str(Path(args.shap_out_dir) / "shap_feature_importance_by_forecast_time.csv")]
        argv3 += ["--out-dir", str(website_dir)]
        if args.prob_col is not None:
            argv3 += ["--prob-col", args.prob_col]
        if args.warn_col is not None:
            argv3 += ["--warn-col", args.warn_col]
        argv3.append("--smooth-xrs-cme" if args.smooth_xrs_cme else "--no-smooth-xrs-cme")
        _run(site, argv3)
    else:
        print("[main] website skipped")

    index = website_dir / "index.html"
    print("\n" + "=" * 72)
    print("[main] done")
    if index.is_file():
        print(f"[main] website: {index}")
        if args.open:
            webbrowser.open(index.resolve().as_uri())
    print("=" * 72)


if __name__ == "__main__":
    main()
