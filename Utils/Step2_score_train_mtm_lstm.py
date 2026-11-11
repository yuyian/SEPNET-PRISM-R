#!/usr/bin/env python3
"""Score the FULL-TRAIN MTM-LSTM ensemble on the historical training feature CSV.

Separate job from NRT scoring (Step1) and from training-set SHAP (Step2_SHAP_train).

Writes under ``Data/Eval_outputs/Train_predictions/`` (paths.json:
``train_predictions_dir``):

  * mtm_lstm_fulltrain_train1h_predictions.csv
      per-split rows (SHARP_label==1 windows × 50 models)
  * mtm_lstm_fulltrain_train1h_ensemble.csv
      median across split_seed (same aggregation as NRT Data_test builder)
  * by_model/mtm_lstm_fulltrain_split_XXX.csv
      one file per model (streaming-friendly intermediate)

Example::

  /opt/anaconda3/envs/py312/bin/python Utils/Step2_score_train_mtm_lstm.py
  /opt/anaconda3/envs/py312/bin/python Utils/Step2_score_train_mtm_lstm.py --max-models 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths_config import P, S  # noqa: E402

try:
    _pycode = S("pycode_dir", "").strip()
    if _pycode:
        _pycode = str(Path(_pycode).expanduser().resolve())
        if _pycode not in sys.path:
            sys.path.insert(0, _pycode)
except Exception:
    pass

from eval_saved_mtm_lstm_models_fulltrain import filter_rows, score_full_file  # noqa: E402
from Step1_fetch_aggregate_eval_nrt import aggregate_predictions_across_splits  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score MTM-LSTM ensemble probabilities on the training feature dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-csv", default=str(P("feature_dataset_csv")),
                   help="Historical rolling feature table")
    p.add_argument("--models-dir", default=str(P("models_dir")))
    p.add_argument("--model-glob", default=S("model_glob", "mtm_lstm_fulltrain_split_*.pt"))
    p.add_argument(
        "--out-dir",
        default=str(P("train_predictions_dir", "Data/Eval_outputs/Train_predictions")),
        help="Output directory for train predictions",
    )
    p.add_argument("--max-models", type=int, default=0, help="0 = all matching split models")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="auto = cuda if available else cpu")
    p.add_argument("--reg-online-debias", action=argparse.BooleanOptionalAction, default=True,
                   help="Online regression de-bias (same default as NRT eval)")
    p.add_argument("--reg-online-window-days", type=float, default=30.0)
    p.add_argument("--reg-online-min-samples", type=int, default=3)
    p.add_argument("--skip-ensemble", action="store_true",
                   help="Only write per-model / concatenated predictions")
    p.add_argument("--keep-by-model", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep per-model CSV shards under out-dir/by_model/")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    models_dir = Path(args.models_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    by_model_dir = out_dir / "by_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_by_model:
        by_model_dir.mkdir(parents=True, exist_ok=True)

    model_files = sorted(models_dir.glob(args.model_glob))
    if not model_files:
        raise FileNotFoundError(f"No model files matching {args.model_glob} in {models_dir}")
    if args.max_models and args.max_models > 0:
        model_files = model_files[: int(args.max_models)]

    input_csv = Path(args.input_csv).expanduser().resolve()
    print(f"[train-score] reading {input_csv}")
    df_in = pd.read_csv(input_csv, low_memory=False)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(
        f"[train-score] {len(model_files)} models | device={device} | "
        f"rows={len(df_in):,} | out={out_dir}"
    )

    concat_path = out_dir / "mtm_lstm_fulltrain_train1h_predictions.csv"
    if concat_path.is_file():
        concat_path.unlink()

    n_written = 0
    header = True
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
            reg_online_debias=bool(args.reg_online_debias),
            reg_online_window_days=float(args.reg_online_window_days),
            reg_online_min_samples=int(args.reg_online_min_samples),
        )
        f_in = filter_rows(df_in, dict(art["row_filter"]), common["wcol_begin"], common["wcol_end"])
        df_out = score_full_file(f_in, **common)
        df_out["model_file"] = mf.name
        df_out["split_seed"] = art.get("split_seed")
        df_out = df_out.reset_index(drop=False).rename(columns={"index": "row_index"})

        if args.keep_by_model:
            shard = by_model_dir / f"{mf.stem}_predictions.csv"
            df_out.to_csv(shard, index=False)

        df_out.to_csv(concat_path, mode="a", header=header, index=False)
        header = False
        n_written += len(df_out)
        print(f"  [{mi + 1}/{len(model_files)}] {mf.name} | n={len(df_out):,} | total_rows={n_written:,}")
        # free GPU/CPU tensors between models
        del model, art, df_out, f_in
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[train-score] wrote {concat_path} rows={n_written:,}")

    if args.skip_ensemble:
        return

    print("[train-score] aggregating ensemble median across splits ...")
    # Chunked read + groupby median via concat of per-model shards is safer than
    # loading the full 50× table when memory is tight; prefer shards if present.
    if args.keep_by_model and by_model_dir.is_dir():
        shards = sorted(by_model_dir.glob("mtm_lstm_fulltrain_split_*_predictions.csv"))
        if not shards:
            shards = sorted(by_model_dir.glob("*_predictions.csv"))
        parts = [pd.read_csv(s, low_memory=False) for s in shards]
        pred = pd.concat(parts, ignore_index=True)
    else:
        pred = pd.read_csv(concat_path, low_memory=False)

    ens = aggregate_predictions_across_splits(pred)
    ens_path = out_dir / "mtm_lstm_fulltrain_train1h_ensemble.csv"
    ens.to_csv(ens_path, index=False)
    print(f"[train-score] wrote {ens_path} rows={len(ens):,}")

    # Slim probability-only export (and a copy next to SHAP_train outputs).
    prob_cols = [
        c for c in (
            "window_begin", "window_end",
            "mtm_lstm_cls_prob", "mtm_lstm_cls_pred", "mtm_lstm_cls_true",
            "mtm_lstm_threshold_used",
        )
        if c in ens.columns
    ]
    prob_path = out_dir / "mtm_lstm_fulltrain_train1h_cls_prob.csv"
    ens.loc[:, prob_cols].to_csv(prob_path, index=False)
    print(f"[train-score] wrote {prob_path}")
    try:
        shap_dir = Path(P("shap_train_outputs_dir"))
        shap_dir.mkdir(parents=True, exist_ok=True)
        ens.loc[:, prob_cols].to_csv(shap_dir / prob_path.name, index=False)
        print(f"[train-score] also wrote {shap_dir / prob_path.name}")
    except Exception as e:
        print(f"[train-score] skip SHAP_train probability copy: {e}")

    if "mtm_lstm_cls_prob" in ens.columns:
        p = pd.to_numeric(ens["mtm_lstm_cls_prob"], errors="coerce")
        print(
            f"[train-score] ensemble cls_prob: "
            f"min={p.min():.4g} median={p.median():.4g} max={p.max():.4g} "
            f"mean={p.mean():.4g}"
        )


if __name__ == "__main__":
    main()
