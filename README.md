# SEPNET-PRISM-R

Real-time **24-hour-ahead** probabilistic forecasting of Solar Energetic Particle (SEP) events, with multitask deep learning (**SEPNET-PRISM**) and Hidden Semi-Markov Model (**HSMM**) post-processing.

This repository contains the **operational Python pipeline** used for near-real-time (NRT) inference, HSMM smoothing, SHAP post-analysis, diagnostic plots, and selected MagPy / SEPVAL scoreboard helpers.

> **Not included here:** large training tables, raw archives, trained `.pt` weights, paper Overleaf sources, or the interactive dashboard website builders. See below for where to get data and models.

## What is included

| Path | Role |
|------|------|
| `Utils/Main_nrt_analysis.py` | End-to-end NRT entry point (fetch → score → HSMM → optional SHAP/plots) |
| `Utils/Step1_fetch_aggregate_eval_nrt.py` | Fetch solar data, build hourly rolling features, ensemble score, HSMM |
| `Utils/fetch_solar_data.py` | NRT fetchers (SHARP, flares, CME, GOES proton / XRS) |
| `Utils/multitask_model.py` | Multitask LSTM model definition and training/eval utilities |
| `Utils/eval_saved_mtm_lstm_models_fulltrain.py` | Score saved full-train ensemble checkpoints |
| `Utils/hsmm_analysis.py` | Apply saved HSMM parameters (`predict_hsmm`) |
| `Utils/Step2_score_train_mtm_lstm.py` | Score ensemble on the historical feature table |
| `Utils/Step2_SHAP_postanalysis.py` | SHAP attributions for NRT forecasts |
| `Utils/Step3_postanalysis_plots.py` | Probability / flux / SHAP figures |
| `Utils/MagPy_period_*.py` | MagPy-period holdout train / HSMM / compare / CCMC JSON export |
| `Utils/SEPVAL_*.py` | SEPVAL scoreboard export / comparison helpers |
| `Utils/paths.json` / `paths.example.json` | Central relative-path config |
| `examples/hsmm_params.json` | Example pretrained HSMM parameter file |

## What is **not** public in this repo

- **`SEP-PRISM` / `SEP-PRISM_Data` historical dataset** (rolling feature CSV, raw magnetograms / flares / protons used for development)
- Trained MTM-LSTM checkpoints (`*.pt`, ~hundreds of MB)
- Local NRT caches, Eval_outputs time series, MagPy/SEPVAL JSON dumps
- Paper LaTeX / response-letter package
- Full dashboard website generators (`Step3_build_*_website.py`)

## Data: use SEP-PRISM (external)

Historical training / development features live in the **SEP-PRISM** dataset (not redistributed here).

- Dataset descriptor / archive: Zenodo **[10.5281/zenodo.21297635](https://doi.org/10.5281/zenodo.21297635)**
- After download, place (or symlink) the analyzed table so that `Utils/paths.json` resolves:

```text
SEP-PRISM_Data/Analyzed Data/rolling_combinded_seq_1hours.csv
SEP-PRISM_Data/Raw Data/sunspot_number_daily.csv   # optional, analogs / sunspot helpers
```

Paths are configurable in `Utils/paths.json` (copy from `paths.example.json` if you edit).

NRT operation can fetch recent SHARP / flare / GOES products via `fetch_solar_data.py` without shipping the full historical archive.

## Models / HSMM parameters

1. Place full-train ensemble checkpoints under (default):

   ```text
   Data/Saved_mtm_lstm_models_fulltrain_sharp_flare/mtm_lstm_fulltrain_split_*.pt
   ```

2. Copy the example HSMM file (or your tuned params) to:

   ```text
   Data/Eval_outputs/hsmm_params.json
   ```

   A starting point is provided at `examples/hsmm_params.json`.

Trained weights are not bundled in this public tree. If you need the paper’s operational S+F ensemble for reproduction, contact the authors or use your own full-train run via `multitask_model.py` / project training scripts.

## Quick start

```bash
# Python 3.10+ recommended (PyTorch + scientific stack)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure paths (optional; defaults are relative to this repo root)
cp Utils/paths.example.json Utils/paths.json

# Ensure SEP-PRISM feature CSV + model checkpoints exist (see above), then:
python Utils/Main_nrt_analysis.py --skip-shap --skip-website
# or stepwise:
python Utils/Step1_fetch_aggregate_eval_nrt.py --skip-fetch   # if NRT CSV already built
python Utils/hsmm_analysis.py
python Utils/Step3_postanalysis_plots.py
```

### MagPy-period / SEPVAL helpers

```bash
python Utils/MagPy_period_run_all.py --help
python Utils/SEPVAL_optimal_hsmm_export.py --help
```

These scripts expect local validation directories under `Data/` as described in each module docstring.

## Pipeline sketch

```text
solar inputs (NRT fetch or SEP-PRISM CSV)
        │
        ▼
  multitask LSTM ensemble  →  SEP probability (+ aux. regressions)
        │
        ▼
  HSMM post-processing     →  hsmm_prob / hsmm_warn
        │
        ├── SHAP / plots
        └── optional MagPy / SEPVAL JSON exports
```

Neural-network inputs for the operational **S+F** configuration are disk-summed SHARP/SMARP + flare features. Recent 1 h proton flux and XRSB, together with the network SEP probability, are **HSMM covariates** (not neural-network inputs).

## Citation

If you use this code, please cite:

- Yu et al., *Real-Time 24-hour Ahead Forecasting of Solar Energetic Particle Events with SEPNET-PRISM-R*, arXiv:2610.28292, https://arxiv.org/abs/2610.28292
- Yu et al., *A multi-source dataset for solar energetic particle forecasting*, Scientific Data, https://doi.org/10.1038/s41597-026-08552-x ; dataset archive: https://doi.org/10.5281/zenodo.21297635

## License

See `LICENSE` (MIT). Third-party data (GOES, SDO/HMI SHARP, HEK, DONKI, etc.) remain subject to their respective providers’ terms.

## Disclaimer

This software is a research / decision-support prototype for space-weather forecasting. It is not an official operational warning product.
