# Public vs private scope

## Included (this GitHub tree)

- Operational NRT Python pipeline (fetch, aggregate, ensemble score, HSMM)
- Model code (`multitask_model.py`) and HSMM predict path
- SHAP + diagnostic plotting scripts used in the paper workflow
- MagPy-period and SEPVAL helper scripts (JSON / comparison), without bundled validation dumps
- Example `hsmm_params.json`

## Excluded on purpose

| Asset | Why | Where to get it |
|-------|-----|-----------------|
| `SEP-PRISM_Data/` historical features & raw products | Large curated dataset | Zenodo [10.5281/zenodo.21297635](https://doi.org/10.5281/zenodo.21297635) |
| `*.pt` full-train ensemble | Large binaries; training artifact | Retrain or request from authors |
| Paper Overleaf / response letter | Manuscript process, not runtime code | Authors / Overleaf project |
| Dashboard website builders | Heavy local UI glue | Private working tree |
| NRT caches, Eval_outputs time series, scoreboard JSON dumps | Regenerable / institutional | Run pipeline locally |

## Suggested local layout after clone

```text
SEPNET-PRISM-R/          # this repo
  Utils/                 # code
  examples/
  Data/                  # create locally (gitignored)
    Saved_mtm_lstm_models_fulltrain_sharp_flare/
    Eval_outputs/hsmm_params.json
  SEP-PRISM_Data/        # symlink or copy from Zenodo extract
    Analyzed Data/rolling_combinded_seq_1hours.csv
```
