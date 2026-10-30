# Asset-level calibration of conformal upper bounds for volatility

Code and results for the paper

> Kalaiarasan G. *Asset-level calibration of conformal upper bounds for volatility: Evidence from a multi-asset
> panel.* Manuscript submitted for publication, 2026.

Conformal prediction gives volatility upper bounds with correct coverage on average over the assets used for
calibration, but not necessarily for each asset. This repository tests conformal volatility bounds asset by asset
on 58 instruments from six asset classes, three horizons (21, 5 and 1 trading days) and four base forecasters
(DLinear, HAR, GARCH(1,1)-t and EWMA). It also contains the proposed bound, in which each forecast error is
standardised by the median and the upper spread (90th percentile minus median) of the same asset's recent errors
before one pooled conformal quantile is computed with adaptive conformal inference (ACI).

All modelling choices were made on data up to 2014. The test period 2015-2025 was run once with the frozen code;
the configuration of that run is in `results/holdout/logs/config_used.yaml`. The development history of the method is
described in [`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md).

## Repository layout

```
config/config.yaml         all design settings and the run switches (mode, signoff, frozen_commit)
config/smoke.yaml          small settings for a quick end-to-end check on synthetic data
src/calvol/                Python package
  data.py                  Yahoo Finance download (cached), Kenneth French factors, synthetic smoke data
  panel.py                 panels, cleaning, features known at t, targets over days t+1..t+h, signal dates
  models.py, forecast.py   DLinear-QLIKE, LSTM, HAR, GARCH(1,1)-t, EWMA, naive, quantile HAR; walk-forward
  calib.py                 conformal bounds: the proposed bound and all comparison methods
  evaluate.py, stats.py    Kupiec and Christoffersen tests, clustered coverage, Holm, Diebold-Mariano (HLN),
                           model confidence set, Ledoit-Wolf variance and Sharpe tests, Romano-Wolf, bootstrap
  vm.py                    volatility-managed factor strategies
  convar.py                conformal value-at-risk (appendix)
  pipeline.py              one function per notebook stage, with the checks of the design
  plots.py                 figures
scripts/                   additional analyses that only read saved forecasts and bounds
  spread_sensitivity.py    sensitivity to q, M and m (notebook 07)
  paired_tests.py          paired tests for all horizons and base forecasters (notebook 07)
  portfolio_risk_budget.py portfolio risk budget (notebook 08)
  exact_coverage_tests.py  exact binomial tests and Clopper-Pearson intervals (notebook 09)
notebooks/00 ... 09        Colab notebooks, run in order
tests/test_calvol.py       24 tests: look-ahead, purging, closed targets in calibration, holdout gate, ...
results/development/       tables, figures and logs of the final development run
results/holdout/           tables, figures and logs of the test-period run and of notebooks 08 and 09
paper/                     scripts that rebuild every table and figure of the manuscript from results/
docs/                      development history and implementation notes
```

## Installation

Python 3.10 or later. The notebooks are written for Google Colab, which already provides numpy, pandas, scipy,
torch, matplotlib and statsmodels; the other packages are installed by the first cell of each notebook. For a local
installation:

```bash
pip install numpy pandas scipy matplotlib torch -r requirements.txt
python -m pytest -q tests      # 24 tests, a few seconds
```

## Reproducing the results

1. Copy the repository folder to Google Drive (`MyDrive/CalVol_project`, or change `PROJECT_DIR` in the first cell
   of each notebook).
2. Optional check: set `project.smoke_test: true` in `config/config.yaml` and run notebooks 00-06 once on synthetic
   data (about 5 minutes). Set it back to `false` afterwards.
3. Development run (`project.mode: development`): run notebooks 00-06 in order.

| Notebook | What it does | Runtime |
|---|---|---|
| 00_setup_and_tests | installs packages, prints the configuration, runs the tests | CPU, 1 min |
| 01_data | Yahoo Finance and Kenneth French downloads | CPU, 10-20 min |
| 02_panels_and_forecasts | panels and walk-forward forecasts (resumable) | GPU (T4), 2-4 h |
| 03_conformal_bounds | bounds for every panel, horizon, level, base forecaster and method; chooses the ACI step | CPU, 10-30 min |
| 04_calibration_results | asset-level calibration tables and figures; conformal VaR | CPU, 5-10 min |
| 05_volatility_management | volatility-managed factor portfolios | CPU, 10-20 min |
| 06_verdicts_and_export | applies the decision rules of the design and exports the results | CPU, 1 min |

4. Test-period run: set `mode: holdout`, `signoff: true` and `frozen_commit` in `config/config.yaml` (as in
   `results/holdout/logs/config_used.yaml`) and run notebooks 01-06 once. The run refuses to start without sign-off,
   refuses if any design setting differs from the development run, and reuses the ACI step sizes and the CVM level
   chosen on the development period (`results/development/tables/selections.json`).
5. Additional analyses on the saved test-period bounds: notebooks 07 (sensitivity and paired tests),
   08 (portfolio risk budget) and 09 (exact coverage tests).
6. Rebuild the manuscript tables and figures from `results/`:

```bash
bash paper/make_all.sh         # writes paper/output/tables/*.tex and paper/output/figures/*.pdf
```

Yahoo Finance data change over time (revisions, delisted symbols), so a new download may give slightly different
numbers. `results/holdout/tables/download_report.csv` lists the first and last date and the number of observations of
each symbol in the run reported in the paper.

## Result files

Every table is saved as CSV and as LaTeX in `results/<period>/tables/`, and every figure as PDF and PNG in
`results/<period>/figures/`. Files referred to in the paper:

| File | Content |
|---|---|
| `assets_h{21,5,1}_calibration_all.csv` | coverage, Kupiec and Christoffersen rejections, width and pinball loss for every method, base forecaster and level |
| `assets_h21_H1.csv`, `hypothesis_verdicts.csv` | pre-specified comparison with pooled split conformal and the decision rules of the design |
| `exact_coverage_assets.csv` | for every instrument, method, horizon and level: exceedances, coverage, 95% Clopper-Pearson interval, exact binomial p-value |
| `exact_coverage_summary.csv` | exact tests, interval counts and dispersion ratio by cell |
| `portfolio_risk_budget.csv`, `portfolio_risk_budget_months.csv`, `portfolio_risk_budget_paired.csv` | portfolio risk budget: summary, monthly detail and paired tests |
| `assets_outliers.csv` | the 46 large daily moves checked by the cleaning rule: 4 removed, 42 kept |
| `assets_instruments.csv`, `download_report.csv` | instruments and data coverage |
| `assets_h*_forecasts.csv` | forecast accuracy (QLIKE, Mincer-Zarnowitz, Diebold-Mariano, model confidence set) |
| `ff_*` | factor panels and volatility-managed portfolios |

The sensitivity analysis of notebook 07 writes `sensitivity_summary.csv` and `sensitivity_check.csv`; copy them to
`results/holdout/tables/` to rebuild the sensitivity table with `paper/make_all.sh`.

## Data

Prices are downloaded from Yahoo Finance and factor returns from the Kenneth R. French Data Library. Raw data are
not included in this repository; the notebooks download them. The result files contain only statistics computed
from the data.

## Leakage and integrity safeguards

* **Targets and features.** Targets use only days t+1..t+h. Features use only days up to and including t. Both
  are tested.
* **Truncated panels.** A panel cut at `dev_end` produces exactly the same features as the full panel (tested).
  In a smoke run, development-period forecasts from the development and holdout runs were identical to the
  last bit.
* **Walk-forward training.** Training at each refit uses only targets whose window has closed. The inner
  validation block is purged and embargoed (tested).
* **Conformal calibration.** The calibration set at t contains only targets closed by t. ACI feedback arrives only
  when a target closes (tested).
* **Volatility management.** Exposures at t use no information after t. The scaling constant c_t is computed in
  real time (tested).
* **Run records.** Every stage writes a manifest (configuration hash, mode, sign-off, commit, package versions)
  to `results/<mode>/logs/`.


## Citation

If you use this code, please cite the paper above. Citation details are also in `CITATION.cff`.

## Licence

The code is released under the MIT licence (see `LICENSE`).
