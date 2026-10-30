# Development history

This file records how the method was developed before the test period was used. All changes below were made on
the development period (data up to 2014). The test period (2015-2025) was run once, with the frozen code archive
`CalVol_frozen_2026-10-08.zip` and the configuration in `results/holdout/logs/config_used.yaml`.

In the original research design the main method scaled each error by the volatility of volatility of the asset.
The development runs showed problems with this scale, and the method was changed twice, as described below. The
earlier scales are kept in the code as the comparators `ours_vov` (volatility of volatility) and `ours_sd` (mean and
standard deviation), and their development results are in `results/development/tables/` and in Appendix A of the
paper. Comments in the code that mention "Sec." refer to sections of the internal research design document; every
setting is in `config/config.yaml`.

## Amendment 1 (after the first development run, before any holdout run)

The first development run (data to 2014) showed that the registered method did not work as intended. These changes
were made before the holdout was touched, and the paper must report them:

1. **Main method.** The vol-of-vol scale was replaced. Each forecast error is now standardised by the mean and sd of
   the *same asset's* recent closed errors: the last 60 monthly, 104 weekly or 252 daily forecasts, with at least
   12, 26 or 60 of them. One pooled conformal quantile with ACI is then taken across assets. Reason: in development,
   vol-of-vol scaling helped a little at h = 21 (Kupiec failures 1/58 vs 5/58 at 90%). It made per-asset calibration
   worse at h = 5 and h = 1 (20/58 vs 1/58 at h = 1, 90%) and had a worse pinball loss everywhere. Per-asset
   conformal was the best-calibrated method at 80-95%, but it failed at 99% for lack of scores. The new method
   combines the two. The original method is kept as the comparator `ours_vov`.
2. **H2 yearly rule.** "No year off by more than 5 pp" failed for every method, because the 2008-2009 shock hits all
   assets at once. The rule is now relative: our mean absolute yearly deviation must not exceed pooled split
   conformal's. The date-clustered overall test is unchanged.
3. **Hypothesis labels.** H3 and H4 are labelled secondary, as in the design document.
4. **Bug fixes.** These do not change the design:
   * HAR's daily lag used the zero return of non-trading days, giving ln(1e-10). It now uses the last traded day.
   * GARCH fits with persistence ≥ 0.999 were discarded, so 10-35% of forecasts fell back to EWMA. These fits now
     keep GARCH with persistence capped at 0.999 (variance targeting). The status of every forecast is reported in
     `*_garch_fallback.csv`.
   * HAR and LSTM are fitted in log volatility by least squares. They now get the Duan smearing correction, so they
     forecast variance like the other models. This makes the H4 comparison fair, and harder for DLinear.

## Amendment 2 (after the second development run, before any holdout run)

The second development run showed that amendment 1 (mean and sd) fixed h = 5 (Kupiec failures at 90%: 0/58 vs
3/58 for pooled split) and the h = 21 99% level (0/58 vs 11/58). It still failed at h = 1 (26/58 vs 1/58 at 90%).
Simulation showed the cause. Daily log-volatility errors differ in *shape* across assets: fat tails differ, and days
with tiny or zero returns create a long left tail. That inflates the sd without saying anything about the upper
tail. The location and scale are therefore now the **median and the upper spread Q90 − median** of the asset's own
recent closed errors (same windows as before). The mean/sd version is kept as the ablation `ours_sd`. This is the
last change to the method before the holdout. Further development runs only check the code.


## Analyses added after the test-period run

These analyses only read the saved forecasts and bounds. They do not change any forecast, bound or earlier result.

* Notebook 07 (`scripts/spread_sensitivity.py`, `scripts/paired_tests.py`): sensitivity to the spread quantile, the
  history window and the minimum history; paired tests.
* Notebook 08 (`scripts/portfolio_risk_budget.py`): portfolio risk budget built from the asset-level bounds. This
  application is outside the frozen design; all its settings are fixed in the script header and it is reported for
  both periods.
* Notebook 09 (`scripts/exact_coverage_tests.py`): exact binomial tests, Clopper-Pearson intervals and a dispersion
  check, as a complement to the asymptotic Kupiec test.
* The paired comparison in the paper uses the exact McNemar test. The research design specified an unpaired
  two-proportion test (`assets_h21_H1.csv`, `hypothesis_verdicts.csv`); both are reported in the paper.

## Other implementation notes

* The parametric comparator uses the residuals of the same base forecaster as every other method. Section 6.2
  says "HAR residuals"; using the same base keeps the comparison like for like.
* The two-sided (lower-bound) variant of Section 6.1 is not implemented. Every hypothesis concerns upper bounds.
