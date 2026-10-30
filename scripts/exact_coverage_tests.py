"""Asset-level coverage: exact binomial tests, Clopper-Pearson intervals and a dispersion check.

Complements the asymptotic Kupiec test. For every base forecaster, method, horizon and level, and for each instrument
with enough observations (same sample and same minimum number of observations as the main evaluation):
  * number of observations n_i and of exceedances k_i, empirical coverage 1 - k_i / n_i;
  * exact two-sided binomial test of H0: P(exceedance) = 1 - level (scipy.stats.binomtest), and the direction of the
    deviation (too many exceedances = under-coverage, too few = over-coverage);
  * 95% Clopper-Pearson interval for the coverage of the instrument;
  * the Kupiec test as in the main results.
Summary per cell: rejections by each test and direction, how often the interval lies entirely below or above the
nominal level, median interval width, the expected number of rejections if every instrument had exactly the nominal
coverage (exact size of each test at each n_i), and a dispersion ratio: the variance of the per-instrument exceedance
rates divided by its expected value under binomial sampling at the nominal rate (1 = binomial, < 1 = the instruments
are closer to the nominal rate than independent sampling would give, > 1 = more spread).

The script only reads saved bounds; no forecast or bound is changed.
Usage (Colab, after the holdout run):   !python scripts/exact_coverage_tests.py
Writes results/<mode>/tables/exact_coverage_summary.csv (+ .tex) and exact_coverage_assets.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def kupiec_reject(k: int, n: int, a: float) -> bool:
    """Same rule as calvol.stats.kupiec_christoffersen (unconditional part)."""
    ph = k / n
    ll0 = (n - k) * np.log(1 - a) + k * np.log(a)
    ll1 = ((n - k) * np.log(max(1 - ph, 1e-12)) + k * np.log(max(ph, 1e-12))) if 0 < k < n else 0.0
    return bool(2 * (ll1 - ll0) > stats.chi2.ppf(0.95, 1))


_size_cache: dict = {}


def exact_sizes(n: int, a: float) -> tuple[float, float]:
    """Probability that the exact test and the Kupiec test reject when P(exceedance) is exactly a."""
    key = (n, round(a, 6))
    if key not in _size_cache:
        ks = np.arange(n + 1)
        pm = stats.binom.pmf(ks, n, a)
        re = np.array([stats.binomtest(int(k), n, a).pvalue < 0.05 for k in ks])
        rk = np.array([kupiec_reject(int(k), n, a) for k in ks])
        _size_cache[key] = (float(pm[re].sum()), float(pm[rk].sum()))
    return _size_cache[key]


def asset_rows(df: pd.DataFrame, col: str, level: float, h: int, min_obs: dict) -> list[dict]:
    a = 1 - level
    d = df[np.isfinite(df[col]) & df["y"].notna()]
    out = []
    for asset, g in d.groupby("asset"):
        n = len(g)
        if n < min_obs.get(h, 36):
            continue
        k = int((g["y"] > g[col]).sum())
        bt = stats.binomtest(k, n, a)
        ci = bt.proportion_ci(confidence_level=0.95, method="exact")
        out.append({"asset": asset, "n": n, "exceed": k, "coverage": 1 - k / n,
                    "cov_ci_low": 1 - ci.high, "cov_ci_high": 1 - ci.low, "p_exact": float(bt.pvalue),
                    "direction": "under" if k / n > a else ("over" if k / n < a else "equal"),
                    "kupiec_reject": kupiec_reject(k, n, a)})
    return out


def summarise_cell(r: pd.DataFrame, level: float) -> dict:
    a = 1 - level
    rej = r["p_exact"] < 0.05
    sizes = np.array([exact_sizes(int(n), a) for n in r["n"]])
    rate = r["exceed"] / r["n"]
    disp = float(np.var(rate, ddof=1) / np.mean(a * (1 - a) / r["n"])) if len(r) > 1 else np.nan
    return {"assets": len(r), "kupiec_rejects": int(r["kupiec_reject"].sum()), "exact_rejects": int(rej.sum()),
            "exact_rejects_under": int((rej & (r["direction"] == "under")).sum()),
            "exact_rejects_over": int((rej & (r["direction"] == "over")).sum()),
            "ci_below_nominal": int((r["cov_ci_high"] < level).sum()),
            "ci_above_nominal": int((r["cov_ci_low"] > level).sum()),
            "median_ci_width": float((r["cov_ci_high"] - r["cov_ci_low"]).median()),
            "expected_rejects_exact": float(sizes[:, 0].sum()), "expected_rejects_kupiec": float(sizes[:, 1].sum()),
            "dispersion_ratio": disp, "min_coverage": float(r["coverage"].min()),
            "median_n": float(r["n"].median())}


def main():
    from calvol.evaluate import MIN_OBS_ASSET, ub_col
    from calvol.pipeline import eval_split
    from calvol.utils import get_paths, load_config, save_table
    cfg = load_config(ROOT)
    P = get_paths(cfg)
    cc = cfg["conformal"]
    rows, assets = [], []
    for h in cfg["panels"]["assets"]["horizons"]:
        f = P.bounds / f"assets_h{h}.parquet"
        if not f.exists():
            print("missing", f)
            continue
        df = pd.read_parquet(f)
        df = df[df["split"] == eval_split(cfg)]
        for b in cc["base_forecasters"]:
            for m in cc["methods"]:
                for lv in cc["levels"]:
                    col = ub_col(b, m, lv)
                    if col not in df:
                        continue
                    r = pd.DataFrame(asset_rows(df, col, lv, h, MIN_OBS_ASSET))
                    if not len(r):
                        continue
                    r.insert(0, "level", lv); r.insert(0, "method", m); r.insert(0, "base", b); r.insert(0, "h", h)
                    assets.append(r)
                    s = summarise_cell(r, lv)
                    s.update({"h": h, "base": b, "method": m, "level": lv})
                    rows.append(s)
            print(f"h={h} base={b} done", flush=True)
    a = pd.concat(assets, ignore_index=True)
    a.to_csv(P.tables / "exact_coverage_assets.csv", index=False)
    s = pd.DataFrame(rows)
    save_table(s.set_index(["h", "base", "method", "level"]), P, "exact_coverage_summary",
               caption="Asset-level coverage: Kupiec and exact binomial tests, Clopper-Pearson intervals, expected "
                       "rejections under exact nominal coverage and dispersion ratio (test period).", float_fmt="{:.3f}")
    pd.set_option("display.width", 220)
    key = s[(s["base"] == "dl") & s["method"].isin(["ours", "pooled_split", "per_asset"])]
    print(key[["h", "method", "level", "assets", "kupiec_rejects", "exact_rejects", "exact_rejects_under",
               "exact_rejects_over", "expected_rejects_exact", "expected_rejects_kupiec", "dispersion_ratio",
               "median_ci_width"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
