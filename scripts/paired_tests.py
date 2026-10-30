"""Paired comparison of asset-level calibration: proposed bound ('ours') vs a comparator, on the same instruments.

For every base forecaster, horizon and level:
  * 2x2 table of Kupiec outcomes (reject at 5%) on the same instruments,
  * one-sided exact McNemar test (H1: fewer rejections for 'ours'; equivalent to a sign-flip permutation test on the
    discordant instruments) and the difference in rejection rates with a Newcombe hybrid-score 95% CI,
  * one-sided Wilcoxon signed-rank test on |per-asset coverage - nominal| (smaller for 'ours').
Holm adjustment is applied across the four levels within each base forecaster and horizon.

Usage (Colab, after the holdout run):   !python scripts/paired_tests.py            (comparator: pooled_split)
                                        !python scripts/paired_tests.py per_asset  (any other comparator)
Writes results/<mode>/tables/paired_tests_<comparator>.csv (and .tex). This is an evaluation of saved bounds only;
it does not change any forecast or bound.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calvol.evaluate import MIN_OBS_ASSET, ub_col  # noqa: E402
from calvol.pipeline import eval_split  # noqa: E402
from calvol.stats import holm, kupiec_christoffersen  # noqa: E402
from calvol.utils import get_paths, load_config, save_table  # noqa: E402


def wilson(x, n, z=1.959964):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_paired(a, b, c, d):
    """95% CI for p1 - p2 with p1 = (a+b)/n (ours rejects), p2 = (a+c)/n (comparator rejects); Newcombe (1998) method 10."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / den if den > 0 else 0.0
    dl = np.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    du = np.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return p1 - p2, p1 - p2 - dl, p1 - p2 + du


def per_asset(df, col, level, h):
    d = df[np.isfinite(df[col]) & df["y"].notna()].sort_values("t_idx")
    out = {}
    for a, g in d.groupby("asset"):
        if len(g) < MIN_OBS_ASSET.get(h, 36):
            continue
        miss = (g["y"] > g[col]).to_numpy()
        rej, _ = kupiec_christoffersen(miss, 1 - level)
        out[a] = (int(rej), abs((1 - miss.mean()) - level))
    return out


def main():
    comp = sys.argv[1] if len(sys.argv) > 1 else "pooled_split"
    cfg = load_config(ROOT)
    P = get_paths(cfg)
    cc = cfg["conformal"]
    rows = []
    for h in cfg["panels"]["assets"]["horizons"]:
        f = P.bounds / f"assets_h{h}.parquet"
        if not f.exists():
            print("missing", f)
            continue
        df = pd.read_parquet(f)
        df = df[df["split"] == eval_split(cfg)]
        for b in cc["base_forecasters"]:
            for lv in cc["levels"]:
                c1, c2 = ub_col(b, "ours", lv), ub_col(b, comp, lv)
                if c1 not in df or c2 not in df:
                    continue
                r1, r2 = per_asset(df, c1, lv, h), per_asset(df, c2, lv, h)
                common = sorted(set(r1) & set(r2))
                x = np.array([r1[a][0] for a in common])
                y = np.array([r2[a][0] for a in common])
                a_ = int(((x == 1) & (y == 1)).sum()); b_ = int(((x == 1) & (y == 0)).sum())
                c_ = int(((x == 0) & (y == 1)).sum()); d_ = int(((x == 0) & (y == 0)).sum())
                m = b_ + c_
                p_mc = float(binom.cdf(b_, m, 0.5)) if m > 0 else 1.0
                diff, lo, hi = newcombe_paired(a_, b_, c_, d_)
                dev1 = np.array([r1[a][1] for a in common]); dev2 = np.array([r2[a][1] for a in common])
                try:
                    p_w = float(wilcoxon(dev1, dev2, alternative="less", zero_method="wilcox").pvalue)
                except ValueError:
                    p_w = np.nan
                rows.append({"base": b, "h": h, "level": lv, "n_assets": len(common), "both": a_, "ours_only": b_,
                             "comparator_only": c_, "neither": d_, "diff_pp": 100 * diff, "ci_low_pp": 100 * lo,
                             "ci_high_pp": 100 * hi, "p_mcnemar_exact": p_mc,
                             "median_abs_dev_ours_pp": 100 * np.median(dev1),
                             "median_abs_dev_comp_pp": 100 * np.median(dev2), "p_wilcoxon": p_w})
    t = pd.DataFrame(rows)
    if not len(t):
        print("no bounds found")
        return
    for col in ["p_mcnemar_exact", "p_wilcoxon"]:
        t[col + "_holm"] = np.nan
        for _, g in t.groupby(["base", "h"]):
            t.loc[g.index, col + "_holm"] = holm(g[col].fillna(1.0).to_numpy())
    save_table(t.set_index(["base", "h", "level"]), P, f"paired_tests_{comp}",
               caption=f"Paired comparison of asset-level calibration, proposed bound vs {comp}: exact McNemar test on "
                       "Kupiec outcomes, paired difference in rejection rates (Newcombe CI), Wilcoxon signed-rank test on "
                       "absolute per-asset coverage error; Holm across levels.", float_fmt="{:.4f}")
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
