"""Portfolio application: asset-level volatility upper bounds used in a portfolio risk budget.

ADDITIONAL ANALYSIS, designed after the main test-period evaluation (requested in peer feedback, October 2026).
All settings below were fixed before this script was run on real data and are not tuned. The script only reads saved
forecasts and bounds; it does not change any forecast, bound or earlier result. Results are reported for the
development period (2005-2014) and the test period (2015-2025).

Setting (monthly horizon h = 21, DLinear base forecaster, rebalancing on the monthly signal dates):
  * Universe at date t: instruments with a target, a forecast, a finite proposed and pooled bound at every level, and
    at least MIN_HIST of the last COV_WINDOW daily returns observed.
  * sigma_i^up = exp(U_i) is the upper bound for the realised volatility (daily RMS) of instrument i over t+1..t+21.
  * Covariance: Ledoit-Wolf shrinkage of the last COV_WINDOW daily returns; C = implied correlation matrix.
  * Portfolios (weights w >= 0, sum w = 1):
      EW     equal weights;
      MV     minimum variance with the Ledoit-Wolf covariance, w_i <= W_CAP (weights do not depend on the bounds);
      UB-MV  minimum variance with the "upper-bound covariance" D C D, D = diag(sigma^up), w_i <= W_CAP
             (weights depend on the bounds of each method).
  * Two portfolio risk bounds for the unit-exposure portfolio w:
      additive   B_add = sum_i w_i sigma_i^up. By the Minkowski inequality the realised portfolio volatility is at most
                 sum_i w_i RV_i, so B_add is never exceeded when every instrument is covered. It needs no assumption on
                 dependence but ignores diversification.
      dependence B_dep = sqrt(w' D C D w). Uses the estimated correlation; no guarantee.
  * Realised portfolio volatility RV_p = sqrt(mean_{d=t+1..t+21} (sum_i w_i r_{i,d})^2), same convention as the target
    (non-traded days count as zero returns).
  * Risk budget: exposure k = R_MAX / B (rest in cash), R_MAX = 10% a year. A breach is RV_p > B, i.e. the realised
    volatility of the scaled portfolio exceeds the budget. Exposure is reported without a cap.

Reported for each period, portfolio, method and level: number of months, breach rates of B_add and B_dep, Kupiec p-value
of the B_dep breaches, share of months with every instrument covered, mean exposure and mean realised volatility of the
scaled portfolio. Paired exact McNemar tests compare the proposed bound with pooled split conformal.

Usage (Colab, after the holdout run):   !python scripts/portfolio_risk_budget.py
Writes results/<mode>/tables/portfolio_risk_budget*.csv (+ .tex).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binom, chi2

H = 21
COV_WINDOW = 252
MIN_HIST = 200
W_CAP = 0.10
R_MAX = 0.10 / np.sqrt(252)               # 10% a year, in daily RMS units
METHODS = ["ours", "pooled_split", "pooled_aci", "per_asset", "mondrian", "parametric"]
REQUIRED = ["ours", "pooled_split"]       # must be finite for an instrument to enter the universe
PORTFOLIOS = ["EW", "MV", "UB-MV"]


# ----------------------------------------------------------------------------- helpers

def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity (same estimator as sklearn.covariance.LedoitWolf)."""
    X = X - X.mean(0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    X2 = X ** 2
    b2 = np.sum(X2.T @ X2 / n - S ** 2) / (n * p)
    b2 = min(b2, d2)
    sh = b2 / d2 if d2 > 0 else 1.0
    return sh * mu * np.eye(p) + (1 - sh) * S


def min_var(S: np.ndarray, cap: float) -> np.ndarray:
    n = len(S)
    cap = max(cap, 1.0 / n)
    w0 = np.full(n, 1.0 / n)
    Ss = S / np.mean(np.diag(S))           # rescale for numerical stability
    res = minimize(lambda w: w @ Ss @ w, w0, jac=lambda w: 2 * Ss @ w, method="SLSQP",
                   bounds=[(0.0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                                          "jac": lambda w: np.ones_like(w)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x if res.success else w0, 0, None)
    return w / w.sum()


def kupiec_p(breach: np.ndarray, alpha: float) -> float:
    n, x = len(breach), int(np.sum(breach))
    if n == 0:
        return np.nan
    ph = x / n

    def ll(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - p) + x * np.log(p)
    return float(chi2.sf(max(-2 * (ll(alpha) - ll(ph)), 0.0), 1))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact McNemar test; returns (a only, b only, p)."""
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    m = n10 + n01
    p = 1.0 if m == 0 else float(min(1.0, 2 * binom.cdf(min(n10, n01), m, 0.5)))
    return n10, n01, p


# ----------------------------------------------------------------------------- core

def run_analysis(ret: np.ndarray, obs: np.ndarray, bounds: pd.DataFrame, levels, methods=METHODS, log=print):
    """ret, obs: [T, N] panel arrays; bounds: saved bounds for h = 21 (columns t_idx, i_idx, split, y, yhat_dl,
    ub_dl_<method>_<level>). Returns (monthly detail, checks)."""
    T, N = ret.shape
    R = np.nan_to_num(ret, nan=0.0)
    seen = np.isfinite(ret) & obs
    methods = [m for m in methods if all(f"ub_dl_{m}_{int(round(lv * 100))}" in bounds for lv in levels)]
    rows, checks = [], []
    for t, g in bounds.groupby("t_idx"):
        t = int(t)
        if t < COV_WINDOW or t + H >= T:
            continue
        ok = np.isfinite(g["y"].to_numpy(float)) & np.isfinite(g["yhat_dl"].to_numpy(float))
        for m in REQUIRED:
            for lv in levels:
                ok &= np.isfinite(g[f"ub_dl_{m}_{int(round(lv * 100))}"].to_numpy(float))
        idx = g["i_idx"].to_numpy(int)
        hist = seen[t - COV_WINDOW + 1: t + 1].sum(0)
        ok &= hist[idx] >= MIN_HIST
        if ok.sum() < 10:
            continue
        g = g[ok]
        idx = g["i_idx"].to_numpy(int)
        n = len(idx)
        past = R[t - COV_WINDOW + 1: t + 1][:, idx]
        fut = R[t + 1: t + H + 1][:, idx]
        Sig = ledoit_wolf(past)
        sd = np.sqrt(np.diag(Sig))
        C = Sig / np.outer(sd, sd)
        rv_i = np.sqrt(np.mean(fut ** 2, 0))
        y = g["y"].to_numpy(float)
        checks.append({"t_idx": t, "n": n, "max_abs_rel_diff_target": float(np.max(np.abs(rv_i / np.exp(y) - 1)))})
        w_fixed = {"EW": np.full(n, 1.0 / n), "MV": min_var(Sig, W_CAP)}
        split = g["split"].iloc[0]
        date = g["date"].iloc[0] if "date" in g else t
        for lv in levels:
            L = int(round(lv * 100))
            for m in methods + ["forecast"]:
                if m == "forecast":
                    su = np.exp(g["yhat_dl"].to_numpy(float))
                else:
                    su = np.exp(g[f"ub_dl_{m}_{L}"].to_numpy(float))
                finite = np.isfinite(su)
                su_f = np.where(finite, su, np.nan)
                cover = np.all(y <= np.log(su))
                for pf in PORTFOLIOS:
                    if pf == "UB-MV":
                        if not finite.all():
                            continue
                        w = min_var(C * np.outer(su, su), W_CAP)
                    else:
                        w = w_fixed[pf]
                    rv_p = float(np.sqrt(np.mean((fut @ w) ** 2)))
                    if finite.all():
                        b_add = float(w @ su)
                        b_dep = float(np.sqrt(w @ (C * np.outer(su, su)) @ w))
                    else:                      # an infinite bound: budget never binding, no exposure
                        b_add = b_dep = np.inf
                    if m == "ours" and lv == levels[0]:
                        checks[-1].setdefault(f"minkowski_slack_{pf}", float(w @ rv_i - rv_p))
                    rows.append({"t_idx": t, "date": date, "split": split, "level": lv, "method": m, "portfolio": pf,
                                 "n_assets": n, "rv_p": rv_p, "b_add": b_add, "b_dep": b_dep,
                                 "breach_add": rv_p > b_add, "breach_dep": rv_p > b_dep, "all_covered": bool(cover),
                                 "exposure_add": R_MAX / b_add, "exposure_dep": R_MAX / b_dep,
                                 "n_infinite": int((~finite).sum()), "max_weight": float(w.max())})
        if len(checks) % 24 == 0:
            log(f"  {len(checks)} months done (t_idx={t}, {n} instruments)")
    return pd.DataFrame(rows), pd.DataFrame(checks)


def summarise(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = d.copy()
    d["period"] = np.where(d["split"] == "dev", "development", "test")
    ann = np.sqrt(252)
    out = []
    for (per, pf, m, lv), g in d.groupby(["period", "portfolio", "method", "level"]):
        a = 1 - lv
        out.append({"period": per, "portfolio": pf, "method": m, "level": lv, "months": len(g),
                    "breach_add_pct": 100 * g["breach_add"].mean(), "breach_dep_pct": 100 * g["breach_dep"].mean(),
                    "kupiec_p_dep": kupiec_p(g["breach_dep"].to_numpy(bool), a),
                    "all_covered_pct": 100 * g["all_covered"].mean(),
                    "exposure_dep": g["exposure_dep"].mean(), "exposure_add": g["exposure_add"].mean(),
                    "realised_vol_dep_ann_pct": 100 * ann * (g["exposure_dep"] * g["rv_p"]).mean(),
                    "realised_vol_add_ann_pct": 100 * ann * (g["exposure_add"] * g["rv_p"]).mean(),
                    "assets_mean": g["n_assets"].mean(), "infinite_bounds_months": int((g["n_infinite"] > 0).sum())})
    s = pd.DataFrame(out)
    pr = []
    for (per, pf, lv), g in d.groupby(["period", "portfolio", "level"]):
        piv = g.pivot_table(index="t_idx", columns="method", values=["breach_dep", "breach_add"], aggfunc="first")
        for kind in ["breach_dep", "breach_add"]:
            if ("ours" not in piv[kind]) or ("pooled_split" not in piv[kind]):
                continue
            x = piv[kind][["ours", "pooled_split"]].dropna().astype(bool)
            n10, n01, p = mcnemar_exact(x["ours"].to_numpy(), x["pooled_split"].to_numpy())
            pr.append({"period": per, "portfolio": pf, "level": lv, "bound": kind.replace("breach_", ""),
                       "months": len(x), "ours_only": n10, "pooled_only": n01, "p_mcnemar_exact": p})
    return s, pd.DataFrame(pr)


# ----------------------------------------------------------------------------- main

def main():
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from calvol.panel import VolPanel
    from calvol.utils import get_paths, load_config, save_table
    cfg = load_config(ROOT)
    P = get_paths(cfg)
    levels = list(cfg["conformal"]["levels"])
    panel = VolPanel.load(P.processed / "assets.npz")
    b = pd.read_parquet(P.bounds / f"assets_h{H}.parquet")
    print(f"panel {panel.T} days x {panel.N} instruments; bounds {len(b)} rows; splits {sorted(b['split'].unique())}")
    d, c = run_analysis(panel.ret, panel.obs, b, levels)
    d.to_csv(P.tables / "portfolio_risk_budget_months.csv", index=False)
    c.to_csv(P.tables / "portfolio_risk_budget_checks.csv", index=False)
    print("checks: max relative difference between recomputed and saved targets:",
          float(c["max_abs_rel_diff_target"].max()))
    mk = [k for k in c if k.startswith("minkowski")]
    print("checks: smallest Minkowski slack (must be >= 0):", float(c[mk].min().min()))
    s, pr = summarise(d)
    save_table(s.set_index(["period", "portfolio", "method", "level"]), P, "portfolio_risk_budget",
               caption="Portfolio risk budget built from asset-level volatility upper bounds (h = 21, DLinear base): "
                       "breach rates of the additive and the dependence-adjusted portfolio bound, Kupiec p-value, "
                       "mean exposure for a 10% annual budget and realised volatility of the scaled portfolio.",
               float_fmt="{:.4f}")
    save_table(pr.set_index(["period", "portfolio", "level", "bound"]), P, "portfolio_risk_budget_paired",
               caption="Paired exact McNemar tests of portfolio budget breaches, proposed bound vs pooled split.",
               float_fmt="{:.4f}")
    pd.set_option("display.width", 200)
    for per in ["development", "test"]:
        x = s[s["period"] == per]
        print(f"\n=== {per}: breach rate of the dependence-adjusted bound (%) ===")
        print(x.pivot_table(index=["portfolio", "method"], columns="level", values="breach_dep_pct").round(1))
        print(f"\n=== {per}: breach rate of the additive bound (%) ===")
        print(x.pivot_table(index=["portfolio", "method"], columns="level", values="breach_add_pct").round(1))
        print(f"\n=== {per}: mean exposure, dependence-adjusted budget ===")
        print(x.pivot_table(index=["portfolio", "method"], columns="level", values="exposure_dep").round(2))
    print("\n=== paired tests ===")
    print(pr.round(4).to_string())


if __name__ == "__main__":
    main()
