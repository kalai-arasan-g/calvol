"""Conformal volatility management of factor portfolios (design document, Section 7).

Exposure set at month end t, applied over the next month:  w = min(w_max, c_t / V_t),  w >= 0
c_t is chosen in real time so that the average exposure over all past months equals 1 (Cederburg et al., 2020).
Inverse-volatility variants use sqrt(V_t) instead of V_t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import VolPanel, signal_days
from .stats import (cvar, lw_sharpe_test, lw_variance_test, max_drawdown, mm_alpha, paired_bootstrap_ci,
                    romano_wolf_variance, sharpe)

STRATS = ["UNMANAGED", "VM_RV", "VM_HAR", "VM_DL", "CVM", "CVM_HAR"]


def monthly_frame(panel: VolPanel, lag: int = 0) -> pd.DataFrame:
    """Per factor and signal month end t: next-month compounded return and last-month realised variance."""
    sig = signal_days(panel, 21)
    r = panel.ret
    rows = []
    for k in range(1, len(sig) - 1):
        t0, t, t1 = sig[k - 1], sig[k], sig[k + 1]
        hold = slice(t + 1 + lag, t1 + 1 + lag)
        past = slice(t0 + 1, t + 1)
        if hold.stop > panel.T:
            break
        fut = np.prod(1 + np.nan_to_num(r[hold], nan=0.0), 0) - 1
        rv = np.nansum(r[past] ** 2, 0)
        ok_f = np.isfinite(r[hold]).sum(0) >= 0.8 * (hold.stop - hold.start)
        ok_p = np.isfinite(r[past]).sum(0) >= 0.8 * (past.stop - past.start)
        for i, a in enumerate(panel.assets):
            if ok_f[i] and ok_p[i]:
                rows.append({"t_idx": int(t), "date": panel.dates[t], "asset": a, "f_next": float(fut[i]),
                             "rv_prev": float(rv[i]) / (past.stop - past.start)})
    return pd.DataFrame(rows)


def build_strategies(mon: pd.DataFrame, pred: pd.DataFrame, cvm_level: float, vcfg: dict) -> pd.DataFrame:
    """Exposures and net returns for every strategy (inverse-variance and inverse-volatility rules)."""
    lv = int(round(cvm_level * 100))
    p = pred[["t_idx", "asset", "yhat_har", "yhat_dl", f"ub_dl_ours_{lv}", f"ub_har_ours_{lv}", "split"]]
    d = mon.merge(p, on=["t_idx", "asset"], how="inner")
    V = {"VM_RV": d["rv_prev"], "VM_HAR": np.exp(2 * d["yhat_har"]), "VM_DL": np.exp(2 * d["yhat_dl"]),
         "CVM": np.exp(2 * d[f"ub_dl_ours_{lv}"]), "CVM_HAR": np.exp(2 * d[f"ub_har_ours_{lv}"])}
    d = d.assign(**{f"V_{k}": v for k, v in V.items()})
    d = d.dropna(subset=[f"V_{k}" for k in V]).sort_values(["asset", "t_idx"]).reset_index(drop=True)
    cost = vcfg["cost_bps"] / 1e4
    out = []
    for a, g in d.groupby("asset"):
        g = g.reset_index(drop=True)
        n = np.arange(1, len(g) + 1)
        for rule, power in [("var", 1.0), ("vol", 0.5)]:
            for s in STRATS:
                if s == "UNMANAGED":
                    w = np.ones(len(g))
                else:
                    inv = 1.0 / g[f"V_{s}"].to_numpy() ** power
                    c = 1.0 / (np.cumsum(inv) / n)            # real-time: mean past exposure = 1
                    w = np.clip(c * inv, 0, vcfg["w_max"])
                    w[n <= vcfg["min_history_months"]] = 1.0
                dw = np.abs(np.diff(np.concatenate([[w[0]], w])))
                gross = w * g["f_next"].to_numpy()
                out.append(pd.DataFrame({"t_idx": g["t_idx"], "date": g["date"], "asset": a, "rule": rule,
                                         "strategy": s, "w": w, "gross": gross, "turnover": dw,
                                         "net": gross - cost * dw, "split": g["split"],
                                         "live": n > vcfg["min_history_months"]}))
    return pd.concat(out, ignore_index=True)


def pooled(st: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight combination of all factors available each month (per rule and strategy)."""
    g = st.groupby(["rule", "strategy", "t_idx", "date", "split"], as_index=False).agg(
        net=("net", "mean"), gross=("gross", "mean"), w=("w", "mean"), turnover=("turnover", "mean"),
        live=("live", "all"))
    g["asset"] = "POOLED"
    return g


def summary(st: pd.DataFrame, split: str) -> pd.DataFrame:
    d = st[(st["split"] == split) & st["live"]]
    rows = []
    for (rule, a, s), g in d.groupby(["rule", "asset", "strategy"]):
        x = g.sort_values("t_idx")["net"].to_numpy()
        if len(x) < 24:
            continue
        rows.append({"rule": rule, "asset": a, "strategy": s, "ann_mean": x.mean() * 12,
                     "ann_vol": x.std(ddof=1) * np.sqrt(12), "sharpe": sharpe(x), "max_drawdown": max_drawdown(x),
                     "cvar95": cvar(x), "mean_exposure": g["w"].mean(), "turnover": g["turnover"].mean(),
                     "months": len(x), "_gross": g["gross"].mean(), "_turn": g["turnover"].mean()})
    out = pd.DataFrame(rows)
    if len(out):
        # break-even cost (bps per unit change in exposure) at which the strategy's mean return equals UNMANAGED's
        ref = out[out["strategy"] == "UNMANAGED"].set_index(["rule", "asset"])["_gross"]
        key = list(zip(out["rule"], out["asset"]))
        gap = out["_gross"].to_numpy() - np.array([ref.get(k, np.nan) for k in key])
        out["breakeven_cost_bps"] = np.where(out["_turn"] > 0, 1e4 * gap / out["_turn"].replace(0, np.nan), np.nan)
        out = out.drop(columns=["_gross", "_turn"])
    return out


def h3_tests(st: pd.DataFrame, split: str, reps: int, seed: int = 0, comparators=("VM_DL", "VM_HAR", "VM_RV")):
    """CVM vs each comparator: Ledoit-Wolf (2011) variance test (primary: pooled panel), bootstrap CIs of the
    max-drawdown and CVaR differences, Ledoit-Wolf (2008) Sharpe p-value (descriptive). Per-factor variance p-values
    are adjusted across factors with the Romano-Wolf (2005) step-down (design document, Section 8.4)."""
    d = st[(st["split"] == split) & st["live"]]
    rows = []
    for rule in d["rule"].unique():
        for a in d["asset"].unique():
            g = d[(d["rule"] == rule) & (d["asset"] == a)]
            piv = g.pivot_table(index="t_idx", columns="strategy", values="net").dropna()
            if "CVM" not in piv or len(piv) < 36:
                continue
            for c in comparators:
                if c not in piv:
                    continue
                x1, x2 = piv["CVM"].values, piv[c].values
                v = lw_variance_test(x1, x2, reps, seed)
                dd = paired_bootstrap_ci(x1, x2, max_drawdown, reps // 5, seed)
                cv = paired_bootstrap_ci(x1, x2, cvar, reps // 5, seed)
                sh = lw_sharpe_test(x1, x2)
                rows.append({"rule": rule, "asset": a, "comparator": c, "vol_ratio": v["vol_ratio"],
                             "p_var_lower": v["p_lower"], "p_var_two_sided": v["p_two_sided"],
                             "mdd_diff": dd["diff"], "mdd_ci_low": dd["ci_low"], "mdd_ci_high": dd["ci_high"],
                             "cvar_diff": cv["diff"], "cvar_ci_low": cv["ci_low"], "cvar_ci_high": cv["ci_high"],
                             "sharpe_cvm": sharpe(x1), "sharpe_comp": sharpe(x2), "p_sharpe_lw2008": sh["p_two_sided"],
                             "months": len(piv)})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_var_romano_wolf"] = np.nan
        f = d[d["asset"] != "POOLED"]
        for rule in t["rule"].unique():
            fr = f[f["rule"] == rule]
            for c in comparators:
                piv = fr[fr["strategy"].isin(["CVM", c])].pivot_table(index="t_idx", columns=["strategy", "asset"],
                                                                        values="net")
                if "CVM" not in piv.columns.get_level_values(0) or c not in piv.columns.get_level_values(0):
                    continue
                assets = sorted(set(piv["CVM"].columns) & set(piv[c].columns))
                R1, R2 = piv["CVM"][assets], piv[c][assets]
                ok = R1.notna().all(1) & R2.notna().all(1)
                if ok.sum() < 36 or len(assets) < 2:
                    continue
                adj = romano_wolf_variance(R1[ok].values, R2[ok].values, reps, seed)
                for a, p in zip(assets, adj):
                    m = (t["rule"] == rule) & (t["comparator"] == c) & (t["asset"] == a)
                    t.loc[m, "p_var_romano_wolf"] = p
    return t


def alphas(st: pd.DataFrame, split: str) -> pd.DataFrame:
    d = st[(st["split"] == split) & st["live"]]
    rows = []
    for (rule, a), g in d.groupby(["rule", "asset"]):
        piv = g.pivot_table(index="t_idx", columns="strategy", values="net").dropna()
        if len(piv) < 36:
            continue
        for s in STRATS[1:]:
            if s in piv:
                r = mm_alpha(piv[s].values, piv["UNMANAGED"].values)
                r.update({"rule": rule, "asset": a, "strategy": s})
                rows.append(r)
    return pd.DataFrame(rows)


def choose_cvm_level(mon: pd.DataFrame, pred: pd.DataFrame, levels, vcfg: dict) -> tuple[float, pd.DataFrame]:
    """Pre-registered rule (Section 7.2): level with the highest development-period Sharpe of the pooled panel
    (inverse-variance rule)."""
    rows = []
    for lv in levels:
        st = build_strategies(mon, pred, lv, vcfg)
        pl = pooled(st)
        x = pl[(pl["split"] == "dev") & pl["live"] & (pl["rule"] == "var") & (pl["strategy"] == "CVM")]
        rows.append({"level": lv, "dev_sharpe_pooled": sharpe(x.sort_values("t_idx")["net"].values),
                     "months": len(x)})
    tab = pd.DataFrame(rows)
    return float(tab.sort_values("dev_sharpe_pooled", ascending=False).iloc[0]["level"]), tab
