"""Calibration and forecast evaluation (design document, Sections 3 and 8.1-8.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calib import pinball
from .stats import (clustered_coverage, diebold_mariano, holm, kupiec_christoffersen, model_confidence_set,
                    qlike, two_proportion_less)

MIN_OBS_ASSET = {21: 36, 5: 100, 1: 250}
BASE_COL = {"dl": "yhat_dl", "har": "yhat_har", "garch": "yhat_garch", "ewma": "yhat_ewma", "lstm": "yhat_lstm",
            "naive": "yhat_naive"}


def ub_col(base: str, method: str, level: float) -> str:
    return f"ub_{base}_{method}_{int(round(level * 100))}"


def _eval_rows(df: pd.DataFrame, col: str, base_col: str, level: float, h: int):
    d = df[np.isfinite(df[col]) & df["y"].notna()]
    if len(d) == 0:
        return None, None
    cov = (d["y"] <= d[col]).to_numpy()
    cl = clustered_coverage(cov, d["t_idx"].to_numpy(), level)
    rej_k = rej_c = n_ok = 0
    per_asset = []
    for a, g in d.assign(miss=~cov).groupby("asset"):
        per_asset.append(1 - g["miss"].mean())
        if len(g) < MIN_OBS_ASSET.get(h, 36):
            continue
        k, c = kupiec_christoffersen(g.sort_values("t_idx")["miss"].to_numpy(), 1 - level)
        rej_k += k
        rej_c += c
        n_ok += 1
    row = {"coverage": cl["coverage"], "clustered_z": cl["z"], "clustered_p": cl["p"],
           "kupiec_rejects": rej_k, "christoffersen_rejects": rej_c, "assets_tested": n_ok,
           "kupiec_share": rej_k / max(n_ok, 1), "christoffersen_share": rej_c / max(n_ok, 1),
           "worst_asset_cov_p5": float(np.quantile(per_asset, 0.05)) if per_asset else np.nan,
           "sharpness_log_ratio": float(np.mean(d[col] - d[base_col])),
           "pinball": float(np.mean(pinball(d["y"], d[col], level))), "n": int(len(d))}
    loss = pd.Series(pinball(d["y"], d[col], level), index=d.index).groupby(d["t_idx"]).mean()
    return row, loss


def calibration_tables(df: pd.DataFrame, h: int, levels, methods, bases, primary_base: str = "dl"):
    """Returns dict of tables: all (every base x method x level), h1 (ours vs comparators), h2 (ours coverage),
    pinball_dm (ours vs comparators), by_group (ours coverage by year / class / regime)."""
    rows, losses = [], {}
    for b in bases:
        ms = methods if b == primary_base else ["ours", "ours_vov", "pooled_split"]
        for m in ms:
            for lv in levels:
                c = ub_col(b, m, lv)
                if c not in df:
                    continue
                r, l = _eval_rows(df, c, BASE_COL[b], lv, h)
                if r is None:
                    continue
                r.update({"base": b, "method": m, "level": lv})
                rows.append(r)
                losses[(b, m, lv)] = l
    allt = pd.DataFrame(rows).set_index(["base", "method", "level"]) if rows else pd.DataFrame()
    # H1: ours vs each comparator (asset-level Kupiec rejection shares), per base and level
    h1 = []
    for (b, m, lv), r in allt.iterrows() if len(allt) else []:
        if m == "ours":
            continue
        if (b, "ours", lv) not in allt.index:
            continue
        o = allt.loc[(b, "ours", lv)]
        t = two_proportion_less(int(o["kupiec_rejects"]), int(o["assets_tested"]), int(r["kupiec_rejects"]),
                                int(r["assets_tested"]))
        h1.append({"base": b, "comparator": m, "level": lv, "ours_share": t["share_1"],
                   "comparator_share": t["share_2"], "z": t["z"], "p_one_sided": t["p_z"],
                   "p_fisher": t["p_fisher"]})
    h1 = pd.DataFrame(h1)
    if len(h1):
        prim = (h1["base"] == primary_base) & (h1["comparator"] == "pooled_split")
        h1["p_holm_primary"] = np.nan
        if prim.any():
            h1.loc[prim, "p_holm_primary"] = holm(h1.loc[prim, "p_one_sided"].fillna(1.0).to_numpy())
    # H2 and grouped coverage for ours (primary base)
    h2, groups = [], []
    for lv in levels:
        c = ub_col(primary_base, "ours", lv)
        if c not in df:
            continue
        d = df[np.isfinite(df[c]) & df["y"].notna()].copy()
        d["covered"] = d["y"] <= d[c]
        d["year"] = d["date"].dt.year
        by_year = d.groupby("year")["covered"].mean()
        r = allt.loc[(primary_base, "ours", lv)]
        row = {"level": lv, "coverage": r["coverage"], "clustered_p": r["clustered_p"],
               "max_year_deviation_pp": float(100 * (by_year - lv).abs().max()),
               "mean_year_deviation_pp": float(100 * (by_year - lv).abs().mean()),
               "worst_year": int((by_year - lv).abs().idxmax())}
        # benchmark for the (amended) relative yearly rule: pooled split conformal on the same rows
        cb = ub_col(primary_base, "pooled_split", lv)
        if cb in df:
            db = df[np.isfinite(df[cb]) & df["y"].notna()]
            yb = (db["y"] <= db[cb]).groupby(db["date"].dt.year).mean()
            row["pooled_max_year_deviation_pp"] = float(100 * (yb - lv).abs().max())
            row["pooled_mean_year_deviation_pp"] = float(100 * (yb - lv).abs().mean())
        h2.append(row)
        if d["regime"].notna().any():
            d["regime_tercile"] = pd.qcut(d["regime"], 3, labels=["low", "mid", "high"])
        for gname in ["year", "aclass", "regime_tercile"]:
            if gname not in d:
                continue
            gg = d.groupby(gname, observed=True)["covered"].agg(["mean", "size"]).reset_index()
            gg.columns = ["group", "coverage", "n"]
            gg["grouping"], gg["level"] = gname, lv
            groups.append(gg)
    h2 = pd.DataFrame(h2)
    if len(h2):
        h2["p_holm"] = holm(h2["clustered_p"].fillna(1.0).to_numpy())
    # pinball DM: ours vs each comparator
    dm = []
    for (b, m, lv), l in losses.items():
        if m == "ours" or (b, "ours", lv) not in losses:
            continue
        lo = losses[(b, "ours", lv)]
        j = pd.concat([l, lo], axis=1, keys=["comp", "ours"]).dropna()
        st, p = diebold_mariano(j["comp"].values, j["ours"].values)
        dm.append({"base": b, "comparator": m, "level": lv, "DM_stat": st, "p": p,
                   "mean_pinball_ours": j["ours"].mean(), "mean_pinball_comp": j["comp"].mean()})
    return {"all": allt, "h1": h1, "h2": h2, "pinball_dm": pd.DataFrame(dm),
            "by_group": pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()}


def forecast_table(df: pd.DataFrame, models: dict, mcs_size: float, mcs_reps: int, seed: int = 0):
    """QLIKE, MSE, Mincer-Zarnowitz; DM vs DLinear and vs HAR (H4); MCS membership."""
    cols = [c for c in models.values() if c in df]
    d = df.dropna(subset=["y"] + cols)
    rows, losses = [], {}
    for name, c in models.items():
        if c not in d:
            continue
        ql = qlike(d["y"], d[c])
        losses[name] = pd.Series(ql, index=d.index).groupby(d["t_idx"]).mean()
        X = np.column_stack([np.ones(len(d)), d[c]])
        b = np.linalg.lstsq(X, d["y"], rcond=None)[0]
        res = d["y"] - X @ b
        rows.append({"model": name, "QLIKE": float(np.mean(ql)), "MSE_logvol": float(((d["y"] - d[c]) ** 2).mean()),
                     "MZ_slope": float(b[1]), "MZ_R2": float(1 - res.var() / d["y"].var())})
    t = pd.DataFrame(rows).set_index("model")
    for ref in ["DLinear-QLIKE", "HAR"]:
        if ref not in losses:
            continue
        for name in t.index:
            if name == ref:
                continue
            st, p = diebold_mariano(losses[ref].values, losses[name].values)
            t.loc[name, f"DM_vs_{ref}"] = st
            t.loc[name, f"p_vs_{ref}"] = p
    try:
        mcs = model_confidence_set(pd.DataFrame(losses), mcs_size, mcs_reps, seed)
        t["MCS_pvalue"] = mcs
        t["in_MCS"] = t["MCS_pvalue"] >= mcs_size
    except Exception as e:  # noqa: BLE001
        t["MCS_note"] = f"MCS failed: {e}"
    return t
