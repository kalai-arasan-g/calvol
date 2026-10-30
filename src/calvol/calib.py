"""Conformal upper volatility bounds and comparators (design document, Section 6).

All bounds are one-sided upper bounds on log volatility y (so sigma_up = exp(ub)). The engine walks forward through
signal dates; at date t it uses only scores whose target window has closed (end_idx <= t) from the last K closed
dates. Each method differs only in the score and in how the quantile is pooled:

  ours            s = (y - yhat - m_i) / s_i, pooled over all assets, ACI    ub = yhat + m_i + Q * s_i
                  m_i = median, s_i = Q90 - median of asset i's own closed residuals over its last M forecasts
                  (amendment 2; amendments replace the registered vol-of-vol scale, see README)
  ours_mondrian   as ours, separate quantile per asset class, ACI
  ours_sd         as ours with mean and standard deviation (amendment 1 version, kept as ablation)
  ours_vov        s = (y - yhat) / vov, pooled, ACI  (the originally registered method, kept as comparator)
  pooled_split    s = y - yhat, pooled, fixed alpha                           ub = yhat + Q
  pooled_aci      s = y - yhat, pooled, ACI
  per_asset       s = y - yhat, each asset's own past scores, fixed alpha
  mondrian        s = y - yhat, separate quantile per asset class, fixed alpha
  cqr             s = y - q_tau (linear quantile regression on HAR features), pooled, fixed alpha
  parametric      ub = yhat + z_(1-alpha) * sd(asset's own past residuals)   (no conformal step)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

METHODS = {
    "ours": dict(norm="resid", kind="upper", group=None, aci=True),
    "ours_mondrian": dict(norm="resid", kind="upper", group="aclass", aci=True),
    "ours_sd": dict(norm="resid", kind="sd", group=None, aci=True),
    "ours_vov": dict(norm="vov", group=None, aci=True),
    "pooled_split": dict(norm=False, group=None, aci=False),
    "pooled_aci": dict(norm=False, group=None, aci=True),
    "per_asset": dict(norm=False, group="asset", aci=False),
    "mondrian": dict(norm=False, group="aclass", aci=False),
    "cqr": dict(norm=False, group=None, aci=False, cqr=True),
    "parametric": dict(parametric=True),
}


def _kth(sorted_s: np.ndarray, alpha: float) -> float:
    n = len(sorted_s)
    if n == 0:
        return np.nan
    a = min(max(alpha, 0.0), 1.0 - 1.0 / (n + 1))
    k = math.ceil((n + 1) * (1 - a))
    return float(sorted_s[min(k, n) - 1])


def residual_scale(pred: pd.DataFrame, base_col: str, window: int, min_n: int, floor: float = 0.02,
                   kind: str = "upper"):
    """Per row: location and scale of the same asset's residuals y - yhat over its last `window` forecasts whose
    target had closed by the row's forecast date (end_idx <= t_idx); at least `min_n` of them.

    kind = "upper" (amendment 2, main method): location = median, scale = Q90 - Q50 (upper-half spread). It ignores
           the long left tail of log-volatility errors (days with tiny or zero returns), which says nothing about
           an upper bound but inflates the standard deviation, and it adapts to differences in tail shape.
    kind = "sd"    (amendment 1): location = mean, scale = standard deviation.
    Rows without enough own history get location 0 and the median scale of the other assets on the same date; if
    none has one yet the row stays NaN (no full-sample statistic is ever used)."""
    loc = np.full(len(pred), np.nan)
    sc = np.full(len(pred), np.nan)
    res_all = pred["y"].to_numpy(dtype=float) - pred[base_col].to_numpy(dtype=float)
    t_all = pred["t_idx"].to_numpy()
    e_all = pred["end_idx"].to_numpy()
    for _, idx in pred.groupby("i_idx").indices.items():
        idx = idx[np.argsort(t_all[idx], kind="stable")]
        t = t_all[idx]
        e = np.maximum.accumulate(e_all[idx])            # end_idx increases with t for a fixed horizon
        roll = pd.Series(res_all[idx]).rolling(window, min_periods=min_n)
        if kind == "upper":
            m_ = roll.quantile(0.5).to_numpy()
            s_ = roll.quantile(0.9).to_numpy() - m_
        else:
            m_ = roll.mean().to_numpy()
            s_ = roll.std(ddof=1).to_numpy()
        c = np.minimum(np.searchsorted(e, t, side="right"), np.arange(len(t)))   # rows closed by t
        j = c - 1                                          # rolling statistic ending at the last closed row
        ok = j >= 0
        loc[idx[ok]] = m_[j[ok]]
        sc[idx[ok]] = s_[j[ok]]
    sc = np.maximum(sc, floor)
    fill = pd.Series(sc).groupby(t_all).transform("median").to_numpy()
    miss = ~np.isfinite(sc)
    sc[miss] = fill[miss]
    loc[~np.isfinite(loc)] = 0.0
    return loc, sc


def upper_bounds(pred: pd.DataFrame, base_col: str, level: float, method: str, gamma: float, K: int,
                 min_closed: int, min_group: int = 30, min_asset: int = 12, scale_window: int = 60,
                 scale_min: int = 12) -> tuple[np.ndarray, pd.DataFrame]:
    """Upper bound ub (log vol) for every row of `pred` (NaN until enough closed dates exist).

    pred must be sorted by t_idx and contain: t_idx, end_idx, y, vov, asset, aclass and the base column
    (and q_cqr_<level> for 'cqr')."""
    spec = METHODS[method]
    target = 1.0 - level
    t_idx = pred["t_idx"].to_numpy()
    dates, starts = np.unique(t_idx, return_index=True)
    stops = np.append(starts[1:], len(pred))
    ends = pred["end_idx"].to_numpy()[starts]
    y = pred["y"].to_numpy(dtype=float)
    if spec.get("cqr"):
        base = pred[f"q_cqr_{int(round(level * 100))}"].to_numpy(dtype=float)
    else:
        base = pred[base_col].to_numpy(dtype=float)
    loc = np.zeros(len(pred))
    if spec.get("norm") == "vov":
        scale = pred["vov"].to_numpy(dtype=float)
    elif spec.get("norm") == "resid":
        loc, scale = residual_scale(pred, base_col, scale_window, scale_min, kind=spec["kind"])
    else:
        scale = np.ones(len(pred))
    base = base + loc
    score = (y - base) / scale
    grp = None
    if spec.get("group"):
        grp = pd.factorize(pred[spec["group"]])[0]
    ub = np.full(len(pred), np.nan)
    alpha = target
    out_rows = []
    issued_miss = {}
    c_prev = 0
    for k, t in enumerate(dates):
        c = int(np.searchsorted(ends, t, side="right"))
        c = min(c, k)                                   # only earlier dates
        # ACI feedback from dates that closed since the last step
        for j in range(c_prev, c):
            if j in issued_miss and spec.get("aci"):
                m = issued_miss.pop(j)
                if np.isfinite(m):
                    alpha = alpha + gamma * (target - m)
            else:
                issued_miss.pop(j, None)
        c_prev = c
        if c < min_closed:
            continue
        lo = max(0, c - K)
        sl = slice(starts[lo], stops[c - 1])
        cal_s = score[sl]
        okc = np.isfinite(cal_s)
        rs = slice(starts[k], stops[k])
        if spec.get("parametric"):
            res = (y - base)[sl]
            ids = pred["i_idx"].to_numpy()[sl]
            sd_by = pd.Series(res[okc]).groupby(ids[okc]).std()
            cnt = pd.Series(res[okc]).groupby(ids[okc]).size()
            sd_pool = np.nanstd(res[okc])
            mine = pred["i_idx"].to_numpy()[rs]
            sd = np.array([sd_by.get(a, sd_pool) if cnt.get(a, 0) >= min_asset else sd_pool for a in mine])
            ub[rs] = base[rs] + norm.ppf(level) * sd
        else:
            a_use = alpha if spec.get("aci") else target
            pooled = np.sort(cal_s[okc])
            if grp is None:
                q = np.full(stops[k] - starts[k], _kth(pooled, a_use))
            else:
                g_cal = grp[sl][okc]
                s_cal = cal_s[okc]
                thr = min_asset if spec["group"] == "asset" else min_group
                cache = {}
                q = np.empty(stops[k] - starts[k])
                for n_, gv in enumerate(grp[rs]):
                    if gv not in cache:
                        arr = np.sort(s_cal[g_cal == gv])
                        cache[gv] = _kth(arr, a_use) if len(arr) >= thr else _kth(pooled, a_use)
                    q[n_] = cache[gv]
            ub[rs] = base[rs] + q * scale[rs]
        yy = y[rs]
        okk = np.isfinite(yy) & np.isfinite(ub[rs])
        miss = float(np.mean(yy[okk] > ub[rs][okk])) if okk.any() else np.nan
        issued_miss[k] = miss
        out_rows.append({"t_idx": int(t), "alpha_t": alpha, "n_calib": int(okc.sum()), "miss_rate": miss})
    return ub, pd.DataFrame(out_rows)


def pinball(y, q, tau):
    """Quantile (pinball) loss of an upper bound q at level tau; lower is better."""
    d = np.asarray(y) - np.asarray(q)
    return np.maximum(tau * d, (tau - 1) * d)


def choose_gamma(pred: pd.DataFrame, base_col: str, level: float, grid, K: int, min_closed: int,
                 split: str = "dev", scale_window: int = 60, scale_min: int = 12) -> tuple[float, pd.DataFrame]:
    """gamma for the ACI methods: minimum mean pinball loss of 'ours' on development rows."""
    rows = []
    for g in grid:
        ub, _ = upper_bounds(pred, base_col, level, "ours", g, K, min_closed, scale_window=scale_window,
                             scale_min=scale_min)
        m = (pred["split"] == split).to_numpy() & np.isfinite(ub) & np.isfinite(pred["y"].to_numpy())
        rows.append({"gamma": g, "pinball": float(np.mean(pinball(pred["y"].to_numpy()[m], ub[m], level))),
                     "coverage": float(np.mean(pred["y"].to_numpy()[m] <= ub[m]))})
    tab = pd.DataFrame(rows)
    return float(tab.sort_values("pinball").iloc[0]["gamma"]), tab
