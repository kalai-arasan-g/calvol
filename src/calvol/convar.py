"""Exploratory extension (design document, Section 6.3): conformal value-at-risk.

Score = standardised h-day loss  s = -R_{i,t} / (sigma_hat_{i,t} * sqrt(h)),  R = sum of log returns t+1..t+h.
The conformal quantile q_t of past closed scores (pooled across assets, with or without ACI) gives
VaR_{i,t} = q_t * sigma_hat_{i,t} * sqrt(h). Compared with the Gaussian VaR z_level * sigma_hat * sqrt(h).
No hypothesis is attached; results are labelled exploratory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .calib import upper_bounds
from .panel import VolPanel
from .stats import kupiec_christoffersen

MIN_OBS = {21: 36, 5: 100, 1: 250}


def h_day_returns(panel: VolPanel, t_idx: np.ndarray, i_idx: np.ndarray, h: int) -> np.ndarray:
    r = np.nan_to_num(panel.ret, nan=0.0)
    cs = np.vstack([np.zeros((1, panel.N)), np.cumsum(r, 0)])
    out = np.full(len(t_idx), np.nan)
    ok = t_idx + h < panel.T
    out[ok] = cs[t_idx[ok] + h + 1, i_idx[ok]] - cs[t_idx[ok] + 1, i_idx[ok]]
    return out


def conformal_var_table(panel: VolPanel, pred: pd.DataFrame, h: int, bases: dict, levels, gamma: float, K: int,
                        min_closed: int, split: str) -> pd.DataFrame:
    p = pred.sort_values(["t_idx", "i_idx"]).reset_index(drop=True)
    R = h_day_returns(panel, p["t_idx"].to_numpy(), p["i_idx"].to_numpy(), h)
    rows = []
    for bname, col in bases.items():
        sig = np.exp(p[col].to_numpy(dtype=float)) * np.sqrt(h)
        frame = p[["t_idx", "i_idx", "asset", "aclass", "end_idx"]].copy()
        frame["y"] = -R / sig
        frame["zero"] = 0.0
        frame["vov"] = 1.0
        for lv in levels:
            var = {"gaussian": norm.ppf(lv) * sig}
            for m in ["pooled_split", "pooled_aci"]:
                q, _ = upper_bounds(frame, "zero", lv, m, gamma, K, min_closed)
                var[f"conformal_{m.split('_')[1]}"] = q * sig
            common = np.all([np.isfinite(v) for v in var.values()], 0) & np.isfinite(R)   # same rows for all
            for name, v in var.items():
                d = p.assign(loss=-R, var=v)
                d = d[(d["split"] == split) & common]
                if not len(d):
                    continue
                exc = (d["loss"] > d["var"]).to_numpy()
                rk = rc = n = 0
                for _, g in d.assign(e=exc).groupby("asset"):
                    if len(g) < MIN_OBS.get(h, 36):
                        continue
                    k, c = kupiec_christoffersen(g.sort_values("t_idx")["e"].to_numpy(), 1 - lv)
                    rk, rc, n = rk + k, rc + c, n + 1
                rows.append({"base": bname, "level": lv, "method": name, "exceed_rate": exc.mean(),
                             "target_rate": 1 - lv, "kupiec_share": rk / max(n, 1),
                             "christoffersen_share": rc / max(n, 1), "assets_tested": n, "n": len(d)})
    return pd.DataFrame(rows)
