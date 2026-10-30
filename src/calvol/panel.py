"""Panels, cleaning, features and targets (design document, Sections 4.4, 5.1, 6.1).

A VolPanel is a dense [T, N] array of daily returns on the union calendar of its instruments.
Conventions: information at day t uses closes up to t; the target for signal day t covers days t+1 .. t+h.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class VolPanel:
    name: str
    dates: pd.DatetimeIndex
    assets: list
    aclass: list
    ret: np.ndarray          # daily return [T, N], NaN before first observation
    obs: np.ndarray          # True where a real (not forward-filled) observation exists
    regime: np.ndarray | None = None   # log VIX on the panel calendar (NaN if unavailable)
    meta: dict = field(default_factory=dict)

    @property
    def T(self):
        return len(self.dates)

    @property
    def N(self):
        return len(self.assets)

    @property
    def first_obs(self):
        return np.where(self.obs.any(0), self.obs.argmax(0), self.T)

    def save(self, f: Path) -> None:
        np.savez_compressed(f, ret=self.ret, obs=self.obs,
                            regime=self.regime if self.regime is not None else np.full(self.T, np.nan))
        with open(str(f).replace(".npz", "_meta.json"), "w") as g:
            json.dump({"name": self.name, "dates": [str(d.date()) for d in self.dates], "assets": self.assets,
                       "aclass": self.aclass, "meta": self.meta}, g)

    @staticmethod
    def load(f: Path) -> "VolPanel":
        z = np.load(f)
        with open(str(f).replace(".npz", "_meta.json")) as g:
            m = json.load(g)
        return VolPanel(m["name"], pd.DatetimeIndex(pd.to_datetime(m["dates"])), m["assets"], m["aclass"],
                        z["ret"], z["obs"], z["regime"], m["meta"])


# ============================================================================ building


def build_yahoo_panel(px: pd.DataFrame, classes: dict, start: str, cutoff: str, ccfg: dict,
                      regime_symbol: str | None = None):
    """Clean Yahoo closes into a VolPanel. Returns (panel, outlier_report)."""
    px = px.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px[(px["date"] >= pd.Timestamp(start)) & (px["date"] <= pd.Timestamp(cutoff))]
    reg = px[px["ticker"] == regime_symbol] if regime_symbol else None
    px = px[px["ticker"].isin(classes)]
    assets = [t for t in classes if t in set(px["ticker"])]
    dates = pd.DatetimeIndex(sorted(px["date"].unique()))
    dates = dates[dates.dayofweek < 5]
    C = px.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").reindex(index=dates, columns=assets)
    C = np.array(C.to_numpy(dtype=np.float64), copy=True)
    C[~(C > 0)] = np.nan
    obs = np.isfinite(C)
    # bad prints: large move reversed the next day (thresholds by asset class)
    cf = pd.DataFrame(C).ffill(limit=ccfg["max_ffill_days"]).to_numpy()
    r = np.full_like(C, np.nan)
    r[1:] = np.log(cf[1:] / cf[:-1])
    rows = []
    for j, a in enumerate(assets):
        fut = classes[a] == "futures"
        thr = ccfg["futures_bad_print_abs"] if fut else ccfg["bad_print_abs"]
        rev = ccfg["futures_bad_print_reversal"] if fut else ccfg["bad_print_reversal"]
        x = r[:, j]
        big = np.abs(x) > thr
        nxt = np.zeros_like(big)
        nxt[:-1] = (np.abs(x[1:]) > rev) & (np.sign(x[1:]) == -np.sign(x[:-1]))
        bad = big & nxt
        for t in np.where(big)[0]:
            rows.append({"date": dates[t].date(), "asset": a, "log_return": float(x[t]),
                         "action": "removed (reversed next day)" if bad[t] else "kept (check)"})
        C[bad, j] = np.nan
        obs[bad, j] = False
    cf = pd.DataFrame(C).ffill(limit=ccfg["max_ffill_days"]).to_numpy()
    ret = np.full_like(C, np.nan)
    ret[1:] = np.log(cf[1:] / cf[:-1])
    regime = None
    if reg is not None and len(reg):
        s = reg.set_index("date")["close"].sort_index()
        regime = np.log(s[~s.index.duplicated()].reindex(dates).ffill().to_numpy())
    panel = VolPanel("assets", dates, assets, [classes[a] for a in assets], ret, obs, regime,
                     {"start": start, "cutoff": cutoff})
    return panel, pd.DataFrame(rows)


def build_factor_panel(fac: pd.DataFrame, names: list, start: str, cutoff: str, name: str,
                       regime: pd.Series | None = None) -> VolPanel:
    f = fac[fac["factor"].isin(names)].copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f[(f["date"] >= pd.Timestamp(start)) & (f["date"] <= pd.Timestamp(cutoff))]
    W = f.pivot_table(index="date", columns="factor", values="ret", aggfunc="last").sort_index()
    names = [n for n in names if n in W.columns]
    W = W[names]
    dates = pd.DatetimeIndex(W.index)
    ret = np.array(W.to_numpy(dtype=np.float64), copy=True)
    obs = np.isfinite(ret)
    reg = None
    if regime is not None:
        reg = regime.reindex(dates).ffill().to_numpy()
    return VolPanel(name, dates, names, ["factor"] * len(names), ret, obs, reg, {"start": start, "cutoff": cutoff})


# ============================================================================ features and targets


def _roll_mean(a: np.ndarray, w: int, minp: int) -> np.ndarray:
    return pd.DataFrame(a).rolling(w, min_periods=minp).mean().to_numpy()


def features(panel: VolPanel, cfg: dict) -> dict:
    """All daily quantities known at the close of t (no look-ahead)."""
    floor = cfg["cleaning"]["var_floor"]
    v = panel.ret ** 2                                      # NaN before first observation
    v0 = np.where(np.isfinite(v), v, np.nan)
    sm = cfg["forecast"]["smooth_days"]
    z = np.log(np.maximum(_roll_mean(v0, sm, max(2, sm - 2)), floor))
    n21 = 0.5 * np.log(np.maximum(_roll_mean(v0, 21, 15), floor))
    # HAR daily lag: last *traded* day's squared return. Non-trading days (holidays, forward-filled prices) give a
    # zero return; using it would put ln(floor) ~ -11.5 into the regression (bug found in the development run).
    traded = panel.obs & np.isfinite(v0) & (v0 > 0)
    v_last = pd.DataFrame(np.where(traded, v0, np.nan)).ffill().to_numpy()
    f1 = 0.5 * np.log(np.maximum(np.nan_to_num(v_last, nan=floor), floor))
    f5 = 0.5 * np.log(np.maximum(_roll_mean(v0, 5, 3), floor))
    vw = cfg["conformal"]["vov_window"]
    vov = pd.DataFrame(n21).rolling(vw, min_periods=vw // 2).std().to_numpy()
    vov = np.maximum(vov, cfg["conformal"]["vov_floor"])
    lam = cfg["forecast"]["ewma_lambda"]
    e = np.full_like(v0, np.nan)
    cur = np.full(panel.N, np.nan)
    for t in range(panel.T):
        x = v0[t]
        fx, fc = np.isfinite(x), np.isfinite(cur)
        cur = np.where(fx & fc, lam * cur + (1 - lam) * np.where(fx, x, 0.0), np.where(fx & ~fc, x, cur))
        e[t] = cur
    ewma = 0.5 * np.log(np.maximum(e, floor))
    f22 = 0.5 * np.log(np.maximum(_roll_mean(v0, 22, 15), floor))
    return {"v": v0, "z": z, "naive": n21, "har1": f1, "har5": f5, "har22": f22, "vov": vov, "ewma": ewma}


def target(panel: VolPanel, h: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """y[t] = 0.5 ln(sum_{d=t+1}^{t+h} r_d^2 / h); NaN if too many days in the window are not observed.
    Returns (y, end_idx) with end_idx[t] = t + h (the last day the target uses)."""
    T, N = panel.T, panel.N
    floor = cfg["cleaning"]["var_floor"]
    v = np.nan_to_num(panel.ret ** 2, nan=0.0)
    o = panel.obs.astype(np.int64)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(v, 0)])
    co = np.vstack([np.zeros((1, N), np.int64), np.cumsum(o, 0)])
    y = np.full((T, N), np.nan)
    t = np.arange(T - h)
    s = cs[t + h + 1] - cs[t + 1]
    n = co[t + h + 1] - co[t + 1]
    need = h if h == 1 else int(np.ceil((1 - cfg["cleaning"]["max_missing_target"]) * h))
    y[: T - h] = np.where(n >= need, 0.5 * np.log(np.maximum(s, floor * h) / h), np.nan)
    started = np.isfinite(panel.ret)
    y[~started] = np.nan
    return y.astype(np.float32), np.arange(T) + h


def window_valid(panel: VolPanel, L: int, max_missing: float) -> np.ndarray:
    T, N = panel.T, panel.N
    o = panel.obs.astype(np.int64)
    cs = np.vstack([np.zeros((1, N), np.int64), np.cumsum(o, 0)])
    valid = np.zeros((T, N), bool)
    t = np.arange(L - 1, T)
    valid[L - 1:] = (cs[t + 1] - cs[t + 1 - L]) >= np.ceil((1 - max_missing) * L)
    hist = np.arange(T)[:, None] - panel.first_obs[None, :] + 1
    return valid & (hist >= L) & panel.obs


def signal_days(panel: VolPanel, h: int) -> np.ndarray:
    """Signal days: month ends (h=21), week ends (h=5), every day (h=1); each needs a following day."""
    s = pd.Series(np.arange(panel.T), index=panel.dates)
    if h >= 21:
        idx = s.groupby([panel.dates.year, panel.dates.month]).max().to_numpy()
    elif h >= 5:
        iso = panel.dates.isocalendar()
        idx = s.groupby([iso["year"].to_numpy(), iso["week"].to_numpy()]).max().to_numpy()
    else:
        idx = s.to_numpy()
    idx = np.sort(idx)
    return idx[idx < panel.T - 1]
