"""Walk-forward volatility forecasting (design document, Section 5).

For each refit year Y (every `refit_every_years`):
  cutoff   = first signal day of Y; training pool = samples whose target window closed by the cutoff
  inner validation = last `inner_val_months` of the pool; inner training purged and embargoed before it
  DLinear-QLIKE: hyper-parameters chosen on inner validation; refit on the full pool with several seeds
  LSTM: pooled, fixed settings; HAR, GARCH: per asset; CQR: pooled linear quantile regression on HAR features
  predictions for every signal day in [Y, Y + refit_every) - nothing from those years enters any fit.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from .models import WindowBank, build_model, fit, predict
from .panel import VolPanel, features, signal_days, target, window_valid


def _t(a, device):
    return torch.as_tensor(np.array(a, dtype=np.int64, copy=True), dtype=torch.long, device=device)


def _pairs(rows_mask: np.ndarray, S: np.ndarray, stride: int):
    days = np.where(rows_mask & (np.arange(len(rows_mask)) % stride == 0))[0]
    dd, ii = np.nonzero(S[days])
    return days[dd], ii


def refit_schedule(panel: VolPanel, sig: np.ndarray, pcfg: dict) -> list[tuple[int, list[int]]]:
    years = sorted({int(y) for y in panel.dates[sig].year if y >= pcfg["first_forecast_year"]})
    step = int(pcfg.get("refit_every_years", 1))
    out = []
    for k in range(0, len(years), step):
        out.append((years[k], years[k:k + step]))
    return out


def _splits(panel, end_idx, fcfg, cutoff):
    T = panel.T
    pool = end_idx <= cutoff
    vs = int(panel.dates.searchsorted(panel.dates[cutoff] - pd.DateOffset(months=fcfg["inner_val_months"])))
    inner_val = pool & (np.arange(T) >= vs)
    inner_tr = pool & (end_idx < vs - fcfg["embargo_days"])
    return pool, inner_tr, inner_val


class LSTMVol(nn.Module):
    def __init__(self, C, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(C, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)


GARCH_CODES = {0: "ok", 1: "short sample", 2: "fit failed", 3: "persistence capped", 4: "far from naive"}


def _garch_one(r, fit_end, window, days, h):
    """GARCH(1,1)-t on the last `window` daily returns up to fit_end; h-day variance forecasts for `days`.
    Returns (log-vol forecasts, status code). Non-stationary fits (alpha + beta >= 0.999) are not discarded: the
    persistence is capped at 0.999 with omega set by variance targeting (code 3), instead of falling back to EWMA."""
    from arch import arch_model
    lo = max(0, fit_end - window + 1)
    x = r[lo: fit_end + 1] * 100.0
    x = x[np.isfinite(x) & (x != 0.0)]          # drop non-trading days (zero returns from forward-filled prices)
    if len(x) < 250:
        return np.full(len(days), np.nan), 1
    code = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(
                disp="off", show_warning=False, options={"maxiter": 500})
        mu, om, al, be = (float(res.params[k]) for k in ["mu", "omega", "alpha[1]", "beta[1]"])
    except Exception:  # noqa: BLE001
        return np.full(len(days), np.nan), 2
    if not all(np.isfinite([mu, om, al, be])) or al < 0 or be < 0:
        return np.full(len(days), np.nan), 2
    pers = al + be
    if pers >= 0.999 or om <= 0:
        f = 0.999 / max(pers, 1e-12)
        al, be = al * f, be * f
        pers = 0.999
        om = np.var(x) * (1 - pers)
        code = 3
    wbar = om / (1 - pers)
    last = int(days.max())
    xs = r[lo: last + 1] * 100.0
    xs = np.where(np.isfinite(xs), xs, mu)       # missing day: no shock
    s2 = np.empty(len(xs) + 1)
    s2[0] = np.var(x)
    for k in range(len(xs)):
        s2[k + 1] = om + al * (xs[k] - mu) ** 2 + be * s2[k]
    j = np.arange(1, h + 1)
    out = np.empty(len(days))
    for n, t in enumerate(days):
        s_next = s2[t - lo + 1]
        out[n] = 0.5 * np.log(np.mean(wbar + pers ** (j - 1) * (s_next - wbar)) / 1e4)
    return out, code


def run_forecasts(panel: VolPanel, h: int, cfg: dict, pcfg: dict, dev_end: str, device, logger,
                  include_lstm: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (predictions, selected DLinear hyper-parameters)."""
    fc = cfg["forecast"]
    dl = fc["dlinear"]
    L = fc["lookback"]
    stride = int(fc["train_stride"][h])
    F = features(panel, cfg)
    y, end_idx = target(panel, h, cfg)
    valid = window_valid(panel, L, cfg["cleaning"]["max_missing_window"])
    S = valid & np.isfinite(y)
    sig = signal_days(panel, h)
    X = np.stack([np.nan_to_num(panel.ret, nan=0.0),
                  pd.DataFrame(F["z"]).ffill().bfill().to_numpy()], -1).astype(np.float32)
    bank = WindowBank(X, np.zeros_like(y), y, h, device)
    chans = ["ret", "logvar"]
    dev_end_ts = pd.Timestamp(dev_end)
    rows, hp = [], []
    har_X = np.stack([np.ones_like(F["har1"]), F["har1"], F["har5"], F["har22"]], -1)
    levels = cfg["conformal"]["levels"]
    r_daily = np.nan_to_num(panel.ret, nan=0.0) if panel.name != "assets" else np.nan_to_num(panel.ret, nan=0.0)

    for Y, years in refit_schedule(panel, sig, pcfg):
        sig_y = sig[np.isin(panel.dates[sig].year, years)]
        if len(sig_y) == 0:
            continue
        cutoff = int(sig_y[0])
        pool, inner_tr, inner_val = _splits(panel, end_idx, fc, cutoff)
        d_tr, i_tr = _pairs(inner_tr, S, stride)
        d_va, i_va = _pairs(inner_val, S, stride)
        d_po, i_po = _pairs(pool, S, stride)
        if len(d_tr) < 300 or len(d_va) < 50:
            logger.warning(f"{panel.name} h={h} {Y}: too few samples (train {len(d_tr)}, val {len(d_va)}) - skipped")
            continue
        pd_, pi_ = [], []
        for t in sig_y:
            ids = np.where(valid[t])[0]
            pd_ += [int(t)] * len(ids)
            pi_ += list(ids)
        pd_, pi_ = np.array(pd_, np.int64), np.array(pi_, np.int64)
        if len(pd_) == 0:
            continue
        T_tr, I_tr, T_va, I_va = (_t(a, device) for a in (d_tr, i_tr, d_va, i_va))
        T_po, I_po, P_d, P_i = (_t(a, device) for a in (d_po, i_po, pd_, pi_))
        # ---------------- DLinear-QLIKE
        best = None
        for wd, k in itertools.product(dl["weight_decay_grid"], dl["kernel_grid"]):
            m = build_model("dlinear", L, 2, kernel=k, channels=chans)
            m, ep, hist = fit(m, bank, T_tr, I_tr, L, "vol", dl["loss"], dl["lr"], wd, dl["batch_size"],
                              dl["max_epochs"], 0, T_va, I_va, dl["patience"])
            v = min(hist) if hist else np.inf
            if best is None or v < best[0]:
                best = (v, wd, k, ep)
        v, wd, k, ep = best
        hp.append({"panel": panel.name, "h": h, "refit_year": Y, "weight_decay": wd, "kernel": k, "epochs": ep,
                   "inner_val_loss": v, "n_pool": len(d_po)})
        outs = []
        for seed in range(dl["seeds"]):
            m = build_model("dlinear", L, 2, kernel=k, channels=chans)
            m, _, _ = fit(m, bank, T_po, I_po, L, "vol", dl["loss"], dl["lr"], wd, dl["batch_size"], ep, 1000 + seed)
            outs.append(predict(m, bank, P_d, P_i, L, "vol"))
        yhat_dl = np.mean(outs, 0)
        # ---------------- LSTM (pooled, 5-day window, MSE on standardised target)
        yhat_lstm = np.full(len(pd_), np.nan)
        if include_lstm:
            lc = fc["lstm"]
            W = lc["window"]
            xs = bank.windows(T_po[:20000], I_po[:20000], W)
            mu_x, sd_x = xs.mean((0, 1), keepdim=True), xs.std((0, 1), keepdim=True) + 1e-6
            yp = bank.y_v[T_po, I_po]
            my, sy = yp.mean(), yp.std() + 1e-8
            preds, nets = [], []
            for seed in range(lc["seeds"]):
                torch.manual_seed(2000 + seed)
                g = torch.Generator().manual_seed(2000 + seed)
                net = LSTMVol(2, lc["hidden"], lc["layers"]).to(device)
                opt = torch.optim.Adam(net.parameters(), lr=1e-3)
                n = len(T_po)
                for _ in range(lc["epochs"]):
                    perm = torch.randperm(n, generator=g).to(device)
                    for b0 in range(0, n, dl["batch_size"]):
                        b = perm[b0:b0 + dl["batch_size"]]
                        xb = (bank.windows(T_po[b], I_po[b], W) - mu_x) / sd_x
                        loss = ((net(xb) - (yp[b] - my) / sy) ** 2).mean()
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                with torch.no_grad():
                    out = [net((bank.windows(P_d[b0:b0 + 8192], P_i[b0:b0 + 8192], W) - mu_x) / sd_x)
                           for b0 in range(0, len(P_d), 8192)]
                    preds.append((torch.cat(out) * sy + my).cpu().numpy())
                nets.append(net)
            yhat_lstm = np.mean(preds, 0)
            # Duan smearing: the LSTM is fitted by MSE in log-vol, which forecasts E[ln vol], not ln E[vol]; the
            # in-sample correction 0.5 ln mean exp(2e) makes it a variance forecast (fair comparison under QLIKE)
            with torch.no_grad():
                sub_i = torch.arange(0, len(T_po), max(1, len(T_po) // 20000), device=T_po.device)
                xb = (bank.windows(T_po[sub_i], I_po[sub_i], W) - mu_x) / sd_x
                fit_in = np.mean([(nt(xb) * sy + my).cpu().numpy() for nt in nets], 0)
            e_in = yp[sub_i].cpu().numpy() - fit_in
            e_in = e_in[np.isfinite(e_in)]
            if len(e_in):
                yhat_lstm = yhat_lstm + 0.5 * np.log(np.mean(np.exp(2 * np.clip(e_in, -10, 10))))
        # ---------------- HAR per asset (pooled fallback), CQR quantiles (pooled)
        dd, ii = np.nonzero(pool[:, None] & S)
        Xh, yh = har_X[dd, ii], y[dd, ii]
        okh = np.isfinite(Xh).all(1) & np.isfinite(yh)
        beta_pool = np.linalg.lstsq(Xh[okh], yh[okh], rcond=None)[0]

        def _smear(e):  # Duan (1983) smearing: HAR in logs forecasts E[ln vol]; QLIKE needs ln E[vol]
            e = e[np.isfinite(e)]
            return 0.5 * np.log(np.mean(np.exp(2 * np.clip(e, -10, 10)))) if len(e) else 0.0
        c_pool = _smear(yh[okh] - Xh[okh] @ beta_pool)
        yhat_har = np.full(len(pd_), np.nan)
        for a in np.unique(pi_):
            sel = okh & (ii == a)
            if sel.sum() >= fc["har_min_obs"]:
                b = np.linalg.lstsq(Xh[sel], yh[sel], rcond=None)[0]
                c_a = _smear(yh[sel] - Xh[sel] @ b)
            else:
                b, c_a = beta_pool, c_pool
            rr = pi_ == a
            yhat_har[rr] = har_X[pd_[rr], a] @ b + c_a
        q_cqr = {}
        import statsmodels.api as sm
        sub = np.where(okh)[0]
        if len(sub) > 150000:
            sub = np.random.default_rng(Y).choice(sub, 150000, replace=False)
        for lv in levels:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qb = sm.QuantReg(yh[sub], Xh[sub]).fit(q=lv, max_iter=2000).params
            q_cqr[lv] = har_X[pd_, pi_] @ qb
        # ---------------- GARCH per asset
        yhat_garch = np.full(len(pd_), np.nan)
        garch_code = np.full(len(pd_), 1, dtype=np.int8)
        if fc["garch"]["enabled"]:
            jobs = [(a, np.where(pi_ == a)[0]) for a in np.unique(pi_)]
            res = Parallel(n_jobs=-1)(delayed(_garch_one)(r_daily[:, a], cutoff - 1, fc["garch"]["fit_window"],
                                                          pd_[rws], h) for a, rws in jobs)
            for (a, rws), (v_, c_) in zip(jobs, res):
                yhat_garch[rws] = v_
                garch_code[rws] = c_
        df = pd.DataFrame({"t_idx": pd_, "i_idx": pi_, "refit_year": Y, "yhat_dl": yhat_dl, "yhat_lstm": yhat_lstm,
                           "yhat_har": yhat_har, "yhat_garch": yhat_garch,
                           "garch_code": garch_code})
        for lv in levels:
            df[f"q_cqr_{int(round(lv * 100))}"] = q_cqr[lv]
        rows.append(df)
        logger.info(f"{panel.name} h={h} refit {Y}: pool {len(d_po):,}; predicted {len(pd_):,} asset-dates "
                    f"(wd={wd}, k={k}, epochs={ep})")
    pred = pd.concat(rows, ignore_index=True)
    t, i = pred["t_idx"].to_numpy(), pred["i_idx"].to_numpy()
    pred["yhat_naive"] = F["naive"][t, i]
    pred["yhat_ewma"] = F["ewma"][t, i]
    bound = fc["garch"]["max_log_ratio"]
    far = pred["yhat_garch"].notna() & ((pred["yhat_garch"] - pred["yhat_naive"]).abs() > bound)
    pred.loc[far, "garch_code"] = 4
    bad = pred["yhat_garch"].isna() | far
    pred["garch_fallback"] = bad
    pred.loc[bad, "yhat_garch"] = pred.loc[bad, "yhat_ewma"]
    pred["y"] = y[t, i]
    pred["vov"] = F["vov"][t, i]
    pred["vov"] = pred["vov"].fillna(pred.groupby("t_idx")["vov"].transform("median")).fillna(0.2)
    pred["end_idx"] = end_idx[t]
    pred["date"] = panel.dates[t]
    pred["asset"] = np.array(panel.assets)[i]
    pred["aclass"] = np.array(panel.aclass)[i]
    pred["split"] = np.where(pred["date"] <= dev_end_ts, "dev", "holdout")
    pred["regime"] = panel.regime[t] if panel.regime is not None else np.nan
    pred["h"] = h
    return pred.sort_values(["t_idx", "i_idx"]).reset_index(drop=True), pd.DataFrame(hp)
