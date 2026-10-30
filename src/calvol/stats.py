"""Statistical tests (design document, Section 8)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================ losses and forecast tests


def qlike(y, yhat):
    """QLIKE on variance written with log-vol inputs: RV^2/s^2 - ln(RV^2/s^2) - 1 (Patton, 2011)."""
    e = np.clip(2.0 * (np.asarray(y) - np.asarray(yhat)), -30, 30)
    return np.exp(e) - e - 1.0


def diebold_mariano(l1, l2, lags: int = 3):
    """DM with HAC variance and HLN correction (horizon 1 on non-overlapping signal dates). Positive statistic:
    model 2 has lower loss."""
    d = np.asarray(l1) - np.asarray(l2)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    m = d.mean()
    x = d - m
    v = x @ x / T
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / T
    if v <= 0:
        return np.nan, np.nan
    st = m / math.sqrt(v / T) * math.sqrt((T - 1) / T)
    return float(st), float(2 * (1 - stats.t.cdf(abs(st), T - 1)))


def model_confidence_set(losses: pd.DataFrame, size: float = 0.10, reps: int = 2000, seed: int = 0) -> pd.Series:
    """Hansen-Lunde-Nason MCS on date-averaged losses; returns the MCS p-value per model."""
    from arch.bootstrap import MCS
    L = losses.dropna()
    b = block_len(L.mean(1).values)
    mcs = MCS(L.values, size=size, reps=reps, block_size=b, method="R", bootstrap="stationary", seed=seed)
    mcs.compute()
    pv = mcs.pvalues.copy()
    pv.index = [L.columns[int(i)] for i in pv.index]
    return pv.iloc[:, 0].reindex(L.columns)


def block_len(x) -> int:
    try:
        from arch.bootstrap import optimal_block_length
        b = optimal_block_length(np.asarray(x))["circular"].max()
        return int(np.clip(round(b), 2, max(2, len(x) // 8)))
    except Exception:  # noqa: BLE001
        return max(2, int(round(len(x) ** (1 / 3))))


# ============================================================================ calibration tests


def kupiec_christoffersen(miss: np.ndarray, alpha: float):
    """Returns (kupiec_reject, christoffersen_reject) at 5% for one asset's miss series."""
    x = np.asarray(miss, int)
    n, k = len(x), x.sum()
    ph = k / n
    ll0 = (n - k) * np.log(1 - alpha) + k * np.log(alpha)
    ll1 = ((n - k) * np.log(max(1 - ph, 1e-12)) + k * np.log(max(ph, 1e-12))) if 0 < k < n else 0.0
    kup = 2 * (ll1 - ll0) > stats.chi2.ppf(0.95, 1)
    n00 = np.sum((x[:-1] == 0) & (x[1:] == 0))
    n01 = np.sum((x[:-1] == 0) & (x[1:] == 1))
    n10 = np.sum((x[:-1] == 1) & (x[1:] == 0))
    n11 = np.sum((x[:-1] == 1) & (x[1:] == 1))
    p01, p11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(c, pr):
        return c * np.log(pr) if (pr > 0 and c > 0) else 0.0
    lr = 2 * ((ll(n00, 1 - p01) + ll(n01, p01) + ll(n10, 1 - p11) + ll(n11, p11))
              - (ll(n00 + n10, 1 - p) + ll(n01 + n11, p)))
    return bool(kup), bool(lr > stats.chi2.ppf(0.95, 1))


def clustered_coverage(covered: np.ndarray, date_ids: np.ndarray, level: float) -> dict:
    """H0: coverage = level; standard errors clustered by date."""
    s = pd.Series(covered.astype(float)).groupby(date_ids).mean()
    se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
    z = (covered.mean() - level) / se if se and se > 0 else np.nan
    return {"coverage": float(covered.mean()), "z": float(z),
            "p": float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan}


def two_proportion_less(k1, n1, k2, n2) -> dict:
    """One-sided test H1: p1 < p2 (rejection share of method 1 lower than method 2) + Fisher exact check."""
    p1, p2 = k1 / max(n1, 1), k2 / max(n2, 1)
    p = (k1 + k2) / max(n1 + n2, 1)
    se = math.sqrt(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1))) if 0 < p < 1 else np.nan
    z = (p1 - p2) / se if se and se > 0 else np.nan
    fisher = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="less")[1]
    return {"share_1": p1, "share_2": p2, "z": z, "p_z": float(stats.norm.cdf(z)) if np.isfinite(z) else np.nan,
            "p_fisher": float(fisher)}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


# ============================================================================ portfolio risk tests


def max_drawdown(r) -> float:
    w = np.cumprod(1 + np.asarray(r))
    return float(np.max(1 - w / np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]))


def cvar(r, q: float = 0.05) -> float:
    r = np.asarray(r)
    return float(r[r <= np.quantile(r, q)].mean())


def sharpe(r, ppy: int = 12) -> float:
    r = np.asarray(r)
    return float(r.mean() / r.std(ddof=1) * math.sqrt(ppy)) if r.std(ddof=1) > 0 else np.nan


def _boot_idx(T, b, rng):
    l = int(math.ceil(T / b))
    starts = rng.integers(0, T, l)
    return ((starts[:, None] + np.arange(b)[None, :]) % T).ravel()[:T]


def lw_variance_test(r1, r2, reps: int = 5000, seed: int = 0) -> dict:
    """Ledoit & Wolf (2011) studentised circular block bootstrap test of H0: Var(r1) = Var(r2), on
    delta = ln Var(r1) - ln Var(r2). One-sided p (r1 has lower variance) and two-sided p."""
    Y = np.column_stack([r1, r2]).astype(float)
    T = len(Y)

    def stat(Y):
        mu = Y.mean(0)
        g = (Y ** 2).mean(0)
        v = g - mu ** 2
        d = np.log(v[0]) - np.log(v[1])
        grad = np.array([-2 * mu[0] / v[0], 2 * mu[1] / v[1], 1 / v[0], -1 / v[1]])
        return d, grad, mu, g

    def psi(Y, mu, g, lags):
        y = np.column_stack([Y[:, 0] - mu[0], Y[:, 1] - mu[1], Y[:, 0] ** 2 - g[0], Y[:, 1] ** 2 - g[1]])
        S = y.T @ y / len(y)
        for k in range(1, lags + 1):
            G = y[k:].T @ y[:-k] / len(y)
            S += (1 - k / (lags + 1)) * (G + G.T)
        return S

    d, gr, mu, g = stat(Y)
    lags = int(math.floor(4 * (T / 100) ** (2 / 9)))
    se = math.sqrt(max(gr @ psi(Y, mu, g, lags) @ gr / T, 1e-18))
    b = max(block_len(r1), block_len(r2))
    l = T // b
    rng = np.random.default_rng(seed)
    t0 = d / se
    cnt_two, cnt_less = 0, 0
    for _ in range(reps):
        starts = rng.integers(0, T, l)
        idx = ((starts[:, None] + np.arange(b)[None, :]) % T).ravel()
        Yb = Y[idx]
        db, gb, mub, ggb = stat(Yb)
        y = np.column_stack([Yb[:, 0] - mub[0], Yb[:, 1] - mub[1], Yb[:, 0] ** 2 - ggb[0], Yb[:, 1] ** 2 - ggb[1]])
        z = math.sqrt(b) * y.reshape(l, b, 4).mean(1)
        seb = math.sqrt(max(gb @ (z.T @ z / l) @ gb / (l * b), 1e-18))
        tb = (db - d) / seb
        cnt_two += abs(tb) >= abs(t0)
        cnt_less += tb <= t0
    return {"log_var_ratio": d, "vol_ratio": float(np.exp(d / 2)), "t": t0,
            "p_two_sided": (cnt_two + 1) / (reps + 1), "p_lower": (cnt_less + 1) / (reps + 1), "block": b}


def paired_bootstrap_ci(r1, r2, fn, reps: int = 5000, seed: int = 0, level: float = 0.95) -> dict:
    """Circular block bootstrap CI of fn(r1) - fn(r2) on paired series."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    T = len(r1)
    b = max(block_len(r1), block_len(r2))
    rng = np.random.default_rng(seed)
    est = fn(r1) - fn(r2)
    draws = np.empty(reps)
    for k in range(reps):
        idx = _boot_idx(T, b, rng)
        draws[k] = fn(r1[idx]) - fn(r2[idx])
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"diff": float(est), "ci_low": float(lo), "ci_high": float(hi), "p_gt0": float(np.mean(draws >= 0))}


def mm_alpha(managed, unmanaged, lags: int = 3, ppy: int = 12) -> dict:
    """Moreira-Muir regression: managed = alpha + beta * unmanaged; HAC t-statistic."""
    import statsmodels.api as sm
    X = sm.add_constant(np.asarray(unmanaged))
    res = sm.OLS(np.asarray(managed), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha_ann": float(res.params[0] * ppy), "alpha_t": float(res.tvalues[0]), "beta": float(res.params[1]),
            "R2": float(res.rsquared)}


def _lv_parts(x1, x2):
    """delta = ln Var(x1) - ln Var(x2) and its gradient w.r.t. (mu1, mu2, E x1^2, E x2^2)."""
    mu = np.array([x1.mean(), x2.mean()])
    g = np.array([(x1 ** 2).mean(), (x2 ** 2).mean()])
    v = g - mu ** 2
    d = np.log(v[0]) - np.log(v[1])
    grad = np.array([-2 * mu[0] / v[0], 2 * mu[1] / v[1], 1 / v[0], -1 / v[1]])
    y = np.column_stack([x1 - mu[0], x2 - mu[1], x1 ** 2 - g[0], x2 ** 2 - g[1]])
    return d, grad, y


def romano_wolf_variance(R1: np.ndarray, R2: np.ndarray, reps: int = 5000, seed: int = 0) -> np.ndarray:
    """Romano & Wolf (2005) step-down adjusted one-sided p-values for H0_k: Var(R1[:,k]) >= Var(R2[:,k]) across
    K series (columns) observed on the same months. Joint circular block bootstrap (same resampled months for every
    series) of the studentised log-variance-ratio statistics, so the dependence between factors is kept."""
    R1, R2 = np.asarray(R1, float), np.asarray(R2, float)
    T, K = R1.shape
    b = max(block_len(R1[:, k] - R2[:, k]) for k in range(K))
    l = T // b
    lags = int(math.floor(4 * (T / 100) ** (2 / 9)))
    d0, s0 = np.empty(K), np.empty(K)
    for k in range(K):
        d, gr, y = _lv_parts(R1[:, k], R2[:, k])
        S = y.T @ y / T
        for j in range(1, lags + 1):
            G = y[j:].T @ y[:-j] / T
            S += (1 - j / (lags + 1)) * (G + G.T)
        d0[k], s0[k] = d, math.sqrt(max(gr @ S @ gr / T, 1e-18))
    stat = -d0 / s0                                   # large = strong evidence CVM variance is lower
    rng = np.random.default_rng(seed)
    boot = np.empty((reps, K))
    for r in range(reps):
        starts = rng.integers(0, T, l)
        idx = ((starts[:, None] + np.arange(b)[None, :]) % T).ravel()
        for k in range(K):
            d, gr, y = _lv_parts(R1[idx, k], R2[idx, k])
            z = math.sqrt(b) * y.reshape(l, b, 4).mean(1)
            se = math.sqrt(max(gr @ (z.T @ z / l) @ gr / (l * b), 1e-18))
            boot[r, k] = -(d - d0[k]) / se
    order = np.argsort(-stat)
    adj = np.empty(K)
    run = 0.0
    for j, k in enumerate(order):
        rest = order[j:]
        p = (np.sum(boot[:, rest].max(1) >= stat[k]) + 1) / (reps + 1)
        run = max(run, p)
        adj[k] = run
    return adj


def lw_sharpe_test(r1, r2) -> dict:
    """Ledoit & Wolf (2008) HAC (Parzen-free Bartlett kernel) test of equal Sharpe ratios; two-sided p. Reported
    descriptively only (design document, Section 8.3)."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    T = len(r1)
    mu = np.array([r1.mean(), r2.mean()])
    g = np.array([(r1 ** 2).mean(), (r2 ** 2).mean()])
    s1, s2 = math.sqrt(g[0] - mu[0] ** 2), math.sqrt(g[1] - mu[1] ** 2)
    diff = mu[0] / s1 - mu[1] / s2
    grad = np.array([g[0] / s1 ** 3, -g[1] / s2 ** 3, -mu[0] / (2 * s1 ** 3), mu[1] / (2 * s2 ** 3)])
    y = np.column_stack([r1 - mu[0], r2 - mu[1], r1 ** 2 - g[0], r2 ** 2 - g[1]])
    lags = int(math.floor(4 * (T / 100) ** (2 / 9)))
    S = y.T @ y / T
    for k in range(1, lags + 1):
        G = y[k:].T @ y[:-k] / T
        S += (1 - k / (lags + 1)) * (G + G.T)
    se = math.sqrt(max(grad @ S @ grad / T, 1e-18))
    z = diff / se
    return {"sharpe_diff_monthly": diff, "z": z, "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(z))))}
