"""Leakage, pre-registration and correctness tests. Run with `python -m pytest -q tests` (notebook 00 does this).

Each test targets one claim a referee could challenge:
  * targets use only days t+1..t+h; features use only days <= t;
  * a panel truncated at dev_end gives bit-identical features to the full panel (holdout data cannot leak back);
  * the walk-forward training pool contains only targets that closed before the refit date (purge + embargo);
  * conformal bounds at t depend only on targets that closed by t;
  * the vol-of-vol normalised conformal bound is valid on exchangeable data and fixes per-asset miscalibration
    that pooled split conformal leaves in place;
  * volatility-managed exposures at t use no information after t;
  * holdout runs are refused without sign-off, and a changed configuration is detected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calvol.calib import upper_bounds  # noqa: E402
from calvol.forecast import _splits  # noqa: E402
from calvol.panel import VolPanel, build_factor_panel, features, signal_days, target  # noqa: E402
from calvol.stats import holm, kupiec_christoffersen, lw_variance_test, qlike  # noqa: E402
from calvol.utils import PreRegistrationError, check_gate, load_config  # noqa: E402
from calvol.vm import build_strategies, monthly_frame  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT, smoke=True)


def _panel(T=1500, N=4, seed=0, start="2000-01-03"):
    rng = np.random.default_rng(seed)
    lv = np.zeros((T, N))
    for t in range(1, T):
        lv[t] = 0.98 * lv[t - 1] + 0.15 * rng.standard_normal(N)
    r = 0.01 * np.exp(lv) * rng.standard_normal((T, N))
    dates = pd.bdate_range(start, periods=T)
    return VolPanel("t", dates, [f"A{i}" for i in range(N)], ["x"] * N, r, np.isfinite(r), None, {})


# ----------------------------------------------------------------------------- targets and features


def test_target_uses_only_future_window(cfg):
    p = _panel()
    h, t = 21, 700
    y, end = target(p, h, cfg)
    assert end[t] == t + h
    expect = 0.5 * np.log(np.sum(p.ret[t + 1:t + h + 1, 0] ** 2) / h)
    assert abs(y[t, 0] - expect) < 1e-5
    q = _panel()
    q.ret[t, :] *= 50          # day t itself (known at signal time) must not enter the target
    q.ret[t + h + 1:, :] *= 50  # days after the window must not enter either
    y2, _ = target(q, h, cfg)
    assert abs(y2[t, 0] - y[t, 0]) < 1e-6


def test_features_are_causal(cfg):
    p = _panel()
    t = 900
    f1 = features(p, cfg)
    q = _panel()
    q.ret[t + 1:] = q.ret[t + 1:] * 10 + 0.05
    f2 = features(q, cfg)
    for k in f1:
        np.testing.assert_allclose(f1[k][: t + 1], f2[k][: t + 1], equal_nan=True, err_msg=k)


def test_truncated_panel_identical_to_full(cfg):
    p = _panel(T=2000)
    long = pd.DataFrame({"date": np.repeat(p.dates, p.N), "factor": np.tile(p.assets, p.T),
                         "ret": p.ret.ravel()})
    cut = str(p.dates[1200].date())
    full = build_factor_panel(long, p.assets, "1990-01-01", "2100-01-01", "f")
    trunc = build_factor_panel(long, p.assets, "1990-01-01", cut, "f")
    ff, ft = features(full, cfg), features(trunc, cfg)
    for k in ff:
        np.testing.assert_array_equal(ff[k][: trunc.T], ft[k], err_msg=k)
    assert trunc.dates[-1] <= pd.Timestamp(cut)


def test_signal_days_month_ends():
    p = _panel(T=600)
    s = signal_days(p, 21)
    d = p.dates[s]
    nxt = p.dates[np.minimum(s + 1, p.T - 1)]
    assert np.all((nxt.month != d.month) | (s == p.T - 1))


# ----------------------------------------------------------------------------- walk-forward purge


def test_walk_forward_purge(cfg):
    p = _panel(T=2500)
    h = 21
    _, end = target(p, h, cfg)
    cutoff = 1800
    pool, inner_tr, inner_val = _splits(p, end, cfg["forecast"], cutoff)
    assert end[pool].max() <= cutoff                       # every training target closed by the refit date
    vs = np.where(inner_val)[0].min()
    assert end[inner_tr].max() < vs - cfg["forecast"]["embargo_days"]   # embargo before inner validation
    assert not np.any(inner_tr & inner_val)


# ----------------------------------------------------------------------------- conformal


def _pred(D=240, N=30, h=1, seed=1, hetero=False, vov_informative=True, bias=False):
    """Exchangeable residuals. hetero=True: each asset has its own residual scale; vov_informative controls whether
    the vov column equals that scale (True) or is uninformative (constant). bias=True adds asset-specific offsets."""
    rng = np.random.default_rng(seed)
    sc = np.linspace(0.1, 0.6, N) if hetero else np.full(N, 0.3)
    mu = np.linspace(-0.2, 0.2, N) if bias else np.zeros(N)
    vv = sc if vov_informative else np.full(N, 0.3)
    rows = []
    for d in range(D):
        e = mu + rng.standard_normal(N) * sc
        for i in range(N):
            rows.append((d, i, f"A{i}", "c" if i < N // 2 else "d", e[i], 0.0, vv[i], d + h))
    return pd.DataFrame(rows, columns=["t_idx", "i_idx", "asset", "aclass", "y", "yhat_dl", "vov", "end_idx"])


def _kupiec_share(pred, method, level=0.9, **kw):
    ub, _ = upper_bounds(pred, "yhat_dl", level, method, 0.01, 60, 12, **kw)
    d = pred.assign(c=pred["y"] <= ub)[np.isfinite(ub)]
    rej = [kupiec_christoffersen((~g.sort_values("t_idx")["c"]).to_numpy(), 1 - level)[0]
           for _, g in d.groupby("asset")]
    return float(np.mean(rej))


def test_conformal_uses_only_closed_targets():
    h = 5
    pred = _pred(D=200, N=10, h=h)
    ub, _ = upper_bounds(pred, "yhat_dl", 0.9, "ours", 0.01, 60, 12)
    t = 120
    q = pred.copy()
    q.loc[q["end_idx"] > t, "y"] += 100.0               # change every target not yet closed at t
    ub2, _ = upper_bounds(q, "yhat_dl", 0.9, "ours", 0.01, 60, 12)
    m = (pred["t_idx"] <= t).to_numpy()
    np.testing.assert_allclose(ub[m], ub2[m], equal_nan=True)
    assert np.isfinite(ub[m]).any()


@pytest.mark.parametrize("method", ["ours", "ours_vov", "ours_sd", "pooled_split", "per_asset", "mondrian",
                                    "pooled_aci"])
def test_marginal_coverage(method):
    pred = _pred(D=400, N=30)
    ub, _ = upper_bounds(pred, "yhat_dl", 0.9, method, 0.01, 60, 12, 30, 12)
    ok = np.isfinite(ub)
    cov = np.mean(pred["y"].to_numpy()[ok] <= ub[ok])
    assert abs(cov - 0.9) < 0.02, cov


def test_vov_normalisation_when_vov_tracks_error_scale():
    pred = _pred(D=400, N=30, hetero=True)
    assert _kupiec_share(pred, "ours_vov") <= 0.15 and _kupiec_share(pred, "pooled_split") >= 0.5


def test_asset_standardised_conformal_fixes_scale_and_bias():
    """Amendment 1: the asset's own residual mean and sd fix per-asset miscalibration even when vol-of-vol carries
    no information about the error scale (the situation found at short horizons in the development data)."""
    pred = _pred(D=400, N=30, hetero=True, vov_informative=False, bias=True)
    ours = _kupiec_share(pred, "ours", scale_window=60, scale_min=12)
    assert ours <= 0.15, ours
    assert _kupiec_share(pred, "ours_vov") >= 0.5
    assert _kupiec_share(pred, "pooled_split") >= 0.5


@pytest.mark.parametrize("kind", ["upper", "sd"])
def test_residual_scale_uses_only_closed_targets(kind):
    from calvol.calib import residual_scale
    h = 5
    pred = _pred(D=200, N=6, h=h, hetero=True)
    loc, sc = residual_scale(pred, "yhat_dl", 60, 12, kind=kind)
    t = 120
    q = pred.copy()
    q.loc[q["end_idx"] > t, "y"] += 50.0
    loc2, sc2 = residual_scale(q, "yhat_dl", 60, 12, kind=kind)
    m = (pred["t_idx"] <= t).to_numpy()
    np.testing.assert_allclose(loc[m], loc2[m])
    np.testing.assert_allclose(sc[m], sc2[m])
    # at row t the statistics are those of that asset's last 60 residuals closed by t
    a = pred[(pred["i_idx"] == 3)]
    r = a[a["end_idx"] <= t]["y"].to_numpy()[-60:]
    k = a.index[a["t_idx"] == t][0]
    if kind == "sd":
        exp_loc, exp_sc = r.mean(), r.std(ddof=1)
    else:
        exp_loc, exp_sc = np.quantile(r, 0.5), np.quantile(r, 0.9) - np.quantile(r, 0.5)
    assert abs(sc[k] - exp_sc) < 1e-9 and abs(loc[k] - exp_loc) < 1e-9


def test_upper_spread_handles_daily_error_shapes():
    """Amendment 2: daily log-vol errors differ in shape across assets (fat tails, days with zero returns giving a
    long left tail). Standardising by the sd mis-calibrates assets; the upper spread Q90 - Q50 does not."""
    rng = np.random.default_rng(0)
    D, N = 900, 20
    dfs = np.r_[np.full(10, 30.0), np.linspace(2.5, 5, 10)]
    zero = np.r_[np.zeros(15), np.full(5, 0.1)]
    rows = []
    for d in range(D):
        r = np.array([rng.standard_t(v) for v in dfs])
        r[rng.random(N) < zero] = 0.0
        y = 0.5 * np.log(np.maximum(r ** 2, 1e-10))
        for i in range(N):
            rows.append((d, i, f"A{i}", "c", y[i], 0.0, 0.3, d + 1))
    pred = pd.DataFrame(rows, columns=["t_idx", "i_idx", "asset", "aclass", "y", "yhat_dl", "vov", "end_idx"])
    spread = {}
    for m in ["ours", "ours_sd"]:
        ub, _ = upper_bounds(pred, "yhat_dl", 0.9, m, 0.01, 252, 84, scale_window=252, scale_min=60)
        d = pred.assign(c=pred["y"] <= ub)[np.isfinite(ub)]
        spread[m] = d.groupby("asset")["c"].mean().std()
    assert spread["ours"] < 0.01 < spread["ours_sd"], spread


def test_har_daily_lag_ignores_non_trading_days(cfg):
    p = _panel(T=400, N=2)
    p.ret[200, 0] = 0.0          # holiday: forward-filled price, zero return
    p.obs[200, 0] = False
    f = features(p, cfg)
    assert f["har1"][200, 0] == f["har1"][199, 0]
    assert f["har1"][200, 0] > -8


# ----------------------------------------------------------------------------- volatility management


def test_vm_exposure_has_no_lookahead(cfg):
    p = _panel(T=2600, N=3)
    mon = monthly_frame(p)
    rng = np.random.default_rng(0)
    pred = mon[["t_idx", "asset"]].copy()
    for c in ["yhat_har", "yhat_dl", "ub_dl_ours_90", "ub_har_ours_90"]:
        pred[c] = 0.5 * np.log(mon["rv_prev"].to_numpy() + 1e-6) + 0.1 * rng.standard_normal(len(mon))
    pred["split"] = "dev"
    vc = dict(cfg["vm"], min_history_months=12)
    st = build_strategies(mon, pred, 0.9, vc)
    tcut = int(np.sort(mon["t_idx"].unique())[60])
    pred2 = pred.copy()
    late = pred2["t_idx"] > tcut
    pred2.loc[late, ["yhat_dl", "ub_dl_ours_90"]] += 3.0     # change every later signal
    mon2 = mon.copy()
    mon2.loc[mon2["t_idx"] > tcut, "f_next"] *= -5           # and every later return
    st2 = build_strategies(mon2, pred2, 0.9, vc)
    a = st[st["t_idx"] <= tcut].sort_values(["asset", "rule", "strategy", "t_idx"])["w"].to_numpy()
    b = st2[st2["t_idx"] <= tcut].sort_values(["asset", "rule", "strategy", "t_idx"])["w"].to_numpy()
    np.testing.assert_allclose(a, b)
    live = st[st["live"] & (st["strategy"] == "CVM") & (st["rule"] == "var")]
    assert 0.6 < live["w"].mean() < 1.4                      # real-time scaling keeps exposure near 1
    assert live["w"].max() <= vc["w_max"] + 1e-12


def test_monthly_frame_holding_period():
    p = _panel(T=400, N=1)
    mon = monthly_frame(p)
    sig = signal_days(p, 21)
    t, t1 = sig[3], sig[4]
    row = mon[mon["t_idx"] == t].iloc[0]
    assert abs(row["f_next"] - (np.prod(1 + p.ret[t + 1:t1 + 1, 0]) - 1)) < 1e-12


# ----------------------------------------------------------------------------- pre-registration gate


def test_holdout_refused_without_signoff():
    c = load_config(ROOT, smoke=False, overrides={"project": {"mode": "holdout", "signoff": False}})
    with pytest.raises(PreRegistrationError):
        check_gate(c)
    c = load_config(ROOT, smoke=False, overrides={"project": {"mode": "holdout", "signoff": True,
                                                               "frozen_commit": ""}})
    with pytest.raises(PreRegistrationError):
        check_gate(c)
    c = load_config(ROOT, smoke=False, overrides={"project": {"mode": "holdout", "signoff": True,
                                                               "frozen_commit": "abc1234"}})
    assert check_gate(c) == "holdout"
    c = load_config(ROOT, smoke=False, overrides={"project": {"mode": "test"}})
    with pytest.raises(PreRegistrationError):
        check_gate(c)


def test_design_hash_detects_config_change():
    from calvol.pipeline import design_hash
    a = load_config(ROOT, smoke=False)
    b = load_config(ROOT, smoke=False, overrides={"project": {"mode": "holdout", "signoff": True,
                                                               "frozen_commit": "x"}})
    c = load_config(ROOT, smoke=False, overrides={"conformal": {"vov_window": 126}})
    assert design_hash(a) == design_hash(b)
    assert design_hash(a) != design_hash(c)


# ----------------------------------------------------------------------------- statistics


def test_stats_sanity():
    assert qlike(np.array([0.3]), np.array([0.3]))[0] == pytest.approx(0.0)
    assert qlike(np.array([0.3]), np.array([0.1]))[0] > 0
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    miss = rng.random(2000) < 0.1
    k, _ = kupiec_christoffersen(miss, 0.1)
    assert not k
    k, _ = kupiec_christoffersen(rng.random(2000) < 0.2, 0.1)
    assert k
    r1, r2 = 0.5 * rng.standard_normal(400), rng.standard_normal(400)
    assert lw_variance_test(r1, r2, reps=300)["p_lower"] < 0.01
