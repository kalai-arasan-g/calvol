"""Stage functions called by the notebooks. Each stage reads the previous stage's files, so notebooks can be run
one at a time and resumed. The pre-registration gate (utils.check_gate) runs at the start of every stage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .calib import METHODS, choose_gamma, upper_bounds
from .data import (download_yahoo, factor_names, instrument_classes, load_french_factors, make_synthetic)
from .evaluate import BASE_COL, calibration_tables, forecast_table, ub_col
from .forecast import run_forecasts
from .panel import VolPanel, build_factor_panel, build_yahoo_panel
from .utils import (PreRegistrationError, check_gate, config_hash, get_device, get_logger, get_paths, save_fig, save_table,
                    set_plot_style, set_seed, timer, write_manifest)
from .vm import alphas, build_strategies, choose_cvm_level, h3_tests, monthly_frame, pooled, summary

FACTOR_PANELS = ("ff_us", "ff_intl")


def _ctx(cfg):
    mode = check_gate(cfg)
    P = get_paths(cfg)
    lg = get_logger("calvol", P.logs)
    set_seed(cfg["project"]["seed"])
    set_plot_style()
    return P, lg, mode


def eval_split(cfg) -> str:
    return "dev" if cfg["project"]["mode"] == "development" else "holdout"


def _cutoff(cfg, pcfg):
    return pcfg["dev_end"] if cfg["project"]["mode"] == "development" else pcfg["end"]


# ============================================================================ selections (frozen after development)


def _sel_path(P, mode):
    return (P.results if mode == "development" else P.dev_results) / "tables" / "selections.json"


def design_hash(cfg) -> str:
    """Hash of every design setting except the run switches (mode, sign-off, commit, device). The holdout run must
    use exactly the configuration that produced the development selections."""
    c = {k: v for k, v in cfg.items() if not k.startswith("_")}
    c = json.loads(json.dumps(c, default=str))
    for k in ("mode", "signoff", "frozen_commit", "device"):
        c["project"].pop(k, None)
    return config_hash(c)


def load_selections(cfg) -> dict:
    P, _, mode = _ctx(cfg)
    f = _sel_path(P, mode)
    if f.exists():
        sel = json.load(open(f))
        if mode == "holdout" and sel.get("design_hash") != design_hash(cfg):
            raise PreRegistrationError("Holdout configuration differs from the configuration used in development "
                                       f"(hash {design_hash(cfg)} vs {sel.get('design_hash')}). Revert config.yaml.")
        return sel
    if mode == "holdout":
        raise PreRegistrationError("Holdout run needs the development-period selections "
                                   f"({f}); run the development pipeline first.")
    return {"gamma": {}, "cvm_level": {}}


def save_selections(cfg, sel: dict) -> None:
    P, _, mode = _ctx(cfg)
    if mode != "development":
        raise PreRegistrationError("Selections can only be written in development mode.")
    sel["design_hash"] = design_hash(cfg)
    with open(_sel_path(P, mode), "w") as f:
        json.dump(sel, f, indent=2)


# ============================================================================ 01 data


def _region_cutoffs(cfg) -> dict:
    """Last date allowed for each French region in the current mode (US -> ff_us, others -> ff_intl)."""
    return {"US": pd.Timestamp(_cutoff(cfg, cfg["panels"]["ff_us"])),
            "INTL": pd.Timestamp(_cutoff(cfg, cfg["panels"]["ff_intl"]))}


def stage_data(cfg: dict) -> dict:
    """Download (or simulate, in smoke mode) and write mode-specific raw files. In development mode the Yahoo request
    itself stops at dev_end and the French factors are cut at each panel's dev_end before being written, so no
    holdout date is present in any file the later stages read (design document, Section 9)."""
    P, lg, mode = _ctx(cfg)
    pa = cfg["panels"]["assets"]
    cls = instrument_classes(pa, include_exploratory=True)
    tick = list(cls) + [pa["regime_symbol"]]
    if cfg["project"]["smoke_test"]:
        make_synthetic(cfg, P)
        px = pd.read_parquet(P.yahoo / "assets_ohlcv.parquet")
    else:
        with timer(lg, "Yahoo download"):
            px = download_yahoo(tick, pa["start"], _cutoff(cfg, pa), P.yahoo / f"cache_{mode}", 20, lg)
    with timer(lg, "French factors"):
        fac, rf = load_french_factors(P.french, lg)
    px = px[pd.to_datetime(px["date"]) <= pd.Timestamp(_cutoff(cfg, pa))]
    px.to_parquet(P.yahoo / f"assets_ohlcv_{mode}.parquet", index=False)
    cut = _region_cutoffs(cfg)
    reg = np.where(fac["factor"].str.startswith("US_"), "US", "INTL")
    lim = np.where(reg == "US", cut["US"], cut["INTL"])
    fac = fac[pd.to_datetime(fac["date"]).to_numpy() <= lim.astype("datetime64[ns]")]
    fac.to_parquet(P.french / f"factors_{mode}.parquet", index=False)
    got = px.groupby("ticker")["date"].agg(["min", "max", "count"]).reset_index()
    got["class"] = got["ticker"].map(cls).fillna("regime")
    missing = sorted(set(tick) - set(got["ticker"]))
    got.to_csv(P.tables / "download_report.csv", index=False)
    fr = fac.groupby("factor")["date"].agg(["min", "max", "count"]).reset_index()
    fr.to_csv(P.tables / "french_factors_report.csv", index=False)
    info = {"mode": mode, "yahoo_instruments": int(len(got)), "yahoo_missing": missing,
            "yahoo_last_date": str(pd.to_datetime(px["date"]).max().date()),
            "french_factors": sorted(fac["factor"].unique().tolist()),
            "french_last_date": {k: str(v.date()) for k, v in cut.items()}}
    with open(P.raw / f"data_info_{mode}.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    write_manifest(cfg, P, "01_data", info)
    return info


# ============================================================================ 02 panels


def stage_panels(cfg: dict) -> dict:
    """Build panels truncated at dev_end (development mode) or end (holdout mode)."""
    P, lg, mode = _ctx(cfg)
    out = {}
    pa = cfg["panels"]["assets"]
    px = pd.read_parquet(P.yahoo / f"assets_ohlcv_{mode}.parquet")
    cut = _cutoff(cfg, pa)
    panel, outl = build_yahoo_panel(px, instrument_classes(pa), pa["start"], cut, cfg["cleaning"], pa["regime_symbol"])
    panel.save(P.processed / "assets.npz")
    outl.to_csv(P.tables / "assets_outliers.csv", index=False)
    summ = pd.DataFrame({"asset": panel.assets, "class": panel.aclass,
                         "first": [str(panel.dates[i].date()) if i < panel.T else "" for i in panel.first_obs],
                         "obs_days": panel.obs.sum(0)})
    summ["missing_share_after_start"] = 1 - summ["obs_days"] / (panel.T - panel.first_obs)
    save_table(summ.set_index("asset"), P, "assets_instruments", caption="Instruments in the calibration panel.",
               float_fmt="{:.3f}")
    out["assets"] = {"T": panel.T, "N": panel.N, "cutoff": cut}
    vix = None
    reg = px[px["ticker"] == pa["regime_symbol"]]
    if len(reg):
        vix = np.log(reg.assign(date=pd.to_datetime(reg["date"])).set_index("date")["close"].sort_index())
        vix = vix[~vix.index.duplicated()]
    fac = pd.read_parquet(P.french / f"factors_{mode}.parquet")
    for name in FACTOR_PANELS:
        pc = cfg["panels"][name]
        fp = build_factor_panel(fac, factor_names(pc["region"], fac), pc["start"], _cutoff(cfg, pc), name, vix)
        fp.save(P.processed / f"{name}.npz")
        out[name] = {"T": fp.T, "N": fp.N, "factors": fp.assets, "cutoff": _cutoff(cfg, pc)}
    write_manifest(cfg, P, "02_panels", out)
    lg.info(json.dumps(out, default=str))
    return out


# ============================================================================ 03 forecasts


def stage_forecasts(cfg: dict, panel_name: str, h: int) -> pd.DataFrame:
    P, lg, mode = _ctx(cfg)
    f = P.preds / f"{panel_name}_h{h}.parquet"
    if f.exists():
        lg.info(f"{f.name} exists - loading (delete to recompute)")
        return pd.read_parquet(f)
    pcfg = cfg["panels"][panel_name]
    panel = VolPanel.load(P.processed / f"{panel_name}.npz")
    with timer(lg, f"forecasts {panel_name} h={h}"):
        pred, hp = run_forecasts(panel, h, cfg, pcfg, pcfg["dev_end"], get_device(cfg), lg)
    pred.to_parquet(f, index=False)
    hp.to_csv(P.tables / f"{panel_name}_h{h}_hyperparams.csv", index=False)
    write_manifest(cfg, P, f"03_forecasts_{panel_name}_h{h}", {"rows": len(pred)})
    return pred


def forecast_tables(cfg: dict, panel_name: str, h: int) -> pd.DataFrame:
    P, lg, mode = _ctx(cfg)
    pred = pd.read_parquet(P.preds / f"{panel_name}_h{h}.parquet")
    d = pred[pred["split"] == eval_split(cfg)]
    models = {"DLinear-QLIKE": "yhat_dl", "HAR": "yhat_har", "GARCH(1,1)-t": "yhat_garch", "EWMA": "yhat_ewma",
              "LSTM": "yhat_lstm", "Naive": "yhat_naive"}
    t = forecast_table(d, models, cfg["evaluation"]["mcs_size"], cfg["evaluation"]["mcs_reps"],
                       cfg["project"]["seed"])
    save_table(t, P, f"{panel_name}_h{h}_forecasts", caption=f"Volatility forecast accuracy, {panel_name}, h={h} "
               f"({eval_split(cfg)} period). DM: positive = the row model has lower QLIKE than the reference.")
    gf = d.groupby(d["date"].dt.year)["garch_fallback"].mean().to_frame("garch_fallback_share")
    if "garch_code" in d:
        from .forecast import GARCH_CODES
        codes = pd.crosstab(d["date"].dt.year, d["garch_code"].map(GARCH_CODES), normalize="index")
        gf = gf.join(codes)
    gf.to_csv(P.tables / f"{panel_name}_h{h}_garch_fallback.csv")
    return t


# ============================================================================ 04 conformal bounds


def _K(cfg, h):
    K = int(cfg["conformal"]["calib_dates"][h])
    return K, max(12, K // 3)


def _scale(cfg, h):
    c = cfg["conformal"]
    return {"scale_window": int(c["scale_window"][h]), "scale_min": int(c["scale_min"][h])}


def stage_bounds(cfg: dict, panel_name: str, h: int) -> pd.DataFrame:
    """Upper bounds for every base forecaster, method and level. gamma per level is chosen on the development
    period (development mode) and reused unchanged in the holdout."""
    P, lg, mode = _ctx(cfg)
    out = P.bounds / f"{panel_name}_h{h}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    cc = cfg["conformal"]
    pred = pd.read_parquet(P.preds / f"{panel_name}_h{h}.parquet").sort_values(["t_idx", "i_idx"]).reset_index(drop=True)
    sel = load_selections(cfg)
    K, mc = _K(cfg, h)
    levels = list(cc["levels"])
    if panel_name in FACTOR_PANELS:
        levels = sorted(set(levels) | set(cfg["vm"]["cvm_levels"]))
    gtabs = []
    for lv in levels:
        key = f"{panel_name}_h{h}_{int(round(lv * 100))}"
        if key not in sel["gamma"]:
            if mode == "holdout":
                raise PreRegistrationError(f"gamma for {key} missing from development selections")
            g, tab = choose_gamma(pred, "yhat_dl", lv, cc["gamma_grid"], K, mc, "dev", **_scale(cfg, h))
            tab["level"] = lv
            gtabs.append(tab)
            sel["gamma"][key] = g
        g = sel["gamma"][key]
        for b in cc["base_forecasters"]:
            # full comparison set only at the pre-registered calibration levels; CVM-only levels need "ours"
            ms = cc["methods"] if (b == "dl" and lv in cc["levels"]) else ["ours", "ours_vov", "pooled_split"]
            for m in ms:
                ub, _ = upper_bounds(pred, BASE_COL[b], lv, m, g, K, mc, cc["min_group_scores"], cc["min_asset_scores"],
                                     **_scale(cfg, h))
                pred[ub_col(b, m, lv)] = ub.astype(np.float32)
    if mode == "development":
        save_selections(cfg, sel)
        if gtabs:
            save_table(pd.concat(gtabs).set_index(["level", "gamma"]), P, f"{panel_name}_h{h}_gamma_selection",
                       caption="ACI step size chosen on the development period (pinball loss of the upper bound).",
                       float_fmt="{:.5f}")
    keep = [c for c in pred.columns if c.startswith("ub_")] + ["t_idx", "i_idx", "date", "asset", "aclass", "split",
                                                              "y", "vov", "regime", "end_idx"] + list(BASE_COL.values())
    pred[keep].to_parquet(out, index=False)
    write_manifest(cfg, P, f"04_bounds_{panel_name}_h{h}", {"levels": levels})
    return pred[keep]


# ============================================================================ 05 calibration evaluation


def stage_calibration(cfg: dict, panel_name: str, h: int) -> dict:
    P, lg, mode = _ctx(cfg)
    cc = cfg["conformal"]
    df = pd.read_parquet(P.bounds / f"{panel_name}_h{h}.parquet")
    df = df[df["split"] == eval_split(cfg)]
    res = calibration_tables(df, h, cc["levels"], cc["methods"], cc["base_forecasters"], cfg["evaluation"]["primary_base"])
    tag = f"{panel_name}_h{h}"
    per = eval_split(cfg)
    save_table(res["all"], P, f"{tag}_calibration_all", caption=f"Calibration of upper volatility bounds, {tag}, "
               f"{per} period: coverage, date-clustered test, share of assets failing Kupiec / Christoffersen at 5%, "
               "5th percentile of per-asset coverage, sharpness ln(sigma_up / sigma_hat), pinball loss.",
               float_fmt="{:.4f}")
    if len(res["h1"]):
        save_table(res["h1"].set_index(["base", "comparator", "level"]), P, f"{tag}_H1", caption="H1: share of assets "
                   "failing Kupiec, ours vs comparator (one-sided two-proportion z-test and Fisher exact test; Holm "
                   "across levels for the primary comparison).", float_fmt="{:.4f}")
    if len(res["h2"]):
        save_table(res["h2"].set_index("level"), P, f"{tag}_H2", caption="H2: overall coverage of our bounds, "
                   "date-clustered test (Holm across levels) and largest yearly deviation.", float_fmt="{:.4f}")
    if len(res["pinball_dm"]):
        save_table(res["pinball_dm"].set_index(["base", "comparator", "level"]), P, f"{tag}_pinball_dm",
                   caption="Diebold-Mariano on date-averaged pinball loss; positive = ours lower loss.",
                   float_fmt="{:.4f}")
    if len(res["by_group"]):
        res["by_group"].to_csv(P.tables / f"{tag}_coverage_by_group.csv", index=False)
    if len(res["all"]):
        save_fig(plots.kupiec_bars(res["all"], "dl", f"Asset-level calibration, {panel_name}, h={h}"), P,
                 f"fig_{tag}_kupiec")
        lv = 0.9 if 0.9 in cc["levels"] else cc["levels"][0]
        cols = {plots.LABEL["ours"]: ub_col("dl", "ours", lv), plots.LABEL["ours_vov"]: ub_col("dl", "ours_vov", lv),
                plots.LABEL["pooled_split"]: ub_col("dl", "pooled_split", lv),
                plots.LABEL["per_asset"]: ub_col("dl", "per_asset", lv), plots.LABEL["cqr"]: ub_col("dl", "cqr", lv)}
        cols = {k: v for k, v in cols.items() if v in df}
        save_fig(plots.per_asset_coverage(df, cols, lv, f"Per-asset coverage of {int(lv * 100)}% bounds, {panel_name}"),
                 P, f"fig_{tag}_per_asset_coverage")
        save_fig(plots.coverage_by_year(df, dict(list(cols.items())[:3]), lv,
                                        f"Coverage by year, {int(lv * 100)}% bounds, {panel_name}"), P,
                 f"fig_{tag}_coverage_by_year")
    write_manifest(cfg, P, f"05_calibration_{tag}")
    return res


# ============================================================================ 06 volatility management


def stage_vm(cfg: dict, panel_name: str) -> dict:
    P, lg, mode = _ctx(cfg)
    vc = cfg["vm"]
    panel = VolPanel.load(P.processed / f"{panel_name}.npz")
    pred = pd.read_parquet(P.bounds / f"{panel_name}_h21.parquet")
    mon = monthly_frame(panel, vc["implementation_lag_days"])
    sel = load_selections(cfg)
    if panel_name not in sel["cvm_level"]:
        if mode == "holdout":
            raise PreRegistrationError(f"CVM level for {panel_name} missing from development selections")
        lv, tab = choose_cvm_level(mon, pred, vc["cvm_levels"], vc)
        sel["cvm_level"][panel_name] = lv
        save_selections(cfg, sel)
        save_table(tab.set_index("level"), P, f"{panel_name}_cvm_level_selection",
                   caption="CVM level chosen on the development period (pooled Sharpe, inverse-variance rule).")
    lv = sel["cvm_level"][panel_name]
    st = build_strategies(mon, pred, lv, vc)
    st = pd.concat([st, pooled(st)], ignore_index=True)
    st.to_parquet(P.vm / f"{panel_name}_strategies.parquet", index=False)
    split = eval_split(cfg)
    S = summary(st, split)
    if not len(S):
        lg.warning(f"{panel_name}: no live months in the {split} period - VM tables skipped")
        write_manifest(cfg, P, f"06_vm_{panel_name}", {"cvm_level": lv, "note": "no live months"})
        return {"summary": S, "H3": pd.DataFrame(), "alpha": pd.DataFrame()}
    save_table(S.set_index(["rule", "asset", "strategy"]), P, f"{panel_name}_vm_summary",
               caption=f"Volatility-managed factor portfolios, {panel_name}, {split} period, after "
                       f"{vc['cost_bps']} bps per unit change in exposure; CVM level {lv}.", float_fmt="{:.4f}")
    T3 = h3_tests(st, split, cfg["evaluation"]["bootstrap_reps"], cfg["project"]["seed"])
    if len(T3):
        save_table(T3.set_index(["rule", "asset", "comparator"]), P, f"{panel_name}_H3",
                   caption="H3: CVM vs point-forecast volatility management. Volatility ratio and Ledoit-Wolf (2011) "
                           "variance test (one-sided: CVM lower); bootstrap 95% CIs of max-drawdown and CVaR differences; "
                           "Romano-Wolf step-down across factors; Ledoit-Wolf (2008) Sharpe p-values are descriptive.",
                   float_fmt="{:.4f}")
    A = alphas(st, split)
    if len(A):
        save_table(A.set_index(["rule", "asset", "strategy"]), P, f"{panel_name}_mm_alpha",
                   caption="Moreira-Muir alphas of managed on unmanaged factors (HAC t).", float_fmt="{:.4f}")
    # cost sensitivity (exposures fixed)
    rows = []
    d = st[(st["split"] == split) & st["live"] & (st["asset"] == "POOLED") & (st["rule"] == "var")]
    for cb in vc["cost_grid_bps"]:
        for s, g in d.groupby("strategy"):
            x = (g["gross"] - cb / 1e4 * g["turnover"]).values
            rows.append({"cost_bps": cb, "strategy": s, "ann_vol": x.std(ddof=1) * np.sqrt(12),
                         "sharpe": x.mean() / x.std(ddof=1) * np.sqrt(12)})
    save_table(pd.DataFrame(rows).pivot(index="cost_bps", columns="strategy", values="sharpe"), P,
               f"{panel_name}_vm_cost_sensitivity", caption="Pooled Sharpe ratio by cost per unit change in exposure.")
    # sensitivity: every pre-specified CVM level (Section 7.2), pooled panel, both rules
    sens = []
    for lv_ in vc["cvm_levels"]:
        s_ = build_strategies(mon, pred, lv_, vc)
        s_ = pooled(s_)
        S_ = summary(s_, split)
        S_ = S_[S_["strategy"].isin(["CVM", "CVM_HAR"])].assign(cvm_level=lv_)
        sens.append(S_)
    if sens:
        sens = pd.concat(sens, ignore_index=True)
        save_table(sens.set_index(["rule", "strategy", "cvm_level"]).drop(columns="asset"), P,
                   f"{panel_name}_cvm_level_sensitivity", caption=f"CVM at every pre-specified coverage level, pooled "
                   f"factors, {split} period (the level used for claims was fixed on the development period: {lv}).",
                   float_fmt="{:.4f}")
    pl = st[(st["asset"] == "POOLED") & (st["rule"] == "var") & (st["split"] == split) & st["live"]]
    if len(pl):
        save_fig(plots.wealth(pl, f"Volatility-managed factors (pooled), {panel_name}"), P, f"fig_{panel_name}_vm_wealth")
        save_fig(plots.exposure(pl, f"Average exposure, {panel_name}"), P, f"fig_{panel_name}_vm_exposure")
    write_manifest(cfg, P, f"06_vm_{panel_name}", {"cvm_level": lv})
    return {"summary": S, "H3": T3, "alpha": A}


# ============================================================================ exploratory: conformal VaR (Section 6.3)


def stage_conformal_var(cfg: dict, panel_name: str = "assets", h: int = 21) -> pd.DataFrame:
    from .convar import conformal_var_table
    P, lg, mode = _ctx(cfg)
    panel = VolPanel.load(P.processed / f"{panel_name}.npz")
    pred = pd.read_parquet(P.preds / f"{panel_name}_h{h}.parquet")
    sel = load_selections(cfg)
    K, mc = _K(cfg, h)
    levels = [0.95, 0.99]
    tabs = []
    for lv in levels:
        g = sel["gamma"].get(f"{panel_name}_h{h}_{int(round(lv * 100))}", 0.01)
        t = conformal_var_table(panel, pred, h, {"dl": "yhat_dl", "har": "yhat_har"}, [lv], g, K, mc, eval_split(cfg))
        tabs.append(t)
    t = pd.concat(tabs, ignore_index=True)
    if len(t):
        save_table(t.set_index(["base", "level", "method"]), P, f"{panel_name}_h{h}_conformal_var_EXPLORATORY",
                   caption=f"Exploratory: conformal vs Gaussian VaR of {h}-day losses, {panel_name}, {eval_split(cfg)} "
                           "period: exceedance rate and share of assets failing Kupiec / Christoffersen at 5%.",
                   float_fmt="{:.4f}")
    return t


# ============================================================================ 07 hypothesis verdicts and export


def stage_summary(cfg: dict) -> pd.DataFrame:
    """Applies the pre-registered pass/fail rules of Section 3 to the saved tables, mechanically."""
    P, lg, mode = _ctx(cfg)
    T = P.tables
    per = eval_split(cfg)
    h = cfg["evaluation"]["primary_horizon"]
    rows = []

    def add(hyp, scope, stat, rule, verdict):
        rows.append({"hypothesis": hyp, "scope": scope, "statistic": stat, "rule (fails if)": rule,
                     "verdict": verdict, "period": per})

    f = T / f"assets_h{h}_H1.csv"
    if f.exists():
        h1 = pd.read_csv(f)
        h1 = h1[(h1["base"] == cfg["evaluation"]["primary_base"]) & (h1["comparator"] == "pooled_split")]
        for _, r in h1.iterrows():
            ok = (r["p_holm_primary"] < 0.05) and (r["ours_share"] <= 0.10)
            add("H1", f"assets h={h}, level {r['level']:.2f}",
                f"Kupiec share ours {100 * r['ours_share']:.1f}% vs pooled {100 * r['comparator_share']:.1f}%, "
                f"p_Holm {r['p_holm_primary']:.4f}", "share not lower at p<0.05 (Holm) or ours > 10%",
                "PASS" if ok else "FAIL")
    f = T / f"assets_h{h}_H2.csv"
    if f.exists():
        h2 = pd.read_csv(f)
        for _, r in h2.iterrows():
            bad_test = (abs(r["level"] - 0.9) < 1e-9) and (r["clustered_p"] < 0.05)
            bad_year = r["mean_year_deviation_pp"] > r.get("pooled_mean_year_deviation_pp", np.inf)
            add("H2", f"assets h={h}, level {r['level']:.2f}",
                f"coverage {100 * r['coverage']:.2f}% (clustered p {r['clustered_p']:.3f}, Holm {r['p_holm']:.3f}); "
                f"mean |yearly deviation| {r['mean_year_deviation_pp']:.1f} pp vs pooled split "
                f"{r.get('pooled_mean_year_deviation_pp', np.nan):.1f} pp (max {r['max_year_deviation_pp']:.1f} vs "
                f"{r.get('pooled_max_year_deviation_pp', np.nan):.1f})",
                "clustered test rejects at 5% (90% level) or mean yearly deviation larger than pooled split's "
                "(amendment 1)", "FAIL" if (bad_year or bad_test) else "PASS")
    for pn in FACTOR_PANELS:
        f = T / f"{pn}_H3.csv"
        if not f.exists():
            continue
        h3 = pd.read_csv(f)
        r = h3[(h3["asset"] == "POOLED") & (h3["rule"] == "var") & (h3["comparator"] == "VM_DL")]
        if len(r):
            r = r.iloc[0]
            add("H3 (secondary)", f"{pn} pooled, CVM vs VM-DL",
                f"vol ratio {r['vol_ratio']:.3f}, LW(2011) one-sided p {r['p_var_lower']:.4f}; "
                f"MDD diff {r['mdd_diff']:.3f} [{r['mdd_ci_low']:.3f}, {r['mdd_ci_high']:.3f}]",
                "variance not significantly lower at 5%", "PASS" if r["p_var_lower"] < 0.05 else "FAIL")
    f = T / f"assets_h{h}_forecasts.csv"
    if f.exists():
        fc = pd.read_csv(f).set_index("model")
        if "HAR" in fc.index and "DM_vs_DLinear-QLIKE" in fc:
            dm, p = fc.loc["HAR", "DM_vs_DLinear-QLIKE"], fc.loc["HAR", "p_vs_DLinear-QLIKE"]
            add("H4 (secondary)", f"assets h={h}", f"QLIKE DLinear {fc.loc['DLinear-QLIKE', 'QLIKE']:.4f} vs HAR "
                f"{fc.loc['HAR', 'QLIKE']:.4f}; DM {dm:.2f} (positive = HAR better), p {p:.4f}",
                "DLinear significantly worse than HAR (two-sided 5%)", "FAIL" if (dm > 0 and p < 0.05) else "PASS")
    out = pd.DataFrame(rows)
    if len(out):
        save_table(out.set_index(["hypothesis", "scope"]), P, "hypothesis_verdicts",
                   caption=f"Pre-registered hypotheses, {per} period, with the pass/fail rules fixed in Section 3.")
    if mode == "development":
        lg.warning("Development-period verdicts are for debugging only; claims are made on the holdout.")
    return out


def export_results(cfg: dict) -> Path:
    """Zip tables, figures, logs and manifests (not the large parquet files) for sharing."""
    import zipfile
    P, lg, mode = _ctx(cfg)
    z = P.results_root / f"calvol_results_{mode}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ["tables", "figures", "logs"]:
            for f in sorted((P.results / sub).rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(P.results_root))
        for f in sorted(P.results.glob("*.json")):
            zf.write(f, f.relative_to(P.results_root))
    lg.info(f"results archive: {z}")
    return z
