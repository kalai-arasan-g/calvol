"""Sensitivity of the proposed bound to its three tuning constants, varied one at a time:
  (a) the upper quantile q that defines the spread s = Q_q(E) - median(E):   q in {0.75, 0.80, 0.85, 0.90, 0.95}
  (b) the history window M of each asset's own errors:                      M/2, M, 2M   (M = 60 / 104 / 252)
  (c) the minimum number of own errors before the asset's scale is used:     m/2, m, 2m   (m = 12 / 26 / 60)

The proposed bound ('ours', DLinear base) is recomputed for every horizon and level with everything else frozen
(calibration window, ACI step sizes from the development selections). Asset-level Kupiec rejections, coverage, width and pinball loss are reported separately for the
development period and the test period. The packaged code is not modified: the spread quantile is changed by
temporarily replacing calvol.calib.residual_scale inside this script. For q = 0.90 the script checks that it
reproduces the saved bounds exactly.

Usage (Colab, after the holdout run):  !python scripts/spread_sensitivity.py
Writes results/<mode>/tables/sensitivity.csv, sensitivity_summary.csv (+ .tex) and sensitivity_check.csv.
Runtime: about 5-10 minutes for h = 21 and 5, and 30-60 minutes for h = 1 on a CPU runtime (9 settings x 4 levels).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import calvol.calib as calib  # noqa: E402
from calvol.calib import pinball  # noqa: E402
from calvol.evaluate import MIN_OBS_ASSET, ub_col  # noqa: E402
from calvol.pipeline import _K, _scale, eval_split  # noqa: E402
from calvol.stats import kupiec_christoffersen  # noqa: E402
from calvol.utils import get_paths, load_config, save_table  # noqa: E402

Q_GRID = [0.75, 0.80, 0.85, 0.90, 0.95]
_original = calib.residual_scale


def make_scale(q):
    """Same as calvol.calib.residual_scale(kind='upper') but with Q_q instead of Q_0.9."""
    def residual_scale(pred, base_col, window, min_n, floor=0.02, kind="upper"):
        if kind != "upper":
            return _original(pred, base_col, window, min_n, floor, kind)
        loc = np.full(len(pred), np.nan)
        sc = np.full(len(pred), np.nan)
        res_all = pred["y"].to_numpy(dtype=float) - pred[base_col].to_numpy(dtype=float)
        t_all = pred["t_idx"].to_numpy()
        e_all = pred["end_idx"].to_numpy()
        for _, idx in pred.groupby("i_idx").indices.items():
            idx = idx[np.argsort(t_all[idx], kind="stable")]
            t = t_all[idx]
            e = np.maximum.accumulate(e_all[idx])
            roll = pd.Series(res_all[idx]).rolling(window, min_periods=min_n)
            m_ = roll.quantile(0.5).to_numpy()
            s_ = roll.quantile(q).to_numpy() - m_
            c = np.minimum(np.searchsorted(e, t, side="right"), np.arange(len(t)))
            j = c - 1
            ok = j >= 0
            loc[idx[ok]] = m_[j[ok]]
            sc[idx[ok]] = s_[j[ok]]
        sc = np.maximum(sc, floor)
        fill = pd.Series(sc).groupby(t_all).transform("median").to_numpy()
        miss = ~np.isfinite(sc)
        sc[miss] = fill[miss]
        loc[~np.isfinite(loc)] = 0.0
        return loc, sc
    return residual_scale


def evaluate(df, ub, level, h):
    d = df.assign(ub=ub)
    d = d[np.isfinite(d["ub"]) & d["y"].notna()]
    rej = n = 0
    for _, g in d.sort_values("t_idx").groupby("asset"):
        if len(g) < MIN_OBS_ASSET.get(h, 36):
            continue
        k, _ = kupiec_christoffersen((g["y"] > g["ub"]).to_numpy(), 1 - level)
        rej += int(k)
        n += 1
    return {"kupiec_rejects": rej, "assets_tested": n, "coverage": float((d["y"] <= d["ub"]).mean()),
            "width": float((d["ub"] - d["yhat_dl"]).mean()),
            "pinball": float(np.mean(pinball(d["y"], d["ub"], level)))}


def main():
    cfg = load_config(ROOT)
    P = get_paths(cfg)
    cc = cfg["conformal"]
    sel_file = P.dev_results / "tables" / "selections.json"
    gam = json.load(open(sel_file))["gamma"]
    rows, checks = [], []
    for h in cfg["panels"]["assets"]["horizons"]:
        f = P.preds / f"assets_h{h}.parquet"
        if not f.exists():
            print("missing", f)
            continue
        pred = pd.read_parquet(f).sort_values(["t_idx", "i_idx"]).reset_index(drop=True)
        saved = None
        fb = P.bounds / f"assets_h{h}.parquet"
        if fb.exists():
            saved = pd.read_parquet(fb).sort_values(["t_idx", "i_idx"]).reset_index(drop=True)
        K, mc = _K(cfg, h)
        base = _scale(cfg, h)                       # {"scale_window": M, "scale_min": m}
        M0, m0 = base["scale_window"], base["scale_min"]
        settings = [("q", q, q, M0, m0) for q in Q_GRID]
        settings += [("M", f"{f}M", 0.90, max(int(round(f * M0)), m0), m0) for f in (0.5, 2.0)]
        settings += [("min", f"{f}m", 0.90, M0, max(int(round(f * m0)), 2)) for f in (0.5, 2.0)]
        for kind, label, q, M, mmin in settings:
            calib.residual_scale = make_scale(q)
            for lv in cc["levels"]:
                g = gam[f"assets_h{h}_{int(round(lv * 100))}"]
                ub, _ = calib.upper_bounds(pred, "yhat_dl", lv, "ours", g, K, mc, cc["min_group_scores"],
                                           cc["min_asset_scores"], scale_window=M, scale_min=mmin)
                if kind == "q" and q == 0.90 and saved is not None:
                    col = ub_col("dl", "ours", lv)
                    diff = np.nanmax(np.abs(ub - saved[col].to_numpy(dtype=float)))
                    checks.append({"h": h, "level": lv, "max_abs_diff_vs_saved": float(diff)})
                for period in ["dev", "holdout"]:
                    m = (pred["split"] == period).to_numpy()
                    if not m.any():
                        continue
                    r = evaluate(pred[m], ub[m], lv, h)
                    r.update({"h": h, "level": lv, "varied": kind, "setting": str(label), "q": q, "M": M,
                              "min_hist": mmin, "period": "development" if period == "dev" else "test"})
                    rows.append(r)
            print(f"h={h} {kind}={label} (q={q}, M={M}, min={mmin}) done", flush=True)
        calib.residual_scale = _original
    t = pd.DataFrame(rows)
    t.to_csv(P.tables / "sensitivity.csv", index=False)
    if checks:
        c = pd.DataFrame(checks)
        print("reproduction check (q = 0.90 vs saved bounds; float32 storage, expect < 1e-5):")
        print(c.to_string())
        c.to_csv(P.tables / "sensitivity_check.csv", index=False)
    s = t.groupby(["period", "h", "varied", "setting", "q", "M", "min_hist"]).agg(kupiec_rejects=("kupiec_rejects", "sum"),
                                             tests=("assets_tested", "sum"), width=("width", "mean"),
                                             pinball=("pinball", "mean")).reset_index()
    s["reject_share_pct"] = 100 * s["kupiec_rejects"] / s["tests"]
    save_table(s.set_index(["period", "h", "varied", "setting"]), P, "sensitivity_summary",
               caption="Sensitivity of the proposed bound (DLinear base) to the spread quantile q, the history window M "
                       "and the minimum history, varied one at a time: Kupiec rejections summed over the four levels, mean "
                       "width and mean pinball loss.", float_fmt="{:.4f}")
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
