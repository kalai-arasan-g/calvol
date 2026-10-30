"""Manuscript figures (PDF + 300-dpi PNG via utils.save_fig)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import PALETTE

LABEL = {"ours": "Ours (asset-standardised, ACI)", "ours_mondrian": "Ours, Mondrian by class",
         "ours_sd": "Mean/sd standardised (amendment 1)", "ours_vov": "Vol-of-vol normalised (original)",
         "pooled_split": "Pooled split conformal", "pooled_aci": "Pooled ACI", "per_asset": "Per-asset conformal",
         "mondrian": "Mondrian by class", "cqr": "CQR (quantile HAR)", "parametric": "Parametric (residual sd)"}
SLABEL = {"UNMANAGED": "Unmanaged", "VM_RV": "VM-RV (Moreira-Muir)", "VM_HAR": "VM-HAR", "VM_DL": "VM-DL",
          "CVM": "CVM (ours)", "CVM_HAR": "CVM-HAR"}


def kupiec_bars(allt: pd.DataFrame, base: str, title: str) -> plt.Figure:
    d = allt.reset_index()
    d = d[d["base"] == base].pivot(index="method", columns="level", values="kupiec_share")
    order = [m for m in LABEL if m in d.index]
    d = d.loc[order]
    d.index = [LABEL[m] for m in order]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    d.columns = [f"{int(round(c * 100))}% bound" for c in d.columns]
    (100 * d).plot(kind="barh", ax=ax, color=PALETTE[: d.shape[1]], width=0.8, legend=False)
    ax.axvline(5, color="k", lw=0.8, ls=":", label="Nominal 5%")
    ax.set_xlabel("Assets failing the Kupiec test at 5% (%)")
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def per_asset_coverage(df: pd.DataFrame, cols: dict, level: float, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    data, labels = [], []
    for lab, c in cols.items():
        d = df[np.isfinite(df[c]) & df["y"].notna()]
        data.append(100 * (d["y"] <= d[c]).groupby(d["asset"]).mean().values)
        labels.append(lab)
    ax.boxplot(data, vert=False, widths=0.5, showfliers=True)
    for k, x in enumerate(data):
        ax.scatter(x, np.full(len(x), k + 1) + np.random.default_rng(k).uniform(-0.15, 0.15, len(x)),
                   s=8, color=PALETTE[k % len(PALETTE)], alpha=0.7, zorder=3)
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.axvline(100 * level, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Per-asset coverage (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def coverage_by_year(df: pd.DataFrame, cols: dict, level: float, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    for k, (lab, c) in enumerate(cols.items()):
        d = df[np.isfinite(df[c]) & df["y"].notna()]
        s = (d["y"] <= d[c]).groupby(d["date"].dt.year).mean()
        ax.plot(s.index, 100 * s.values, marker="o", ms=3, color=PALETTE[k], label=lab)
    ax.axhline(100 * level, color="k", lw=0.8, ls=":", label=f"Target {int(level * 100)}%")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


def wealth(pl: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for k, s in enumerate([s for s in SLABEL if s in set(pl["strategy"])]):
        g = pl[pl["strategy"] == s].sort_values("date")
        w = np.cumprod(1 + g["net"].values)
        dd = 1 - w / np.maximum.accumulate(w)
        lw = 1.8 if s == "CVM" else 1.0
        ax[0].plot(g["date"], w, color=PALETTE[k], lw=lw, label=SLABEL[s])
        ax[1].plot(g["date"], -100 * dd, color=PALETTE[k], lw=lw)
    ax[0].set_yscale("log")
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
    ax[0].yaxis.set_major_locator(MaxNLocator(6))
    ax[0].yaxis.set_minor_locator(NullLocator())
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[0].set_ylabel("Wealth of 1 (log scale)")
    ax[1].set_ylabel("Drawdown (%)")
    ax[0].set_title(title)
    ax[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def exposure(pl: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for k, s in enumerate(["VM_DL", "CVM"]):
        g = pl[pl["strategy"] == s].sort_values("date")
        ax.plot(g["date"], g["w"], color=PALETTE[k], lw=1.6 if s == "CVM" else 1.0, label=SLABEL[s])
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_ylabel("Average exposure")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def bars(tab: pd.DataFrame, ylabel: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * len(tab) + 2.5), 2.9))
    tab.plot(kind="bar", ax=ax, color=PALETTE[: tab.shape[1]], width=0.8, legend=tab.shape[1] > 1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    if tab.shape[1] > 1:
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig
