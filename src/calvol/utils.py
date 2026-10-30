"""Shared utilities: configuration, pre-registration gate, paths, seeding, logging, tables/figures, manifests."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ----------------------------------------------------------------------------- config


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(project_dir: str | Path, smoke: bool | None = None, overrides: dict | None = None) -> dict:
    """Load config/config.yaml; apply config/smoke.yaml if smoke mode; then explicit overrides."""
    project_dir = Path(project_dir)
    with open(project_dir / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    if smoke is None:
        env = os.environ.get("CALVOL_SMOKE")
        smoke = (env == "1") if env is not None else bool(cfg["project"].get("smoke_test", False))
    if smoke:
        with open(project_dir / "config" / "smoke.yaml") as f:
            sm = yaml.safe_load(f)
        cfg = _deep_merge(cfg, sm)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    env_mode = os.environ.get("CALVOL_MODE")
    if env_mode and not (overrides or {}).get("project", {}).get("mode"):   # explicit overrides win
        cfg["project"]["mode"] = env_mode
    cfg["project"]["smoke_test"] = bool(smoke)
    cfg["_project_dir"] = str(project_dir)
    return cfg


class PreRegistrationError(RuntimeError):
    pass


def check_gate(cfg: dict) -> str:
    """Enforce the pre-registration rules (design document, Section 9).

    development: data are truncated at each panel's dev_end; holdout dates are never loaded.
    holdout:     allowed only with signoff = true and a recorded frozen_commit, and only if the
                 development-period selections (gamma, CVM level) exist and are reused unchanged."""
    p = cfg["project"]
    mode = p["mode"]
    if mode not in ("development", "holdout"):
        raise PreRegistrationError(f"project.mode must be 'development' or 'holdout', got {mode!r}")
    if mode == "holdout" and not p.get("smoke_test"):
        if not p.get("signoff") or not str(p.get("frozen_commit", "")).strip():
            raise PreRegistrationError(
                "Holdout run refused: set project.signoff = true and record project.frozen_commit after "
                "both authors sign the design document (Section 9).")
    return mode


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:10]


@dataclass
class Paths:
    root: Path
    smoke: bool
    mode: str

    @property
    def data(self) -> Path:
        return self.root / ("data_smoke" if self.smoke else "data")

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def yahoo(self) -> Path:
        return self.raw / "yahoo"

    @property
    def french(self) -> Path:
        return self.raw / "french"

    @property
    def processed(self) -> Path:
        return self.data / "processed" / self.mode

    @property
    def results_root(self) -> Path:
        return self.root / ("results_smoke" if self.smoke else "results")

    @property
    def results(self) -> Path:
        return self.results_root / self.mode

    @property
    def dev_results(self) -> Path:
        return self.results_root / "development"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def preds(self) -> Path:
        return self.results / "predictions"

    @property
    def bounds(self) -> Path:
        return self.results / "bounds"

    @property
    def vm(self) -> Path:
        return self.results / "vm"

    @property
    def logs(self) -> Path:
        return self.results / "logs"

    def make(self) -> "Paths":
        for p in [self.yahoo, self.french, self.processed, self.tables, self.figures, self.preds, self.bounds,
                  self.vm, self.logs]:
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(cfg: dict) -> Paths:
    return Paths(Path(cfg["_project_dir"]), bool(cfg["project"]["smoke_test"]), cfg["project"]["mode"]).make()


# ----------------------------------------------------------------------------- reproducibility


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(cfg: dict):
    import torch
    want = cfg["project"].get("device", "auto")
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


@contextmanager
def timer(logger: logging.Logger, label: str):
    t0 = time.time()
    yield
    logger.info(f"{label} finished in {time.time() - t0:,.1f}s")


def write_manifest(cfg: dict, paths: Paths, stage: str, extra: dict | None = None) -> None:
    """Record versions, config hash and timestamp for every stage (reproducibility log)."""
    import importlib
    versions = {}
    for m in ["numpy", "pandas", "scipy", "torch", "arch", "statsmodels", "yfinance", "matplotlib"]:
        try:
            versions[m] = importlib.import_module(m).__version__
        except Exception:
            versions[m] = None
    man = {
        "stage": stage,
        "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "config_hash": config_hash(cfg),
        "smoke_test": cfg["project"]["smoke_test"],
        "mode": cfg["project"]["mode"],
        "signoff": cfg["project"].get("signoff"),
        "frozen_commit": cfg["project"].get("frozen_commit"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "extra": extra or {},
    }
    with open(paths.logs / f"manifest_{stage}.json", "w") as f:
        json.dump(man, f, indent=2, default=str)
    with open(paths.logs / "config_used.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)


# ----------------------------------------------------------------------------- outputs


def _latex_escape(s: str) -> str:
    rep = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    for a, b in rep.items():
        s = s.replace(a, b)
    return s


def df_to_latex(df: pd.DataFrame, caption: str = "", label: str = "", float_fmt: str = "{:.3f}",
                index: bool = True) -> str:
    """Booktabs LaTeX table without requiring jinja2."""
    d = df.copy()
    cols = list(d.columns)
    header = ([_latex_escape(str(d.index.name or ""))] if index else []) + [_latex_escape(str(c)) for c in cols]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small"]
    if caption:
        lines.append(r"\caption{" + caption + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + ("l" if index else "") + "r" * len(cols) + "}")
    lines += [r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for idx, row in d.iterrows():
        cells = [_latex_escape(str(idx))] if index else []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else float_fmt.format(v))
            else:
                cells.append(_latex_escape(str(v)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, paths: Paths, name: str, caption: str = "", float_fmt: str = "{:.3f}",
               index: bool = True) -> None:
    """Save a manuscript table as CSV (full precision) and LaTeX (formatted)."""
    df.to_csv(paths.tables / f"{name}.csv", index=index)
    with open(paths.tables / f"{name}.tex", "w") as f:
        f.write(df_to_latex(df, caption=caption, label=f"tab:{name}", float_fmt=float_fmt, index=index))


def set_plot_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "lines.linewidth": 1.2,
        "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_fig(fig, paths: Paths, name: str) -> None:
    """Save a manuscript figure as vector PDF and 300-dpi PNG."""
    fig.savefig(paths.figures / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(paths.figures / f"{name}.png", bbox_inches="tight")


# colour-blind-safe palette (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
