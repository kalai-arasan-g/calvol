"""Data acquisition (design document, Section 4).

Yahoo instruments  -> <data>/raw/yahoo/<panel>_ohlcv.parquet   long: date, ticker, open, high, low, close, volume
French factors     -> <data>/raw/french/factors_daily.parquet  long: date, factor, ret (decimal); plus RF per region
Downloads always fetch the full history; the pre-registration gate is applied when panels are BUILT
(development mode truncates at dev_end), so a download never exposes holdout results.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "US_FF3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "US_FF5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "US_MOM": "F-F_Momentum_Factor_daily_CSV.zip",
}
INTL_REGIONS = {"DEV": "Developed", "NA": "North_America", "EU": "Europe", "JP": "Japan",
                "APXJ": "Asia_Pacific_ex_Japan"}
for _k, _v in INTL_REGIONS.items():
    FRENCH_FILES[f"{_k}_FF5"] = f"{_v}_5_Factors_Daily_CSV.zip"
    FRENCH_FILES[f"{_k}_MOM"] = f"{_v}_Mom_Factor_Daily_CSV.zip"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (academic research script)", "Accept": "*/*"}


# ============================================================================ Yahoo


def _yf_frame_to_long(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=keep)
    fields = {"Open", "High", "Low", "Close", "Volume"}
    if isinstance(df.columns, pd.MultiIndex):
        if set(map(str, df.columns.get_level_values(0))) & fields:
            df = df.swaplevel(0, 1, axis=1)
        parts = []
        for t in sorted(set(df.columns.get_level_values(0))):
            sub = df[t].copy()
            sub.columns = [str(c).lower() for c in sub.columns]
            sub["ticker"] = t
            parts.append(sub)
        out = pd.concat(parts)
    else:
        out = df.copy()
        out.columns = [str(c).lower() for c in out.columns]
        out["ticker"] = tickers[0]
    out = out.reset_index().rename(columns={"Date": "date", "index": "date", "Datetime": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep].dropna(subset=["close"])


def download_yahoo(tickers: list[str], start: str, end: str, cache_dir: Path, chunk: int = 20, logger=None,
                   retries: int = 3) -> pd.DataFrame:
    """Adjusted daily OHLCV with per-chunk caching (resumable after a Colab disconnect)."""
    import yfinance as yf
    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers = list(dict.fromkeys(tickers))
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = []
    for k in range(0, len(tickers), chunk):
        batch = tickers[k:k + chunk]
        f = cache_dir / f"chunk_{k // chunk:03d}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            continue
        long = None
        for a in range(retries):
            try:
                raw = yf.download(batch, start=start, end=end_excl, auto_adjust=True, actions=False,
                                  group_by="ticker", threads=True, progress=False)
                long = _yf_frame_to_long(raw, batch)
                break
            except Exception as e:  # noqa: BLE001
                if logger:
                    logger.warning(f"yahoo chunk {k // chunk} attempt {a + 1}: {e}")
                time.sleep(5 * (a + 1))
        long = long if long is not None else _yf_frame_to_long(None, batch)
        for t in [t for t in batch if t not in set(long["ticker"])]:
            try:
                one = _yf_frame_to_long(yf.download(t, start=start, end=end_excl, auto_adjust=True,
                                                    actions=False, progress=False), [t])
                if len(one):
                    one["ticker"] = t
                    long = pd.concat([long, one])
            except Exception:  # noqa: BLE001
                pass
        long.to_parquet(f, index=False)
        frames.append(long)
        if logger:
            logger.info(f"yahoo chunk {k // chunk + 1}: {long['ticker'].nunique()}/{len(batch)} with data")
        time.sleep(1.0)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["date", "ticker"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def instrument_classes(pcfg: dict, include_exploratory: bool = False) -> dict:
    """ticker -> asset class."""
    out = {t: c for c, ts in pcfg["instruments"].items() for t in ts}
    if include_exploratory:
        out.update({t: c for c, ts in pcfg.get("exploratory_instruments", {}).items() for t in ts})
    return out


# ============================================================================ Kenneth French library


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the first daily table of a French-library CSV (decimal returns)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.split(",")[0].strip().isdigit() and len(ln.split(",")[0].strip()) == 8)
    header = [c.strip() for c in lines[start - 1].split(",")]
    rows = []
    for ln in lines[start:]:
        parts = [p.strip() for p in ln.split(",")]
        if not (parts[0].isdigit() and len(parts[0]) == 8):
            break
        rows.append(parts)
    ncol = len(rows[0])
    cols = ["date"] + [h if h else f"c{j}" for j, h in enumerate(header[1:ncol], 1)]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def _french_text(key: str, french_dir: Path, logger=None) -> str:
    """Use a manually placed zip if present (data/raw/french/<file>), else download it."""
    fname = FRENCH_FILES[key]
    local = french_dir / fname
    if not local.exists():
        import requests
        r = requests.get(FRENCH_BASE + fname, headers=HTTP_HEADERS, timeout=90)
        r.raise_for_status()
        local.write_bytes(r.content)
        if logger:
            logger.info(f"downloaded {fname}")
    z = zipfile.ZipFile(local)
    return z.read(z.namelist()[0]).decode("latin-1")


def load_french_factors(french_dir: Path, logger=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long factor table (date, factor, ret) and risk-free table (date, region, rf).

    US: Mkt-RF, SMB, HML from the 3-factor file (from 1926); RMW, CMA from the 5-factor file (from 1963);
    Mom from the momentum file. International regions: Mkt-RF, SMB, HML, RMW, CMA, WML."""
    out = french_dir / "factors_daily.parquet"
    rf_out = french_dir / "rf_daily.parquet"
    if out.exists() and rf_out.exists():
        return pd.read_parquet(out), pd.read_parquet(rf_out)
    parts, rfs = [], []
    ff3 = parse_french_csv(_french_text("US_FF3", french_dir, logger))
    ff5 = parse_french_csv(_french_text("US_FF5", french_dir, logger))
    mom = parse_french_csv(_french_text("US_MOM", french_dir, logger))
    mom = mom.rename(columns={[c for c in mom.columns if c != "date"][0]: "Mom"})
    for c in ["Mkt-RF", "SMB", "HML"]:
        parts.append(pd.DataFrame({"date": ff3["date"], "factor": f"US_{c}", "ret": ff3[c]}))
    for c in ["RMW", "CMA"]:
        parts.append(pd.DataFrame({"date": ff5["date"], "factor": f"US_{c}", "ret": ff5[c]}))
    parts.append(pd.DataFrame({"date": mom["date"], "factor": "US_Mom", "ret": mom["Mom"]}))
    rfs.append(pd.DataFrame({"date": ff3["date"], "region": "US", "rf": ff3["RF"]}))
    for k in INTL_REGIONS:
        f5 = parse_french_csv(_french_text(f"{k}_FF5", french_dir, logger))
        mm = parse_french_csv(_french_text(f"{k}_MOM", french_dir, logger))
        mm = mm.rename(columns={[c for c in mm.columns if c != "date"][0]: "WML"})
        for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
            parts.append(pd.DataFrame({"date": f5["date"], "factor": f"{k}_{c}", "ret": f5[c]}))
        parts.append(pd.DataFrame({"date": mm["date"], "factor": f"{k}_WML", "ret": mm["WML"]}))
        rfs.append(pd.DataFrame({"date": f5["date"], "region": k, "rf": f5["RF"]}))
    fac = pd.concat(parts, ignore_index=True).dropna()
    rf = pd.concat(rfs, ignore_index=True).dropna()
    fac.to_parquet(out, index=False)
    rf.to_parquet(rf_out, index=False)
    return fac, rf


def factor_names(region: str, factors: pd.DataFrame) -> list[str]:
    names = sorted(factors["factor"].unique())
    if region == "US":
        return [n for n in names if n.startswith("US_")]
    return [n for n in names if not n.startswith("US_")]


# ============================================================================ synthetic data (smoke test)


def _sim_logvol(rng, T, N, mean, persist=0.985, vv=0.08):
    lv = np.empty((T, N))
    cur = mean + rng.normal(0, 0.2, N)
    for t in range(T):
        cur = persist * cur + (1 - persist) * mean + rng.normal(0, vv, N) * rng.uniform(0.5, 1.5, N)
        lv[t] = cur
    return lv


def make_synthetic(cfg: dict, paths) -> None:
    """Synthetic Yahoo panel and factor files in the real raw layout (pipeline check only)."""
    rng = np.random.default_rng(cfg["project"]["seed"])
    pa = cfg["panels"]["assets"]
    dates = pd.bdate_range(pa["start"], pa["end"])
    cls = instrument_classes(pa)
    tick = list(cls)[:24]
    T, N = len(dates), len(tick)
    vov = rng.uniform(0.03, 0.15, N)
    lv = np.empty((T, N))
    cur = np.log(rng.uniform(0.005, 0.02, N))
    base = cur.copy()
    for t in range(T):
        cur = 0.98 * cur + 0.02 * base + rng.normal(0, 1, N) * vov
        lv[t] = cur
    r = np.exp(lv) * rng.standard_t(5, (T, N)) / np.sqrt(5 / 3)
    close = 100 * np.exp(np.cumsum(r, 0))
    start_idx = rng.integers(0, T // 4, N)
    rows = []
    for j, tk in enumerate(tick):
        sl = slice(start_idx[j], T)
        c = close[sl, j]
        rows.append(pd.DataFrame({"date": dates[sl], "ticker": tk, "open": c, "high": c * 1.01, "low": c * 0.99,
                                  "close": c, "volume": 0.0}))
    vix = 100 * np.sqrt(252) * np.exp(lv.mean(1))
    rows.append(pd.DataFrame({"date": dates, "ticker": pa["regime_symbol"], "open": vix, "high": vix, "low": vix,
                              "close": vix, "volume": 0.0}))
    pd.concat(rows).to_parquet(paths.yahoo / "assets_ohlcv.parquet", index=False)
    # factors
    parts, rfs = [], []
    for region, keyp in [("US", cfg["panels"]["ff_us"]), ("INTL", cfg["panels"]["ff_intl"])]:
        d = pd.bdate_range(keyp["start"], keyp["end"])
        names = (["US_Mkt-RF", "US_SMB", "US_HML", "US_RMW", "US_CMA", "US_Mom"] if region == "US" else
                 [f"{k}_{c}" for k in ["DEV", "EU"] for c in ["Mkt-RF", "SMB", "HML", "WML"]])
        lvf = _sim_logvol(rng, len(d), len(names), np.log(0.007))
        ret = np.exp(lvf) * rng.standard_normal((len(d), len(names))) + 0.0002
        for j, n in enumerate(names):
            parts.append(pd.DataFrame({"date": d, "factor": n, "ret": ret[:, j]}))
        rfs.append(pd.DataFrame({"date": d, "region": "US" if region == "US" else "DEV", "rf": 0.0001}))
    pd.concat(parts).to_parquet(paths.french / "factors_daily.parquet", index=False)
    pd.concat(rfs).to_parquet(paths.french / "rf_daily.parquet", index=False)
