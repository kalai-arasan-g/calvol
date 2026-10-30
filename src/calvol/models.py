"""Forecasting models (methodology Section 6) and a generic trainer.

All models map a window x[B, L, C] of raw channels to a scalar z. Target scaling (Section 6.3):
    return model   z_r = y_r / s,  s = std(daily returns in window) * sqrt(h)       -> mu_hat = s * z
    vol model      z_v = y_v - a,  a = 0.5 * ln(mean of exp(logvar) over last 21 days) -> lv_hat = a + z
Channel 0 must be 'ret' and channel 1 'logvar' (enforced by forecast.py).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5


CENTERED_CHANNELS = {"logvar", "logdvol", "logvix"}


def center_mask(channels: list[str]) -> torch.Tensor:
    """1 for level-type channels (centred by RevIN), 0 for return-type channels (scaled only, so the
    window's mean return - momentum/drift information - is preserved)."""
    return torch.tensor([1.0 if c in CENTERED_CHANNELS else 0.0 for c in channels])


def revin(x: torch.Tensor, cmask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sample, per-channel normalisation using the window's own statistics only. Level-type channels are
    centred and scaled; return-type channels are scaled only."""
    sd = x.std(1, keepdim=True, unbiased=False)
    mu = x.mean(1, keepdim=True)
    if cmask is not None:
        mu = mu * cmask.to(x.device).view(1, 1, -1)
    return (x - mu) / (sd + EPS)


class LinearModel(nn.Module):
    """Plain linear map over the normalised window (all channels), as in the 'Linear' baseline of Zeng et al."""

    def __init__(self, L: int, C: int, cmask=None, **_):
        super().__init__()
        self.lin = nn.Linear(L * C, 1)
        self.register_buffer("cmask", cmask if cmask is not None else torch.ones(C))
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(revin(x, self.cmask).flatten(1)).squeeze(-1)


class DLinear(nn.Module):
    """DLinear (Zeng et al., 2023): moving-average trend/remainder decomposition, one linear map each.
    Forecasts the scalar aggregate target directly (Section 6.1)."""

    def __init__(self, L: int, C: int, kernel: int = 25, cmask=None, **_):
        super().__init__()
        self.k = int(kernel)
        self.register_buffer("cmask", cmask if cmask is not None else torch.ones(C))
        self.lin_t = nn.Linear(L * C, 1)
        self.lin_s = nn.Linear(L * C, 1, bias=False)
        for m in (self.lin_t, self.lin_s):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.lin_t.bias)

    def decompose(self, x):
        # x [B, L, C] -> moving average along time with edge (replicate) padding
        xt = x.transpose(1, 2)
        left = (self.k - 1) // 2
        right = self.k - 1 - left
        xp = F.pad(xt, (left, right), mode="replicate")
        trend = F.avg_pool1d(xp, kernel_size=self.k, stride=1).transpose(1, 2)
        return trend, x - trend

    def forward(self, x):
        xn = revin(x, self.cmask)
        tr, res = self.decompose(xn)
        return (self.lin_t(tr.flatten(1)) + self.lin_s(res.flatten(1))).squeeze(-1)


class NLinear(nn.Module):
    """NLinear (Zeng et al., 2023) with a robust anchor: each channel is shifted by the mean of its last
    `anchor` values (Section 6.2) and scaled by its window standard deviation."""

    def __init__(self, L: int, C: int, anchor: int = 5, **_):
        super().__init__()
        self.a = int(anchor)
        self.lin = nn.Linear(L * C, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        anc = x[:, -self.a:, :].mean(1, keepdim=True)
        sd = x.std(1, keepdim=True, unbiased=False)
        return self.lin(((x - anc) / (sd + EPS)).flatten(1)).squeeze(-1)


class LSTMBase(nn.Module):
    """Base-paper model (Ameri et al., 2026): 2-layer LSTM (50 units) on a 5-day window, three heads
    (return, volatility, confidence). Inputs are standardised with training-pool statistics."""

    def __init__(self, C: int, hidden: int = 50, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(C, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x):
        o, _ = self.lstm(x)
        z = self.head(o[:, -1, :])
        return z[:, 0], z[:, 1], torch.sigmoid(z[:, 2])


def build_model(kind: str, L: int, C: int, kernel: int = 25, anchor: int = 5, channels: list | None = None) -> nn.Module:
    cm = center_mask(channels) if channels is not None else None
    if kind == "dlinear":
        return DLinear(L, C, kernel=kernel, cmask=cm)
    if kind == "nlinear":
        return NLinear(L, C, anchor=anchor)
    if kind == "linear":
        return LinearModel(L, C, cmask=cm)
    raise ValueError(kind)


# ============================================================================ data access on device


class WindowBank:
    """Holds the channel tensor X[T, N, C] and targets on the device; gathers windows by (day, asset)."""

    def __init__(self, X: np.ndarray, y_r: np.ndarray, y_v: np.ndarray, h: int, device):
        self.device = device
        self.X = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.y_r = torch.as_tensor(np.nan_to_num(y_r), dtype=torch.float32, device=device)
        self.y_v = torch.as_tensor(np.nan_to_num(y_v), dtype=torch.float32, device=device)
        self.h = h

    def windows(self, d: torch.Tensor, i: torch.Tensor, L: int) -> torch.Tensor:
        off = torch.arange(L - 1, -1, -1, device=self.device)
        tidx = d[:, None] - off[None, :]
        return self.X[tidx, i[:, None], :]

    def scales(self, xw: torch.Tensor):
        """Target scaling from the raw window only (channels 0=ret, 1=logvar):
        s  return scale  = std(daily returns) * sqrt(h)
        a  log-vol anchor = 0.5 * ln(mean of the variance proxy over the last 21 days)
        v  log-vol scale  = 0.5 * std(log variance proxy over the window)
        Inputs are standardised by their window statistics (RevIN), so outputs are de-normalised with the
        same statistics: mu_hat = s * z_r,  lv_hat = a + v * z_v."""
        s = xw[:, :, 0].std(1, unbiased=False).clamp_min(1e-4) * math.sqrt(self.h)
        a = 0.5 * torch.log(torch.exp(xw[:, -21:, 1]).mean(1))
        v = (0.5 * xw[:, :, 1].std(1, unbiased=False)).clamp_min(0.05)
        return s, a, v


# ============================================================================ losses


def loss_fn(kind: str, z_hat: torch.Tensor, z: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Loss on normalised targets. For 'qlike' `scale` converts z back to log-vol units."""
    if scale is not None and kind in ("huber", "mse"):
        z_hat, z = z_hat * scale, z * scale
    if kind == "huber":
        return F.huber_loss(z_hat, z, delta=1.0)
    if kind == "mse":
        return F.mse_loss(z_hat, z)
    if kind == "qlike":
        # on variance: RV^2/sig^2 - ln(RV^2/sig^2) - 1 with e = 2 (y_v - y_v_hat)
        sc = scale if scale is not None else 1.0
        e = (2.0 * sc * (z - z_hat)).clamp(-20, 20)
        return (torch.exp(e) - e - 1.0).mean()
    raise ValueError(kind)


# ============================================================================ trainer


def _batches(n: int, bs: int, gen: torch.Generator | None, device, shuffle: bool):
    idx = torch.randperm(n, generator=gen, device="cpu").to(device) if shuffle else torch.arange(n, device=device)
    for k in range(0, n, bs):
        yield idx[k:k + bs]


def _target(bank: WindowBank, which: str, xw, d, i):
    """Normalised target and the scale that maps it back to target units."""
    s, a, v = bank.scales(xw)
    if which == "ret":
        return bank.y_r[d, i] / s, None          # returns: loss on the scale-free target
    return (bank.y_v[d, i] - a) / v, v           # volatility: loss evaluated in log-vol units


@torch.no_grad()
def evaluate(model, bank: WindowBank, d, i, L: int, which: str, loss: str, bs: int = 8192) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in _batches(len(d), bs, None, bank.device, False):
        xw = bank.windows(d[b], i[b], L)
        z, sc = _target(bank, which, xw, d[b], i[b])
        tot += float(loss_fn(loss, model(xw), z, sc)) * len(b)
        n += len(b)
    return tot / max(n, 1)


def fit(model, bank: WindowBank, d_tr, i_tr, L: int, which: str, loss: str, lr: float, wd: float,
        bs: int, epochs: int, seed: int, d_va=None, i_va=None, patience: int = 4):
    """Train with AdamW. With a validation set: early stopping, returns (best_model, best_epoch, history).
    Without: trains exactly `epochs` epochs."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = model.to(bank.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_ep, best_state, bad, hist = float("inf"), 0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        for b in _batches(len(d_tr), bs, gen, bank.device, True):
            xw = bank.windows(d_tr[b], i_tr[b], L)
            z, sc = _target(bank, which, xw, d_tr[b], i_tr[b])
            opt.zero_grad(set_to_none=True)
            lo = loss_fn(loss, model(xw), z, sc)
            lo.backward()
            opt.step()
        if d_va is not None:
            v = evaluate(model, bank, d_va, i_va, L, which, loss)
            hist.append(v)
            if v < best - 1e-7:
                best, best_ep, best_state, bad = v, ep, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if d_va is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, max(best_ep, 1), hist


@torch.no_grad()
def predict(model, bank: WindowBank, d, i, L: int, which: str, bs: int = 8192) -> np.ndarray:
    """Return predictions in target units: mu_hat (horizon log excess return) or lv_hat (log daily vol)."""
    model.eval()
    out = []
    for b in _batches(len(d), bs, None, bank.device, False):
        xw = bank.windows(d[b], i[b], L)
        s, a, v = bank.scales(xw)
        z = model(xw)
        out.append((s * z if which == "ret" else a + v * z).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
