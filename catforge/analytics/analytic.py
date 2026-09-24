"""Analytic EP curves from an Event Loss Table (ELT) — an independent cross-check of the simulation.

Each event's loss is modelled as  L_i = 0 w.p. p0_i,  else  max_i · Beta(α_i, β_i)  matched to the
ELT's conditional mean and variance.  For peril p with mixed-Poisson frequency (pgf P_p) and total
rate Λ_p, the severity is the rate-weighted mixture  f_p = Σ_i (λ_i/Λ_p) f_i.  Then, perils being
independent,

    OEP(x) = 1 - Π_p P_p(F_p(x))                             (max occurrence)
    AEP:  g = IFFT[ Π_p P_p( FFT(f_p · e^{-θk}) ) ] · e^{θk}   (aggregate; exponentially tilted FFT)

The tilt θ ≈ 20/N suppresses wrap-around aliasing (Grübel & Hermesmeier 1999; Embrechts & Frei 2009).
"""

from __future__ import annotations

import numpy as np
from scipy.special import betainc

from ..hazard.base import FrequencyModel


def _beta_params(mean_pos, var_pos, cap):
    m = np.clip(mean_pos / cap, 1e-9, 1 - 1e-9)
    v = np.clip(var_pos / cap**2, 1e-12, None)
    v = np.minimum(v, 0.98 * m * (1 - m))
    s = m * (1 - m) / v - 1.0
    return m * s, (1 - m) * s


def discretise_severity(mean, sd, cap, p0, weights, h, n_grid, n_local: int = 256) -> np.ndarray:
    """Rate-weighted mixture pmf on grid {0, h, 2h, ...} with mean-preserving linear mass splitting."""
    f = np.zeros(n_grid)
    mean = np.asarray(mean, float)
    ok = (mean > 0) & (cap > 0)
    f[0] += float(np.sum(weights[~ok]))
    mean, sd, cap, p0, w = mean[ok], np.asarray(sd)[ok], np.asarray(cap)[ok], np.asarray(p0)[ok], weights[ok]
    p0 = np.clip(p0, 0.0, 0.999)
    ex2 = sd**2 + mean**2
    mpos = mean / (1 - p0)
    vpos = np.maximum(ex2 / (1 - p0) - mpos**2, 1e-12)
    cap = np.maximum(cap, mpos * 1.0001)
    a, b = _beta_params(mpos, vpos, cap)
    f[0] += float(np.sum(w * p0))
    u = np.linspace(0, 1, n_local + 1)
    m = a / (a + b)
    cdf = betainc(a[:, None], b[:, None], u[None, :])
    cdf1 = betainc(a[:, None] + 1.0, b[:, None], u[None, :])  # E[X 1{X<u}] = m I_u(a+1, b)
    wpos = (w * (1 - p0))[:, None]
    pm = np.diff(cdf, axis=1) * wpos
    pmx = cap[:, None] * m[:, None] * np.diff(cdf1, axis=1) * wpos
    mid = cap[:, None] * 0.5 * (u[:-1] + u[1:])[None, :]
    x = np.where(pm > 1e-300, pmx / np.maximum(pm, 1e-300), mid)  # exact conditional mean of each local bin
    pos = x / h
    i0 = np.floor(pos).astype(np.int64)
    fr = pos - i0
    i0 = np.clip(i0, 0, n_grid - 2)
    np.add.at(f, i0.ravel(), (pm * (1 - fr)).ravel())
    np.add.at(f, (i0 + 1).ravel(), (pm * fr).ravel())
    return f


def analytic_ep(elt, freq_models: dict[str, FrequencyModel], n_grid: int = 1 << 16, x_max: float | None = None,
                rps=(2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)) -> dict:
    """``elt`` columns: peril, rate, mean, sd, cap, p0."""
    if len(elt) == 0:
        return {"rp": [], "oep": [], "aep": []}
    cap_all = elt["cap"].to_numpy(float)
    if x_max is None:
        x_max = float(min(4.0 * np.quantile(cap_all, 0.999), cap_all.sum()) + 1.0)
    h = x_max / n_grid
    theta = 20.0 / n_grid
    k = np.arange(n_grid)
    tilt = np.exp(-theta * k)
    F_prod = np.ones(n_grid)
    G_hat = np.ones(n_grid, dtype=complex)
    for peril, grp in elt.groupby("peril"):
        lam = grp["rate"].to_numpy(float)
        Lam = float(lam.sum())
        if Lam <= 0:
            continue
        fm = freq_models.get(peril, FrequencyModel())
        f = discretise_severity(grp["mean"].to_numpy(), grp["sd"].to_numpy(), grp["cap"].to_numpy(),
                                grp["p0"].to_numpy(), lam / Lam, h, n_grid)
        F = np.minimum(np.cumsum(f), 1.0)
        F_prod *= np.real(fm.pgf(F.astype(complex), Lam))
        G_hat *= fm.pgf(np.fft.fft(f * tilt), Lam)
    oep = 1.0 - F_prod
    g = np.real(np.fft.ifft(G_hat)) / tilt
    g = np.clip(g, 0.0, None)
    aep = 1.0 - np.minimum(np.cumsum(g), 1.0)
    x = k * h  # oep/aep[k] = P(L > k h)

    def at_rp(ep, rp):
        p = 1.0 / rp
        idx = np.searchsorted(-ep, -p)  # ep is non-increasing
        return float(x[min(idx, n_grid - 1)])

    table = [{"rp": rp, "oep": at_rp(oep, rp), "aep": at_rp(aep, rp)} for rp in rps]
    mean_agg = float(np.sum(g * k * h))
    sel = np.unique(np.geomspace(1, n_grid - 1, 400).astype(int))
    return {"table": table, "aal": mean_agg, "x": x[sel].tolist(), "oep": oep[sel].tolist(), "aep": aep[sel].tolist(),
            "grid_step": h, "x_max": x_max}
