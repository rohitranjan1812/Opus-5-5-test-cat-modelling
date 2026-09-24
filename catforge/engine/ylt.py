"""Year-occurrence table simulation (mixed-Poisson frequency, seasonality, perils independent)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PERILS
from ..hazard.base import FrequencyModel

SEASON = {  # (kind, params) day-of-year distribution
    "TC": ("normal", 250.0, 26.0, 152.0, 334.0),
    "EQ": ("uniform", 0.0, 365.0, 0.0, 365.0),
}
YLT_KEY_BASE = 1 << 33
ELT_KEY_BASE = 1 << 52


@dataclass
class YLT:
    n_years: int
    year: np.ndarray  # int64 (K,)
    event: np.ndarray  # int64 global event index (K,)
    day: np.ndarray  # float (K,)
    peril: np.ndarray  # int8 (K,)
    key: np.ndarray  # uint64 (K,)
    theta: np.ndarray  # (n_perils, n_years) frequency mixing variable
    regime: np.ndarray  # (n_perils, n_years) int16 regime index (-1 none)

    @property
    def n(self) -> int:
        return int(self.year.shape[0])


def simulate_ylt(ev_rate: np.ndarray, ev_peril: np.ndarray, relevant: np.ndarray,
                 freq: dict[str, FrequencyModel], n_years: int, seed: int) -> YLT:
    """Simulate occurrences of *relevant* events only (valid by mixed-Poisson thinning)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x59E7]))
    years, events, days, perils = [], [], [], []
    theta_all = np.ones((len(PERILS), n_years))
    regime_all = np.full((len(PERILS), n_years), -1, dtype=np.int16)
    for pi, peril in enumerate(PERILS):
        idx = np.nonzero(relevant & (ev_peril == pi))[0]
        if idx.size == 0:
            continue
        lam = ev_rate[idx]
        Lam = float(lam.sum())
        if Lam <= 0:
            continue
        fm = freq.get(peril, FrequencyModel())
        theta, reg = fm.sample_theta(rng, n_years)
        theta_all[pi], regime_all[pi] = theta, reg
        counts = rng.poisson(theta * Lam)
        k = int(counts.sum())
        cdf = np.cumsum(lam) / Lam
        pick = idx[np.minimum(np.searchsorted(cdf, rng.random(k), side="right"), idx.size - 1)]
        yr = np.repeat(np.arange(n_years), counts)
        kind, a, b, lo, hi = SEASON[peril]
        d = np.clip(rng.normal(a, b, k), lo, hi) if kind == "normal" else rng.uniform(a, b, k)
        years.append(yr)
        events.append(pick)
        days.append(d)
        perils.append(np.full(k, pi, dtype=np.int8))
    if years:
        year = np.concatenate(years)
        event = np.concatenate(events)
        day = np.concatenate(days)
        peril = np.concatenate(perils)
    else:
        year = np.zeros(0, np.int64)
        event = np.zeros(0, np.int64)
        day = np.zeros(0)
        peril = np.zeros(0, np.int8)
    order = np.lexsort((day, year))
    year, event, day, peril = year[order], event[order], day[order], peril[order]
    key = (YLT_KEY_BASE + np.arange(year.size)).astype(np.uint64)
    return YLT(n_years, year.astype(np.int64), event.astype(np.int64), day, peril, key, theta_all, regime_all)


def rate_change_weights(ylt: YLT, ev_rate: np.ndarray, new_rate: np.ndarray, relevant: np.ndarray,
                        ev_peril: np.ndarray) -> np.ndarray:
    """Exact per-year likelihood ratios for a change of event rates λ → λ'.

    Conditional on the simulated mixing variable Θ_y, each peril's occurrence process is Poisson
    with intensity Θ_y λ, so   w_y = Π_{k∈y} (λ'_{e_k}/λ_{e_k}) · Π_p exp(-Θ_{p,y} (Λ'_p - Λ_p)).
    """
    ratio = np.where(ev_rate > 0, new_rate / np.maximum(ev_rate, 1e-300), 1.0)
    logw = np.zeros(ylt.n_years)
    if ylt.n:
        np.add.at(logw, ylt.year, np.log(np.maximum(ratio[ylt.event], 1e-300)))
    for pi in range(len(PERILS)):
        m = relevant & (ev_peril == pi)
        dLam = float(new_rate[m].sum() - ev_rate[m].sum())
        logw -= ylt.theta[pi] * dLam
    w = np.exp(logw - logw.max())
    return w * (ylt.n_years / w.sum())
