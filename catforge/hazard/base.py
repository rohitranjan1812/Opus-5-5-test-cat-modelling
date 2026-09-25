"""Stochastic event catalog abstractions and frequency (occurrence) models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Regime:
    """A discrete climate state (e.g. ENSO phase) that scales the annual event rate."""

    name: str
    prob: float
    multiplier: float


@dataclass
class FrequencyModel:
    """Mixed-Poisson annual occurrence model.

    Annual count  N | Θ ~ Poisson(Θ Λ),   Θ = M_R · G,
      R ~ Categorical(regime probs)            (discrete climate regimes, E[M_R] normalised to 1)
      G ~ Gamma(shape=r, scale=1/r)            (residual over-dispersion; r = inf → Poisson)

    Because thinning a mixed Poisson process preserves the mixing law, only the portfolio-relevant
    subset of events needs to be simulated. The probability generating function is
        P_N(z) = Σ_R p_R · (1 - M_R Λ (z - 1) / r)^(-r)
    which the analytic engine uses for FFT-based aggregate distributions.
    """

    dispersion_r: float = float("inf")
    regimes: list[Regime] = field(default_factory=list)

    def __post_init__(self):
        if self.regimes:
            p = np.array([g.prob for g in self.regimes], float)
            p = p / p.sum()
            m = np.array([g.multiplier for g in self.regimes], float)
            m = m / float(np.dot(p, m))  # normalise so E[Θ] = 1 (Λ keeps its meaning)
            self.regimes = [Regime(g.name, float(pi), float(mi)) for g, pi, mi in zip(self.regimes, p, m)]

    @property
    def is_poisson(self) -> bool:
        return not self.regimes and not np.isfinite(self.dispersion_r)

    def sample_theta(self, rng: np.random.Generator, n_years: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (Θ_y, regime index_y) for each simulated year (regime index = -1 if no regimes)."""
        theta = np.ones(n_years)
        reg = np.full(n_years, -1, dtype=np.int16)
        if self.regimes:
            p = np.array([g.prob for g in self.regimes])
            reg = rng.choice(len(p), size=n_years, p=p).astype(np.int16)
            theta *= np.array([g.multiplier for g in self.regimes])[reg]
        if np.isfinite(self.dispersion_r):
            r = self.dispersion_r
            theta *= rng.gamma(shape=r, scale=1.0 / r, size=n_years)
        return theta, reg

    def pgf(self, z: np.ndarray, lam: float) -> np.ndarray:
        """Probability generating function E[z^N] (complex z allowed)."""
        comps = self.regimes or [Regime("all", 1.0, 1.0)]
        out = np.zeros_like(z, dtype=complex)
        for g in comps:
            lm = g.multiplier * lam
            if np.isfinite(self.dispersion_r):
                r = self.dispersion_r
                out += g.prob * (1.0 - lm * (z - 1.0) / r) ** (-r)
            else:
                out += g.prob * np.exp(lm * (z - 1.0))
        return out

    def variance_to_mean(self, lam: float) -> float:
        comps = self.regimes or [Regime("all", 1.0, 1.0)]
        e_t2 = sum(g.prob * g.multiplier ** 2 for g in comps)
        if np.isfinite(self.dispersion_r):
            e_t2 *= 1.0 + 1.0 / self.dispersion_r
        var_theta = e_t2 - 1.0
        return 1.0 + lam * var_theta

    def to_dict(self) -> dict:
        return {
            "dispersion_r": None if not np.isfinite(self.dispersion_r) else self.dispersion_r,
            "regimes": [g.__dict__ for g in self.regimes],
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> FrequencyModel:
        if not d:
            return cls()
        r = d.get("dispersion_r")
        return cls(dispersion_r=float("inf") if r in (None, 0) else float(r),
                   regimes=[Regime(**g) for g in d.get("regimes", [])])


ENSO_REGIMES = [Regime("La Niña", 0.25, 1.30), Regime("Neutral", 0.50, 1.00), Regime("El Niño", 0.25, 0.65)]


@dataclass
class HazardUncertainty:
    """Primary (hazard) uncertainty and dependence for a peril, in log-intensity space.

    ln I_ij = ln m_ij + η_i + W_i(s_j),  η_i ~ N(0, σ_b²) shared by all sites in an event.

    * ``dependence="grf"`` (default): W_i = σ_w·Z_i with Z_i a Matérn Gaussian random field
      (smoothness ``grf_nu``, practical range ``grf_range_km``) sampled explicitly at the sites;
      the remaining damage (vulnerability) uncertainty is coupled by a weak two-level copula
      (``dmg_rho_event``, ``dmg_rho_cell``) representing construction-quality/claims correlation.
    * ``dependence="copula"`` (legacy): W is folded analytically into the vulnerability tables and
      the combined (W, damage) variable is coupled by a two-level Gaussian copula
      (``rho_event``, ``rho_cell``).
    """

    sigma_between: float
    sigma_within: float
    rho_event: float
    rho_cell: float
    cell_deg: float = 0.25
    grf_nu: float = 0.5
    grf_range_km: float = 30.0
    dmg_rho_event: float = 0.04
    dmg_rho_cell: float = 0.10


@dataclass
class EventCatalog:
    """A stochastic event set for one peril.

    ``events`` is a DataFrame with at least ``event_id`` and ``rate`` columns plus peril-specific
    descriptors.  ``geometry`` holds CSR-packed arrays used by the footprint kernels.
    """

    peril: str
    events: pd.DataFrame
    geometry: dict[str, np.ndarray]
    frequency: FrequencyModel
    uncertainty: HazardUncertainty
    intensity_unit: str
    meta: dict = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def rates(self) -> np.ndarray:
        return self.events["rate"].to_numpy(float)

    @property
    def total_rate(self) -> float:
        return float(self.rates.sum())

    def summary(self) -> dict:
        return {
            "peril": self.peril,
            "n_events": self.n_events,
            "total_rate": self.total_rate,
            "intensity_unit": self.intensity_unit,
            "frequency": self.frequency.to_dict(),
            "uncertainty": self.uncertainty.__dict__,
            "meta": self.meta,
        }
