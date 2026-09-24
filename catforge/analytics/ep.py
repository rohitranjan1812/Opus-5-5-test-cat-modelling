"""Exceedance-probability curves and tail risk metrics from simulated annual losses.

* Empirical EP: loss at return period T is the (1 - 1/T) quantile of the annual (AEP) or annual
  maximum-occurrence (OEP) loss distribution.
* Confidence intervals for quantiles are *distribution-free* order-statistic intervals: the rank of
  the q-quantile among n years is Binomial(n, q), so [x_(l), x_(u)] with l, u the 2.5%/97.5%
  binomial quantiles has ≥95% coverage without bootstrap.
* TVaR (CTE) CIs use a vectorised bootstrap over years.
* Weighted versions support importance-reweighted years (climate / frequency what-ifs).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

RETURN_PERIODS = (2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)


def quantile_at_rp(sorted_asc: np.ndarray, rp: float) -> float:
    n = sorted_asc.size
    if n == 0:
        return 0.0
    q = 1.0 - 1.0 / rp
    pos = q * n - 0.5  # Hazen plotting position
    if pos <= 0:
        return float(sorted_asc[0])
    if pos >= n - 1:
        return float(sorted_asc[-1])
    i = int(np.floor(pos))
    f = pos - i
    return float(sorted_asc[i] * (1 - f) + sorted_asc[i + 1] * f)


def tvar_sorted(sorted_asc: np.ndarray, rp: float) -> float:
    n = sorted_asc.size
    k = max(int(np.floor(n / rp)), 1)
    return float(sorted_asc[-k:].mean())


def order_stat_ci(sorted_asc: np.ndarray, rp: float, level: float = 0.95) -> tuple[float, float]:
    n = sorted_asc.size
    q = 1.0 - 1.0 / rp
    a = (1.0 - level) / 2
    lo = int(binom.ppf(a, n, q)) - 1
    hi = int(binom.ppf(1 - a, n, q))
    lo, hi = max(lo, 0), min(hi, n - 1)
    return float(sorted_asc[lo]), float(sorted_asc[hi])


def ep_table(values: np.ndarray, rps=RETURN_PERIODS, with_ci: bool = True, n_boot: int = 200,
             seed: int = 0) -> list[dict]:
    x = np.sort(np.asarray(values, float))
    n = x.size
    rows = []
    boot = None
    if with_ci and n_boot > 0 and n > 0:
        rng = np.random.default_rng(seed)
        boot = np.sort(x[rng.integers(0, n, size=(n_boot, n))], axis=1)
    for rp in rps:
        if rp > n:
            continue
        r = {"rp": rp, "prob": 1.0 / rp, "loss": quantile_at_rp(x, rp), "tvar": tvar_sorted(x, rp)}
        if with_ci:
            r["loss_lo"], r["loss_hi"] = order_stat_ci(x, rp)
            if boot is not None:
                k = max(int(np.floor(n / rp)), 1)
                bs = boot[:, -k:].mean(axis=1)
                r["tvar_lo"], r["tvar_hi"] = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        rows.append(r)
    return rows


def ep_curve(values: np.ndarray, n_points: int = 120, max_rp: float | None = None) -> dict:
    """EP curve sampled on a log-spaced return-period grid (for plotting)."""
    x = np.sort(np.asarray(values, float))
    n = x.size
    max_rp = min(max_rp or n, n)
    rps = np.unique(np.geomspace(1.0001, max(max_rp, 1.01), n_points))
    loss = np.array([quantile_at_rp(x, r) for r in rps])
    return {"rp": rps.round(4).tolist(), "prob": (1 / rps).tolist(), "loss": loss.tolist()}


def summary_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, float)
    n = v.size
    mean = float(v.mean()) if n else 0.0
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "cov": sd / mean if mean > 0 else None,
            "se_mean": sd / np.sqrt(n) if n else None, "p_nonzero": float((v > 0).mean()) if n else 0.0,
            "max": float(v.max()) if n else 0.0}


# ------------------------------------------------------------------ weighted (reweighted years)
def weighted_quantile_at_rp(values: np.ndarray, weights: np.ndarray, rp: float) -> float:
    order = np.argsort(values)[::-1]
    v = values[order]
    w = weights[order] / weights.sum()
    ex = np.cumsum(w)  # P(X >= v_i)
    target = 1.0 / rp
    i = np.searchsorted(ex, target)
    return float(v[min(i, v.size - 1)])


def weighted_tvar(values: np.ndarray, weights: np.ndarray, rp: float) -> float:
    order = np.argsort(values)[::-1]
    v = values[order]
    w = weights[order] / weights.sum()
    ex = np.cumsum(w)
    target = 1.0 / rp
    i = int(np.searchsorted(ex, target))
    i = min(i, v.size - 1)
    wt = w[: i + 1].copy()
    wt[-1] -= max(ex[i] - target, 0.0)
    return float((v[: i + 1] * wt).sum() / max(wt.sum(), 1e-300))


def weighted_ep_table(values: np.ndarray, weights: np.ndarray, rps=RETURN_PERIODS) -> list[dict]:
    ess = float(weights.sum() ** 2 / np.maximum((weights**2).sum(), 1e-300))
    return [{"rp": rp, "prob": 1 / rp, "loss": weighted_quantile_at_rp(values, weights, rp),
             "tvar": weighted_tvar(values, weights, rp)} for rp in rps if rp <= ess]


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.maximum((weights**2).sum(), 1e-300))
