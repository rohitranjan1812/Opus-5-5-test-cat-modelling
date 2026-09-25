"""Counter-based (stateless) random numbers for reproducible, embarrassingly-parallel Monte Carlo.

Every random draw in the loss engine is a *pure function* of integer keys::

    h  = stream(seed, key)            # one 64-bit stream per (seed, occurrence)
    u  = uniform(h, index)            # index = location id, cell id, or a reserved tag

using the SplitMix64 finaliser (Steele, Lea & Flood 2014) as the mixing function.  Consequences:

* bit-for-bit reproducibility regardless of thread count or scheduling;
* any single (occurrence, location) draw can be regenerated in isolation, which lets the
  engine re-simulate only tail years at full location detail for Euler allocation;
* common random numbers (CRN) across what-if runs: changing one account does not perturb the
  draws of any other location, so marginal / sensitivity deltas are nearly noise-free.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)
_ONE = np.uint64(1)
_INV53 = 1.0 / 9007199254740992.0  # 2**-53

# Reserved stream indices.  Location draws use index = location id + TAG_BASE.
TAG_ETA = 0  # event-level (inter-event) hazard residual
TAG_Z = 1  # event-wide copula factor
TAG_CELL_BASE = 1 << 40  # spatial cell factors live in a disjoint index range
TAG_LOC_BASE = 16
TAG_SURGE_E = 2  # event-level storm-surge water-level error
TAG_SURGE_BASE = 1 << 38  # node-level surge water-level errors (disjoint range)
TAG_SURGE_WET = 1 << 39  # node-level surge connectivity draws (disjoint range)


@nb.njit(inline="always", cache=True)
def mix64(x):
    """SplitMix64 finaliser: a bijective avalanche mix of a uint64."""
    z = x + _GOLDEN
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    return z ^ (z >> _S31)


@nb.njit(inline="always", cache=True)
def stream(seed, key):
    """Derive an independent 64-bit stream handle from (seed, key)."""
    return mix64(np.uint64(seed) ^ mix64(np.uint64(key)))


@nb.njit(inline="always", cache=True)
def uniform(h, index):
    """U(0,1) (open interval) for position ``index`` of stream ``h``."""
    z = mix64(h + (np.uint64(index) + _ONE) * _GOLDEN)
    return (float(z >> _S11) + 0.5) * _INV53


# Acklam's rational approximation to the normal quantile + one Halley refinement step
# (|rel err| < 1e-15 after refinement).
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)
_P_LOW = 0.02425
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


@nb.njit(cache=True)
def norm_ppf_fast(p):
    """Acklam's approximation without refinement (|rel err| < 1.2e-9) — used in hot loops."""
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
        ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)


@nb.njit(cache=True)
def norm_ppf(p):
    """Standard normal quantile Φ⁻¹(p) for p in (0, 1) (Acklam + Halley refinement).

    For p > 1/2 the reflection Φ⁻¹(p) = -Φ⁻¹(1-p) is used; 1-p is exact there (Sterbenz), which
    avoids cancellation in the refinement residual deep in the upper tail.
    """
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    if p > 0.5:
        return -norm_ppf(1.0 - p)
    x = norm_ppf_fast(p)
    e = 0.5 * math.erfc(-x / _SQRT2) - p
    u = e * _SQRT2PI * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


@nb.njit(inline="always", cache=True)
def norm_cdf(x):
    return 0.5 * math.erfc(-x / _SQRT2)


@nb.njit(cache=True)
def _uniform_block(seed, keys, n):
    out = np.empty((keys.shape[0], n))
    for i in range(keys.shape[0]):
        h = stream(seed, keys[i])
        for j in range(n):
            out[i, j] = uniform(h, j)
    return out


def uniform_block(seed: int, keys: np.ndarray, n: int) -> np.ndarray:
    """Convenience (numpy-facing) helper: an (len(keys), n) block of uniforms."""
    return _uniform_block(np.uint64(seed), np.asarray(keys, dtype=np.uint64), int(n))
