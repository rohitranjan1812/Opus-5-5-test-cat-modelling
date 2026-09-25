"""Gaussian random fields for intra-event hazard residuals.

Model
-----
Within one event, the log-intensity residual at site s is W(s) = φ·Z(s) with Z a zero-mean,
unit-variance Gaussian random field with isotropic Matérn correlation

    ρ(h) = 2^{1-ν}/Γ(ν) (√(2ν) h/ℓ)^ν K_ν(√(2ν) h/ℓ),   ν ∈ {1/2, 3/2, 5/2} (closed forms),

parameterised by the *practical range* r₀.₀₅ (ρ = 0.05).  For earthquakes this is the
Jayaram & Baker (2009) exponential model (ν = 1/2) for PGA intra-event residuals.

Sampling at portfolio sites — Vecchia / nearest-neighbour Gaussian process
--------------------------------------------------------------------------
Order sites by an (approximate) maximin ordering and factor the joint density as
    p(z) ≈ Π_j p(z_j | z_{N(j)}),  N(j) = the m nearest *earlier* sites  (Vecchia 1988),
    z_j = b_jᵀ z_{N(j)} + d_j ε_j,   b_j = C_NN⁻¹ c_jN,  d_j² = 1 − c_jNᵀ b_j,   ε_j ~ N(0, 1).
The coefficients depend only on geometry, so they are computed once per portfolio (batched
m×m solves); each occurrence then costs O(n·m) and ε_j come from the counter-based RNG keyed
by site, which preserves bit-reproducibility and common random numbers.  Neighbours whose
correlation is below a threshold are pruned, so each event only needs the *ancestor closure*
of its affected sites (computed once per event), which stays spatially local.

Maximin ordering + m ≈ 16–30 neighbours gives near-exact fields for Matérn covariances
(Guinness 2018; Katzfuss & Guinness 2021) — verified in the test-suite against the exact
covariance and by Monte Carlo.

Regular grids use exact circulant embedding (Dietrich & Newsam 1997).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba as nb
import numpy as np
from scipy.spatial import cKDTree

from ..rng import norm_ppf_fast, stream, uniform

EARTH_R_KM = 6371.0088
TAG_W_BASE = 1 << 36  # RNG index range for field innovations (disjoint from other draws)


@dataclass(frozen=True)
class Matern:
    nu: float = 0.5
    range_km: float = 30.0  # practical range: ρ(range) = 0.05

    @property
    def ell(self) -> float:
        """Scale ℓ such that ρ(range_km) = 0.05 (found by bisection on the closed form)."""
        lo, hi = 1e-6, 50.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _matern_u(mid, self.nu) > 0.05:
                lo = mid
            else:
                hi = mid
        return self.range_km / (0.5 * (lo + hi))

    def corr(self, h: np.ndarray) -> np.ndarray:
        return _matern_u(np.asarray(h, float) / self.ell, self.nu)

    def to_dict(self) -> dict:
        return {"nu": self.nu, "range_km": self.range_km}


def _matern_u(u, nu):
    u = np.abs(u)
    if nu == 0.5:
        return np.exp(-u)
    if nu == 1.5:
        x = math.sqrt(3.0) * u
        return (1.0 + x) * np.exp(-x)
    if nu == 2.5:
        x = math.sqrt(5.0) * u
        return (1.0 + x + x * x / 3.0) * np.exp(-x)
    raise ValueError("nu must be 0.5, 1.5 or 2.5")


def xyz_km(lat, lon) -> np.ndarray:
    """Earth-centred Cartesian coordinates (km); chord distance ≈ arc distance at these scales."""
    la = np.radians(np.asarray(lat, float))
    lo = np.radians(np.asarray(lon, float))
    return EARTH_R_KM * np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)


def maximin_order(lat, lon) -> np.ndarray:
    """Approximate maximin ordering by coarse-to-fine hierarchical gridding (O(n log n)).

    At each level the site nearest each occupied cell's centre is taken, so every prefix of the
    ordering is a quasi-uniform cover of the domain at a halving length scale.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    n = lat.size
    if n == 0:
        return np.zeros(0, np.int64)
    kx = 111.32 * math.cos(math.radians(float(lat.mean())))
    xy = np.stack([lon * kx, lat * 110.574], axis=1)
    lo = xy.min(axis=0)
    s = float((xy.max(axis=0) - lo).max()) + 1e-6
    picked = np.zeros(n, bool)
    parts = []
    while True:
        idx = np.nonzero(~picked)[0]
        if idx.size == 0:
            break
        cell = np.floor((xy[idx] - lo) / s).astype(np.int64)
        key = cell[:, 0] * 4_194_304 + cell[:, 1]
        d = ((xy[idx] - (lo + (cell + 0.5) * s)) ** 2).sum(axis=1)
        o = np.lexsort((d, key))
        first = np.ones(o.size, bool)
        first[1:] = key[o][1:] != key[o][:-1]
        sel = idx[o[first]]
        parts.append(sel)
        picked[sel] = True
        if s < 1e-3:  # co-located remainder
            rest = np.nonzero(~picked)[0]
            parts.append(rest)
            break
        s *= 0.5
    return np.concatenate(parts).astype(np.int64)


def ordered_neighbors(P: np.ndarray, order: np.ndarray, m: int) -> np.ndarray:
    """For each site, indices (site ids) of its ≤ m nearest sites that come earlier in ``order``.

    Returns an (n, m) int64 array padded with -1, sorted by distance.
    """
    n = P.shape[0]
    out = np.full((n, m), -1, np.int64)
    if n <= 1:
        return out
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    tree = cKDTree(P)
    k = int(min(n, 4 * m + 1))
    _, nbr = tree.query(P, k=k)
    nbr = np.atleast_2d(nbr)
    valid = rank[nbr] < rank[:, None]
    pos = np.argsort(~valid, axis=1, kind="stable")[:, :m]
    sel = np.take_along_axis(nbr, pos, axis=1)
    ok = np.take_along_axis(valid, pos, axis=1)
    out[:, : sel.shape[1]] = np.where(ok, sel, -1)
    need = ok.sum(axis=1) < np.minimum(rank, m)
    for j in np.nonzero(need)[0]:  # early (coarse) sites: brute force over all predecessors
        prev = order[: rank[j]]
        d = ((P[prev] - P[j]) ** 2).sum(axis=1)
        take = prev[np.argsort(d)[:m]]
        out[j] = -1
        out[j, : take.size] = take
    return out


@dataclass
class VecchiaField:
    """CSR Vecchia factor for one correlation model over a fixed set of sites."""

    model: Matern
    order: np.ndarray  # (n,) site ids in sampling order
    rank: np.ndarray  # (n,)
    ptr: np.ndarray  # (n+1,) per site id
    nbr: np.ndarray  # (nnz,) parent site ids
    coef: np.ndarray  # (nnz,) b coefficients
    cond_sd: np.ndarray  # (n,) d_j
    m: int

    @property
    def n(self) -> int:
        return int(self.order.size)

    def implied_covariance(self) -> np.ndarray:
        """Dense covariance of the Vecchia approximation, Σ = (I-B)⁻¹ D² (I-B)⁻ᵀ (small n only)."""
        n = self.n
        B = np.zeros((n, n))
        for j in range(n):
            B[j, self.nbr[self.ptr[j]:self.ptr[j + 1]]] = self.coef[self.ptr[j]:self.ptr[j + 1]]
        A = np.linalg.inv(np.eye(n) - B)
        return A @ np.diag(self.cond_sd**2) @ A.T


def build_vecchia(lat, lon, model: Matern, m: int = 16, prune: float = 1e-3, nugget: float = 1e-6,
                  order: np.ndarray | None = None, neighbors: np.ndarray | None = None,
                  chunk: int = 20000) -> VecchiaField:
    P = xyz_km(lat, lon)
    n = P.shape[0]
    order = maximin_order(lat, lon) if order is None else order
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    nb_all = ordered_neighbors(P, order, m) if neighbors is None else neighbors
    ptr = np.zeros(n + 1, np.int64)
    nbr_parts, coef_parts = [], []
    cond_sd = np.ones(n)
    for a in range(0, n, chunk):
        z = min(a + chunk, n)
        NB = nb_all[a:z]
        mask = NB >= 0
        idx = np.where(mask, NB, 0)
        Pj = P[a:z]
        Pn = P[idx]  # (c, m, 3)
        c = model.corr(np.sqrt(((Pn - Pj[:, None, :]) ** 2).sum(-1)))
        mask &= c >= prune
        c = np.where(mask, c, 0.0)
        C = model.corr(np.sqrt(((Pn[:, :, None, :] - Pn[:, None, :, :]) ** 2).sum(-1)))
        mm = mask[:, :, None] & mask[:, None, :]
        eye = np.eye(NB.shape[1])[None]
        C = np.where(mm, C, 0.0) + eye * np.where(mask[:, :, None], nugget, 1.0)
        b = np.linalg.solve(C, c[..., None])[..., 0]
        b = np.where(mask, b, 0.0)
        cond_sd[a:z] = np.sqrt(np.clip(1.0 - (b * c).sum(axis=1), 1e-6, 1.0))
        cnt = mask.sum(axis=1)
        ptr[a + 1:z + 1] = cnt
        nbr_parts.append(NB[mask])
        coef_parts.append(b[mask])
    ptr = np.cumsum(ptr)
    return VecchiaField(model, order, rank, ptr, np.concatenate(nbr_parts).astype(np.int64),
                        np.concatenate(coef_parts), cond_sd, m)


# ---------------------------------------------------------------------------- per-event closures
@nb.njit(cache=True)
def _closure_one(sites, ptr, nbr, rank, mark, buf):
    """Ancestor closure of ``sites`` (sorted by rank) written into buf; returns its length."""
    n = 0
    for s in sites:
        if not mark[s]:
            mark[s] = True
            buf[n] = s
            n += 1
    i = 0
    while i < n:
        j = buf[i]
        for t in range(ptr[j], ptr[j + 1]):
            p = nbr[t]
            if not mark[p]:
                mark[p] = True
                buf[n] = p
                n += 1
        i += 1
    for i in range(n):
        mark[buf[i]] = False
    keys = np.empty(n, np.int64)
    for i in range(n):
        keys[i] = rank[buf[i]]
    o = np.argsort(keys)
    tmp = buf[:n].copy()
    for i in range(n):
        buf[i] = tmp[o[i]]
    return n


@nb.njit(cache=True)
def event_closures(ev_ptr, pair_loc, ev_field, ptr2, nbr, rank2, n_loc):
    """CSR of per-event ancestor closures.  ``ev_field[e]`` selects the field (peril) or -1.

    ptr2: (n_fields, n_loc+1) offsets into ``nbr``; rank2: (n_fields, n_loc).
    """
    E = ev_ptr.shape[0] - 1
    sizes = np.zeros(E, np.int64)
    mark = np.zeros(n_loc, np.bool_)
    buf = np.empty(n_loc, np.int64)
    out = np.empty(max(16, 2 * pair_loc.shape[0]), np.int64)
    pos = 0
    for e in range(E):
        f = ev_field[e]
        a = ev_ptr[e]
        z = ev_ptr[e + 1]
        if f < 0 or z == a:
            continue
        k = _closure_one(pair_loc[a:z], ptr2[f], nbr, rank2[f], mark, buf)
        if pos + k > out.shape[0]:
            bigger = np.empty(max(2 * out.shape[0], pos + k), np.int64)
            bigger[:pos] = out[:pos]
            out = bigger
        out[pos:pos + k] = buf[:k]
        pos += k
        sizes[e] = k
    cptr = np.zeros(E + 1, np.int64)
    for e in range(E):
        cptr[e + 1] = cptr[e] + sizes[e]
    return cptr, out[:pos].copy()


@nb.njit(cache=True)
def sample_sites(keys, seed, order, ptr, nbr, coef, cond_sd):
    """Monte Carlo draws of Z at all sites for each key → (len(keys), n). (Validation helper.)"""
    n = order.shape[0]
    out = np.zeros((keys.shape[0], n))
    for k in range(keys.shape[0]):
        h = stream(np.uint64(seed), np.uint64(keys[k]))
        for r in range(n):
            j = order[r]
            s = 0.0
            for t in range(ptr[j], ptr[j + 1]):
                s += coef[t] * out[k, nbr[t]]
            out[k, j] = s + cond_sd[j] * norm_ppf_fast(uniform(h, TAG_W_BASE + j))
    return out


# ---------------------------------------------------------------------------- regular grids
def circulant_field(ny: int, nx: int, dy_km: float, dx_km: float, model: Matern, rng: np.random.Generator,
                    n_fields: int = 1) -> tuple[np.ndarray, float]:
    """Exact stationary GRF samples on a regular grid by circulant embedding.

    Returns (fields[n_fields, ny, nx], min_eigenvalue_ratio) — a negative ratio means the embedding
    was not non-negative definite and was clipped (it is ~0 for the ranges used here).
    """
    My, Mx = 2 * ny, 2 * nx
    iy = np.minimum(np.arange(My), My - np.arange(My)) * dy_km
    ix = np.minimum(np.arange(Mx), Mx - np.arange(Mx)) * dx_km
    c = model.corr(np.sqrt(iy[:, None] ** 2 + ix[None, :] ** 2))
    lam = np.fft.fft2(c).real
    ratio = float(lam.min() / lam.max())
    lam = np.clip(lam, 0.0, None)
    fields = []
    for _ in range((n_fields + 1) // 2):
        xi = rng.standard_normal((My, Mx)) + 1j * rng.standard_normal((My, Mx))
        Z = np.fft.fft2(np.sqrt(lam / (My * Mx)) * xi)
        fields.append(Z.real[:ny, :nx])
        fields.append(Z.imag[:ny, :nx])
    return np.stack(fields[:n_fields]), ratio


def von_karman_field(nz: int, nx: int, dz_km: float, dx_km: float, az_km: float, ax_km: float, hurst: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Anisotropic von Kármán random field (Mai & Beroza 2002) by spectral synthesis; unit variance."""
    kz = np.fft.fftfreq(nz, d=dz_km) * 2 * np.pi
    kx = np.fft.fftfreq(nx, d=dx_km) * 2 * np.pi
    k2 = (az_km * kz[:, None]) ** 2 + (ax_km * kx[None, :]) ** 2
    psd = ax_km * az_km / (1.0 + k2) ** (hurst + 1.0)
    noise = np.fft.fft2(rng.standard_normal((nz, nx)))
    f = np.fft.ifft2(noise * np.sqrt(psd)).real
    f -= f.mean()
    return f / max(f.std(), 1e-12)
