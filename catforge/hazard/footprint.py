"""Sparse event × site hazard footprints, hazard curves and hazard maps.

A footprint is stored as CSR: for catalog event e, sites ``site[ev_ptr[e]:ev_ptr[e+1]]`` receive a
median intensity ``exp(log_i[...])``.  Only pairs above a damage-relevant threshold are kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from ..data.coast import land_polygon
from ..geo import SpatialGrid, points_in_polygon
from . import earthquake as eqm
from . import tropical_cyclone as tcm
from .base import EventCatalog

MIN_INTENSITY = {"TC": 20.0, "EQ": 0.02}
DOMAINS = {
    "TC": [(-98.5, 24.0, -66.5, 46.0)],
    "EQ": [(-125.0, 31.5, -110.0, 49.5), (-92.5, 31.5, -78.5, 39.5)],
}


@dataclass
class Sites:
    lat: np.ndarray
    lon: np.ndarray
    terrain_k: np.ndarray  # TC gust factor
    vs30: np.ndarray  # EQ site condition

    @classmethod
    def plain(cls, lat, lon, terrain_k: float = 1.10, vs30: float = 400.0) -> Sites:
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        return cls(lat, lon, np.full(lat.shape, terrain_k), np.full(lat.shape, vs30))

    @property
    def n(self) -> int:
        return int(self.lat.shape[0])


@dataclass
class Pairs:
    ev_ptr: np.ndarray  # int64 (n_events + 1)
    site: np.ndarray  # int32
    log_i: np.ndarray  # float32

    @property
    def n_pairs(self) -> int:
        return int(self.site.shape[0])

    def event_slice(self, e: int) -> slice:
        return slice(int(self.ev_ptr[e]), int(self.ev_ptr[e + 1]))

    def counts(self) -> np.ndarray:
        return np.diff(self.ev_ptr)


def _dense(cat: EventCatalog, ev_idx: np.ndarray, sites: Sites, grid: SpatialGrid, min_intensity: float):
    if cat.peril == "TC":
        return tcm.tc_footprint_dense(ev_idx, *tcm.tc_kernel_args(cat), sites.lat, sites.lon, sites.terrain_k,
                                      *grid.as_tuple(), 15.0)
    if cat.peril == "EQ":
        return eqm.eq_footprint_dense(ev_idx, *eqm.eq_kernel_args(cat), sites.lat, sites.lon, sites.vs30,
                                      *grid.as_tuple(), float(np.log(min_intensity)))
    raise ValueError(f"unknown peril {cat.peril}")


def compute_pairs(cat: EventCatalog, sites: Sites, min_intensity: float | None = None,
                  progress=None) -> Pairs:
    """Evaluate the footprint of every catalog event at every site (sparse CSR output)."""
    thr = float(min_intensity if min_intensity is not None else MIN_INTENSITY[cat.peril])
    n_ev = cat.n_events
    if sites.n == 0:
        return Pairs(np.zeros(n_ev + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32))
    grid = SpatialGrid(sites.lat, sites.lon, cell_deg=0.5)
    batch = int(np.clip(3e7 / max(sites.n, 1), 8, 2048))
    counts = np.zeros(n_ev, np.int64)
    site_parts, val_parts = [], []
    for b0 in range(0, n_ev, batch):
        idx = np.arange(b0, min(b0 + batch, n_ev), dtype=np.int64)
        dense = _dense(cat, idx, sites, grid, thr)
        rows, cols = np.nonzero(dense >= thr)
        counts[idx] = np.bincount(rows, minlength=idx.size)
        site_parts.append(cols.astype(np.int32))
        val_parts.append(np.log(dense[rows, cols]).astype(np.float32))
        if progress is not None:
            progress(min(b0 + batch, n_ev) / n_ev)
    ev_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return Pairs(ev_ptr, np.concatenate(site_parts), np.concatenate(val_parts))


def event_footprint_grid(cat: EventCatalog, event_index: int, res_deg: float = 0.05, pad_deg: float = 3.0,
                         max_cells: int = 60000) -> dict:
    """Median-intensity footprint of one event on a regular grid (for maps)."""
    g = cat.geometry
    a, z = int(g["ptr"][event_index]), int(g["ptr"][event_index + 1])
    lat, lon = g["lat"][a:z], g["lon"][a:z]
    if cat.peril == "TC":
        keep = g["vmax"][a:z] >= 17.0
        lat, lon = (lat[keep], lon[keep]) if keep.any() else (lat, lon)
    la0, la1 = float(lat.min()) - pad_deg, float(lat.max()) + pad_deg
    lo0, lo1 = float(lon.min()) - pad_deg, float(lon.max()) + pad_deg
    while ((la1 - la0) / res_deg) * ((lo1 - lo0) / res_deg) > max_cells:
        res_deg *= 1.25
    glat = np.arange(la0, la1 + 1e-9, res_deg)
    glon = np.arange(lo0, lo1 + 1e-9, res_deg)
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    k_tc = tcm.TERRAIN_GUST_FACTOR["open"]
    sites = Sites.plain(LA.ravel(), LO.ravel(), terrain_k=k_tc, vs30=400.0)
    grid = SpatialGrid(sites.lat, sites.lon, cell_deg=0.5)
    dense = _dense(cat, np.array([event_index], np.int64), sites, grid, MIN_INTENSITY[cat.peril] * 0.5)[0]
    return {"lat0": la0, "lon0": lo0, "res_deg": res_deg, "ny": glat.size, "nx": glon.size,
            "values": np.round(dense.reshape(glat.size, glon.size), 4 if cat.peril == "EQ" else 2).tolist(),
            "unit": cat.intensity_unit}


def exceedance_rate(pairs: Pairs, rates: np.ndarray, sigma: float, n_sites: int, x: np.ndarray) -> np.ndarray:
    """Annual exceedance rate λ(x) at each site for thresholds x → (n_sites, len(x))."""
    ev_of_pair = np.repeat(np.arange(len(rates)), np.diff(pairs.ev_ptr))
    lam_pair = rates[ev_of_pair]
    out = np.zeros((n_sites, x.size))
    lx = np.log(x)
    for j, l in enumerate(lx):
        p = 1.0 - ndtr((l - pairs.log_i.astype(float)) / sigma)
        out[:, j] = np.bincount(pairs.site, weights=lam_pair * p, minlength=n_sites)
    return out


def hazard_curve(cat: EventCatalog, lat: float, lon: float, terrain_k: float = 1.10, vs30: float = 400.0,
                 n_points: int = 60) -> dict:
    """Site hazard curve: intensity vs annual exceedance probability, with lognormal aleatory σ."""
    sites = Sites(np.array([lat]), np.array([lon]), np.array([terrain_k]), np.array([vs30]))
    thr = MIN_INTENSITY[cat.peril] * 0.4
    pairs = compute_pairs(cat, sites, min_intensity=thr)
    sig = float(np.hypot(cat.uncertainty.sigma_between, cat.uncertainty.sigma_within))
    x = np.geomspace(20.0, 110.0, n_points) if cat.peril == "TC" else np.geomspace(0.02, 2.5, n_points)
    lam = exceedance_rate(pairs, cat.rates, sig, 1, x)[0]
    aep = 1.0 - np.exp(-lam)
    rps = [10, 25, 50, 100, 250, 500, 1000]
    rp_int = {}
    for rp in rps:
        target = -np.log1p(-1.0 / rp)
        if lam[0] < target:
            rp_int[str(rp)] = None
        else:
            rp_int[str(rp)] = float(np.exp(np.interp(-np.log(target), -np.log(np.maximum(lam, 1e-300)), np.log(x))))
    return {"peril": cat.peril, "unit": cat.intensity_unit, "intensity": x.round(4).tolist(),
            "annual_exceedance_prob": aep.tolist(), "rate": lam.tolist(), "return_period_intensity": rp_int,
            "n_events_contributing": int(pairs.n_pairs)}


def hazard_map(cat: EventCatalog, return_period: float = 100.0, res_deg: float = 0.25) -> dict:
    """Return-period intensity on a land-masked grid across the peril's domains."""
    lat_all, lon_all = [], []
    for lo0, la0, lo1, la1 in DOMAINS[cat.peril]:
        glat = np.arange(la0, la1 + 1e-9, res_deg)
        glon = np.arange(lo0, lo1 + 1e-9, res_deg)
        LA, LO = np.meshgrid(glat, glon, indexing="ij")
        lat_all.append(LA.ravel())
        lon_all.append(LO.ravel())
    lat = np.concatenate(lat_all)
    lon = np.concatenate(lon_all)
    if cat.peril == "TC":
        plon, plat = land_polygon()
        m = points_in_polygon(lon, lat, plon, plat)
        lat, lon = lat[m], lon[m]
    sites = Sites.plain(lat, lon, terrain_k=tcm.TERRAIN_GUST_FACTOR["open"], vs30=400.0)
    pairs = compute_pairs(cat, sites, min_intensity=MIN_INTENSITY[cat.peril] * 0.5)
    sig = float(np.hypot(cat.uncertainty.sigma_between, cat.uncertainty.sigma_within))
    x = np.geomspace(15.0, 120.0, 80) if cat.peril == "TC" else np.geomspace(0.01, 3.0, 80)
    lam = exceedance_rate(pairs, cat.rates, sig, sites.n, x)
    target = -np.log1p(-1.0 / return_period)
    val = np.full(sites.n, np.nan)
    for i in range(sites.n):
        li = lam[i]
        if li[0] < target:
            continue
        val[i] = float(np.exp(np.interp(-np.log(target), -np.log(np.maximum(li, 1e-300)), np.log(x))))
    ok = np.isfinite(val)
    return {"peril": cat.peril, "return_period": return_period, "res_deg": res_deg, "unit": cat.intensity_unit,
            "lat": lat[ok].round(3).tolist(), "lon": lon[ok].round(3).tolist(),
            "value": val[ok].round(4).tolist()}
