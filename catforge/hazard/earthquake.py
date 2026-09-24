"""Earthquake hazard: fault & area sources, truncated Gutenberg-Richter recurrence, finite ruptures,
and a Boore & Atkinson (2008)-form ground-motion model for PGA with linear+nonlinear site terms.

Rupture length scaling: Wells & Coppersmith (1994) (crustal, all slip types),
Strasser et al. (2010) (subduction interface).  Distances: Joyner-Boore distance to the rupture
trace, combined with a source pseudo-depth h:  R = sqrt(Rjb² + h²).

Aleatory variability is split into inter-event τ (sampled per occurrence) and intra-event φ
(folded into vulnerability and correlated spatially through the loss-engine copula).
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
import pandas as pd

from ..data.seismic_sources import AREAS, FAULTS
from ..geo import DEG2RAD, Polyline, destination, local_xy_km, point_segment_dist
from .base import EventCatalog, FrequencyModel, HazardUncertainty

# BA08-form coefficients for PGA (illustrative; see module docstring)
E1, E5, E6, E7, MH = -0.53804, 0.28805, -0.10164, 0.0, 6.75
C1, C2, C3, MREF, RREF = -0.66050, 0.11970, -0.01151, 4.5, 1.0
BLIN, B1, B2, V1, V2, VREF = -0.360, -0.640, -0.14, 180.0, 300.0, 760.0
TAU, PHI = 0.26, 0.50

CEUS_REGIONS = {"Central US", "South Carolina"}


def rupture_length_km(m, kind: str = "crustal"):
    m = np.asarray(m, float)
    if kind == "subduction":
        return 10 ** (-2.477 + 0.585 * m)
    return 10 ** (-3.22 + 0.69 * m)


def gr_bin_rates(mmin, mmax, rate, b, dm):
    """Truncated Gutenberg-Richter: (bin centres, annual rate per bin)."""
    edges = np.arange(mmin, mmax + 1e-9, dm)
    if edges[-1] < mmax - 1e-9:
        edges = np.append(edges, mmax)
    beta = b * math.log(10.0)
    denom = 1.0 - math.exp(-beta * (mmax - mmin))
    ncum = rate * (np.exp(-beta * (edges - mmin)) - math.exp(-beta * (mmax - mmin))) / denom
    return 0.5 * (edges[:-1] + edges[1:]), ncum[:-1] - ncum[1:]


@nb.njit(inline="always", cache=True)
def ln_pga_median(m, rjb, h, vs30, anelastic):
    """ln(PGA [g]) median, BA08 functional form with nonlinear site amplification."""
    r = math.sqrt(rjb * rjb + h * h)
    if m <= MH:
        fm = E1 + E5 * (m - MH) + E6 * (m - MH) ** 2
    else:
        fm = E1 + E7 * (m - MH)
    fd = (C1 + C2 * (m - MREF)) * math.log(r / RREF) + anelastic * C3 * (r - RREF)
    ln_rock = fm + fd
    flin = BLIN * math.log(vs30 / VREF)
    if vs30 <= V1:
        bnl = B1
    elif vs30 <= V2:
        bnl = (B1 - B2) * math.log(vs30 / V2) / math.log(V1 / V2) + B2
    elif vs30 < VREF:
        bnl = B2 * math.log(vs30 / VREF) / math.log(V2 / VREF)
    else:
        bnl = 0.0
    pga_rock = math.exp(ln_rock)
    fnl = bnl * math.log(max(pga_rock, 0.1) / 0.1)
    return ln_rock + flin + fnl


def _sub_trace(line: Polyline, s0: float, s1: float):
    """Vertices of the polyline between arc lengths s0 < s1."""
    inner = [(la, lo) for la, lo, c in zip(line.lat, line.lon, line.cum_km) if s0 < c < s1]
    la0, lo0 = line.point_at(s0)
    la1, lo1 = line.point_at(s1)
    pts = [(float(la0), float(lo0))] + inner + [(float(la1), float(lo1))]
    return [p[0] for p in pts], [p[1] for p in pts]


def generate_eq_catalog(seed: int = 7, fault_dm: float = 0.1, area_dm: float = 0.2, max_fault_positions: int = 14,
                        area_position_density: float = 1.0) -> EventCatalog:
    rng = np.random.default_rng(seed)
    rows, rup_lat, rup_lon, ptr = [], [], [], [0]

    def add(src, kind, region, m, rate, lats, lons, h, strike, length):
        rows.append({"source": src, "source_type": kind, "region": region, "mag": round(float(m), 3),
                     "rate": float(rate), "lat": float(np.mean(lats)), "lon": float(np.mean(lons)),
                     "strike": float(strike), "length_km": float(length), "depth_h": float(h),
                     "anelastic": 0.3 if region in CEUS_REGIONS else (0.6 if kind == "subduction" else 1.0)})
        rup_lat.extend(lats)
        rup_lon.extend(lons)
        ptr.append(ptr[-1] + len(lats))

    for f in FAULTS:
        tr = np.array(f["trace"])
        line = Polyline(tr[:, 0], tr[:, 1])
        kind = "subduction" if "Subduction" in f["name"] else "fault"
        mags, brates = gr_bin_rates(f["mmin"], f["mmax"], f["rate"], f["b"], fault_dm)
        for m, br in zip(mags, brates):
            L = float(min(rupture_length_km(m, kind), line.length_km))
            span = line.length_km - L
            npos = 1 if span < 1.0 else int(np.clip(round(line.length_km / (0.5 * L)), 2, max_fault_positions))
            starts = [0.0] if npos == 1 else np.linspace(0, span, npos) + rng.uniform(-0.3, 0.3, npos) * span / npos
            for s0 in np.clip(starts, 0, max(span, 0)):
                lats, lons = _sub_trace(line, float(s0), float(s0) + L)
                strike = float(line.bearing_at(s0 + 0.5 * L))
                add(f["name"], kind, f["region"], m, br / npos, lats, lons, f["h"], strike, L)

    for a in AREAS:
        lo0, la0, lo1, la1 = a["bbox"]
        area_km2 = (lo1 - lo0) * 111.32 * math.cos(math.radians(0.5 * (la0 + la1))) * (la1 - la0) * 110.57
        mags, brates = gr_bin_rates(a["mmin"], a["mmax"], a["rate"], a["b"], area_dm)
        for m, br in zip(mags, brates):
            r_dmg = 10 ** (0.5 * m - 1.5)
            npos = int(np.clip(area_position_density * area_km2 / (math.pi * r_dmg**2), 6, 60))
            # stratified (Latin-hypercube) epicentres
            ux = (rng.permutation(npos) + rng.random(npos)) / npos
            uy = (rng.permutation(npos) + rng.random(npos)) / npos
            for x, y in zip(ux, uy):
                clat, clon = la0 + y * (la1 - la0), lo0 + x * (lo1 - lo0)
                L = float(rupture_length_km(m)) if m >= 5.5 else 1.0
                strike = float(rng.uniform(0, 180))
                p_lat, p_lon = destination(np.array([clat, clat]), np.array([clon, clon]),
                                           np.array([strike, strike + 180.0]), np.array([L / 2, L / 2]))
                add(a["name"], "area", a["region"], m, br / npos, [float(p_lat[1]), float(p_lat[0])],
                    [float(p_lon[1]), float(p_lon[0])], a["h"], strike, L)

    events = pd.DataFrame(rows)
    events.insert(0, "event_id", np.arange(1, len(events) + 1, dtype=np.int64) + 1_000_000)
    events["name"] = [f"EQ-{i - 1_000_000:05d} M{m:.1f} {s}" for i, m, s in
                      zip(events["event_id"], events["mag"], events["source"])]
    geometry = {"ptr": np.asarray(ptr, np.int64), "lat": np.asarray(rup_lat, float), "lon": np.asarray(rup_lon, float)}
    return EventCatalog(
        peril="EQ", events=events, geometry=geometry, frequency=FrequencyModel(),
        uncertainty=HazardUncertainty(sigma_between=TAU, sigma_within=PHI, rho_event=0.12, rho_cell=0.40),
        intensity_unit="g (PGA)",
        meta={"seed": seed, "gmpe": "BA08-form (illustrative coefficients)", "n_faults": len(FAULTS),
              "n_areas": len(AREAS)},
    )


def single_rupture(lat: float, lon: float, mag: float, strike: float = 0.0, depth_h: float = 6.0,
                   length_km: float | None = None, region: str = "scenario", trace=None,
                   event_id: int = 1_000_001, name: str = "scenario") -> EventCatalog:
    """One-event EQ catalog. Either an explicit trace [(lon, lat), ...] or a centred straight rupture."""
    if trace is not None:
        lons = [p[0] for p in trace]
        lats = [p[1] for p in trace]
        L = Polyline(np.array(lons), np.array(lats)).length_km
    else:
        L = float(length_km or (rupture_length_km(mag) if mag >= 5.5 else 1.0))
        p_lat, p_lon = destination(np.array([lat, lat]), np.array([lon, lon]), np.array([strike, strike + 180.0]),
                                   np.array([L / 2, L / 2]))
        lats, lons = [float(p_lat[1]), float(p_lat[0])], [float(p_lon[1]), float(p_lon[0])]
    events = pd.DataFrame([{
        "event_id": event_id, "source": name, "source_type": "scenario", "region": region, "mag": mag,
        "rate": 1.0, "lat": float(np.mean(lats)), "lon": float(np.mean(lons)), "strike": strike, "length_km": L,
        "depth_h": depth_h, "anelastic": 0.3 if region in CEUS_REGIONS else 1.0, "name": name}])
    geometry = {"ptr": np.array([0, len(lats)], np.int64), "lat": np.asarray(lats, float),
                "lon": np.asarray(lons, float)}
    return EventCatalog(peril="EQ", events=events, geometry=geometry, frequency=FrequencyModel(),
                        uncertainty=HazardUncertainty(TAU, PHI, 0.12, 0.40), intensity_unit="g (PGA)",
                        meta={"scenario": True})


@nb.njit(cache=True)
def _cutoff_km(m, h, anelastic, ln_min):
    """Largest distance at which the soft-soil (Vs30=250) median exceeds exp(ln_min)."""
    r = 1.0
    while r < 800.0:
        if ln_pga_median(m, r, h, 250.0, anelastic) < ln_min:
            return r
        r += 5.0
    return 800.0


@nb.njit(parallel=True, cache=True)
def eq_footprint_dense(ev_idx, ptr, rlat, rlon, mag, depth, anel, slat, slon, svs30,
                       glat0, glon0, gdeg, gny, gnx, cell_ptr, cell_items, ln_min):
    """Median PGA (g) at every site for each event in ``ev_idx`` → dense (len(ev_idx), n_sites)."""
    n_ev = ev_idx.shape[0]
    out = np.zeros((n_ev, slat.shape[0]), dtype=np.float32)
    for q in nb.prange(n_ev):
        e = ev_idx[q]
        a = ptr[e]
        z = ptr[e + 1]
        m = mag[e]
        rcut = _cutoff_km(m, depth[e], anel[e], ln_min)
        la_min, la_max, lo_min, lo_max = 90.0, -90.0, 180.0, -180.0
        for k in range(a, z):
            la_min = min(la_min, rlat[k])
            la_max = max(la_max, rlat[k])
            lo_min = min(lo_min, rlon[k])
            lo_max = max(lo_max, rlon[k])
        lat0 = 0.5 * (la_min + la_max)
        lon0 = 0.5 * (lo_min + lo_max)
        dlat = rcut / 110.574
        dlon = rcut / (111.32 * max(math.cos(lat0 * DEG2RAD), 0.2))
        iy0 = max(int(math.floor((la_min - dlat - glat0) / gdeg)), 0)
        iy1 = min(int(math.floor((la_max + dlat - glat0) / gdeg)), gny - 1)
        ix0 = max(int(math.floor((lo_min - dlon - glon0) / gdeg)), 0)
        ix1 = min(int(math.floor((lo_max + dlon - glon0) / gdeg)), gnx - 1)
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                c = iy * gnx + ix
                for p in range(cell_ptr[c], cell_ptr[c + 1]):
                    s = cell_items[p]
                    px, py = local_xy_km(slat[s], slon[s], lat0, lon0)
                    dmin = 1e9
                    if z - a == 1:
                        ax, ay = local_xy_km(rlat[a], rlon[a], lat0, lon0)
                        dmin = math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
                    for k in range(a, z - 1):
                        ax, ay = local_xy_km(rlat[k], rlon[k], lat0, lon0)
                        bx, by = local_xy_km(rlat[k + 1], rlon[k + 1], lat0, lon0)
                        d = point_segment_dist(px, py, ax, ay, bx, by)
                        if d < dmin:
                            dmin = d
                    if dmin > rcut:
                        continue
                    lp = ln_pga_median(m, dmin, depth[e], svs30[s], anel[e])
                    if lp >= ln_min:
                        out[q, s] = math.exp(lp)
    return out


def eq_kernel_args(cat: EventCatalog):
    g = cat.geometry
    ev = cat.events
    return (g["ptr"], g["lat"], g["lon"], ev["mag"].to_numpy(float), ev["depth_h"].to_numpy(float),
            ev["anelastic"].to_numpy(float))


def rupture_of(cat: EventCatalog, event_index: int) -> dict:
    g = cat.geometry
    a, z = int(g["ptr"][event_index]), int(g["ptr"][event_index + 1])
    return {"lat": np.round(g["lat"][a:z], 4).tolist(), "lon": np.round(g["lon"][a:z], 4).tolist()}
