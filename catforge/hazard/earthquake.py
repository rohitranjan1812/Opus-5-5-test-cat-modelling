"""Earthquake hazard: fault & area sources, truncated Gutenberg-Richter recurrence, finite ruptures,
and a Boore & Atkinson (2008)-form ground-motion model for PGA with linear+nonlinear site terms.

Rupture size scaling: Wells & Coppersmith (1994) (crustal, all slip types) for length and down-dip
width, Strasser et al. (2010) for the subduction interface; width is capped by the seismogenic
thickness, W ≤ (z_bot − z_tor)/sin δ.  Each rupture is a (possibly dipping) planar surface whose
*surface projection polygon* is stored; the Joyner–Boore distance is the distance to that polygon
(zero for hanging-wall sites above the rupture), combined with a pseudo-depth h:
R = sqrt(Rjb² + h²).

Aleatory variability is split into inter-event τ (sampled per occurrence) and intra-event φ
(an exponential-correlation random field, Jayaram & Baker 2009, sampled by the loss engine).
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
# Intra-event residual field: Jayaram & Baker (2009) exponential model for PGA, practical range ~30 km
EQ_UNCERTAINTY = HazardUncertainty(sigma_between=TAU, sigma_within=PHI, rho_event=0.12, rho_cell=0.40,
                                   grf_nu=0.5, grf_range_km=30.0, dmg_rho_event=0.04, dmg_rho_cell=0.10)

CEUS_REGIONS = {"Central US", "South Carolina"}


def rupture_length_km(m, kind: str = "crustal"):
    m = np.asarray(m, float)
    if kind == "subduction":
        return 10 ** (-2.477 + 0.585 * m)
    return 10 ** (-3.22 + 0.69 * m)


def rupture_width_km(m, kind: str = "crustal"):
    m = np.asarray(m, float)
    if kind == "subduction":
        return 10 ** (-0.882 + 0.351 * m)
    return 10 ** (-1.01 + 0.32 * m)


def plane_polygon(tlat, tlon, dip: float, ztor: float, width: float):
    """Surface projection of a planar rupture below a trace (right-hand rule: dips to the right).

    Returns the closed ring (top edge in trace order, then bottom edge reversed) and the local dip
    azimuth at each trace vertex.
    """
    tlat = np.asarray(tlat, float)
    tlon = np.asarray(tlon, float)
    n = tlat.size
    brg = np.zeros(n)
    if n >= 2:
        from ..geo import initial_bearing

        seg = initial_bearing(tlat[:-1], tlon[:-1], tlat[1:], tlon[1:])
        brg[0], brg[-1] = seg[0], seg[-1]
        for i in range(1, n - 1):  # average of adjacent strikes (circular mean)
            a, b = np.radians(seg[i - 1]), np.radians(seg[i])
            brg[i] = np.degrees(np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b)))
    dipdir = (brg + 90.0) % 360.0
    if dip >= 89.9:
        top_la, top_lo, bot_la, bot_lo = tlat, tlon, tlat, tlon
    else:
        t = math.tan(math.radians(dip))
        top_la, top_lo = destination(tlat, tlon, dipdir, np.full(n, ztor / t))
        zb = ztor + width * math.sin(math.radians(dip))
        bot_la, bot_lo = destination(tlat, tlon, dipdir, np.full(n, zb / t))
    ring_la = np.concatenate([top_la, bot_la[::-1]])
    ring_lo = np.concatenate([top_lo, bot_lo[::-1]])
    return ring_la, ring_lo, dipdir


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
    rows = []
    geo = {"ptr": [0], "lat": [], "lon": [], "tptr": [0], "tlat": [], "tlon": []}

    def add(src, kind, region, m, rate, lats, lons, h, strike, length, dip, ztor, width, mech):
        rows.append({"source": src, "source_type": kind, "region": region, "mag": round(float(m), 3),
                     "rate": float(rate), "lat": float(np.mean(lats)), "lon": float(np.mean(lons)),
                     "strike": float(strike), "length_km": float(length), "depth_h": float(h),
                     "dip": float(dip), "ztor": float(ztor), "width_km": float(width), "mech": mech,
                     "anelastic": 0.3 if region in CEUS_REGIONS else (0.6 if kind == "subduction" else 1.0)})
        rla, rlo, _ = plane_polygon(lats, lons, dip, ztor, width)
        geo["lat"].extend(rla.tolist())
        geo["lon"].extend(rlo.tolist())
        geo["ptr"].append(geo["ptr"][-1] + rla.size)
        geo["tlat"].extend(lats)
        geo["tlon"].extend(lons)
        geo["tptr"].append(geo["tptr"][-1] + len(lats))

    for f in FAULTS:
        tr = np.array(f["trace"])
        line = Polyline(tr[:, 0], tr[:, 1])
        kind = "subduction" if f.get("mech") == "SUB" else "fault"
        dip, ztor, zbot = f.get("dip", 90.0), f.get("ztor", 0.0), f.get("zbot", 15.0)
        wmax = (zbot - ztor) / math.sin(math.radians(dip))
        mags, brates = gr_bin_rates(f["mmin"], f["mmax"], f["rate"], f["b"], fault_dm)
        for m, br in zip(mags, brates):
            L = float(min(rupture_length_km(m, kind), line.length_km))
            W = float(min(rupture_width_km(m, kind), wmax))
            span = line.length_km - L
            npos = 1 if span < 1.0 else int(np.clip(round(line.length_km / (0.5 * L)), 2, max_fault_positions))
            starts = [0.0] if npos == 1 else np.linspace(0, span, npos) + rng.uniform(-0.3, 0.3, npos) * span / npos
            for s0 in np.clip(starts, 0, max(span, 0)):
                lats, lons = _sub_trace(line, float(s0), float(s0) + L)
                strike = float(line.bearing_at(s0 + 0.5 * L))
                add(f["name"], kind, f["region"], m, br / npos, lats, lons, f["h"], strike, L, dip, ztor, W,
                    f.get("mech", "SS"))

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
            W = float(min(rupture_width_km(m), 15.0))
            ztor = max(0.0, 8.0 - 0.5 * W)
            for x, y in zip(ux, uy):
                clat, clon = la0 + y * (la1 - la0), lo0 + x * (lo1 - lo0)
                L = float(rupture_length_km(m)) if m >= 5.5 else 1.0
                strike = float(rng.uniform(0, 180))
                p_lat, p_lon = destination(np.array([clat, clat]), np.array([clon, clon]),
                                           np.array([strike, strike + 180.0]), np.array([L / 2, L / 2]))
                add(a["name"], "area", a["region"], m, br / npos, [float(p_lat[1]), float(p_lat[0])],
                    [float(p_lon[1]), float(p_lon[0])], a["h"], strike, L, 90.0, ztor, W, "SS")

    events = pd.DataFrame(rows)
    events.insert(0, "event_id", np.arange(1, len(events) + 1, dtype=np.int64) + 1_000_000)
    events["name"] = [f"EQ-{i - 1_000_000:05d} M{m:.1f} {s}" for i, m, s in
                      zip(events["event_id"], events["mag"], events["source"])]
    geometry = {k: np.asarray(v, np.int64 if k.endswith("ptr") else float) for k, v in geo.items()}
    return EventCatalog(
        peril="EQ", events=events, geometry=geometry, frequency=FrequencyModel(),
        uncertainty=EQ_UNCERTAINTY,
        intensity_unit="g (PGA)",
        meta={"seed": seed, "gmpe": "BA08-form (illustrative coefficients)", "n_faults": len(FAULTS),
              "n_areas": len(AREAS), "distance": "Rjb to rupture surface projection"},
    )


def single_rupture(lat: float, lon: float, mag: float, strike: float = 0.0, depth_h: float = 6.0,
                   length_km: float | None = None, region: str = "scenario", trace=None,
                   event_id: int = 1_000_001, name: str = "scenario", dip: float = 90.0, ztor: float = 0.0,
                   width_km: float | None = None, zbot: float = 15.0, mech: str = "SS",
                   hypo_depth_km: float | None = None, hypo_along: float | None = None) -> EventCatalog:
    """One-event EQ catalog. Either an explicit trace [(lon, lat), ...] or a centred straight rupture."""
    kind = "subduction" if mech == "SUB" else "crustal"
    if trace is not None:
        lons = [p[0] for p in trace]
        lats = [p[1] for p in trace]
        L = Polyline(np.array(lons), np.array(lats)).length_km
        strike = float(Polyline(np.array(lons), np.array(lats)).seg_bearing[0])
    else:
        L = float(length_km or (rupture_length_km(mag, kind) if mag >= 5.5 else 1.0))
        p_lat, p_lon = destination(np.array([lat, lat]), np.array([lon, lon]), np.array([strike, strike + 180.0]),
                                   np.array([L / 2, L / 2]))
        lats, lons = [float(p_lat[1]), float(p_lat[0])], [float(p_lon[1]), float(p_lon[0])]
    W = float(width_km or min(rupture_width_km(mag, kind), (zbot - ztor) / math.sin(math.radians(dip))))
    rla, rlo, _ = plane_polygon(lats, lons, dip, ztor, W)
    events = pd.DataFrame([{
        "event_id": event_id, "source": name, "source_type": "scenario", "region": region, "mag": mag,
        "rate": 1.0, "lat": float(np.mean(lats)), "lon": float(np.mean(lons)), "strike": strike, "length_km": L,
        "depth_h": depth_h, "dip": dip, "ztor": ztor, "width_km": W, "mech": mech,
        "hypo_depth_km": hypo_depth_km, "hypo_along": hypo_along,
        "anelastic": 0.3 if region in CEUS_REGIONS else (0.6 if mech == "SUB" else 1.0), "name": name}])
    geometry = {"ptr": np.array([0, rla.size], np.int64), "lat": rla, "lon": rlo,
                "tptr": np.array([0, len(lats)], np.int64), "tlat": np.asarray(lats, float),
                "tlon": np.asarray(lons, float)}
    return EventCatalog(peril="EQ", events=events, geometry=geometry, frequency=FrequencyModel(),
                        uncertainty=EQ_UNCERTAINTY, intensity_unit="g (PGA)",
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


@nb.njit(inline="always", cache=True)
def _ring_rjb(px, py, rlat, rlon, a, z, lat0, lon0):
    """Joyner–Boore distance: 0 inside the rupture's surface projection, else distance to its ring."""
    n = z - a
    if n == 1:
        ax, ay = local_xy_km(rlat[a], rlon[a], lat0, lon0)
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    dmin = 1e9
    inside = False
    for k in range(n):
        i0 = a + k
        i1 = a + (k + 1) % n
        ax, ay = local_xy_km(rlat[i0], rlon[i0], lat0, lon0)
        bx, by = local_xy_km(rlat[i1], rlon[i1], lat0, lon0)
        d = point_segment_dist(px, py, ax, ay, bx, by)
        if d < dmin:
            dmin = d
        if ((ay > py) != (by > py)) and (px < (bx - ax) * (py - ay) / (by - ay + 1e-300) + ax):
            inside = not inside
    return 0.0 if inside else dmin


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
                    dmin = _ring_rjb(px, py, rlat, rlon, a, z, lat0, lon0)
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
    """Surface-projection ring (closed) of the rupture plus its top trace."""
    g = cat.geometry
    a, z = int(g["ptr"][event_index]), int(g["ptr"][event_index + 1])
    lat = np.round(g["lat"][a:z], 4).tolist()
    lon = np.round(g["lon"][a:z], 4).tolist()
    out = {"lat": lat + lat[:1], "lon": lon + lon[:1]}
    if "tptr" in g:
        ta, tz = int(g["tptr"][event_index]), int(g["tptr"][event_index + 1])
        out["trace_lat"] = np.round(g["tlat"][ta:tz], 4).tolist()
        out["trace_lon"] = np.round(g["tlon"][ta:tz], 4).tolist()
    return out
