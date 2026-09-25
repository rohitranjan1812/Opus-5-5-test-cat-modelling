"""Time-resolved hurricane physics for event development: wind, pressure, storm surge, rainfall.

Wind / pressure
    The same Holland/B_s surface field used by the loss engine (``hazard.tropical_cyclone.wind_uv``)
    evaluated at arbitrary times by linear interpolation of the hourly track state.

Storm surge — bathystrophic storm tide (Freeman, Baer & Jung 1957; Bodine 1971)
    For every coastal point a shore-normal transect crosses the continental shelf (width W to the
    100 m isobath) with depth h(d) = 2 + 98 (d/W)^p m at distance d from the coast; p > 1 describes
    the broad, shallow inner shelves of the Gulf coast, p ≈ 1 the Atlantic.  With x landward, q the
    alongshore transport (m²/s) along the coast tangent t̂ and n̂ the landward normal:

        ∂q/∂t      = τ_t/ρ_w − C_f q|q|/D²                          (alongshore momentum)
        ∂η_eq/∂x   = (τ_n/ρ_w − f q) / (g D),   D = h + η            (cross-shore balance)
        η(x=shelf) = Δp(x)/(ρ_w g)                                   (inverse barometer)
        ∂η/∂t      = (η_eq − η)/T_adj,   T_adj = W/√(g h̄)             (shelf adjustment)
        η ≥ −h + 0.3 m                                               (dry-bed limit on set-down)

    Wind stress τ = ρ_a C_d |U| U with C_d = (0.75 + 0.067 U)·10⁻³ (Garratt 1977) capped at
    2.5·10⁻³ (Powell, Vickery & Reinhold 2003).  Astronomical tide and wave set-up are omitted.

Rainfall — R-CLIPER (Tuleya, DeMaria & Kuligowski 2007)
    Axisymmetric rain-rate profile T(r) = T₀ + (T_m − T₀) r/r_m (r < r_m), T_m e^{−(r−r_m)/r_e}
    (r ≥ r_m), with T₀, T_m, r_m, r_e linear in U = 1 + (V_max[kt] − 35)/33, accumulated in time.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ..data.coast import COASTLINE
from ..geo import DEG2RAD, Polyline, destination
from ..hazard.tropical_cyclone import pressure_deficit, wind_uv

RHO_W = 1025.0
RHO_A = 1.15
G = 9.81
CF = 0.003
OMEGA = 7.292e-5

# continental shelf by coastal region: (width km coast → ~100 m isobath, profile exponent p) —
# an approximate bathymetry parameterisation (broad shallow Gulf shelves: p > 1)
SHELF = {"TX_S": (60.0, 1.2), "TX_N": (110.0, 1.5), "LA": (90.0, 1.6), "MS": (110.0, 1.8), "AL": (70.0, 1.5),
         "FL_NW": (50.0, 1.2), "FL_BB": (180.0, 1.8), "FL_SW": (170.0, 1.6), "FL_SE": (6.0, 1.0),
         "FL_E": (50.0, 1.0), "GA": (110.0, 1.3), "SC": (90.0, 1.2), "NC": (45.0, 1.0), "MA": (90.0, 1.1),
         "NJ": (120.0, 1.2), "NY": (150.0, 1.3), "NE": (120.0, 1.0), "ME": (80.0, 0.8)}

# R-CLIPER coefficients (rain rate in inches/day, radii in km)
RC_A1, RC_B1, RC_A2, RC_B2, RC_A3, RC_B3, RC_A4, RC_B4 = -1.10, 3.96, -1.60, 4.80, 64.5, -13.0, 150.0, -16.0
MS_TO_KT = 1.943844


# ---------------------------------------------------------------------------------- track state
@nb.njit(inline="always", cache=True)
def track_state(t, tt, tlat, tlon, tdp, trm, tb, thdg, tvt, tvmax):
    """Linear interpolation of the hourly track state at time t (hours)."""
    n = tt.shape[0]
    if t <= tt[0]:
        i, w = 0, 0.0
    elif t >= tt[n - 1]:
        i, w = n - 2, 1.0
    else:
        i = int(np.searchsorted(tt, t)) - 1
        i = min(max(i, 0), n - 2)
        w = (t - tt[i]) / max(tt[i + 1] - tt[i], 1e-9)
    dh = ((thdg[i + 1] - thdg[i] + 540.0) % 360.0) - 180.0
    return (tlat[i] + w * (tlat[i + 1] - tlat[i]), tlon[i] + w * (tlon[i + 1] - tlon[i]),
            tdp[i] + w * (tdp[i + 1] - tdp[i]), trm[i] + w * (trm[i + 1] - trm[i]),
            tb[i] + w * (tb[i + 1] - tb[i]), thdg[i] + w * dh, tvt[i] + w * (tvt[i + 1] - tvt[i]),
            tvmax[i] + w * (tvmax[i + 1] - tvmax[i]))


def track_arrays(cat, event_index: int = 0):
    g = cat.geometry
    a, z = int(g["ptr"][event_index]), int(g["ptr"][event_index + 1])
    f = np.float64
    return (g["t"][a:z].astype(f), g["lat"][a:z].astype(f), g["lon"][a:z].astype(f), g["dp"][a:z].astype(f),
            g["rmax"][a:z].astype(f), g["b"][a:z].astype(f), g["heading"][a:z].astype(f), g["vt"][a:z].astype(f),
            g["vmax"][a:z].astype(f))


# ---------------------------------------------------------------------------------- site time series
@nb.njit(parallel=True, cache=True)
def site_gust_series(times, tr, slat, slon, skfac, smult):
    """3-s gust (m/s) at each site and time: K_terrain · |V| · exp(η + σ_w W(s)) → (n_sites, n_t)."""
    tt, tlat, tlon, tdp, trm, tb, thdg, tvt, tvmax = tr
    ns = slat.shape[0]
    nt = times.shape[0]
    out = np.zeros((ns, nt))
    for k in nb.prange(nt):
        clat, clon, dp, rm, bh, hdg, vt, _ = track_state(times[k], tt, tlat, tlon, tdp, trm, tb, thdg, tvt, tvmax)
        for j in range(ns):
            u, v = wind_uv(slat[j], slon[j], clat, clon, dp, rm, bh, hdg, vt)
            out[j, k] = skfac[j] * smult[j] * math.sqrt(u * u + v * v)
    return out


# ---------------------------------------------------------------------------------- storm surge
def coast_points(spacing_km: float = 5.0):
    """Coastline resampled every ``spacing_km`` with landward normal and shelf width."""
    lon = np.array([c[0] for c in COASTLINE])
    lat = np.array([c[1] for c in COASTLINE])
    reg = np.array([c[2] for c in COASTLINE])
    line = Polyline(lon, lat)
    s = np.arange(0.0, line.length_km, spacing_km)
    plat, plon = line.point_at(s)
    seg, _ = line.locate(s)
    normal = (line.seg_bearing[seg] - 90.0) % 360.0
    region = reg[seg]
    shelf = np.array([SHELF.get(r, (80.0, 1.2))[0] for r in region])
    pexp = np.array([SHELF.get(r, (80.0, 1.2))[1] for r in region])
    return plat, plon, normal, region, shelf, pexp


@nb.njit(inline="always", cache=True)
def _drag(u):
    return min((0.75 + 0.067 * u) * 1e-3, 2.5e-3)


@nb.njit(parallel=True, cache=True)
def bathystrophic(times, tr, plat, plon, normal, shelf_km, pexp, nx):
    """Storm tide at the coast for each coastal point and time → (surge, ib, wind, coriolis) (n_pts, n_t)."""
    tt, tlat, tlon, tdp, trm, tb, thdg, tvt, tvmax = tr
    npnt = plat.shape[0]
    nt = times.shape[0]
    surge = np.zeros((npnt, nt))
    ib_out = np.zeros((npnt, nt))
    wind_out = np.zeros((npnt, nt))
    cor_out = np.zeros((npnt, nt))
    for i in nb.prange(npnt):
        W = shelf_km[i]
        nrm = normal[i] * DEG2RAD
        nx_, ny_ = math.sin(nrm), math.cos(nrm)  # landward unit vector (east, north)
        tx_, ty_ = -ny_, nx_  # alongshore unit vector (land on its left)
        f = 2.0 * OMEGA * math.sin(plat[i] * DEG2RAD)
        dx = W * 1000.0 / nx
        # transect points from the shelf edge (k=0) to the coast (k=nx)
        tlat_ = np.empty(nx + 1)
        tlon_ = np.empty(nx + 1)
        h = np.empty(nx + 1)
        for k in range(nx + 1):
            d = W * (1.0 - k / nx)  # distance from coast (km), seaward along −n
            tlat_[k] = plat[i] - d * ny_ / 110.574
            tlon_[k] = plon[i] - d * nx_ / (111.32 * math.cos(plat[i] * DEG2RAD))
            h[k] = 2.0 + 98.0 * (d / W) ** pexp[i]
        hbar = 0.0
        for k in range(nx + 1):
            hbar += h[k]
        hbar /= nx + 1
        tadj = W * 1000.0 / math.sqrt(G * hbar)
        q = np.zeros(nx + 1)
        eta = np.zeros(nx + 1)
        for s in range(nt):
            dt = (times[s] - times[s - 1]) * 3600.0 if s > 0 else 0.0
            clat, clon, dp, rm, bh, hdg, vt, _ = track_state(times[s], tt, tlat, tlon, tdp, trm, tb, thdg, tvt,
                                                              tvmax)
            tn = np.empty(nx + 1)
            ib = np.empty(nx + 1)
            for k in range(nx + 1):
                u, v = wind_uv(tlat_[k], tlon_[k], clat, clon, dp, rm, bh, hdg, vt)
                sp = math.sqrt(u * u + v * v)
                cd = _drag(sp)
                tauu = RHO_A * cd * sp * u
                tauv = RHO_A * cd * sp * v
                tn[k] = tauu * nx_ + tauv * ny_
                tt_ = tauu * tx_ + tauv * ty_
                D = max(h[k] + eta[k], 0.5)
                if dt > 0:
                    q[k] = (q[k] + dt * tt_ / RHO_W) / (1.0 + dt * CF * abs(q[k]) / (D * D))
                ib[k] = pressure_deficit(tlat_[k], tlon_[k], clat, clon, dp, rm, bh) / (RHO_W * G)
            # cross-shore equilibrium, integrated from the shelf edge landward
            e_w = 0.0
            e_c = 0.0
            eq_prev = ib[0]
            for k in range(1, nx + 1):
                D = max(h[k] + eta[k], 0.5)
                e_w += 0.5 * (tn[k] + tn[k - 1]) / RHO_W / (G * D) * dx
                e_c += -f * 0.5 * (q[k] + q[k - 1]) / (G * D) * dx
                eq_k = max(ib[k] + e_w + e_c, -h[k] + 0.3)
                a = 1.0 - math.exp(-dt / tadj) if dt > 0 else 1.0
                eta[k] = eta[k] + (eq_k - eta[k]) * a
                eq_prev = eq_k
            eta[0] = ib[0]
            surge[i, s] = eta[nx]
            ib_out[i, s] = ib[nx]
            wind_out[i, s] = e_w
            cor_out[i, s] = e_c
            _ = eq_prev
    return surge, ib_out, wind_out, cor_out


# ---------------------------------------------------------------------------------- rainfall
@nb.njit(inline="always", cache=True)
def rclipper_mm_h(r_km, vmax_ms):
    U = 1.0 + (vmax_ms * MS_TO_KT - 35.0) / 33.0
    t0 = max(RC_A1 + RC_B1 * U, 0.0)
    tm = max(RC_A2 + RC_B2 * U, 0.0)
    rm = max(RC_A3 + RC_B3 * U, 5.0)
    re = max(RC_A4 + RC_B4 * U, 20.0)
    if r_km < rm:
        rate = t0 + (tm - t0) * r_km / rm
    else:
        rate = tm * math.exp(-(r_km - rm) / re)
    return rate * 25.4 / 24.0  # in/day → mm/h


@nb.njit(parallel=True, cache=True)
def rain_accumulation(times, tr, glat, glon, frame_idx):
    """Accumulated rainfall (mm) on grid points at the requested time indices → (n_frames, n_pts)."""
    tt, tlat, tlon, tdp, trm, tb, thdg, tvt, tvmax = tr
    npnt = glat.shape[0]
    nf = frame_idx.shape[0]
    out = np.zeros((nf, npnt))
    for p in nb.prange(npnt):
        acc = 0.0
        fi = 0
        for s in range(times.shape[0]):
            if s > 0:
                clat, clon, _, _, _, _, _, vmax = track_state(0.5 * (times[s] + times[s - 1]), tt, tlat, tlon, tdp,
                                                              trm, tb, thdg, tvt, tvmax)
                x = (glon[p] - clon) * 111.32 * math.cos(0.5 * (glat[p] + clat) * DEG2RAD)
                y = (glat[p] - clat) * 110.574
                r = math.sqrt(x * x + y * y)
                if r < 600.0:
                    acc += rclipper_mm_h(r, vmax) * (times[s] - times[s - 1])
            while fi < nf and frame_idx[fi] == s:
                out[fi, p] = acc
                fi += 1
    return out


def shore_offsets(plat, plon, normal, dist_km):
    """Points ``dist_km`` inland from each coastal point (for rendering surge 'walls')."""
    return destination(plat, plon, normal, np.full(plat.shape, dist_km))
