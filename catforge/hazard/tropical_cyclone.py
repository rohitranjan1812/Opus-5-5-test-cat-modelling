"""Tropical cyclone (hurricane) hazard: landfall-gate stochastic track model + parametric wind field.

Event generation (per synthetic landfall)
-----------------------------------------
* landfall point  s ~ rate density along the coastline gates (regional climatology), drawn by
  importance sampling  g(s) ∝ f(s)^α  (α<1 flattens coverage so low-rate / high-value coasts such
  as New York get enough events; each event carries weight f/g in its rate);
* landfall intensity  Vmax = 33 m/s + Weibull(k, c·scale_region)  for hurricanes, with the Weibull
  quantile drawn from a tail-tilted proposal  u = 1-(1-v)^γ  (importance weight γ(1-u)^(1-1/γ)),
  giving many more intense storms per catalog-size at unchanged expected frequencies;
* translation speed, heading (relative to the coast's landward normal) and Rmax (Vickery & Wadhera
  2008 functional form), with the surface Holland-B_s of Holland (2008);
* central-pressure deficit Δp by fixed-point inversion of  Vmax = sqrt(B_s Δp / (ρ e)) + 0.55 Vt;
* track integrated along great circles with constant curvature (recurvature rate ω), and
  over-land filling by the Kaplan & DeMaria (1995) exponential decay law.

Wind field (per track point, per site)
--------------------------------------
Holland (1980) radial profile evaluated at the surface with B_s (Holland 2008) and a Coriolis
correction, 20° inflow, and a translational asymmetry scaled by the radial profile; the site's 3-second gust is  K_terrain · |V_surface|.
Footprint = maximum gust over the storm's lifetime, evaluated hourly then refined at 7.5-minute
sub-steps around the peak.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
import pandas as pd

from ..data.coast import COASTLINE, REGIONS, land_polygon
from ..geo import DEG2RAD, Polyline, destination, local_xy_km, points_in_polygon, wrap180
from .base import ENSO_REGIMES, EventCatalog, FrequencyModel, HazardUncertainty

RHO_AIR = 1.15  # kg/m^3
E = math.e
OMEGA = 7.292e-5
SURFACE_FACTOR = 1.0  # profile is evaluated at surface level (Holland 2008 B_s)
ASYM_FACTOR = 0.55
INFLOW_DEG = 20.0
KD_ALPHA = 0.095  # 1/h   Kaplan-DeMaria decay rate
KD_VB = 13.75  # m/s  background wind
HURRICANE_VMIN = 33.0
WEIBULL_K = 1.6
WEIBULL_C = 15.5
VMAX_CAP = 88.0
TS_RATE_FACTOR = 0.85
T_START_H = -36
T_END_H = 60

TERRAIN_GUST_FACTOR = {"coastal": 1.18, "open": 1.10, "suburban": 1.00, "urban": 0.94}

SAFFIR_SIMPSON = [(0, 0.0), (1, 33.0), (2, 43.0), (3, 50.0), (4, 58.0), (5, 70.0)]


def saffir_simpson(vmax_ms: np.ndarray) -> np.ndarray:
    v = np.asarray(vmax_ms)
    cat = np.zeros(v.shape, dtype=np.int8)
    for c, thr in SAFFIR_SIMPSON[1:]:
        cat[v >= thr] = c
    return cat


def _coast() -> tuple[Polyline, np.ndarray]:
    lon = np.array([c[0] for c in COASTLINE])
    lat = np.array([c[1] for c in COASTLINE])
    reg = np.array([c[2] for c in COASTLINE])
    return Polyline(lon, lat), reg


def holland_bs(dp_hpa, lat, vt):
    """Holland (2008) surface B_s (with ∂p/∂t = 0)."""
    dp_hpa = np.asarray(dp_hpa, float)
    x = 0.6 * (1.0 - dp_hpa / 215.0)
    b = -4.4e-5 * dp_hpa**2 + 0.01 * dp_hpa - 0.014 * np.abs(lat) + 0.15 * np.asarray(vt, float) ** x + 1.0
    return np.clip(b, 1.0, 2.2)


def dp_from_vmax(vmax, vt, lat, eps_r, eps_b, n_iter: int = 6):
    """Invert Vmax = sqrt(B_s Δp / (ρ e)) + 0.55 Vt jointly with B_s(Δp); then Rmax(Δp, φ).

    Returns (Δp [hPa], Rmax [km], B_s).
    """
    vs = np.maximum(np.asarray(vmax) - ASYM_FACTOR * np.asarray(vt), 8.0) / SURFACE_FACTOR
    b = np.full_like(vs, 1.3)
    dp = RHO_AIR * E * vs**2 / b / 100.0
    for _ in range(n_iter):
        b = np.clip(holland_bs(dp, lat, vt) + 0.12 * eps_b, 1.0, 2.2)
        dp = RHO_AIR * E * vs**2 / b / 100.0
    rmax = np.clip(np.exp(3.015 - 6.291e-5 * dp**2 + 0.0337 * lat + 0.40 * eps_r), 8.0, 120.0)
    return dp, rmax, b


def _build_tracks(lf_lat, lf_lon, heading0, omega, vt, vmax0, rmax, bhol, lp_lon, lp_lat):
    """Integrate tracks hourly from T_START_H to T_END_H (vectorised over events)."""
    n = lf_lat.shape[0]
    times = np.arange(T_START_H, T_END_H + 1, dtype=float)
    nt = times.size
    i0 = -T_START_H
    lat = np.empty((n, nt))
    lon = np.empty((n, nt))
    lat[:, i0], lon[:, i0] = lf_lat, lf_lon
    step_km = vt * 3.6  # km per hour
    for k in range(i0, nt - 1):
        hdg = heading0 + omega * (times[k] + 0.5)
        lat[:, k + 1], lon[:, k + 1] = destination(lat[:, k], lon[:, k], hdg, step_km)
    for k in range(i0, 0, -1):
        hdg = heading0 + omega * (times[k] - 0.5)
        lat[:, k - 1], lon[:, k - 1] = destination(lat[:, k], lon[:, k], hdg + 180.0, step_km)
    heading = (heading0[:, None] + omega[:, None] * times[None, :]) % 360.0
    land = points_in_polygon(lon.ravel(), lat.ravel(), lp_lon, lp_lat).reshape(n, nt)
    # Intensity: constant before landfall, Kaplan-DeMaria filling whenever over land afterwards.
    v = np.empty((n, nt))
    v[:, : i0 + 1] = vmax0[:, None]
    decay = math.exp(-KD_ALPHA)
    for k in range(i0 + 1, nt):
        prev = v[:, k - 1]
        v[:, k] = np.where(land[:, k], KD_VB + (prev - KD_VB) * decay, prev)
    return times, lat, lon, heading, land, v


def generate_tc_catalog(n_hurricanes: int = 4000, n_tropical_storms: int = 800, seed: int = 2024,
                        position_alpha: float = 0.5, tail_gamma: float = 2.2,
                        frequency: FrequencyModel | None = None) -> EventCatalog:
    """Generate an importance-sampled synthetic landfall catalog for the US Gulf & Atlantic coasts."""
    rng = np.random.default_rng(seed)
    coast, vreg = _coast()
    seg_reg = vreg[:-1]
    reg_len = {r: coast.seg_km[seg_reg == r].sum() for r in REGIONS}
    seg_density = np.array([REGIONS[r][0] / reg_len[r] for r in seg_reg])  # hurricanes / yr / km
    lp_lon, lp_lat = land_polygon()

    frames, geo = [], []
    for storm_class, n, rate_factor in (("HU", n_hurricanes, 1.0), ("TS", n_tropical_storms, TS_RATE_FACTOR)):
        if n <= 0:
            continue
        # ---- landfall position: stratified importance sampling along the coastline ----
        f_mass = seg_density * coast.seg_km
        g_mass = seg_density**position_alpha * coast.seg_km
        g_mass = g_mass / g_mass.sum()
        g_cdf = np.concatenate([[0.0], np.cumsum(g_mass)])
        v = (rng.permutation(n) + rng.random(n)) / n
        seg = np.clip(np.searchsorted(g_cdf, v, side="right") - 1, 0, len(g_mass) - 1)
        frac = (v - g_cdf[seg]) / np.maximum(g_mass[seg], 1e-15)
        s = coast.cum_km[seg] + frac * coast.seg_km[seg]
        w_pos = (f_mass[seg] / f_mass.sum()) / g_mass[seg]
        lf_lat, lf_lon = coast.point_at(s)
        regions = seg_reg[seg]
        rinfo = np.array([REGIONS[r] for r in regions], dtype=object)
        clim_heading = rinfo[:, 1].astype(float)
        iscale = rinfo[:, 2].astype(float)
        normal = (coast.bearing_at(s) - 90.0) % 360.0

        # ---- intensity ----
        if storm_class == "HU":
            vq = (rng.permutation(n) + rng.random(n)) / n
            u = 1.0 - (1.0 - vq) ** tail_gamma
            w_int = tail_gamma * (1.0 - u) ** (1.0 - 1.0 / tail_gamma)
            c = WEIBULL_C * iscale
            fmax = 1.0 - np.exp(-(((VMAX_CAP - HURRICANE_VMIN) / c) ** WEIBULL_K))
            x = c * (-np.log1p(-u * fmax)) ** (1.0 / WEIBULL_K)
            vmax = HURRICANE_VMIN + x
        else:
            vmax = rng.uniform(18.0, HURRICANE_VMIN, n)
            w_int = np.ones(n)

        # ---- kinematics & structure ----
        vt = np.clip(np.exp(np.log(4.5 + 0.35 * np.maximum(lf_lat - 25.0, 0.0)) + 0.35 * rng.standard_normal(n)),
                     1.5, 22.0)
        mu_delta = wrap180(clim_heading - normal)
        delta = np.clip(mu_delta + 25.0 * rng.standard_normal(n), -75.0, 75.0)
        heading0 = (normal + delta) % 360.0
        omega_mean = np.where(lf_lat < 27.0, 0.0, np.where(lf_lat < 33.0, 0.25, 0.45))
        omega = omega_mean + 0.35 * rng.standard_normal(n)
        eps_r, eps_b = rng.standard_normal(n), rng.standard_normal(n)
        dp, rmax, bhol = dp_from_vmax(vmax, vt, lf_lat, eps_r, eps_b)

        rate = TOTAL_RATE_HU * rate_factor * w_pos * w_int / n
        times, tlat, tlon, thdg, tland, tv = _build_tracks(lf_lat, lf_lon, heading0, omega, vt, vmax, rmax,
                                                           bhol, lp_lon, lp_lat)
        frames.append(pd.DataFrame({
            "storm_class": storm_class, "landfall_lat": lf_lat, "landfall_lon": lf_lon, "region": regions,
            "heading": heading0, "vmax": vmax, "category": saffir_simpson(vmax), "dp_hpa": dp, "rmax_km": rmax,
            "holland_b": bhol, "vt": vt, "omega": omega, "rate": rate, "importance_weight": w_pos * w_int,
        }))
        geo.append((times, tlat, tlon, thdg, tland, tv, vt, rmax, bhol))

    events = pd.concat(frames, ignore_index=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1, dtype=np.int64))
    events["name"] = [f"TC-{i:05d}" for i in events["event_id"]]
    geometry = _pack_tracks(geo)
    return EventCatalog(
        peril="TC", events=events, geometry=geometry,
        frequency=frequency or FrequencyModel(dispersion_r=20.0, regimes=list(ENSO_REGIMES)),
        uncertainty=HazardUncertainty(sigma_between=0.08, sigma_within=0.10, rho_event=0.10, rho_cell=0.25),
        intensity_unit="m/s (3-s gust)",
        meta={"seed": seed, "n_hurricanes": n_hurricanes, "n_tropical_storms": n_tropical_storms,
              "model": "landfall-gate + Holland (1980)", "position_alpha": position_alpha,
              "tail_gamma": tail_gamma},
    )


TOTAL_RATE_HU = float(sum(r[0] for r in REGIONS.values()))


def intensity_rate_multiplier(events: pd.DataFrame, scale_mult: float = 1.0) -> np.ndarray:
    """Likelihood ratio f'(V)/f(V) re-weighting event rates for a Weibull-scale change of landfall
    intensity (climate what-if) — the catalog itself is unchanged, only its rates.
    """
    mult = np.ones(len(events))
    if abs(scale_mult - 1.0) < 1e-12:
        return mult
    hu = (events["storm_class"] == "HU").to_numpy() & events["region"].isin(list(REGIONS)).to_numpy()
    c = WEIBULL_C * np.array([REGIONS[r][2] if r in REGIONS else 1.0 for r in events["region"]])
    c2 = c * scale_mult
    x = np.maximum(events["vmax"].to_numpy(float) - HURRICANE_VMIN, 1e-9)
    xm = VMAX_CAP - HURRICANE_VMIN
    k = WEIBULL_K
    log_ratio = k * np.log(c / c2) - (x / c2) ** k + (x / c) ** k
    norm = (1 - np.exp(-((xm / c) ** k))) / (1 - np.exp(-((xm / c2) ** k)))
    mult[hu] = (np.exp(log_ratio) * norm)[hu]
    return mult


def _pack_tracks(geo) -> dict[str, np.ndarray]:
    """Pack per-class dense track arrays into CSR form, trimming weak post-landfall tails."""
    ptr = [0]
    cols: dict[str, list] = {k: [] for k in ("t", "lat", "lon", "heading", "land", "vmax", "dp", "rmax", "b", "vt")}
    for times, tlat, tlon, thdg, tland, tv, vt, rmax, bhol in geo:
        for i in range(tlat.shape[0]):
            keep = np.ones(times.size, dtype=bool)
            weak = np.nonzero((times > 0) & (tv[i] < 17.0))[0]
            if weak.size:
                keep[weak[0] + 1:] = False
            vt_i = np.full(keep.sum(), vt[i])
            rm_i = np.full(keep.sum(), rmax[i])
            b_i = np.full(keep.sum(), bhol[i])
            vg = np.maximum(tv[i][keep] - ASYM_FACTOR * vt[i], 5.0) / SURFACE_FACTOR
            dp_i = RHO_AIR * E * vg**2 / b_i  # Pa
            cols["t"].append(times[keep])
            cols["lat"].append(tlat[i][keep])
            cols["lon"].append(tlon[i][keep])
            cols["heading"].append(thdg[i][keep])
            cols["land"].append(tland[i][keep])
            cols["vmax"].append(tv[i][keep])
            cols["dp"].append(dp_i)
            cols["rmax"].append(rm_i)
            cols["b"].append(b_i)
            cols["vt"].append(vt_i)
            ptr.append(ptr[-1] + int(keep.sum()))
    out = {k: np.concatenate(v).astype(np.float32 if k not in ("land",) else np.bool_) for k, v in cols.items()}
    out["ptr"] = np.asarray(ptr, dtype=np.int64)
    return out


def single_track(landfall_lat: float, landfall_lon: float, heading: float, vmax: float, rmax_km: float,
                 vt: float, holland_b: float | None = None, omega: float = 0.0, event_id: int = 1,
                 name: str = "scenario") -> EventCatalog:
    """Build a one-event catalog from explicit landfall parameters (deterministic scenarios)."""
    lat = np.array([landfall_lat], float)
    lon = np.array([landfall_lon], float)
    vt_a = np.array([vt], float)
    vmax_a = np.array([vmax], float)
    rmax_a = np.array([rmax_km], float)
    if holland_b is None:
        dp, _, b_a = dp_from_vmax(vmax_a, vt_a, lat, np.zeros(1), np.zeros(1))
    else:
        b_a = np.array([holland_b], float)
        vg = np.maximum(vmax_a - ASYM_FACTOR * vt_a, 8.0) / SURFACE_FACTOR
        dp = RHO_AIR * E * vg**2 / b_a / 100.0
    lp_lon, lp_lat = land_polygon()
    times, tlat, tlon, thdg, tland, tv = _build_tracks(lat, lon, np.array([heading], float),
                                                       np.array([omega], float), vt_a, vmax_a, rmax_a, b_a,
                                                       lp_lon, lp_lat)
    events = pd.DataFrame({
        "event_id": [event_id], "storm_class": ["HU" if vmax >= HURRICANE_VMIN else "TS"],
        "landfall_lat": lat, "landfall_lon": lon, "region": ["scenario"], "heading": [heading],
        "vmax": vmax_a, "category": saffir_simpson(vmax_a), "dp_hpa": dp, "rmax_km": rmax_a,
        "holland_b": b_a, "vt": vt_a, "omega": [omega], "rate": [1.0], "importance_weight": [1.0],
        "name": [name],
    })
    geometry = _pack_tracks([(times, tlat, tlon, thdg, tland, tv, vt_a, rmax_a, b_a)])
    return EventCatalog(peril="TC", events=events, geometry=geometry, frequency=FrequencyModel(),
                        uncertainty=HazardUncertainty(0.08, 0.10, 0.10, 0.25), intensity_unit="m/s (3-s gust)",
                        meta={"scenario": True})


# ----------------------------------------------------------------------------------------------
# Wind-field kernels
# ----------------------------------------------------------------------------------------------

@nb.njit(inline="always", cache=True)
def _gust(slat, slon, clat, clon, dp, rm, bh, hdg, vt, kfac):
    x, y = local_xy_km(slat, slon, clat, clon)
    r = math.sqrt(x * x + y * y)
    if r < 0.5:
        r = 0.5
    f = 2.0 * OMEGA * abs(math.sin(clat * DEG2RAD))
    rr = (rm / r) ** bh
    rf = r * 1000.0 * f * 0.5
    vg = math.sqrt(bh * dp / RHO_AIR * rr * math.exp(-rr) + rf * rf) - rf
    vs = SURFACE_FACTOR * vg
    vgm = math.sqrt(bh * dp / (RHO_AIR * E))
    vsm = SURFACE_FACTOR * vgm
    cb = math.cos(INFLOW_DEG * DEG2RAD)
    sb = math.sin(INFLOW_DEG * DEG2RAD)
    ux = x / r
    uy = y / r
    wx = vs * (cb * (-uy) - sb * ux)
    wy = vs * (cb * ux - sb * uy)
    sc = ASYM_FACTOR * vs / max(vsm, 1.0)
    wx += sc * vt * math.sin(hdg * DEG2RAD)
    wy += sc * vt * math.cos(hdg * DEG2RAD)
    return kfac * math.sqrt(wx * wx + wy * wy)


@nb.njit(parallel=True, cache=True)
def tc_footprint_dense(ev_idx, ptr, tlat, tlon, tdp, trm, tb, thdg, tvt, tvmax,
                       slat, slon, skfac, glat0, glon0, gdeg, gny, gnx, cell_ptr, cell_items, vmin_track):
    """Max 3-s gust (m/s) at every site for each event in ``ev_idx`` → dense (len(ev_idx), n_sites)."""
    n_ev = ev_idx.shape[0]
    n_sites = slat.shape[0]
    out = np.zeros((n_ev, n_sites), dtype=np.float32)
    for q in nb.prange(n_ev):
        e = ev_idx[q]
        a = ptr[e]
        z = ptr[e + 1]
        # bounding box of damaging part of the track
        la_min, la_max, lo_min, lo_max = 90.0, -90.0, 180.0, -180.0
        rinf_max = 0.0
        for k in range(a, z):
            if tvmax[k] < vmin_track:
                continue
            rinf = min(700.0, 300.0 + 3.0 * trm[k])
            rinf_max = max(rinf_max, rinf)
            la_min = min(la_min, tlat[k])
            la_max = max(la_max, tlat[k])
            lo_min = min(lo_min, tlon[k])
            lo_max = max(lo_max, tlon[k])
        if la_max < la_min:
            continue
        dlat = rinf_max / 110.574
        dlon = rinf_max / (111.32 * max(math.cos(max(abs(la_min), abs(la_max)) * DEG2RAD), 0.2))
        iy0 = max(int(math.floor((la_min - dlat - glat0) / gdeg)), 0)
        iy1 = min(int(math.floor((la_max + dlat - glat0) / gdeg)), gny - 1)
        ix0 = max(int(math.floor((lo_min - dlon - glon0) / gdeg)), 0)
        ix1 = min(int(math.floor((lo_max + dlon - glon0) / gdeg)), gnx - 1)
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                c = iy * gnx + ix
                for p in range(cell_ptr[c], cell_ptr[c + 1]):
                    s = cell_items[p]
                    best = 0.0
                    kbest = -1
                    for k in range(a, z):
                        if tvmax[k] < vmin_track:
                            continue
                        x, y = local_xy_km(slat[s], slon[s], tlat[k], tlon[k])
                        rinf = min(700.0, 300.0 + 3.0 * trm[k])
                        if x * x + y * y > rinf * rinf:
                            continue
                        g = _gust(slat[s], slon[s], tlat[k], tlon[k], tdp[k], trm[k], tb[k], thdg[k], tvt[k],
                                  skfac[s])
                        if g > best:
                            best = g
                            kbest = k
                    if kbest < 0:
                        continue
                    # refine at 7.5-minute sub-steps within +-1 h of the hourly peak
                    k0 = max(kbest - 1, a)
                    k1 = min(kbest + 1, z - 1)
                    for k in range(k0, k1):
                        for j in range(1, 8):
                            w = j / 8.0
                            g = _gust(slat[s], slon[s],
                                      tlat[k] + w * (tlat[k + 1] - tlat[k]),
                                      tlon[k] + w * (tlon[k + 1] - tlon[k]),
                                      tdp[k] + w * (tdp[k + 1] - tdp[k]), trm[k], tb[k],
                                      thdg[k] + w * (((thdg[k + 1] - thdg[k] + 540.0) % 360.0) - 180.0),
                                      tvt[k], skfac[s])
                            if g > best:
                                best = g
                    out[q, s] = best
    return out


def tc_kernel_args(cat: EventCatalog):
    g = cat.geometry
    return (g["ptr"], g["lat"].astype(np.float64), g["lon"].astype(np.float64), g["dp"].astype(np.float64),
            g["rmax"].astype(np.float64), g["b"].astype(np.float64), g["heading"].astype(np.float64),
            g["vt"].astype(np.float64), g["vmax"].astype(np.float64))


def track_of(cat: EventCatalog, event_index: int) -> dict:
    g = cat.geometry
    a, z = int(g["ptr"][event_index]), int(g["ptr"][event_index + 1])
    return {
        "t": g["t"][a:z].tolist(), "lat": np.round(g["lat"][a:z], 4).tolist(),
        "lon": np.round(g["lon"][a:z], 4).tolist(), "vmax": np.round(g["vmax"][a:z], 2).tolist(),
        "land": g["land"][a:z].astype(bool).tolist(),
    }
