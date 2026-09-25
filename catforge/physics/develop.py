"""Event development: one physically consistent realization of an event, resolved in time.

Hurricane: time-varying wind field (client evaluates the identical analytic field; server probes
provide a parity check), 2-D storm surge with inundation, R-CLIPER rainfall, per-building damage
evolving with the running-maximum gust and inundation depth, and the loss timeline.

Earthquake: kinematic finite-fault rupture (von Kármán slip, rupture times), P/S first-arrival
isochrones and significant-shaking windows on a ground grid, a realized PGA field
(GMPE median × inter-event × spatially correlated intra-event residual), per-building damage timed
by wave arrival, and on-demand stochastic finite-fault seismograms.

Heavy grids are returned as base64-encoded little-endian int16/uint8 arrays with a scale factor.
"""

from __future__ import annotations

import base64
import math

import numba as nb
import numpy as np

from ..engine.kernel import _sample_damage
from ..exposure.portfolio import Portfolio
from ..hazard.base import EventCatalog
from ..hazard.earthquake import rupture_of
from ..hazard.footprint import Sites, compute_pairs
from ..hazard.tropical_cyclone import TERRAIN_GUST_FACTOR, saffir_simpson
from ..rng import norm_cdf
from ..scenario import ANALOGS, build_event
from ..vulnerability.damage import BIN_HI, BIN_LO, build_tables, surge_damage_ratio
from .dem import elevation
from .grf import Matern, circulant_field
from .tc_dynamics import rain_accumulation, site_gust_series, track_arrays

TC_CONSTANTS = {"rho_air": 1.15, "e": math.e, "omega": 7.292e-5, "asym": 0.55, "surface_factor": 1.0,
                "inflow": {"inner_deg": 10.0, "outer_deg": 25.0, "r1": 1.0, "r2": 1.2},
                "gust_open": TERRAIN_GUST_FACTOR["open"], "gust_marine": TERRAIN_GUST_FACTOR["coastal"]}


def b64(arr, dtype: str = "int16", scale: float = 1.0, nan_value: int = -32768) -> dict:
    a = np.asarray(arr, float) * scale
    if dtype == "int16":
        out = np.where(np.isfinite(a), np.clip(np.rint(a), -32767, 32767), nan_value).astype("<i2")
    elif dtype == "uint8":
        out = np.where(np.isfinite(a), np.clip(np.rint(a), 0, 255), 0).astype("u1")
    else:
        out = np.where(np.isfinite(a), a, np.nan).astype("<f4")
    return {"b64": base64.b64encode(out.tobytes()).decode(), "dtype": dtype, "scale": scale,
            "shape": list(out.shape), "nan": nan_value if dtype == "int16" else None}


# ------------------------------------------------------------------------------------ events
def one_event(model, peril: str, event_id: int) -> EventCatalog:
    """Slice a single event out of a stochastic catalog as its own EventCatalog."""
    cat = model.catalogs[peril]
    idx = np.nonzero(cat.events["event_id"].to_numpy() == event_id)[0]
    if idx.size == 0:
        raise ValueError(f"event {event_id} not in {peril} catalog")
    i = int(idx[0])
    g = cat.geometry
    a, z = int(g["ptr"][i]), int(g["ptr"][i + 1])
    geo = {"ptr": np.array([0, z - a], np.int64)}
    for k, v in g.items():
        if k not in ("ptr", "tptr", "tlat", "tlon") and v.shape[0] == g["ptr"][-1]:
            geo[k] = v[a:z]
    if "tptr" in g:
        ta, tz = int(g["tptr"][i]), int(g["tptr"][i + 1])
        geo["tptr"] = np.array([0, tz - ta], np.int64)
        geo["tlat"], geo["tlon"] = g["tlat"][ta:tz], g["tlon"][ta:tz]
    ev = cat.events.iloc[[i]].reset_index(drop=True).copy()
    ev["rate"] = 1.0
    return EventCatalog(peril=peril, events=ev, geometry=geo, frequency=cat.frequency, uncertainty=cat.uncertainty,
                        intensity_unit=cat.intensity_unit, meta={"from_catalog": True})


def resolve_event(model, peril: str | None, analog: str | None, params: dict | None, event_id: int | None):
    if analog:
        a = ANALOGS[analog]
        peril = a["peril"]
        p = {**a["params"], "name": a["label"]}
        return peril, build_event(peril, p), a["label"]
    if event_id is not None:
        cat = one_event(model, peril, int(event_id))
        return peril, cat, str(cat.events["name"].iloc[0])
    if peril and params:
        return peril, build_event(peril, params), str(params.get("name", "custom event"))
    raise ValueError("provide analog, event_id (+peril) or peril+params")


@nb.njit(parallel=True, cache=True)
def _damage_quantiles(cdf, vidx, li, u, li0, dli, bin_lo, bin_hi):
    out = np.zeros(li.shape)
    for s in nb.prange(li.shape[0]):
        v = vidx[s]
        for k in range(li.shape[1]):
            out[s, k] = _sample_damage(cdf, v, li[s, k], li0[v], dli[v], u[s], bin_lo, bin_hi)
    return out


def _grid_field(lat0, lat1, lon0, lon1, model: Matern, rng, max_n: int = 220):
    """Realization of the intra-event field on a regular grid (circulant embedding)."""
    km_lat = (lat1 - lat0) * 110.574
    km_lon = (lon1 - lon0) * 111.32 * math.cos(math.radians(0.5 * (lat0 + lat1)))
    h = max(model.range_km / 8.0, max(km_lat, km_lon) / max_n)
    ny = max(int(km_lat / h) + 2, 4)
    nx = max(int(km_lon / h) + 2, 4)
    F, _ = circulant_field(ny, nx, h, h, model, rng)
    dlat = (lat1 - lat0) / (ny - 1)
    dlon = (lon1 - lon0) / (nx - 1)
    return {"lat0": lat0, "lon0": lon0, "dlat": dlat, "dlon": dlon, "ny": ny, "nx": nx, "values": F[0], "h_km": h}


def _sample_grid(g, lat, lon):
    y = np.clip((np.asarray(lat) - g["lat0"]) / g["dlat"], 0, g["ny"] - 1.0001)
    x = np.clip((np.asarray(lon) - g["lon0"]) / g["dlon"], 0, g["nx"] - 1.0001)
    i, j = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - i, x - j
    v = g["values"]
    return (1 - fy) * ((1 - fx) * v[i, j] + fx * v[i, j + 1]) + fy * ((1 - fx) * v[i + 1, j] + fx * v[i + 1, j + 1])


def _field_payload(g, decimals=3):
    return {"lat0": g["lat0"], "lon0": g["lon0"], "dlat": g["dlat"], "dlon": g["dlon"], "ny": g["ny"], "nx": g["nx"],
            "h_km": g["h_km"], "values": b64(g["values"], "int16", 1000.0)}


def _damage_uniforms(n, rng, rho_e, rho_c, cells):
    z = rng.standard_normal()
    uc = {c: rng.standard_normal() for c in np.unique(cells)}
    zc = np.array([uc[c] for c in cells])
    x = math.sqrt(rho_e) * z + math.sqrt(rho_c) * zc + math.sqrt(max(1 - rho_e - rho_c, 0)) * rng.standard_normal(n)
    return np.array([norm_cdf(v) for v in x])


def _site_ground(lat, lon, measured=None) -> np.ndarray:
    """Building ground elevation: measured where available (enrichment), else the 2′ DEM."""
    z = elevation(lat, lon, fill=0.0)
    if measured is None:
        return z
    g = np.asarray(measured, float)
    return np.where(np.isfinite(g), g, z)


def _cities(bbox) -> list[dict]:
    from ..data.cities import CITIES

    lo0, la0, lo1, la1 = bbox
    return [{"name": c[0], "state": c[1], "lat": c[2], "lon": c[3], "weight": c[4]}
            for c in CITIES if la0 <= c[2] <= la1 and lo0 <= c[3] <= lo1]


def _coverage_loss(tiv, cov, d):
    """GU loss for damage ratio array d (sites × times)."""
    cont = np.minimum(1.0, cov[:, 0:1] * d ** cov[:, 1:2])
    bi = np.minimum(1.0, (d / cov[:, 2:3]) ** cov[:, 3:4])
    return tiv[:, 0:1] * d + tiv[:, 1:2] * cont + tiv[:, 2:3] * bi


# ------------------------------------------------------------------------------------ hurricane
def develop_tc(cat: EventCatalog, name: str, portfolio: Portfolio | None, seed: int = 1, frame_h: float = 1.0,
               with_surge: bool = True, with_rain: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    tr = track_arrays(cat)
    tt, tlat, tlon, tdp, trm, tb, thdg, tvt, tvmax = tr
    t0 = float(max(tt[0], -30.0))
    t1 = float(min(tt[-1], 48.0))
    frames = np.round(np.arange(t0, t1 + 1e-9, frame_h), 3)
    act = (tt >= t0 - 1) & (tt <= t1 + 1)
    la0, la1 = float(tlat[act].min()) - 4.0, float(tlat[act].max()) + 4.0
    lo0, lo1 = float(tlon[act].min()) - 4.5, float(tlon[act].max()) + 4.5
    u = cat.uncertainty
    eta = float(u.sigma_between * rng.standard_normal())
    field = _grid_field(la0, la1, lo0, lo1, Matern(u.grf_nu, u.grf_range_km), rng)
    out = {
        "peril": "TC", "name": name, "seed": seed, "t0": t0, "t1": t1, "frame_h": frame_h, "frames": frames.tolist(),
        "bbox": [lo0, la0, lo1, la1], "constants": TC_CONSTANTS,
        "track": {"t": tt.tolist(), "lat": tlat.round(4).tolist(), "lon": tlon.round(4).tolist(),
                  "dp_pa": tdp.round(1).tolist(), "rmax_km": trm.round(2).tolist(), "b": tb.round(4).tolist(),
                  "heading": thdg.round(2).tolist(), "vt": tvt.round(3).tolist(), "vmax": tvmax.round(2).tolist(),
                  "category": saffir_simpson(tvmax).tolist()},
        "realization": {"eta": eta, "sigma_w": u.sigma_within, "grf": {"nu": u.grf_nu, "range_km": u.grf_range_km},
                        "field": _field_payload(field)},
        "event": {k: (v.item() if hasattr(v, "item") else v) for k, v in cat.events.iloc[0].to_dict().items()},
    }
    # land mask on the field grid (so the client applies marine vs open-terrain gust factors)
    glat = field["lat0"] + field["dlat"] * np.arange(field["ny"])
    glon = field["lon0"] + field["dlon"] * np.arange(field["nx"])
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    zg = elevation(LA, LO, fill=-50.0)
    out["land"] = b64(zg > 0.0, "uint8")
    # parity probes (server-evaluated gust, open terrain, no realization noise)
    k_probe = np.linspace(0, frames.size - 1, 6).astype(int)
    pl, po, pt = [], [], []
    for k in k_probe:
        from .tc_dynamics import track_state

        c = track_state(float(frames[k]), *[np.asarray(a, float) for a in tr])
        for dist, brg in ((0.8, 30.0), (1.5, 120.0), (3.0, 250.0)):
            la = c[0] + dist * c[3] / 110.574 * math.cos(math.radians(brg))
            lo = c[1] + dist * c[3] / (111.32 * math.cos(math.radians(c[0]))) * math.sin(math.radians(brg))
            pl.append(la)
            po.append(lo)
            pt.append(float(frames[k]))
    gp = site_gust_series(np.array(sorted(set(pt))), tuple(np.asarray(a, float) for a in tr), np.array(pl), np.array(po),
                          np.full(len(pl), TERRAIN_GUST_FACTOR["open"]), np.ones(len(pl)))
    tl = sorted(set(pt))
    out["probes"] = [{"lat": pl[i], "lon": po[i], "t": pt[i], "gust": float(gp[i, tl.index(pt[i])])}
                     for i in range(len(pl))]
    # rainfall (R-CLIPER) — accumulated every 3 h on a 0.1° grid
    if with_rain:
        rstep = 0.15
        rlat = np.arange(la0, la1 + 1e-9, rstep)
        rlon = np.arange(lo0, lo1 + 1e-9, rstep)
        while rlat.size * rlon.size > 20000:
            rstep *= 1.25
            rlat = np.arange(la0, la1 + 1e-9, rstep)
            rlon = np.arange(lo0, lo1 + 1e-9, rstep)
        RLA, RLO = np.meshgrid(rlat, rlon, indexing="ij")
        tfine = np.arange(t0, t1 + 1e-9, 1.0 / 6.0)
        every = int(round(3.0 * 6))
        fidx = np.arange(0, tfine.size, every)
        if fidx[-1] != tfine.size - 1:
            fidx = np.append(fidx, tfine.size - 1)
        acc = rain_accumulation(tfine, tuple(np.asarray(a, float) for a in tr), RLA.ravel(), RLO.ravel(), fidx)
        out["rain"] = {"lat0": float(rlat[0]), "lon0": float(rlon[0]), "dlat": rstep, "dlon": rstep,
                       "ny": int(rlat.size), "nx": int(rlon.size), "t": tfine[fidx].round(3).tolist(),
                       "frames": b64(acc.reshape(len(fidx), rlat.size, rlon.size), "int16", 1.0),
                       "max_mm": float(acc.max()), "model": "R-CLIPER (Tuleya et al. 2007)"}
    # sites & damage
    sites_payload, loss = None, None
    site_lat = site_lon = None
    if portfolio is not None:
        pairs = compute_pairs(cat, portfolio.sites(), min_intensity=18.0)
        idx = pairs.site[pairs.ev_ptr[0]:pairs.ev_ptr[1]].astype(np.int64)
        L = portfolio.locations.iloc[idx]
        site_lat, site_lon = L["lat"].to_numpy(float), L["lon"].to_numpy(float)
        site_ground = L["ground_elev_m"].to_numpy(float) if "ground_elev_m" in L else None
    else:
        site_ground = None
    surge = None
    if with_surge:
        from .surge2d import coastal_peak, run_surge

        try:
            sres = run_surge(tuple(np.asarray(a, float) for a in tr), max(t0, -24.0), min(t1, 24.0),
                             sites_lat=site_lat, sites_lon=site_lon, sites_ground=site_ground)
            surge = sres
            sub = max(1, int(math.ceil(max(sres["z"].shape) / 110)))
            fr = np.stack(sres["frames"])[:, ::sub, ::sub]
            pk, pla, plo = coastal_peak(sres)
            land_depth = np.where(sres["z"] > 0, sres["depth_max"], np.nan)
            out["surge"] = {
                "lat0": float(sres["lat"][0]), "lon0": float(sres["lon"][0]),
                "dlat": float(sres["lat"][1] - sres["lat"][0]) * sub, "dlon": float(sres["lon"][1] - sres["lon"][0]) * sub,
                "ny": int(fr.shape[1]), "nx": int(fr.shape[2]), "t": sres["frame_t"].round(3).tolist(),
                "frames_cm": b64(fr, "int16", 100.0), "z": b64(sres["z"][::sub, ::sub], "int16", 1.0),
                "max": {"lat0": float(sres["lat"][0]), "lon0": float(sres["lon"][0]),
                        "dlat": float(sres["lat"][1] - sres["lat"][0]), "dlon": float(sres["lon"][1] - sres["lon"][0]),
                        "ny": int(sres["z"].shape[0]), "nx": int(sres["z"].shape[1]),
                        "eta_cm": b64(np.where(sres["depth_max"] > 0.05, sres["eta_max"], np.nan), "int16", 100.0),
                        "land_depth_cm": b64(land_depth, "int16", 100.0)},
                "peak_m": pk, "peak_lat": pla, "peak_lon": plo,
                "inundated_km2": float(sres["inundated_land"].sum() * (sres["stride_deg"] * 111.0) ** 2 *
                                       math.cos(math.radians(pla))),
                "model": "2-D depth-averaged shallow water on ETOPO1 (2′), wetting/drying",
            }
        except ValueError as exc:  # outside DEM coverage
            out["surge"] = {"error": str(exc)}
    if portfolio is not None and site_lat is not None and site_lat.size:
        L = portfolio.locations.iloc[idx]
        sites = portfolio.sites()
        smult = np.exp(eta + u.sigma_within * _sample_grid(field, site_lat, site_lon))
        tr64 = tuple(np.asarray(a, float) for a in tr)
        gust = site_gust_series(frames.astype(float), tr64, site_lat, site_lon, sites.terrain_k[idx], smult)
        runmax = np.maximum.accumulate(gust, axis=1)
        vt = build_tables(L, {"TC": 0.0}, perils=["TC"])
        vidx = vt.loc_vuln[:, 0].astype(np.int64)
        cells = np.floor(site_lat / 0.25).astype(int) * 10000 + np.floor(site_lon / 0.25).astype(int)
        uu = _damage_uniforms(site_lat.size, rng, u.dmg_rho_event, u.dmg_rho_cell, cells)
        li = np.log(np.maximum(runmax, 1e-3))
        d_wind = _damage_quantiles(vt.cdf, vidx, li, uu, vt.log_i0, vt.dlog, BIN_LO, BIN_HI)
        d_surge = np.zeros_like(d_wind)
        depth_frames = None
        if surge is not None and surge.get("site_depth_frames") is not None:
            sd = surge["site_depth_frames"]  # (n_sites, n_surge_frames), hourly
            st = surge["frame_t"]
            runmax_depth = np.maximum.accumulate(sd, axis=1)
            depth_frames = np.stack([np.interp(frames, st, runmax_depth[i], left=0.0, right=runmax_depth[i, -1])
                                     for i in range(site_lat.size)])
            ffh = L["first_floor_height_m"].to_numpy(float)[:, None] if "first_floor_height_m" in L else None
            d_surge = surge_damage_ratio(depth_frames, L["construction"].to_numpy()[:, None],
                                         L["occupancy"].to_numpy()[:, None], L["stories"].to_numpy()[:, None],
                                         L["year_built"].to_numpy()[:, None], first_floor_m=ffh)
        d_tot = 1.0 - (1.0 - d_wind) * (1.0 - d_surge)
        tiv = portfolio.tiv[idx]
        gu_t = _coverage_loss(tiv, vt.cov[vidx], d_tot)
        gu_w = _coverage_loss(tiv, vt.cov[vidx], d_wind)
        gu_site_final = gu_t[:, -1]
        loss = {"t": frames.tolist(), "gu": gu_t.sum(axis=0).round(0).tolist(),
                "gu_wind_only": gu_w.sum(axis=0).round(0).tolist(),
                "n_damaged": (d_tot > 0.02).sum(axis=0).tolist(), "tiv_affected": float(tiv.sum())}
        sites_payload = {
            "n": int(idx.size), "loc_id": L["loc_id"].tolist(), "lat": site_lat.round(5).tolist(),
            "lon": site_lon.round(5).tolist(), "tiv": tiv.sum(axis=1).round(0).tolist(),
            "construction": L["construction"].tolist(), "occupancy": L["occupancy"].tolist(),
            "elev": _site_ground(site_lat, site_lon, site_ground).round(2).tolist(),
            "ground_measured": int(np.isfinite(site_ground).sum()) if site_ground is not None else 0,
            "gust": b64(gust, "int16", 10.0), "damage": b64(d_tot, "int16", 1000.0),
            "gu": gu_site_final.round(0).tolist(), "damage_final": d_tot[:, -1].round(4).tolist(),
            "surge_depth": b64(depth_frames if depth_frames is not None else np.zeros_like(d_tot), "int16", 100.0),
        }
    out["sites"] = sites_payload
    out["loss"] = loss
    out["cities"] = _cities(out["bbox"])
    out["stats"] = {
        "vmax_landfall": float(np.interp(0.0, tt, tvmax)), "min_pressure_hpa": float(1013.0 - tdp.max() / 100.0),
        "peak_surge_m": out.get("surge", {}).get("peak_m"), "max_rain_mm": out.get("rain", {}).get("max_mm"),
        "final_gu": (loss["gu"][-1] if loss else None), "n_sites": int(sites_payload["n"]) if sites_payload else 0,
    }
    return out


# ------------------------------------------------------------------------------------ earthquake
def develop_eq(cat: EventCatalog, name: str, portfolio: Portfolio | None, seed: int = 1, grid_n: int = 120) -> dict:
    from .eq_dynamics import arrivals, build_fault

    rng = np.random.default_rng(seed)
    ev = cat.events.iloc[0].to_dict()
    g = cat.geometry
    tlat = g.get("tlat", g["lat"])
    tlon = g.get("tlon", g["lon"])
    fault = build_fault(ev, tlat, tlon, np.random.default_rng(seed + 7919))
    M = fault["M"]
    ring = rupture_of(cat, 0)
    reach = float(np.clip(40.0 + 45.0 * (M - 5.0) ** 1.6, 60.0, 450.0))
    la0 = min(ring["lat"]) - reach / 110.574
    la1 = max(ring["lat"]) + reach / 110.574
    cl = math.cos(math.radians(0.5 * (la0 + la1)))
    lo0 = min(ring["lon"]) - reach / (111.32 * cl)
    lo1 = max(ring["lon"]) + reach / (111.32 * cl)
    glat = np.linspace(la0, la1, grid_n)
    glon = np.linspace(lo0, lo1, grid_n)
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    tP, tS, tE, tM = arrivals(fault, LA.ravel(), LO.ravel())
    u = cat.uncertainty
    eta = float(u.sigma_between * rng.standard_normal())
    field = _grid_field(la0, la1, lo0, lo1, Matern(u.grf_nu, u.grf_range_km), rng)
    # median PGA on the grid (Rjb to the rupture surface projection), Vs30 = 400 m/s
    gs = Sites.plain(LA.ravel(), LO.ravel(), vs30=400.0)
    gp = compute_pairs(cat, gs, min_intensity=0.005)
    med = np.zeros(LA.size)
    med[gp.site] = np.exp(gp.log_i)
    pga = med * np.exp(eta + u.sigma_within * _sample_grid(field, LA.ravel(), LO.ravel()))
    zg = elevation(LA, LO, fill=0.0)
    out = {
        "peril": "EQ", "name": name, "seed": seed, "bbox": [lo0, la0, lo1, la1],
        "event": {k: (v.item() if hasattr(v, "item") else v) for k, v in ev.items()},
        "fault": {
            "nL": fault["nL"], "nW": fault["nW"], "L_km": fault["L"], "W_km": fault["W"], "dip": fault["dip"],
            "ztor": fault["ztor"], "mech": fault["mech"], "M0": fault["M0"], "Mw": M, "D_mean": fault["D_mean"],
            "D_max": fault["D_max"], "vr_kms": fault["vr"], "duration_s": fault["duration"], "hypo": fault["hypo"],
            "corner_lat": fault["corner_lat"].round(5).tolist(), "corner_lon": fault["corner_lon"].round(5).tolist(),
            "corner_depth_km": fault["corner_depth"].round(3).tolist(), "slip_m": fault["slip"].round(3).tolist(),
            "t_rupture_s": fault["t_rupture"].round(3).tolist(), "ring": ring,
        },
        "grid": {"lat0": float(la0), "lon0": float(lo0), "dlat": float(glat[1] - glat[0]),
                 "dlon": float(glon[1] - glon[0]), "ny": grid_n, "nx": grid_n,
                 "t_p": b64(tP.reshape(LA.shape), "int16", 10.0), "t_s": b64(tS.reshape(LA.shape), "int16", 10.0),
                 "t_end": b64(tE.reshape(LA.shape), "int16", 10.0), "t_peak": b64(tM.reshape(LA.shape), "int16", 10.0),
                 "pga": b64(pga.reshape(LA.shape), "int16", 10000.0), "pga_median": b64(med.reshape(LA.shape), "int16", 10000.0),
                 "elev": b64(zg, "int16", 1.0)},
        "realization": {"eta": eta, "sigma_w": u.sigma_within, "grf": {"nu": u.grf_nu, "range_km": u.grf_range_km}},
        "waves": {"alpha_kms": 6.2, "beta_kms": 3.5},
        "t_max": float(np.nanpercentile(tE[np.isfinite(tE) & (tE > 0)], 95) + 15.0) if np.isfinite(tE).any() else 60.0,
    }
    sites_payload, loss = None, None
    if portfolio is not None:
        pairs = compute_pairs(cat, portfolio.sites(), min_intensity=0.02)
        idx = pairs.site[pairs.ev_ptr[0]:pairs.ev_ptr[1]].astype(np.int64)
        if idx.size:
            L = portfolio.locations.iloc[idx]
            slat, slon = L["lat"].to_numpy(float), L["lon"].to_numpy(float)
            med_s = np.exp(pairs.log_i[pairs.ev_ptr[0]:pairs.ev_ptr[1]].astype(float))
            pga_s = med_s * np.exp(eta + u.sigma_within * _sample_grid(field, slat, slon))
            sp, ss, se, sm = arrivals(fault, slat, slon)
            vt = build_tables(L, {"EQ": 0.0}, perils=["EQ"])
            vidx = vt.loc_vuln[:, 1].astype(np.int64)
            cells = np.floor(slat / 0.25).astype(int) * 10000 + np.floor(slon / 0.25).astype(int)
            uu = _damage_uniforms(slat.size, rng, u.dmg_rho_event, u.dmg_rho_cell, cells)
            d = _damage_quantiles(vt.cdf, vidx, np.log(np.maximum(pga_s, 1e-6))[:, None], uu, vt.log_i0, vt.dlog,
                                  BIN_LO, BIN_HI)[:, 0]
            tiv = portfolio.tiv[idx]
            gu = _coverage_loss(tiv, vt.cov[vidx], d[:, None])[:, 0]
            tt = np.linspace(0.0, out["t_max"], 200)
            order = np.argsort(sm)
            cum = np.interp(tt, np.concatenate([[0.0], sm[order]]), np.concatenate([[0.0], np.cumsum(gu[order])]))
            loss = {"t": tt.round(2).tolist(), "gu": cum.round(0).tolist(),
                    "n_damaged": np.searchsorted(np.sort(sm[d > 0.02]), tt).tolist(), "tiv_affected": float(tiv.sum())}
            sites_payload = {"n": int(idx.size), "loc_id": L["loc_id"].tolist(), "lat": slat.round(5).tolist(),
                             "lon": slon.round(5).tolist(), "tiv": tiv.sum(axis=1).round(0).tolist(),
                             "vs30": L["vs30"].round(0).tolist(), "construction": L["construction"].tolist(),
                             "elev": _site_ground(slat, slon, L["ground_elev_m"].to_numpy(float) if "ground_elev_m" in L else None).round(2).tolist(),
                             "pga": np.round(pga_s, 4).tolist(), "pga_median": np.round(med_s, 4).tolist(),
                             "t_p": np.round(sp, 2).tolist(), "t_s": np.round(ss, 2).tolist(),
                             "t_peak": np.round(sm, 2).tolist(), "damage": np.round(d, 4).tolist(),
                             "gu": np.round(gu, 0).tolist()}
    out["sites"] = sites_payload
    out["loss"] = loss
    out["cities"] = _cities(out["bbox"])
    out["stats"] = {"Mw": M, "M0": fault["M0"], "rupture_duration_s": fault["duration"],
                    "max_pga_g": float(pga.max()), "final_gu": loss["gu"][-1] if loss else None,
                    "n_sites": sites_payload["n"] if sites_payload else 0}
    return out


def seismogram(cat: EventCatalog, lat: float, lon: float, vs30: float = 400.0, seed: int = 1) -> dict:
    from .eq_dynamics import build_fault, simulate_site

    ev = cat.events.iloc[0].to_dict()
    g = cat.geometry
    fault = build_fault(ev, g.get("tlat", g["lat"]), g.get("tlon", g["lon"]), np.random.default_rng(seed + 7919))
    sim = simulate_site(fault, lat, lon, vs30=vs30, seed=seed)
    # GMPE comparison at the site (median, ±1σ total)
    pairs = compute_pairs(cat, Sites.plain([lat], [lon], vs30=vs30), min_intensity=1e-4)
    med = float(np.exp(pairs.log_i[0])) if pairs.n_pairs else float("nan")
    sig = math.hypot(cat.uncertainty.sigma_between, cat.uncertainty.sigma_within)
    n = sim["n"]
    step = max(1, int(math.ceil(n / 6000)))
    t_end = min(n, int((sim["t_s"] + 90.0 + fault["duration"]) / sim["dt"]))

    def dec(a):
        a = a[:t_end]
        m = (a.size // step) * step
        blk = a[:m].reshape(-1, step)
        return blk[np.arange(blk.shape[0]), np.abs(blk).argmax(axis=1)]

    fm = (sim["freq"] > 0.05) & (sim["freq"] < 50.0)
    fsel = np.unique(np.geomspace(1, fm.sum() - 1, 300).astype(int))
    return {
        "lat": lat, "lon": lon, "vs30": vs30, "dt": sim["dt"] * step, "acc_g": np.round(dec(sim["acc_g"]), 5).tolist(),
        "vel_cms": np.round(dec(sim["vel_cms"]), 3).tolist(), "pga_g": sim["pga_g"], "pgv_cms": sim["pgv_cms"],
        "t_p": sim["t_p"], "t_s": sim["t_s"], "periods": sim["periods"].round(4).tolist(),
        "psa_g": np.round(sim["psa_g"], 5).tolist(), "freq": sim["freq"][fm][fsel].round(4).tolist(),
        "fas": sim["fas"][fm][fsel].round(5).tolist(), "gmpe_median_g": med,
        "gmpe_lo_g": med * math.exp(-sig), "gmpe_hi_g": med * math.exp(sig), "n_subfaults": sim["n_subfaults"],
        "epicentral_km": sim["epicentral_km"], "method": "EXSIM-style stochastic finite fault (Motazedian & Atkinson 2005)",
        "params": sim["params"], "Mw": fault["M"],
    }


def develop(model, portfolio: Portfolio | None, peril=None, analog=None, params=None, event_id=None, seed: int = 1,
            with_surge: bool = True) -> dict:
    peril, cat, name = resolve_event(model, peril, analog, params, event_id)
    if peril in model.catalogs:
        cat.uncertainty = model.catalogs[peril].uncertainty
    if peril == "TC":
        return develop_tc(cat, name, portfolio, seed=seed, with_surge=with_surge)
    return develop_eq(cat, name, portfolio, seed=seed)

