"""Storm surge for the stochastic engine — a multi-fidelity hazard.

**Low fidelity (every event).** The validated 2-D shallow-water solver (``physics/surge2d``) at 4′ (stride
2 on ETOPO1 2′), with a CFL-sized time step, over the full model's window (−18 h…+12 h). The domain is
the landfall box united with the padded bounding box of the track's near-coast points, so alongshore
and bypassing storms get the coast they flood. The whole time loop is fused into one ``nogil`` numba
call, and events run concurrently on threads.

**Event product (portfolio-independent, disk-cached per catalog).** The wet near-shore cells, with
their peak water-surface elevation η (m above MSL).

**Site water level.** A friction-limited inland penetration:

    WSE_s = max_c [ η_c − α · max(d(c, s) − r₀, 0) ]    over cells c within R of s

**Full-model correction (``hazard/surge_calib``).** A two-part model calibrated against the 2′ model
on a design set (``scripts/calibrate_surge.py``):

- a logistic connectivity probability P(wet | x, Δ, z, shore), where Δ is the friction loss the site
  rule applied (x at α = 0 minus x);
- a wet-level mixed model s(x) + γ·covariates + δ_node, whose event (σ_e) and site (σ_s) residual terms
  the loss kernel samples.

Here z is the node's 2′ ground and shore says whether the node's stencil touches the sea. δ_node is a
calibrated site response on the 2′ node grid, so harbours and bays the 4′ grid cannot resolve get
their own term.

Water level is ground-independent, so building ground (measured by enrichment, or the 2′ DEM floored at
1 m) is subtracted in the kernel. Depth then maps through the depth–damage curve and combines with wind
damage as 1 − (1 − d_wind)(1 − d_surge).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np

from ..physics import surge2d as S
from ..physics.dem import subgrid

LF_VERSION = 2
# v2: the domain follows the track along the coast (bypassing and alongshore storms flood far from landfall)
LF = {"stride": 2, "dt": 90.0, "forcing_every": 12, "t0": -18.0, "t1": 12.0, "half_lat": 2.0, "half_lon": 2.5,
      "coast_km": 250.0, "pad_lat": 1.25, "pad_lon": 1.5}
CELL_MIN_Z = -20.0  # keep shelf + land cells (drop open ocean)
CELL_MIN_ETA = 0.1  # m above MSL
_CAL_PATH = Path(__file__).resolve().parent.parent / "data" / "surge_calibration.json"


def coastal_mask(lat, lon, max_km: float = 30.0, max_elev_m: float = 20.0) -> np.ndarray:
    """Locations that storm surge can reach: within ``max_km`` of the coast and below ``max_elev_m`` (2′ DEM)."""
    from ..exposure.synthetic import _dist_to_coast_km
    from ..physics.dem import elevation

    z = elevation(lat, lon, fill=np.inf)
    return (_dist_to_coast_km(np.asarray(lat, float), np.asarray(lon, float)) < max_km) & (z < max_elev_m)


def _serial(f, name):
    """Serial, GIL-releasing clone of a parallel solver kernel (renamed so numba's cache doesn't collide)."""
    py = f.py_func
    g = types.FunctionType(py.__code__, py.__globals__, name, py.__defaults__, py.__closure__)
    g.__qualname__ = name
    return nb.njit(cache=True, nogil=True)(g)


_forcing_s = _serial(S._forcing, "_forcing_serial")
_blend_s = _serial(S._blend, "_blend_serial")
_step_s = _serial(S._step, "_step_serial")
_track_max_s = _serial(S._track_max, "_track_max_serial")


@nb.njit(cache=True, nogil=True)
def _integrate(tr, lat, lon, z, zc, dx, dxn, dy, fcor, nman, active, openb, t0, t1, dt, fevery):
    ny, nx = z.shape
    M = np.zeros((ny, nx + 1))
    N = np.zeros((ny + 1, nx))
    taux = np.zeros((ny, nx))
    tauy = np.zeros((ny, nx))
    ib = np.zeros((ny, nx))
    ta0 = np.zeros((ny, nx))
    tb0 = np.zeros((ny, nx))
    ib0 = np.zeros((ny, nx))
    fa = np.zeros((ny, nx))
    fb = np.zeros((ny, nx))
    fi = np.zeros((ny, nx))
    _forcing_s(t0, tr, lat, lon, z, taux, tauy, ib)
    eta = np.empty((ny, nx))
    for i in range(ny):
        for j in range(nx):
            eta[i, j] = ib[i, j] if z[i, j] < 0 else z[i, j]
    eta_max = eta.copy()
    t_max = np.full((ny, nx), np.nan)
    n_steps = int((t1 - t0) * 3600.0 / dt)
    for step in range(n_steps + 1):
        t = t0 + step * dt / 3600.0
        if step % fevery == 0:
            ta0[:] = taux
            tb0[:] = tauy
            ib0[:] = ib
            _forcing_s(t + fevery * dt / 3600.0, tr, lat, lon, z, taux, tauy, ib)
        w = (step % fevery) / fevery
        _blend_s(ta0, taux, w, fa)
        _blend_s(tb0, tauy, w, fb)
        _blend_s(ib0, ib, w, fi)
        _step_s(eta, M, N, z, zc, dx, dxn, dy, fcor, fa, fb, fi, nman, dt, active, openb)
        _track_max_s(eta, z, eta_max, t_max, t)
    return eta_max


def _domain(lat, lon, z):
    """Grid metrics shared with ``surge2d.run_surge``."""
    z = z.astype(np.float64)
    dlat = math.radians(float(lat[1] - lat[0]))
    dlon = math.radians(float(lon[1] - lon[0]))
    dy = S.R_EARTH * dlat
    dx = S.R_EARTH * dlon * np.cos(np.radians(lat))
    lat_faces = np.concatenate([[lat[0] - 0.5 * (lat[1] - lat[0])], 0.5 * (lat[1:] + lat[:-1]),
                                [lat[-1] + 0.5 * (lat[1] - lat[0])]])
    dxn = S.R_EARTH * dlon * np.cos(np.radians(lat_faces))
    fcor = 2 * 7.292e-5 * np.sin(np.radians(lat))
    active = z < S.Z_WALL
    zc = np.where(z < -S.H_CAP, -S.H_CAP, z)
    nman = np.where(z > 0, 0.045, 0.025)
    rim = np.zeros_like(active)
    rim[0, :] = rim[-1, :] = True
    rim[:, 0] = rim[:, -1] = True
    openb = rim & (z < -20.0)
    return z, zc, dx, dxn, dy, fcor, nman, active, openb


def landfall_point(tr) -> tuple[float, float]:
    i = int(np.argmin(np.abs(tr[0])))
    return float(tr[1][i]), float(tr[2][i])


def lf_domain(tr, lf: dict = LF) -> tuple[float, float, float, float]:
    """(lat0, lat1, lon0, lon1) of the low-fidelity run: the landfall box, united with the padded bounding
    box of the track points within ``coast_km`` of the coast over the run window. An alongshore or
    bypassing storm (one passing Tampa Bay offshore on its way to the Panhandle) floods coast far from
    its landfall, and the full model's track-following domain sees it."""
    from ..exposure.synthetic import _dist_to_coast_km

    la, lo = landfall_point(tr)
    box = [la - lf["half_lat"], la + lf["half_lat"], lo - lf["half_lon"], lo + lf["half_lon"]]
    if "coast_km" in lf:
        m = (tr[0] >= lf["t0"]) & (tr[0] <= lf["t1"])
        m[m] = _dist_to_coast_km(np.asarray(tr[1][m], float), np.asarray(tr[2][m], float)) <= lf["coast_km"]
        if m.any():
            box = [min(box[0], float(tr[1][m].min()) - lf["pad_lat"]), max(box[1], float(tr[1][m].max()) + lf["pad_lat"]),
                   min(box[2], float(tr[2][m].min()) - lf["pad_lon"]), max(box[3], float(tr[2][m].max()) + lf["pad_lon"])]
    return box[0], box[1], box[2], box[3]


def lf_event_cells(tr, lf: dict = LF) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low-fidelity run for one track → wet near-shore cells (lat, lon, peak water surface η)."""
    try:
        lat, lon, z = subgrid(*lf_domain(tr, lf))
    except ValueError:  # outside DEM coverage
        return np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    st = lf["stride"]
    lat, lon, z = lat[::st], lon[::st], z[::st, ::st]
    if lat.size < 4 or lon.size < 4:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    z, zc, dx, dxn, dy, fcor, nman, active, openb = _domain(lat, lon, z)
    trn = tuple(np.asarray(a, np.float64) for a in tr)
    eta_max = _integrate(trn, lat, lon, z, zc, dx, dxn, dy, fcor, nman, active, openb,
                         float(lf["t0"]), float(lf["t1"]), float(lf["dt"]), int(lf["forcing_every"]))
    keep = (eta_max - z > S.H_DRY) & (z > CELL_MIN_Z) & (eta_max > CELL_MIN_ETA)
    I, J = np.nonzero(keep)
    return lat[I].astype(np.float32), lon[J].astype(np.float32), eta_max[I, J].astype(np.float32)


@dataclass
class SurgeCells:
    """CSR over catalog events: wet near-shore cells with peak water level."""
    ptr: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    eta: np.ndarray
    lf: dict

    @property
    def n_cells(self) -> int:
        return int(self.ptr[-1])


def _tracks(cat, i):
    g = cat.geometry
    a, b = int(g["ptr"][i]), int(g["ptr"][i + 1])
    f = np.float64
    return (g["t"][a:b].astype(f), g["lat"][a:b].astype(f), g["lon"][a:b].astype(f), g["dp"][a:b].astype(f),
            g["rmax"][a:b].astype(f), g["b"][a:b].astype(f), g["heading"][a:b].astype(f), g["vt"][a:b].astype(f),
            g["vmax"][a:b].astype(f))


def catalog_signature(cat, lf: dict = LF, version: int = LF_VERSION) -> str:
    """Cache key of a catalog's surge product: the track geometry, the model spec and its version."""
    g = cat.geometry
    h = hashlib.sha1(json.dumps({"v": version, "lf": lf}, sort_keys=True).encode())
    h.update(np.ascontiguousarray(cat.events["event_id"].to_numpy(np.int64)).tobytes())
    for k in ("lat", "lon", "dp", "rmax"):
        h.update(np.ascontiguousarray(np.asarray(g[k], np.float32)).tobytes())
    return h.hexdigest()[:16]


def cache_dir() -> Path:
    return Path(os.environ.get("CATFORGE_CACHE_DIR") or Path.home() / ".cache" / "catforge") / "surge"


def catalog_surge(cat, events: np.ndarray | None = None, lf: dict = LF, threads: int | None = None,
                  progress=None, use_cache: bool = True) -> SurgeCells:
    """Low-fidelity surge for catalog events (all, or the subset ``events``), with a per-catalog disk cache."""
    n_ev = len(cat.events)
    todo = np.arange(n_ev) if events is None else np.unique(np.asarray(events, np.int64))
    path = cache_dir() / f"lf_{catalog_signature(cat, lf)}.npz"
    have: dict[int, tuple] = {}
    use_cache = use_cache and n_ev > 20  # single-event scenario catalogs are cheap; don't litter the cache
    if use_cache and path.exists():
        # materialise each array once: indexing a lazy NpzFile re-reads (and re-decompresses) the whole
        # array on every access, and each slice would pin its own full copy
        with np.load(path) as z:
            done, cptr, clat, clon, ceta = (z[k] for k in ("done", "ptr", "lat", "lon", "eta"))
        for k, i in enumerate(done):
            a, b = int(cptr[k]), int(cptr[k + 1])
            have[int(i)] = (clat[a:b], clon[a:b], ceta[a:b])
    missing = [int(i) for i in todo if int(i) not in have]
    if missing:
        threads = threads or max(1, min(os.cpu_count() or 1, 16))
        lf_event_cells(_tracks(cat, missing[0]), lf)  # compile once before fanning out
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = {i: ex.submit(lf_event_cells, _tracks(cat, i), lf) for i in missing}
            for k, (i, f) in enumerate(futs.items()):
                have[i] = f.result()
                if progress and (k % 50 == 0 or k == len(futs) - 1):
                    progress((k + 1) / len(futs))
        if use_cache:
            done = np.array(sorted(have), np.int64)
            lens = np.array([have[i][0].size for i in done], np.int64)
            ptr = np.concatenate([[0], np.cumsum(lens)])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, done=done, ptr=ptr,
                                lat=np.concatenate([have[i][0] for i in done]) if done.size else np.zeros(0, np.float32),
                                lon=np.concatenate([have[i][1] for i in done]) if done.size else np.zeros(0, np.float32),
                                eta=np.concatenate([have[i][2] for i in done]) if done.size else np.zeros(0, np.float32))
            os.replace(tmp, path)
    ptr = np.zeros(n_ev + 1, np.int64)
    for i in range(n_ev):
        ptr[i + 1] = ptr[i] + (have[i][0].size if i in have else 0)
    cat_ = lambda k: np.concatenate([have[i][k] for i in range(n_ev) if i in have]) if have else np.zeros(0, np.float32)  # noqa: E731
    return SurgeCells(ptr=ptr, lat=cat_(0), lon=cat_(1), eta=cat_(2), lf=dict(lf))


@nb.njit(parallel=True, cache=True)
def _site_wse(ev_idx, pair_ptr, pair_site, cptr, clat, clon, ceta, slat, slon, scoastal, alpha, r0, rmax):
    """WSE for every (event, site) pair: max over the event's cells of η − α·max(d − r0, 0), within rmax (km)."""
    out = np.full(pair_site.shape[0], np.nan, np.float32)
    for q in nb.prange(ev_idx.shape[0]):
        e = ev_idx[q]
        a, b = cptr[e], cptr[e + 1]
        if b <= a:
            continue
        for p in range(pair_ptr[e], pair_ptr[e + 1]):
            s = pair_site[p]
            if not scoastal[s]:
                continue
            best = -1e9
            kx = 111.32 * math.cos(slat[s] * 0.017453292519943295)
            for c in range(a, b):
                dy = (clat[c] - slat[s]) * 110.574
                if dy > rmax or dy < -rmax:
                    continue
                dxk = (clon[c] - slon[s]) * kx
                d = math.sqrt(dxk * dxk + dy * dy)
                if d > rmax:
                    continue
                v = ceta[c] - alpha * max(d - r0, 0.0)
                if v > best:
                    best = v
            if best > -1e8:
                out[p] = best
    return out


def site_wse(cells: SurgeCells, ev_ptr, pair_site, slat, slon, scoastal, alpha: float, r0: float, rmax: float,
             ev_idx=None) -> np.ndarray:
    """Pair-aligned low-fidelity WSE (NaN where no wet cell reaches the site)."""
    ev_idx = np.arange(len(ev_ptr) - 1, dtype=np.int64) if ev_idx is None else np.asarray(ev_idx, np.int64)
    return _site_wse(ev_idx, np.asarray(ev_ptr, np.int64), np.asarray(pair_site, np.int64), cells.ptr,
                     cells.lat, cells.lon, cells.eta, np.asarray(slat, float), np.asarray(slon, float),
                     np.asarray(scoastal, np.bool_), float(alpha), float(r0), float(rmax))


# ------------------------------------------------------------------------------------------ calibration
DEFAULT_CAL = {"model": "identity", "alpha_m_per_km": 0.3, "r0_km": 3.7, "rmax_km": 15.0, "sigma_event_m": 0.3,
               "sigma_site_m": 0.4, "tau_site_response_m": 0.0, "source": "uncalibrated default"}
KNOTS = (1.0, 2.0, 3.0, 4.0, 6.0)  # m: linear-spline knots of the low-fidelity → full-model level map
Z_CLIP = 10.0
_SITE_PATH = _CAL_PATH.with_name("surge_site_response.npz")
_site_map: dict | None = None


def calibration() -> dict:
    """LF → HF mapping and residual structure (``catforge/data/surge_calibration.json``)."""
    try:
        return {**DEFAULT_CAL, **json.loads(_CAL_PATH.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULT_CAL)


def node_key(lat, lon) -> np.ndarray:
    """2′ DEM cell of a location (the stencil centre ``run_surge`` rounds a site to), as one int64."""
    iy = np.rint((np.asarray(lat, float) + 90.0) * 30.0).astype(np.int64)
    ix = np.rint((np.asarray(lon, float) + 180.0) * 30.0).astype(np.int64)
    return iy * 20_000 + ix


def site_covariates(lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(node key, node ground clipped to 0…Z_CLIP m, shore flag) from the 2′ DEM at each location's node."""
    from ..physics.dem import elevation

    k = node_key(lat, lon)
    nlat, nlon = (k // 20_000) / 30.0 - 90.0, (k % 20_000) / 30.0 - 180.0
    z = elevation(nlat, nlon, fill=0.0)
    off = np.array([-1.0, 0.0, 1.0]) / 30.0
    smin = np.min([elevation(nlat + a, nlon + b, fill=0.0) for a in off for b in off], axis=0)
    return k, np.clip(z, 0.0, Z_CLIP), (smin < 0.0).astype(float)


def spline_basis(x, knots=KNOTS) -> np.ndarray:
    """Linear-spline design [1, x, (x − κ₁)₊, …]: continuous, linear beyond the last knot."""
    x = np.asarray(x, float)
    return np.stack([np.ones_like(x), x] + [np.maximum(x - k, 0.0) for k in knots], -1)


def level_design(x, dlt, zc, shore, knots=KNOTS) -> np.ndarray:
    """Wet-level design: spline(x) + [shore, z, x·shore, x·z] + [Δ, Δ·shore, min(Δ, 3)·x].

    Δ = x(α = 0) − x(α) is the friction loss the site rule applied. It is large when the site draws on
    distant water; in a bay that water travels over water and amplifies rather than attenuates.
    """
    x = np.asarray(x, float)
    dlt = np.asarray(dlt, float)
    return np.concatenate([spline_basis(x, knots), np.stack([shore, zc, x * shore, x * zc, dlt, dlt * shore,
                                                             np.minimum(dlt, 3.0) * x], -1)], -1)


def wet_features(x, dlt, zc, shore) -> np.ndarray:
    """Connectivity design: [1, x, z, shore, x·z, x − z, (x − z)₊, Δ, Δ·shore]."""
    x = np.asarray(x, float)
    dlt = np.asarray(dlt, float)
    return np.stack([np.ones_like(x), x, zc, shore, x * zc, x - zc, np.maximum(x - zc, 0.0), dlt, dlt * shore], -1)


def surge_levels(x, x_free, zc, shore, cal: dict) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated wet water level (without the site term) and P(wet), NaN/0 where no low-fidelity water.

    ``x`` is the site level at the calibrated α; ``x_free`` is the same extraction with α = 0.
    """
    x = np.asarray(x, float)
    fin = np.isfinite(x)
    xf = np.where(fin, x, 0.0)
    dlt = np.where(fin, np.asarray(x_free, float) - xf, 0.0)
    if cal.get("model") == "hurdle":
        lvl = level_design(xf, dlt, zc, shore, tuple(cal["knots"])) @ np.asarray(cal["level_beta"])
        eta = wet_features(xf, dlt, zc, shore) @ np.asarray(cal["wet_logit"])
        pw = 1.0 / (1.0 + np.exp(-eta))
    else:  # uncalibrated: the low-fidelity level as is, always connected
        lvl, pw = xf, np.ones_like(xf)
    return np.where(fin, lvl, np.nan).astype(np.float32), np.where(fin, pw, 0.0).astype(np.float32)


def event_error_chol(cal: dict) -> tuple[float, float, float, float]:
    """(x0, L11, L21, L22): Cholesky factor of the event error covariance of (u0, u1), where the error at
    level x is u0 + u1·(x − x0). Falls back to the intercept-only σ_e."""
    ee = cal.get("event_error")
    if not ee:
        return 2.0, float(cal.get("sigma_event_m", 0.0)), 0.0, 0.0
    L = np.linalg.cholesky(np.asarray(ee["cov"], float) + 1e-12 * np.eye(2))
    return float(ee["x0"]), float(L[0, 0]), float(L[1, 0]), float(L[1, 1])


def _site_response_map() -> dict:
    global _site_map
    if _site_map is None:
        try:
            d = np.load(_SITE_PATH)
            _site_map = {k: d[k] for k in ("key", "delta", "pv", "m")}
        except OSError:
            _site_map = {"key": np.zeros(0, np.int64), "delta": np.zeros(0, np.float32), "pv": np.zeros(0, np.float32),
                         "m": np.zeros(0, np.int16)}
    return _site_map


def site_response(lat, lon, cal: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrated site term at each location's 2′ node: (δ̂ m, its variance m², events it was seen in).

    Nodes outside the calibration set get the prior (0, τ², 0).
    """
    m = _site_response_map()
    k = node_key(lat, lon)
    tau2 = float(cal.get("tau_site_response_m", 0.0)) ** 2
    if not m["key"].size:
        return np.zeros(k.size), np.full(k.size, tau2), np.zeros(k.size, int)
    pos = np.minimum(np.searchsorted(m["key"], k), m["key"].size - 1)
    hit = m["key"][pos] == k
    return (np.where(hit, m["delta"][pos], 0.0).astype(float), np.where(hit, m["pv"][pos], tau2).astype(float),
            np.where(hit, m["m"][pos], 0).astype(int))


FIELD_COS_REF = math.cos(math.radians(30.0))  # knot lattice: fixed lon spacing so knots are shared by all sites
FIELD_RADIUS = 2.5  # kernel support, in widths (the dropped weight² mass is < 0.2 %)


def field_tables(node_keys: np.ndarray, active: np.ndarray, cal: dict, scale: float = 1.0):
    """Process-convolution weights of the correlated surge residual, from each 2′ node to lattice knots.

    For scale i (fraction s_i of σ², width h_i km), knots sit on a lattice of spacing h_i. A node's
    field value is σ√s_i · Σ_m w_m Z_m with Gaussian weights w_m ∝ exp(−d²/2h_i²), normalised so that
    Σ w_m² = 1. That gives each node exactly the variance σ²s_i and correlation
    ≈ exp(−d²/4h_i²) between nodes. Returns CSR arrays over the dense node index:
    (kptr, knot index, weight in metres, knot RNG key).
    """
    from ..rng import TAG_SURGE_FIELD

    n = node_keys.size
    fld = cal.get("field") or {}
    scales = fld.get("scales") or []
    sig = float(cal.get("sigma_site_m", 0.0)) * float(scale)
    kptr = np.zeros(n + 1, np.int64)
    if not scales or sig <= 0.0:
        return kptr, np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
    nlat = (node_keys // 20_000) / 30.0 - 90.0
    nlon = (node_keys % 20_000) / 30.0 - 180.0
    keys, wts, counts = [], [], np.zeros(n, np.int64)
    for i in np.nonzero(active)[0]:
        kx = 111.32 * math.cos(math.radians(nlat[i]))
        ks, ws = [], []
        for si, sc in enumerate(scales):
            h, amp = float(sc["h_km"]), sig * math.sqrt(float(sc["frac"]))
            if amp <= 0.0:
                continue
            dlat, dlon = h / 110.574, h / (111.32 * FIELD_COS_REF)
            r = FIELD_RADIUS * h
            iy = np.arange(math.floor((nlat[i] - r / 110.574) / dlat), math.ceil((nlat[i] + r / 110.574) / dlat) + 1)
            ix = np.arange(math.floor((nlon[i] - r / kx) / dlon), math.ceil((nlon[i] + r / kx) / dlon) + 1)
            IY, IX = np.meshgrid(iy, ix, indexing="ij")
            IY, IX = IY.ravel(), IX.ravel()
            d2 = ((IY * dlat - nlat[i]) * 110.574) ** 2 + ((IX * dlon - nlon[i]) * kx) ** 2
            m = d2 <= r * r
            w = np.exp(-0.5 * d2[m] / (h * h))
            ks.append(TAG_SURGE_FIELD + (si << 28) + (IY[m] + 8192) * 16384 + (IX[m] + 8192))
            ws.append(amp * w / math.sqrt(float(np.sum(w * w))))
        if ks:
            keys.append(np.concatenate(ks))
            wts.append(np.concatenate(ws))
            counts[i] = keys[-1].size
    kptr[1:] = np.cumsum(counts)
    if not keys:
        return kptr, np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
    uk, knot = np.unique(np.concatenate(keys), return_inverse=True)
    return kptr, knot.astype(np.int64), np.concatenate(wts), uk.astype(np.int64)


# ------------------------------------------------------------------------------------------ full fidelity
HF = {"t0": -18.0, "t1": 12.0, "max_cells": 400_000, "dt": 30.0, "forcing_every": 20, "pad_deg": 2.5}


HF_VERSION = 1  # independent of LF_VERSION: the full model's cache survives low-fidelity changes


def _hf_path(cat, i: int) -> Path:
    return cache_dir() / f"hf_{catalog_signature(cat, HF, HF_VERSION)}" / f"{int(i)}.npz"


def _hf_run(tr) -> dict:
    """The reference 2′ model (``run_surge`` with the HF spec): peak water surface, NaN where dry.

    Events run one after another: each run is parallel inside (numba ``prange``), which keeps the
    grid in cache. Several concurrent serial runs were measured 2–3× slower on 4 cores.
    """
    res = S.run_surge(tr, HF["t0"], HF["t1"], pad_deg=HF["pad_deg"], dt=HF["dt"], forcing_every=HF["forcing_every"],
                      max_cells=HF["max_cells"], keep_frames=False)
    return {"lat": res["lat"], "lon": res["lon"],
            "wse": np.where(res["depth_max"] > S.H_DRY, res["eta_max"], np.nan).astype(np.float32)}


def hf_event_fields(cat, events, use_cache: bool = True, progress=None) -> dict:
    """Full 2′ model for catalog events: {i: peak water surface on its grid, NaN where dry}.

    Portfolio-independent and disk-cached per catalog signature and event.
    """
    out, todo = {}, []
    for i in dict.fromkeys(int(v) for v in events):
        path = _hf_path(cat, i)
        if use_cache and path.exists():
            with np.load(path) as zf:
                out[i] = {"lat": zf["lat"], "lon": zf["lon"], "wse": zf["wse"]}
        else:
            todo.append(i)
    for k, i in enumerate(todo):
        out[i] = _hf_run(_tracks(cat, i))
        if use_cache:
            path = _hf_path(cat, i)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **out[i])
            os.replace(tmp, path)
        if progress:
            progress((k + 1) / len(todo), i)
    return out


def hf_event_field(cat, i: int, use_cache: bool = True) -> dict:
    """Full 2′ model for one catalog event (see ``hf_event_fields``)."""
    return hf_event_fields(cat, [i], use_cache=use_cache)[int(i)]


def hf_site_wse(fld: dict, lat, lon) -> np.ndarray:
    """Peak water surface in each site's 3×3 stencil of the full-model grid (the ``run_surge`` rule);
    NaN if the stencil stayed dry or the site lies outside the domain."""
    la, lo, w = fld["lat"], fld["lon"], fld["wse"]
    ny, nx = w.shape
    cy = np.rint((np.asarray(lat, float) - la[0]) / (la[1] - la[0])).astype(np.int64)
    cx = np.rint((np.asarray(lon, float) - lo[0]) / (lo[1] - lo[0])).astype(np.int64)
    out = np.full(cy.size, -np.inf, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            v = w[np.clip(cy + dy, 0, ny - 1), np.clip(cx + dx, 0, nx - 1)]
            out = np.where(np.isfinite(v), np.maximum(out, v), out)
    inside = (cy >= 0) & (cy < ny) & (cx >= 0) & (cx < nx)
    return np.where(inside & np.isfinite(out), out, np.nan).astype(np.float32)
