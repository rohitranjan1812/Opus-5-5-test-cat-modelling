"""2-D depth-averaged storm-surge model on ETOPO1 bathymetry/topography (with inundation).

Equations (vertically integrated shallow water, flux form, advection neglected — the SLOSH /
linear-momentum approximation appropriate for storm surge), on a spherical Arakawa C-grid:

    ∂η/∂t + ∇·(M, N) = 0
    ∂M/∂t = −g H ∂(η − η_ib)/∂x + f N + τ_sx/ρ_w − C_f M |𝐌| / H²
    ∂N/∂t = −g H ∂(η − η_ib)/∂y − f M + τ_sy/ρ_w − C_f N |𝐌| / H²

η water level (m above MSL), (M, N) volume fluxes (m²/s), H total depth, η_ib = Δp/(ρ_w g) the
inverse-barometer level of the Holland pressure field, τ = ρ_a C_d |U|U (Garratt 1977, capped at
2.5·10⁻³ per Powell et al. 2003) with the 1-min marine wind converted to a 10-min mean (×0.93)
and reduced ×0.6 over land, C_f = g n²/H^{1/3} (Manning n = 0.025 sea, 0.045 land).

Numerics: forward–backward explicit time stepping with semi-implicit friction; face depths use
the "highest bed" rule H_f = max(η_L, η_R) − max(z_L, z_R) for robust wetting and drying; a
positivity limiter caps the outflow of each cell to its available volume.  Open-ocean boundary
cells are clamped to η_ib.  Depths are capped at 200 m for the gravity-wave speed (the deep ocean
contributes only the inverse barometer), allowing Δt ≈ 30 s at 2 arc-minutes.
Land up to +15 m can flood.  Omitted: astronomical tide, wave set-up, riverine inflow.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ..hazard.tropical_cyclone import pressure_deficit, wind_uv
from .dem import elevation, subgrid
from .tc_dynamics import track_state

RHO_W = 1025.0
RHO_A = 1.15
G = 9.81
H_DRY = 0.05
SITE_GROUND_MIN = 1.0  # m: a building stands on land even where its coarse cell averages to sea
H_CAP = 200.0
Z_WALL = 15.0
R_EARTH = 6371000.0


@nb.njit(parallel=True, cache=True)
def _forcing(t, tr, lat, lon, z, taux, tauy, ib):
    tt, tlat, tlon, tdp, trm, tb, thdg, tvt, tvmax = tr
    clat, clon, dp, rm, bh, hdg, vt, _ = track_state(t, tt, tlat, tlon, tdp, trm, tb, thdg, tvt, tvmax)
    ny, nx = z.shape
    for i in nb.prange(ny):
        for j in range(nx):
            dyk = (lat[i] - clat) * 110.574
            dxk = (lon[j] - clon) * 111.32 * math.cos(lat[i] * 0.0174533)
            if dxk * dxk + dyk * dyk > 800.0 * 800.0:
                taux[i, j] = 0.0
                tauy[i, j] = 0.0
                ib[i, j] = 0.0
                continue
            u, v = wind_uv(lat[i], lon[j], clat, clon, dp, rm, bh, hdg, vt)
            u *= 0.93
            v *= 0.93
            if z[i, j] > 0.0:
                u *= 0.6
                v *= 0.6
            sp = math.sqrt(u * u + v * v)
            cd = min((0.75 + 0.067 * sp) * 1e-3, 2.5e-3)
            taux[i, j] = RHO_A * cd * sp * u
            tauy[i, j] = RHO_A * cd * sp * v
            ib[i, j] = pressure_deficit(lat[i], lon[j], clat, clon, dp, rm, bh) / (RHO_W * G)


@nb.njit(parallel=True, cache=True)
def _blend(a0, a1, w, out):
    ny, nx = a0.shape
    for i in nb.prange(ny):
        for j in range(nx):
            out[i, j] = (1.0 - w) * a0[i, j] + w * a1[i, j]


@nb.njit(parallel=True, cache=True)
def _track_max(eta, z, eta_max, t_max, t):
    ny, nx = z.shape
    for i in nb.prange(ny):
        for j in range(nx):
            if eta[i, j] > eta_max[i, j]:
                eta_max[i, j] = eta[i, j]
                if eta[i, j] - z[i, j] > H_DRY:
                    t_max[i, j] = t


@nb.njit(parallel=True, cache=True)
def _step(eta, M, N, z, zc, dx, dxn, dy, fcor, taux, tauy, ib, nman, dt, active, openb):
    ny, nx = z.shape
    # ---- x-fluxes on interior east faces (between j-1 and j) ----
    for i in nb.prange(ny):
        for j in range(1, nx):
            if not (active[i, j] and active[i, j - 1]):
                M[i, j] = 0.0
                continue
            hf = max(eta[i, j], eta[i, j - 1]) - max(zc[i, j], zc[i, j - 1])
            if hf <= H_DRY:
                M[i, j] = 0.0
                continue
            hf = min(hf, H_CAP)
            nv = 0.25 * (N[i, j] + N[i + 1, j] + N[i, j - 1] + N[i + 1, j - 1])
            grad = ((eta[i, j] - ib[i, j]) - (eta[i, j - 1] - ib[i, j - 1])) / dx[i]
            tx = 0.5 * (taux[i, j] + taux[i, j - 1])
            n_ = 0.5 * (nman[i, j] + nman[i, j - 1])
            cf = G * n_ * n_ / hf ** (1.0 / 3.0)
            m = M[i, j] + dt * (-G * hf * grad + fcor[i] * nv + tx / RHO_W)
            M[i, j] = m / (1.0 + dt * cf * math.sqrt(M[i, j] ** 2 + nv * nv) / (hf * hf))
    # ---- y-fluxes on interior north faces (between i-1 and i) ----
    for i in nb.prange(1, ny):
        for j in range(nx):
            if not (active[i, j] and active[i - 1, j]):
                N[i, j] = 0.0
                continue
            hf = max(eta[i, j], eta[i - 1, j]) - max(zc[i, j], zc[i - 1, j])
            if hf <= H_DRY:
                N[i, j] = 0.0
                continue
            hf = min(hf, H_CAP)
            mv = 0.25 * (M[i, j] + M[i, j + 1] + M[i - 1, j] + M[i - 1, j + 1])
            grad = ((eta[i, j] - ib[i, j]) - (eta[i - 1, j] - ib[i - 1, j])) / dy
            ty = 0.5 * (tauy[i, j] + tauy[i - 1, j])
            n_ = 0.5 * (nman[i, j] + nman[i - 1, j])
            cf = G * n_ * n_ / hf ** (1.0 / 3.0)
            fc = 0.5 * (fcor[i] + fcor[i - 1])
            nn = N[i, j] + dt * (-G * hf * grad - fc * mv + ty / RHO_W)
            N[i, j] = nn / (1.0 + dt * cf * math.sqrt(N[i, j] ** 2 + mv * mv) / (hf * hf))
    # ---- positivity limiter: a cell cannot export more water than it holds ----
    for i in nb.prange(ny):
        for j in range(nx):
            if not active[i, j]:
                continue
            h_avail = eta[i, j] - z[i, j]
            if h_avail <= 0.0:
                h_avail = 0.0
            area = dx[i] * dy
            out = (max(M[i, j + 1], 0.0) * dy + max(-M[i, j], 0.0) * dy +
                   max(N[i + 1, j], 0.0) * dxn[i + 1] + max(-N[i, j], 0.0) * dxn[i]) * dt
            vol = h_avail * area
            if out > vol and out > 0.0:
                s = vol / out
                if M[i, j + 1] > 0.0:
                    M[i, j + 1] *= s
                if M[i, j] < 0.0:
                    M[i, j] *= s
                if N[i + 1, j] > 0.0:
                    N[i + 1, j] *= s
                if N[i, j] < 0.0:
                    N[i, j] *= s
    # ---- continuity ----
    for i in nb.prange(ny):
        for j in range(nx):
            if not active[i, j]:
                continue
            if openb[i, j]:
                eta[i, j] = ib[i, j]
                continue
            div = ((M[i, j + 1] - M[i, j]) * dy + N[i + 1, j] * dxn[i + 1] - N[i, j] * dxn[i]) / (dx[i] * dy)
            e = eta[i, j] - dt * div
            if e < z[i, j]:
                e = z[i, j]
            eta[i, j] = e


def run_surge(tr, t_start: float, t_end: float, pad_deg: float = 2.5, dt: float = 30.0, forcing_every: int = 20,
              frame_hours: float = 1.0, max_cells: int = 160_000, sites_lat=None, sites_lon=None, sites_ground=None):
    """Integrate the surge model along a track; returns frames, maxima and site water levels."""
    tt, tlat, tlon = tr[0], tr[1], tr[2]
    m = (tt >= t_start - 1) & (tt <= t_end + 1)
    la0, la1 = float(tlat[m].min()) - pad_deg, float(tlat[m].max()) + pad_deg
    lo0, lo1 = float(tlon[m].min()) - pad_deg - 0.5, float(tlon[m].max()) + pad_deg + 0.5
    lat, lon, z = subgrid(la0, la1, lo0, lo1)
    stride = 1
    while (lat.size // stride) * (lon.size // stride) > max_cells:
        stride += 1
    if stride > 1:
        lat, lon, z = lat[::stride], lon[::stride], z[::stride, ::stride]
    z = z.astype(np.float64)
    ny, nx = z.shape
    dlat = math.radians(float(lat[1] - lat[0]))
    dlon = math.radians(float(lon[1] - lon[0]))
    dy = R_EARTH * dlat
    dx = R_EARTH * dlon * np.cos(np.radians(lat))
    lat_faces = np.concatenate([[lat[0] - 0.5 * (lat[1] - lat[0])], 0.5 * (lat[1:] + lat[:-1]),
                                [lat[-1] + 0.5 * (lat[1] - lat[0])]])
    dxn = R_EARTH * dlon * np.cos(np.radians(lat_faces))
    fcor = 2 * 7.292e-5 * np.sin(np.radians(lat))
    active = z < Z_WALL
    zc = np.where(z < -H_CAP, -H_CAP, z)  # capped bed used for face depths (wave-speed cap)
    nman = np.where(z > 0, 0.045, 0.025)
    # open boundary: deep-water cells on the domain rim
    rim = np.zeros_like(active)
    rim[0, :] = rim[-1, :] = True
    rim[:, 0] = rim[:, -1] = True
    openb = rim & (z < -20.0)
    eta = np.where(z < 0, 0.0, z)
    M = np.zeros((ny, nx + 1))
    N = np.zeros((ny + 1, nx))
    taux = np.zeros((ny, nx))
    tauy = np.zeros((ny, nx))
    ib = np.zeros((ny, nx))
    ta0, tb0 = np.zeros_like(taux), np.zeros_like(taux)
    ib0 = np.zeros_like(ib)
    trn = tuple(np.asarray(a, np.float64) for a in tr)
    _forcing(t_start, trn, lat, lon, z, taux, tauy, ib)
    eta = np.where(z < 0, ib, z)
    eta_max = eta.copy()
    t_max = np.full(z.shape, np.nan)
    frames, frame_t = [], []
    next_frame = t_start
    n_steps = int((t_end - t_start) * 3600.0 / dt)
    sy = sx = zs = None
    if sites_lat is not None and len(sites_lat):
        # sub-grid inundation: building ground (full-res DEM) vs the water surface of wet cells in its 3×3 stencil
        cy = np.rint((np.asarray(sites_lat) - lat[0]) / (lat[1] - lat[0])).astype(int)
        cx = np.rint((np.asarray(sites_lon) - lon[0]) / (lon[1] - lon[0])).astype(int)
        off = np.array([-1, 0, 1])
        sy = np.clip(np.repeat(cy[:, None] + off[None, :], 3, axis=1), 0, ny - 1)
        sx = np.clip(np.tile(cx[:, None] + off[None, :], (1, 3)), 0, nx - 1)
        zs = np.maximum(elevation(np.asarray(sites_lat, float), np.asarray(sites_lon, float), fill=SITE_GROUND_MIN),
                        SITE_GROUND_MIN)
        if sites_ground is not None:  # measured building ground (e.g. 3DEP) replaces the coarse-cell estimate
            g = np.asarray(sites_ground, float)
            zs = np.where(np.isfinite(g), g, zs)

    def site_depth(level, wet):
        ws = np.where(wet[sy, sx], level[sy, sx], -np.inf).max(axis=1)
        return np.maximum(ws - zs, 0.0)
    site_frames = []
    fa = np.zeros_like(taux)
    fb = np.zeros_like(tauy)
    fi = np.zeros_like(ib)
    for step in range(n_steps + 1):
        t = t_start + step * dt / 3600.0
        if step % forcing_every == 0:
            ta0[:], tb0[:], ib0[:] = taux, tauy, ib
            _forcing(t + forcing_every * dt / 3600.0, trn, lat, lon, z, taux, tauy, ib)
        w = (step % forcing_every) / forcing_every
        _blend(ta0, taux, w, fa)
        _blend(tb0, tauy, w, fb)
        _blend(ib0, ib, w, fi)
        _step(eta, M, N, z, zc, dx, dxn, dy, fcor, fa, fb, fi, nman, dt, active, openb)
        _track_max(eta, z, eta_max, t_max, t)
        if t >= next_frame - 1e-9:
            frames.append(np.where(eta - z > H_DRY, eta, np.nan).astype(np.float32))
            frame_t.append(t)
            if sy is not None:
                site_frames.append(site_depth(eta, eta - z > H_DRY))
            next_frame += frame_hours
    depth_max = np.maximum(eta_max - z, 0.0)
    return {
        "lat": lat, "lon": lon, "z": z, "frames": frames, "frame_t": np.array(frame_t), "eta_max": eta_max,
        "depth_max": depth_max, "t_max": t_max,
        "inundated_land": (z > 0) & (depth_max > H_DRY),
        "site_depth_frames": np.array(site_frames).T if site_frames else None,
        "site_depth_max": (site_depth(eta_max, depth_max > H_DRY) if sy is not None else None),
        "dt": dt, "stride_deg": float(lat[1] - lat[0]),
    }


def coastal_peak(res: dict, band_km: float = 25.0):
    """Peak water level over sea/shore cells near the coast (m above MSL) and its location."""
    z = res["z"]
    em = res["eta_max"]
    coastal = (z < 5.0) & (z > -10.0) & (res["depth_max"] > H_DRY)  # only cells that were actually wet
    val = np.where(coastal & np.isfinite(em), em, -np.inf)
    i, j = np.unravel_index(int(np.argmax(val)), val.shape)
    return float(val[i, j]), float(res["lat"][i]), float(res["lon"][j])
