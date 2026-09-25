"""Kinematic finite-fault earthquake physics for event development.

Source
    Planar rupture (strike from the trace, dip δ, top depth z_tor, width W) discretised into
    sub-faults.  Seismic moment M₀ = 10^{1.5 M + 9.05} N·m, rigidity μ = 3.3·10¹⁰ Pa (4.0·10¹⁰ for
    subduction).  Slip is a heterogeneous von Kármán random field (Mai & Beroza 2002):
    a_x = 10^{−2.5 + M/2} km, a_z = 10^{−1.5 + M/3} km, Hurst H = 0.75, lognormal marginal with
    (σ_ln = 0.5), edge tapering (except a surface-breaking top edge), rescaled to conserve M₀.
    Rupture nucleates at the hypocentre and spreads at V_r = 0.8 β: t_r = |ξ − ξ_h| / V_r.

Wave arrivals
    t_P(x) = min_i [t_r,i + R_i/α],  t_S(x) = min_i [t_r,i + R_i/β]   (α = 6.2, β = 3.5 km/s),
    with R_i the 3-D distance from sub-fault i to the surface point — the exact first-arrival
    isochrones of a finite kinematic source in a uniform half-space (directivity included).

Synthetic ground motion — stochastic finite-fault method (EXSIM; Motazedian & Atkinson 2005)
    Each sub-fault is a Brune ω² point source with dynamic corner frequency
        f₀ᵢ(t) = N_R(t)^{−1/3} · 4.9·10⁶ β (Δσ / M₀,ave)^{1/3}
    and high-frequency scaling H_i = √(N Σ_f [f²/(1+(f/f₀)²)]² / Σ_f [f²/(1+(f/f₀ᵢ)²)]²).
    Its Fourier amplitude spectrum
        A_i(f) = C M₀ᵢ H_i (2πf)² / (1 + (f/f₀ᵢ)²) · G(R_i) · e^{−πfR_i/(Q(f)β)} · e^{−πκ₀f} · A_site(f)
    shapes windowed Gaussian noise (Boore 2003, Saragoni–Hart window), delayed by t_r,i + R_i/β
    and summed.  WNA: Q = 180 f^0.45, κ₀ = 0.035 s, G = 1/R (R ≤ 40 km) then R^{−0.5},
    Δσ = 50 bar; CEUS: Q = 680 f^0.36, κ₀ = 0.006 s, G: R^{−1.3}/flat/R^{−0.5} hinges at 70/140 km,
    Δσ = 140 bar.  Outputs acceleration, velocity (frequency-domain integration with a 0.05 Hz
    low-cut), Fourier spectrum and 5 %-damped pseudo-spectral acceleration.
"""

from __future__ import annotations

import math

import numpy as np

from ..geo import KM_PER_DEG_LAT, KM_PER_DEG_LON_EQ, Polyline, destination
from ..hazard.earthquake import CEUS_REGIONS, ln_pga_median
from .grf import von_karman_field

ALPHA = 6.2  # km/s
BETA = 3.5  # km/s (source & path S velocity)
VR_FACTOR = 0.8


def local_xy_km(lat, lon, lat0, lon0):
    """Vectorised equirectangular projection (km) about (lat0, lon0)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    x = (lon - lon0) * KM_PER_DEG_LON_EQ * np.cos(np.radians(0.5 * (lat + lat0)))
    y = (lat - lat0) * KM_PER_DEG_LAT
    return x, y


def build_fault(ev: dict, trace_lat, trace_lon, rng: np.random.Generator, max_subfaults: int = 900) -> dict:
    """Discretise the rupture plane, draw slip and compute rupture times."""
    M = float(ev["mag"])
    dip = float(ev.get("dip", 90.0) or 90.0)
    ztor = float(ev.get("ztor", 0.0) or 0.0)
    W = float(ev.get("width_km") or 10 ** (-1.01 + 0.32 * M))
    mech = str(ev.get("mech", "SS"))
    line = Polyline(np.asarray(trace_lon, float), np.asarray(trace_lat, float))
    L = max(float(line.length_km), 1.0)
    dx = max(1.0, math.sqrt(L * W / max_subfaults), L / 80.0)
    nL = int(np.clip(round(L / dx), 2, 120))
    nW = int(np.clip(round(W / dx), 2, 40))
    dL, dW = L / nL, W / nW
    s_c = (np.arange(nL) + 0.5) * dL
    w_c = (np.arange(nW) + 0.5) * dW
    lat_top, lon_top = line.point_at(s_c)
    strike = line.bearing_at(s_c)
    dipdir = (strike + 90.0) % 360.0
    sd, cd = math.sin(math.radians(dip)), math.cos(math.radians(dip))
    # sub-fault centres
    horiz = w_c[:, None] * cd + (ztor / math.tan(math.radians(dip)) if dip < 89.9 else 0.0)
    clat, clon = destination(np.broadcast_to(lat_top, (nW, nL)), np.broadcast_to(lon_top, (nW, nL)),
                             np.broadcast_to(dipdir, (nW, nL)), np.broadcast_to(horiz, (nW, nL)))
    depth = ztor + w_c[:, None] * sd * np.ones((1, nL))
    # corners (for 3-D rendering)
    s_e = np.arange(nL + 1) * dL
    w_e = np.arange(nW + 1) * dW
    la_e, lo_e = line.point_at(np.clip(s_e, 0, L - 1e-6))
    dd_e = (line.bearing_at(np.clip(s_e, 0, L - 1e-6)) + 90.0) % 360.0
    off = w_e[:, None] * cd + (ztor / math.tan(math.radians(dip)) if dip < 89.9 else 0.0)
    elat, elon = destination(np.broadcast_to(la_e, (nW + 1, nL + 1)), np.broadcast_to(lo_e, (nW + 1, nL + 1)),
                             np.broadcast_to(dd_e, (nW + 1, nL + 1)), np.broadcast_to(off, (nW + 1, nL + 1)))
    edep = ztor + w_e[:, None] * sd * np.ones((1, nL + 1))
    # slip: von Karman random field, lognormal marginal, edge taper, moment conservation
    ax = 10 ** (-2.5 + M / 2.0)
    az = 10 ** (-1.5 + M / 3.0)
    f = von_karman_field(nW, nL, dW, dL, min(az, 3 * W), min(ax, 3 * L), 0.75, rng)
    slip_rel = np.exp(0.5 * f - 0.125)
    tl = np.minimum(1.0, np.minimum(s_c, L - s_c) / (0.15 * L)) ** 0.5
    tw_bot = np.minimum(1.0, (W - w_c) / (0.2 * W)) ** 0.5
    tw_top = np.ones_like(w_c) if ztor <= 0.5 else np.minimum(1.0, w_c / (0.2 * W)) ** 0.5
    slip_rel *= (tw_bot * tw_top)[:, None] * tl[None, :]
    mu = 4.0e10 if mech == "SUB" else 3.3e10
    M0 = 10 ** (1.5 * M + 9.05)
    area = dL * dW * 1e6
    D_mean = M0 / (mu * L * W * 1e6)
    slip = slip_rel / slip_rel.mean() * D_mean
    # hypocentre and rupture times
    ha = ev.get("hypo_along")
    hd = ev.get("hypo_depth_km")
    ha = float(ha) if ha is not None and np.isfinite(ha) else float(rng.uniform(0.2, 0.8))
    wh = (float(hd) - ztor) / sd if hd is not None and np.isfinite(hd) else 0.65 * W
    wh = float(np.clip(wh, 0.5 * dW, W - 0.5 * dW))
    sh = ha * L
    vr = VR_FACTOR * BETA
    t_r = np.sqrt((s_c[None, :] - sh) ** 2 + (w_c[:, None] - wh) ** 2) / vr
    hlat, hlon = destination(np.array([float(line.point_at(sh)[0])]), np.array([float(line.point_at(sh)[1])]),
                             np.array([float((line.bearing_at(sh) + 90.0) % 360.0)]),
                             np.array([wh * cd + (ztor / math.tan(math.radians(dip)) if dip < 89.9 else 0.0)]))
    return {
        "M": M, "M0": M0, "mu": mu, "L": L, "W": W, "dip": dip, "ztor": ztor, "mech": mech, "nL": nL, "nW": nW,
        "dL": dL, "dW": dW, "lat": clat, "lon": clon, "depth": depth, "slip": slip, "t_rupture": t_r,
        "moment": mu * area * slip, "corner_lat": elat, "corner_lon": elon, "corner_depth": edep,
        "hypo": {"lat": float(hlat[0]), "lon": float(hlon[0]), "depth": ztor + wh * sd, "along": ha},
        "vr": vr, "duration": float(t_r.max()), "D_mean": D_mean, "D_max": float(slip.max()),
        "region": ev.get("region", ""), "anelastic": float(ev.get("anelastic", 1.0)),
    }


def arrivals(fault: dict, lat: np.ndarray, lon: np.ndarray, weight_frac: float = 0.15):
    """First P and S arrival times and end of significant direct-S window at surface points."""
    flat = fault["lat"].ravel()
    flon = fault["lon"].ravel()
    fdep = fault["depth"].ravel()
    tr = fault["t_rupture"].ravel()
    mom = fault["moment"].ravel()
    sig = mom >= weight_frac * mom.max()
    lat0, lon0 = float(flat.mean()), float(flon.mean())
    fx, fy = local_xy_km(flat, flon, lat0, lon0)
    px, py = local_xy_km(np.asarray(lat, float), np.asarray(lon, float), lat0, lon0)
    tP = np.full(px.shape, np.inf)
    tS = np.full(px.shape, np.inf)
    tE = np.zeros(px.shape)
    wsum = np.zeros(px.shape)
    tmean = np.zeros(px.shape)
    for k in range(flat.size):
        R = np.sqrt((px - fx[k]) ** 2 + (py - fy[k]) ** 2 + fdep[k] ** 2)
        tP = np.minimum(tP, tr[k] + R / ALPHA)
        ts = tr[k] + R / BETA
        tS = np.minimum(tS, ts)
        if sig[k]:
            tE = np.maximum(tE, ts)
            w = mom[k] / np.maximum(R, 1.0)
            wsum += w
            tmean += w * ts
    return tP, tS, tE, tmean / np.maximum(wsum, 1e-30)


# ---------------------------------------------------------------------------- stochastic simulation
def _path_params(region: str):
    if region in CEUS_REGIONS:
        return {"dsigma": 140.0, "kappa": 0.006, "q0": 680.0, "qn": 0.36, "ceus": True}
    return {"dsigma": 50.0, "kappa": 0.035, "q0": 180.0, "qn": 0.45, "ceus": False}


def _geom(R, ceus):
    R = np.maximum(R, 1.0)
    if ceus:
        return np.where(R <= 70, R ** -1.3, np.where(R <= 140, 70 ** -1.3, 70 ** -1.3 * (R / 140.0) ** -0.5))
    return np.where(R <= 40, 1.0 / R, (1.0 / 40.0) * (R / 40.0) ** -0.5)


def _site_amp(f, vs30):
    """Generic crustal amplification (Boore–Joyner-like shape) scaled to Vs30 (BA08 linear term)."""
    a = 1.0 + 1.9 * (f / (f + 1.5)) ** 0.9
    return a * (vs30 / 620.0) ** -0.36


def _window(n, dt, T):
    eps, eta = 0.2, 0.05
    b = -eps * math.log(eta) / (1.0 + eps * (math.log(eps) - 1.0))
    c = b / eps
    a = (math.e / eps) ** b
    tn = 2.0 * T
    t = np.arange(n) * dt
    return a * (t / tn) ** b * np.exp(-c * t / tn)


def simulate_site(fault: dict, lat: float, lon: float, vs30: float = 400.0, seed: int = 1, dt: float = 0.01,
                  max_subfaults: int = 400) -> dict:
    """EXSIM-style stochastic finite-fault accelerogram at a surface site."""
    rng = np.random.default_rng(seed)
    pp = _path_params(fault.get("region", ""))
    flat = fault["lat"].ravel()
    flon = fault["lon"].ravel()
    fdep = fault["depth"].ravel()
    tr = fault["t_rupture"].ravel()
    mom = fault["moment"].ravel()
    # coarsen very fine sources by merging neighbours (keeps the method fast for M9)
    N = flat.size
    if N > max_subfaults:
        step = int(math.ceil(math.sqrt(N / max_subfaults)))
        nW, nL = fault["nW"], fault["nL"]
        idx = [(i, j) for i in range(0, nW, step) for j in range(0, nL, step)]
        def agg(a):
            a = a.reshape(nW, nL)
            return np.array([a[i:i + step, j:j + step].mean() for i, j in idx])
        flat, flon, fdep, tr = agg(fault["lat"]), agg(fault["lon"]), agg(fault["depth"]), agg(fault["t_rupture"])
        mom = np.array([fault["moment"][i:i + step, j:j + step].sum() for i, j in idx])
        N = flat.size
    x, y = local_xy_km(flat, flon, lat, lon)
    R = np.sqrt(x ** 2 + y ** 2 + fdep ** 2)
    M0_dyne = fault["M0"] * 1e7
    beta_cm = 3.7e5
    C = 0.55 * 0.7071 * 2.0 / (4 * math.pi * 2.8 * beta_cm ** 3 * 1e5)
    M0ave = M0_dyne / N
    f0_static = 4.9e6 * 3.7 * (pp["dsigma"] / M0_dyne) ** (1.0 / 3.0)
    order = np.argsort(tr)
    nr = np.empty(N)
    nr[order] = np.arange(1, N + 1)
    f0i = nr ** (-1.0 / 3.0) * 4.9e6 * 3.7 * (pp["dsigma"] / M0ave) ** (1.0 / 3.0)
    f0i = np.maximum(f0i, f0_static)
    t_arr = tr + R / BETA
    T_src = 1.0 / f0i
    T_path = 0.05 * R if not pp["ceus"] else 0.05 * R
    Tw = T_src + T_path
    total = float((t_arr + 2.5 * Tw).max()) + 10.0
    n = int(2 ** math.ceil(math.log2(total / dt)))
    acc = np.zeros(n)
    fgrid = np.fft.rfftfreq(n, dt)
    fr = np.maximum(fgrid, 1e-3)
    sub_hf_num = (fr ** 2 / (1 + (fr / f0_static) ** 2)) ** 2
    for k in range(N):
        nw = int(max(2.5 * Tw[k] / dt, 64))
        nwin = 2 ** int(math.ceil(math.log2(nw)))
        noise = rng.standard_normal(nwin) * _window(nwin, dt, Tw[k])
        spec = np.fft.rfft(noise)
        fk = np.fft.rfftfreq(nwin, dt)
        fk_ = np.maximum(fk, 1e-3)
        spec /= np.sqrt(np.mean(np.abs(spec) ** 2))
        H = math.sqrt(N * sub_hf_num.sum() / ((fr ** 2 / (1 + (fr / f0i[k]) ** 2)) ** 2).sum())
        m0k = M0_dyne * mom[k] / mom.sum()
        Q = pp["q0"] * fk_ ** pp["qn"]
        amp = (C * m0k * H * (2 * math.pi * fk_) ** 2 / (1 + (fk_ / f0i[k]) ** 2) * _geom(R[k], pp["ceus"]) *
               np.exp(-math.pi * fk_ * R[k] / (Q * 3.7)) * np.exp(-math.pi * pp["kappa"] * fk_) * _site_amp(fk_, vs30))
        amp[0] = 0.0
        a_k = np.fft.irfft(spec * amp, nwin) / dt
        i0 = int(round(t_arr[k] / dt))
        i1 = min(i0 + nwin, n)
        if i0 < n:
            acc[i0:i1] += a_k[: i1 - i0]
    acc_g = acc / 981.0
    A = np.fft.rfft(acc)
    w = 2 * math.pi * np.maximum(fgrid, 1e-6)
    lowcut = 1.0 / (1.0 + (0.05 / np.maximum(fgrid, 1e-6)) ** 8)
    vel = np.fft.irfft(A * lowcut / (1j * w), n)  # cm/s
    vel[0] = 0.0
    periods = np.geomspace(0.03, 5.0, 36)
    psa = []
    for Tn in periods:
        wn = 2 * math.pi / Tn
        Hs = -1.0 / (wn ** 2 - w ** 2 + 2j * 0.05 * wn * w)
        u = np.fft.irfft(A * Hs, n)
        psa.append(float(wn ** 2 * np.abs(u).max() / 981.0))
    fas = np.abs(A) * dt
    x0, y0 = local_xy_km(np.array([fault["hypo"]["lat"]]), np.array([fault["hypo"]["lon"]]), lat, lon)
    return {"dt": dt, "n": n, "acc_g": acc_g, "vel_cms": vel, "pga_g": float(np.abs(acc_g).max()),
            "pgv_cms": float(np.abs(vel).max()), "periods": periods, "psa_g": np.array(psa),
            "freq": fgrid, "fas": fas, "t_p": float((tr + np.sqrt(x ** 2 + y ** 2 + fdep ** 2) / ALPHA).min()),
            "t_s": float(t_arr.min()), "n_subfaults": N,
            "epicentral_km": float(math.hypot(x0[0], y0[0])), "params": pp}


def gmpe_median(fault: dict, lat: float, lon: float, vs30: float, rjb: float, h: float) -> float:
    return float(math.exp(ln_pga_median(fault["M"], rjb, h, vs30, fault["anelastic"])))
