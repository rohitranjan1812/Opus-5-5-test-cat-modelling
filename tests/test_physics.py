import math
import struct
import zlib

import numpy as np
import pytest

from catforge.hazard.earthquake import single_rupture
from catforge.physics.dem import elevation, terrarium_tile
from catforge.physics.eq_dynamics import arrivals, build_fault, simulate_site
from catforge.physics.surge2d import _step, coastal_peak, run_surge
from catforge.physics.tc_dynamics import rclipper_mm_h, track_arrays
from catforge.scenario import ANALOGS, build_event


def test_dem_samples_known_relief():
    assert -5 < elevation(25.77, -80.19) < 15  # Miami: low-lying coast
    assert elevation(25.0, -90.0) < -2000  # deep Gulf of Mexico
    assert elevation(36.6, -118.3) > 2000  # Sierra Nevada
    assert np.isnan(elevation(0.0, 0.0))


def test_terrarium_tile_roundtrip():
    png = terrarium_tile(7, 34, 53)  # covers the Florida peninsula
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    idat = png.index(b"IDAT")
    ln = struct.unpack(">I", png[idat - 4:idat])[0]
    raw = zlib.decompress(png[idat + 4: idat + 4 + ln])
    rows = np.frombuffer(raw, np.uint8).reshape(h, 1 + 3 * w)[:, 1:].reshape(h, w, 3).astype(float)
    elev = rows[..., 0] * 256 + rows[..., 1] + rows[..., 2] / 256 - 32768
    assert elev.min() < -10 and elev.max() > 10  # both sea and land in the tile


def test_surge_solver_closed_basin_setup_and_mass():
    """Uniform wind stress on a closed flat basin: steady set-up Δη = τ L / (ρ g h), mass conserved."""
    ny, nx, dx_m, h, tau = 6, 60, 2000.0, 10.0, 1.0
    z = np.full((ny, nx), -h)
    eta = np.zeros((ny, nx))
    M = np.zeros((ny, nx + 1))
    N = np.zeros((ny + 1, nx))
    dx = np.full(ny, dx_m)
    dxn = np.full(ny + 1, dx_m)
    taux = np.full((ny, nx), tau)
    zero = np.zeros((ny, nx))
    nman = np.full((ny, nx), 0.08)
    active = np.ones((ny, nx), bool)
    openb = np.zeros((ny, nx), bool)
    acc = np.zeros(nx)
    n_avg = 0
    for step in range(30000):  # dt = 10 s → 83 h, damped seiches
        _step(eta, M, N, z, z, dx, dxn, dx_m, np.zeros(ny), taux, zero, zero, nman, 10.0, active, openb)
        if step >= 26000:
            acc += eta.mean(axis=0)
            n_avg += 1
    prof = acc / n_avg
    expected = tau * (nx - 1) * dx_m / (1025.0 * 9.81 * h)
    assert prof[-1] - prof[0] == pytest.approx(expected, rel=0.06)
    assert abs(eta.mean()) < 1e-9  # volume conserved in a closed basin


def test_surge_hugo_analog_peak_near_observed():
    cat = build_event("TC", ANALOGS["hugo_1989"]["params"])
    res = run_surge(track_arrays(cat), -18.0, 12.0)
    pk, la, lo = coastal_peak(res)
    assert 4.0 < pk < 8.0  # observed ~6 m near Cape Romain / Bulls Bay
    assert 32.5 < la < 33.6 and -80.2 < lo < -79.0
    assert res["inundated_land"].sum() >= 3  # coarse 4′ grid: only low-lying cells flood


def test_rclipper_profile():
    assert rclipper_mm_h(20.0, 60.0) > rclipper_mm_h(200.0, 60.0) > 0
    assert rclipper_mm_h(50.0, 70.0) > rclipper_mm_h(50.0, 40.0)


@pytest.fixture(scope="module")
def northridge():
    cat = single_rupture(34.28, -118.56, 6.7, strike=122, dip=40, ztor=5, zbot=21, width_km=21, mech="RV",
                         hypo_depth_km=17.5, hypo_along=0.55)
    ev = cat.events.iloc[0].to_dict()
    return cat, build_fault(ev, cat.geometry["tlat"], cat.geometry["tlon"], np.random.default_rng(4))


def test_fault_moment_and_rupture_times(northridge):
    _, f = northridge
    assert f["moment"].sum() == pytest.approx(f["M0"], rel=1e-9)
    assert f["t_rupture"].min() < 1.0 and f["duration"] > 3.0
    assert f["hypo"]["depth"] == pytest.approx(17.5, abs=1.5)
    tp, ts, te, tm = arrivals(f, np.array([f["hypo"]["lat"]]), np.array([f["hypo"]["lon"]]))
    assert tp[0] < ts[0] <= te[0]
    assert ts[0] == pytest.approx(f["hypo"]["depth"] / 3.5, rel=0.25)  # vertical S travel from the hypocentre


def test_stochastic_seismogram_consistent_with_gmpe(northridge):
    cat, f = northridge
    sims = [simulate_site(f, 34.20, -118.30, vs30=400.0, seed=s)["pga_g"] for s in range(6)]
    from catforge.hazard.footprint import Sites, compute_pairs

    p = compute_pairs(cat, Sites.plain([34.20], [-118.30], vs30=400.0), min_intensity=1e-4)
    gm = float(np.exp(p.log_i[0]))
    ratio = math.exp(np.mean(np.log(sims))) / gm
    assert 0.33 < ratio < 3.0
    out = simulate_site(f, 34.20, -118.30, vs30=400.0, seed=0)
    assert out["t_p"] < out["t_s"] and out["pgv_cms"] > 0 and out["psa_g"].max() > out["pga_g"]
