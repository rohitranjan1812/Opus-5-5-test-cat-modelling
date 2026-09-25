import numpy as np
import pandas as pd
import pytest

from catforge.engine.model import AnalysisConfig, CatModel, build_context
from catforge.exposure.portfolio import Portfolio
from catforge.hazard.surge import (
    SurgeCells,
    calibration,
    coastal_mask,
    lf_event_cells,
    site_response,
    site_wse,
)
from catforge.physics.tc_dynamics import track_arrays
from catforge.scenario import ANALOGS, build_event, run_scenario


def test_site_wse_friction_limited_penetration():
    # one event, two wet cells; site 0 sits 1 km from the 3 m cell, site 1 is 10 km inland, site 2 is not coastal
    cells = SurgeCells(ptr=np.array([0, 2]), lat=np.array([30.0, 30.0], np.float32),
                       lon=np.array([-90.0, -90.2], np.float32), eta=np.array([3.0, 2.0], np.float32), lf={})
    slat = np.array([30.009, 30.09, 30.0])
    slon = np.array([-90.0, -90.0, -90.0])
    w = site_wse(cells, np.array([0, 3]), np.array([0, 1, 2]), slat, slon, np.array([True, True, False]),
                 alpha=0.3, r0=2.0, rmax=15.0)
    d1 = np.hypot(0, 0.09 * 110.574)  # ≈ 9.95 km
    assert w[0] == pytest.approx(3.0, abs=1e-5)  # inside r0: no attenuation
    assert w[1] == pytest.approx(3.0 - 0.3 * (d1 - 2.0), abs=0.02)
    assert np.isnan(w[2])  # not surge-reachable


def test_low_fidelity_surge_tracks_the_2d_model():
    tr = track_arrays(build_event("TC", ANALOGS["hugo_1989"]["params"]))
    la, lo, eta = lf_event_cells(tr)
    assert eta.size > 100 and 4.5 < eta.max() < 7.5  # full model 5.9 m, observed ≈6 m near Bulls Bay
    k = int(np.argmax(eta))
    assert 32.6 < la[k] < 33.4 and -80.2 < lo[k] < -79.0


def _book():
    # Charleston waterfront (low), the same spot on 12 m high ground, and inland Columbia SC
    df = pd.DataFrame({"loc_id": ["low", "high", "inland"], "lat": [32.776, 32.776, 34.00], "lon": [-79.93, -79.93, -81.03],
                       "tiv_building": [5e5, 5e5, 5e5], "construction": "WOOD", "occupancy": "RES_SF", "year_built": 1980,
                       "ground_elev_m": [1.0, 12.0, np.nan]})
    return Portfolio.from_frame(df, name="surge-test")


def test_kernel_adds_surge_only_through_water_above_ground():
    pf = _book()
    assert list(coastal_mask(pf.locations["lat"], pf.locations["lon"])) == [True, True, False]
    cat = build_event("TC", ANALOGS["hugo_1989"]["params"])
    model = CatModel({"TC": cat})
    on = build_context(model, pf, AnalysisConfig(perils=["TC"], seed=5))
    off = build_context(model, pf, AnalysisConfig(perils=["TC"], seed=5, tc_surge=False))
    assert np.isfinite(on.pair_wse).any() and not np.isfinite(off.pair_wse).any()
    keys = np.arange(400, dtype=np.uint64)
    ev = np.zeros(400, np.int64)
    gu_on, *_, sg = on.kernel(ev, keys, want_loc=True, split=True)
    gu_off = off.kernel(ev, keys, want_loc=True)[0]
    loc_on = on.kernel(ev, keys, np.full(400, 1 / 400), want_loc=True)[5]
    loc_off = off.kernel(ev, keys, np.full(400, 1 / 400), want_loc=True)[5]
    j = {lid: i for i, lid in enumerate(pf.locations["loc_id"])}
    assert loc_on[j["low"]] > 1.5 * loc_off[j["low"]]  # waterfront at 1 m: surge dominates
    assert loc_on[j["high"]] == pytest.approx(loc_off[j["high"]], rel=1e-9)  # same water, 12 m ground: dry
    assert loc_on[j["inland"]] == pytest.approx(loc_off[j["inland"]], rel=1e-9)
    assert np.allclose(gu_on - sg, gu_off)  # wind draws are unchanged: the split is exact
    again = on.kernel(ev, keys, split=True)
    assert np.array_equal(again[0], gu_on)  # counter RNG: bit-identical


def test_scenario_and_calibration_defaults():
    cal = calibration()
    assert cal["model"] == "hurdle" and {"wet_logit", "level_beta", "knots", "alpha_m_per_km", "sigma_event_m",
                                        "sigma_site_m", "tau_site_response_m"} <= set(cal)
    # Charleston harbour: the 4′ grid barely resolves it, so the calibrated site term lifts the water level
    delta, var, seen = site_response(np.array([32.776]), np.array([-79.93]), cal)
    assert seen[0] > 0 and delta[0] > 0.1 and var[0] < cal["tau_site_response_m"] ** 2
    pf = _book()
    with_s = run_scenario(pf, "TC", ANALOGS["hugo_1989"]["params"], n_samples=200, with_footprint=False)
    base = CatModel({"TC": build_event("TC", ANALOGS["hugo_1989"]["params"])})
    ctx = build_context(base, pf, AnalysisConfig(perils=["TC"], seed=99, tc_surge=False))
    wind_only = ctx.kernel(np.zeros(200, np.int64), np.arange(200, dtype=np.uint64) + np.uint64(7 << 40))[0].mean()
    assert with_s["gu"]["mean"] > wind_only


def test_crossed_mixed_model_recovers_truth_where_naive_ols_is_biased():
    from catforge.hazard.surge_calib import fit_mixed, naive_fit

    rng = np.random.default_rng(1)
    ne, nn, per = 60, 800, 300
    ev = np.repeat(np.arange(ne), per)
    node = np.concatenate([rng.choice(nn, per, replace=False) for _ in range(ne)])
    x = rng.gamma(2.0, 1.0, ev.size)
    u, d = rng.normal(0, 0.3, ne), rng.normal(0, 0.5, nn)
    y_star = 0.4 + 0.9 * x + u[ev] + d[node] + rng.normal(0, 0.35, x.size)
    c = rng.uniform(0.5, 3.0, x.size)  # building ground: water below it leaves the stencil dry (censored)
    y = np.where(y_star > c, y_star, np.nan)
    p = fit_mixed(np.stack([np.ones_like(x), x], 1), y, ev, node, c=c)  # Tobit with crossed effects
    assert p["beta"][1] == pytest.approx(0.9, abs=0.03) and p["sigma"] == pytest.approx(0.35, abs=0.03)
    assert p["tau"] == pytest.approx(0.5, abs=0.05) and p["sigma_event"] == pytest.approx(0.3, abs=0.08)
    assert np.corrcoef(p["delta"], d)[0, 1] > 0.95
    nv = naive_fit(x, y)  # truncated sample: intercept up, slope down
    assert nv["a"] > 0.8 and nv["b"] < 0.85


def test_hurdle_depth_is_calibrated_where_connectivity_and_level_separate():
    """Sheltered nodes stay dry whatever the level; connected nodes sit above the pooled fit. Only the
    two-part model gets the expected flood depth right in deep water."""
    from catforge.hazard.surge_calib import GROUND_MIN, cross_validate

    rng = np.random.default_rng(3)
    ne, nn, per = 40, 600, 200
    ev = np.repeat(np.arange(ne), per)
    node = np.concatenate([rng.choice(nn, per, replace=False) for _ in range(ne)])
    keys = node.astype(np.int64)
    z = rng.uniform(0.0, 6.0, nn)[node]
    shore = (z < 1.0).astype(float)
    x = rng.gamma(2.5, 1.0, ev.size)
    connected = rng.random(ev.size) < 1 / (1 + np.exp(-(1.5 * (x - z) + 2 * shore)))
    y = np.where(connected, 0.3 + 1.0 * x + rng.normal(0, 0.3, nn)[node] + rng.normal(0, 0.3, ev.size), np.nan)
    c = z + 0.05
    g = np.maximum(z, GROUND_MIN)
    zc = np.clip(z, 0, 10)
    res = {k: cross_validate(k, x, y, c, g, zc, shore, ev, keys) for k in ("hurdle", "tobit")}
    deep = x >= 3
    ratio = {k: r["pred"][deep].sum() / r["true"][deep].sum() for k, r in res.items()}
    assert abs(ratio["hurdle"] - 1) < 0.06 and abs(ratio["hurdle"] - 1) < abs(ratio["tobit"] - 1)
