import numpy as np
import pandas as pd
import pytest

from catforge.engine import fidelity as F
from catforge.engine.model import AnalysisConfig
from catforge.exposure.portfolio import Portfolio
from catforge.hazard.surge import calibration, field_tables, node_key
from catforge.hazard.surge_calib import gauss2


def test_plan_ranks_by_tail_weighted_sd_and_stops_at_tolerance():
    rng = np.random.default_rng(0)
    n_ev, S, n_years = 6, 50, 20000
    annual = np.sort(rng.gamma(0.3, 50.0, n_years))
    rate = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    loss = np.array([900.0, 900.0, 5.0, 900.0, 900.0, 900.0])[:, None] + rng.normal(0, 1, (n_ev, S))
    sd = np.array([40.0, 5.0, 80.0, 20.0, 0.0, 60.0])
    surge = 100 + sd[:, None] * rng.standard_normal((n_ev, S))
    eligible = np.array([True, True, True, True, True, False])
    p = F.plan(rate, loss, surge, annual, eligible, budget=10, tol=0.0, rp=250)
    # event 2 barely reaches the tail (a ≈ λ), 4 has no uncertainty, 5 is not a hurricane
    assert list(p["rows"]) == [0, 3, 1, 2]
    assert p["a"][0] > 50 * p["a"][2] and p["u_after"] == 0.0 and p["tol_met"]
    capped = F.plan(rate, loss, surge, annual, eligible, budget=1, tol=0.0, rp=250)
    assert list(capped["rows"]) == [0] and capped["u_after"] < capped["u_before"] and not capped["tol_met"]
    loose = F.plan(rate, loss, surge, annual, eligible, budget=10, tol=1.0, rp=250)
    assert loose["rows"].size == 0  # already within tolerance: spend nothing
    corr = F.plan(rate, loss, surge, annual, eligible, budget=10, tol=0.0, rp=250, rho=1.0)
    assert corr["u_before"] >= p["u_before"]  # correlated event errors add coherently


def test_realised_tail_participation_is_the_euler_gradient_with_a_taper():
    annual = np.array([10.0, 50.0, 30.0, 5.0, 40.0, 20.0, 0.0, 1.0])  # worst first: years 1, 4, 2, 5, 0, 3, 7, 6
    year = np.array([1, 1, 4, 2, 0, 5, 7])
    event = np.array([0, 1, 1, 2, 3, 0, 2])
    a = F.tail_participation(year, event, annual, rp=4, n_events=4, taper=1.0)  # k = 2 worst years: 1 and 4
    assert np.allclose(a, [0.5, 1.0, 0.0, 0.0])  # event 1 occurs in both tail years, event 0 in one
    t = F.tail_participation(year, event, annual, rp=4, n_events=4, taper=2.0)  # ranks 2–3 get weights 1, 0.5
    assert np.allclose(t, [(1 + 0.5) / 2, (1 + 1) / 2, 1.0 / 2, 0.0])  # year 2 (rank 2): 1; year 5 (rank 3): 0.5


def test_process_convolution_field_has_the_calibrated_correlation():
    cal = calibration()
    sc = cal["field"]["scales"]
    lat = np.full(30, 29.3)
    lon = -94.9 + np.arange(30) / 30.0
    keys = np.unique(node_key(lat, lon))
    kp, kn, kw, kk = field_tables(keys, np.ones(keys.size, bool), cal)
    V = np.zeros((keys.size, kk.size))
    for i in range(keys.size):
        np.add.at(V[i], kn[kp[i]:kp[i + 1]], kw[kp[i]:kp[i + 1]])
    var = (V * V).sum(1)
    frac = sc[0]["frac"] + sc[1]["frac"]
    assert np.allclose(np.sqrt(var), cal["sigma_site_m"] * np.sqrt(frac), rtol=1e-9)
    d = np.arange(keys.size) * 111.32 * np.cos(np.radians(29.3)) / 30.0
    target = gauss2(d, sc[0]["frac"], sc[0]["h_km"], sc[1]["frac"], sc[1]["h_km"]) / frac
    assert np.max(np.abs(V @ V[0] / np.sqrt(var * var[0]) - target)) < 0.03


def _gulf_book():
    rng = np.random.default_rng(4)
    hubs = [(27.95, -82.46), (29.95, -90.07), (29.30, -94.80), (30.40, -86.60)]  # Tampa, NOLA, Galveston, Destin
    rows = []
    for h, (la, lo) in enumerate(hubs):
        for k in range(40):
            rows.append({"loc_id": f"L{h}_{k}", "lat": la + rng.normal(0, 0.05), "lon": lo + rng.normal(0, 0.05),
                         "tiv_building": 4e5, "construction": "WOOD", "occupancy": "RES_SF", "year_built": 1985})
    return Portfolio.from_frame(pd.DataFrame(rows), name="gulf")


def test_fidelity_allocation_upgrades_tail_events_with_exact_resimulation(small_model):
    pf = _gulf_book()
    base = small_model.run(pf, AnalysisConfig(perils=["TC"], n_years=3000, elt_samples=12, seed=11))
    r = small_model.run(pf, AnalysisConfig(perils=["TC"], n_years=3000, elt_samples=12, seed=11,
                                           surge_fidelity={"budget": 2, "tol": 0.0}))
    fid = r.extras["surge"]["fidelity"]
    up = [u["event"] for u in fid["upgraded"]]
    assert len(up) == 2 and fid["u_after"] < fid["u_before"]
    assert set(r.elt.loc[r.elt["surge_full_fidelity"], "global_index"]) == set(up)
    touched = np.isin(r.ylt.event, up)
    assert np.array_equal(r.occ_gu[~touched], base.occ_gu[~touched])  # common random numbers elsewhere
    assert not np.array_equal(r.occ_gu[touched], base.occ_gu[touched])
    # the patched location AAL still sums exactly to the ELT AAL
    assert r.loc_aal_gu.sum() == pytest.approx((r.elt["rate"] * r.elt["mean_gu"]).sum(), rel=1e-9)
    assert base.extras["surge"]["fidelity"] is None
