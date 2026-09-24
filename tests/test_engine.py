import numpy as np
import pandas as pd
import pytest

from catforge.analytics import allocation, sensitivity
from catforge.analytics.analytic import analytic_ep
from catforge.analytics.ep import ep_table, quantile_at_rp
from catforge.engine import AnalysisConfig
from catforge.engine.model import rerun_ylt, reweighted_metrics
from catforge.hazard.base import ENSO_REGIMES, FrequencyModel
from catforge.scenario import ANALOGS, run_scenario


def test_run_consistency(small_result):
    r = small_result
    s = r.summary
    # three independent AAL estimators agree (YLT vs ELT vs location aggregation)
    assert s["elt_aal"] == pytest.approx(s["location_aal_sum"], rel=1e-9)
    assert s["aal"]["gross"] == pytest.approx(s["elt_aal"], rel=0.2)
    assert s["aal"]["gross"] <= s["aal"]["gu"]
    assert (r.occ_gross <= r.occ_gu + 1e-6).all()
    # Euler allocation adds up exactly and tail re-simulation is bit-identical
    tc = s["tail_check"]
    assert tc["tvar_alloc_sum"] == pytest.approx(tc["tvar_direct"], rel=1e-9)
    assert tc["max_abs_occ_diff"] == 0.0
    # EP monotone
    losses = [row["gross_aep"] for row in s["rp_table"]]
    assert all(b >= a for a, b in zip(losses, losses[1:]))
    assert r.insights and all({"title", "severity"} <= set(i) for i in r.insights)


def test_reproducible_and_seed_sensitive(small_model, small_portfolio):
    cfg = AnalysisConfig(n_years=800, elt_samples=4, seed=11)
    a = small_model.run(small_portfolio, cfg)
    b = small_model.run(small_portfolio, cfg)
    assert np.array_equal(a.occ_gross, b.occ_gross)
    c = small_model.run(small_portfolio, AnalysisConfig(n_years=800, elt_samples=4, seed=12))
    assert not np.array_equal(a.annual("gross"), c.annual("gross"))


def test_financial_terms_hold_per_location(small_result):
    r = small_result
    ctx = r.ctx
    occ = np.argsort(r.occ_gross)[-5:]
    ev = r.ylt.event[occ]
    counts = np.diff(ctx.ev_ptr)[ev]
    dptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    _, gross, _, _, _, _, det_g, det_gu = ctx.kernel(ev, r.ylt.key[occ], detail_ptr=dptr)
    fin = ctx.fin
    for i, e in enumerate(ev):
        sl = slice(dptr[i], dptr[i + 1])
        locs = ctx.pair_loc[ctx.ev_ptr[e]:ctx.ev_ptr[e + 1]]
        per = ctx.ev_peril[e]
        x = np.minimum(np.maximum(det_gu[sl] - fin.loc_ded[locs, per], 0), fin.loc_lim[locs, per])
        pol = fin.loc_pol[locs]
        tot = 0.0
        for p in np.unique(pol):
            s = x[pol == p].sum()
            g1 = min(max(s - fin.pol_ded[p, per], 0), fin.pol_lim[p, per])
            g2 = min(max(g1 - fin.pol_att[p], 0), fin.pol_llim[p]) * fin.pol_share[p]
            assert det_g[sl][pol == p].sum() == pytest.approx(g2, rel=1e-9, abs=1e-6)
            tot += g2
        assert gross[i] == pytest.approx(tot, rel=1e-9)


def test_common_random_numbers(small_result):
    base = small_result.annual("gross")
    same = rerun_ylt(small_result)["annual"]
    assert np.array_equal(base, same)
    d = small_result.ctx.p_dmg.copy() * 1.1
    up = rerun_ylt(small_result, p_dmg=d)["annual"]
    assert (up >= base - 1e-6).all() and up.mean() > base.mean()


def test_rate_reweighting_scales_aal(small_result):
    r = small_result
    nr = r.ctx.ev_rate * 1.2
    rw = reweighted_metrics(r, nr)
    assert rw["aal"] == pytest.approx(1.2 * r.annual("gross").mean(), rel=0.06)
    assert rw["weights_ess"] > 0.5 * r.n_years


def test_analytic_ep_simple_compound_poisson():
    # one event, deterministic loss 100 at rate 0.05/yr
    elt = pd.DataFrame({"peril": ["TC"], "rate": [0.05], "mean": [100.0], "sd": [1.0], "cap": [200.0], "p0": [0.0]})
    out = analytic_ep(elt, {"TC": FrequencyModel()}, n_grid=1 << 12, x_max=1000.0)
    tab = {r["rp"]: r for r in out["table"]}
    assert tab[100]["oep"] == pytest.approx(100.0, rel=0.02)
    assert tab[10]["oep"] == pytest.approx(0.0, abs=1.0)
    # P(N >= 2) = 1 - e^-λ(1+λ) ≈ 0.00121; at p=0.002 the single-loss quantile is 100 + 2.13σ
    assert tab[500]["aep"] == pytest.approx(102.1, abs=0.6)
    assert tab[1000]["aep"] == pytest.approx(200.0, rel=0.02)
    assert out["aal"] == pytest.approx(5.0, rel=0.01)


def test_mixed_poisson_pgf_and_dispersion():
    fm = FrequencyModel(dispersion_r=10.0, regimes=list(ENSO_REGIMES))
    rng = np.random.default_rng(0)
    th, reg = fm.sample_theta(rng, 200000)
    assert th.mean() == pytest.approx(1.0, abs=0.01)
    lam = 3.0
    n = rng.poisson(th * lam)
    assert n.var() / n.mean() == pytest.approx(fm.variance_to_mean(lam), rel=0.03)
    z = np.array([0.3, 0.7], complex)
    emp = np.array([(zz ** n).mean() for zz in z])
    assert np.allclose(fm.pgf(z, lam).real, emp.real, atol=0.005)


def test_ep_table_quantiles():
    x = np.arange(1, 10001, dtype=float)
    t = {r["rp"]: r for r in ep_table(x, n_boot=0)}
    assert t[100]["loss"] == pytest.approx(9900.5, rel=1e-3)
    assert t[100]["loss_lo"] <= t[100]["loss"] <= t[100]["loss_hi"]
    assert quantile_at_rp(np.sort(x), 2) == pytest.approx(5000.5, rel=1e-3)


def test_allocation_and_sensitivity(small_result):
    r = small_result
    a = allocation.allocate(r, "state")
    assert sum(x["cotvar_share"] for x in a["rows"]) == pytest.approx(1.0)
    assert sum(x["aal"] for x in a["rows"]) == pytest.approx(r.loc_aal_gross.sum())
    p = allocation.allocate(r, "peril")
    assert {x["key"] for x in p["rows"]} <= {"TC", "EQ"}
    seg = allocation.segment_ep(r)
    assert 0 <= seg["portfolio"]["250"]["diversification_benefit"] < 1
    enso = sensitivity.enso_conditional(r)
    assert len(enso["regimes"]) == 3
    c = sensitivity.climate_scenario(r, tc_frequency=1.1)
    assert c["scenario"]["aal"] > c["base"]["aal"]
    m = sensitivity.mitigation(r, preset="shutters_fl_pre2002")
    assert m["n_locations"] == 0 or m["aal_saving"] >= -1e-6


def test_scenario_andrew(small_portfolio, small_model):
    s = run_scenario(small_portfolio, "TC", ANALOGS["andrew_1992"]["params"], n_samples=200, base_model=small_model,
                     with_footprint=False)
    assert s["n_locations_affected"] > 0
    assert s["gross"]["p95"] >= s["gross"]["p50"] >= s["gross"]["p5"] >= 0
    assert s["gu"]["mean"] >= s["gross"]["mean"]
