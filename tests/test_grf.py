import numpy as np
import pytest

from catforge.engine import AnalysisConfig
from catforge.physics.grf import (
    Matern,
    build_vecchia,
    circulant_field,
    event_closures,
    sample_sites,
    von_karman_field,
    xyz_km,
)


@pytest.fixture(scope="module")
def clustered_sites():
    rng = np.random.default_rng(1)
    c = np.array([[25.8, -80.2], [27.9, -82.5], [26.6, -81.9]])
    k = rng.integers(0, 3, 400)
    return c[k, 0] + rng.normal(0, 0.15, 400), c[k, 1] + rng.normal(0, 0.15, 400)


def test_matern_practical_range():
    for nu in (0.5, 1.5, 2.5):
        m = Matern(nu, 40.0)
        assert m.corr(0.0) == pytest.approx(1.0)
        assert m.corr(40.0) == pytest.approx(0.05, abs=1e-6)
        assert np.all(np.diff(m.corr(np.linspace(0, 200, 50))) < 0)


def test_vecchia_matches_exact_covariance(clustered_sites):
    lat, lon = clustered_sites
    model = Matern(0.5, 30.0)
    V = build_vecchia(lat, lon, model, m=30)
    P = xyz_km(lat, lon)
    R = model.corr(np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)))
    S = V.implied_covariance()
    assert np.abs(S - R).max() < 0.03
    n = lat.size
    M = np.linalg.solve(S, R)
    kl = 0.5 * (np.trace(M) - n - np.linalg.slogdet(M)[1])
    assert kl < 0.05
    # parents always precede children in the sampling order
    for j in range(n):
        assert np.all(V.rank[V.nbr[V.ptr[j]:V.ptr[j + 1]]] < V.rank[j])


def test_vecchia_monte_carlo_correlation(clustered_sites):
    lat, lon = clustered_sites
    model = Matern(1.5, 60.0)
    V = build_vecchia(lat, lon, model, m=30)
    Z = sample_sites(np.arange(12000, dtype=np.uint64), np.uint64(3), V.order, V.ptr, V.nbr, V.coef, V.cond_sd)
    assert Z.var(axis=0).mean() == pytest.approx(1.0, abs=0.03)
    P = xyz_km(lat, lon)
    R = model.corr(np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)))
    iu = np.triu_indices(lat.size, 1)
    assert np.abs(np.corrcoef(Z.T)[iu] - R[iu]).mean() < 0.02


def test_closure_is_ancestor_complete(clustered_sites):
    lat, lon = clustered_sites
    V = build_vecchia(lat, lon, Matern(0.5, 30.0), m=16)
    n = lat.size
    rng = np.random.default_rng(0)
    events = [np.sort(rng.choice(n, size=s, replace=False)) for s in (5, 40, 200)]
    ev_ptr = np.concatenate([[0], np.cumsum([e.size for e in events])]).astype(np.int64)
    pair_loc = np.concatenate(events).astype(np.int64)
    cptr, clo = event_closures(ev_ptr, pair_loc, np.zeros(3, np.int64), V.ptr[None, :], V.nbr, V.rank[None, :], n)
    for e in range(3):
        c = clo[cptr[e]:cptr[e + 1]]
        s = set(c.tolist())
        assert set(events[e].tolist()) <= s
        assert np.all(np.diff(V.rank[c]) > 0)  # sorted by rank → parents computed first
        for j in c:
            assert set(V.nbr[V.ptr[j]:V.ptr[j + 1]].tolist()) <= s


def test_circulant_embedding_exact_on_grid():
    F, ratio = circulant_field(48, 64, 4.0, 4.0, Matern(0.5, 30.0), np.random.default_rng(2), n_fields=300)
    assert ratio > -1e-6
    assert F.var(axis=0).mean() == pytest.approx(1.0, abs=0.05)
    lag = (F[:, :, :-3] * F[:, :, 3:]).mean()
    assert lag == pytest.approx(float(Matern(0.5, 30.0).corr(12.0)), abs=0.03)


def test_von_karman_field_unit_variance():
    f = von_karman_field(16, 64, 2.0, 2.0, 5.0, 20.0, 0.75, np.random.default_rng(3))
    assert f.shape == (16, 64) and abs(f.mean()) < 1e-9 and f.std() == pytest.approx(1.0)


def test_engine_grf_preserves_marginals(small_model, small_portfolio):
    a = small_model.run(small_portfolio, AnalysisConfig(n_years=3000, elt_samples=16, seed=5, dependence="copula"))
    b = small_model.run(small_portfolio, AnalysisConfig(n_years=3000, elt_samples=16, seed=5, dependence="grf"))
    # same site marginals ⇒ same expected loss (ELT estimates, MC tolerance)
    assert b.summary["elt_aal"] == pytest.approx(a.summary["elt_aal"], rel=0.08)
    assert b.summary["dependence"]["mode"] == "grf"
    assert b.summary["tail_check"]["max_abs_occ_diff"] == 0.0
    assert b.summary["tail_check"]["tvar_alloc_sum"] == pytest.approx(b.summary["tail_check"]["tvar_direct"], rel=1e-9)
