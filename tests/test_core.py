import numpy as np
from scipy.stats import norm

from catforge.geo import destination, haversine_km, points_in_polygon
from catforge.rng import norm_ppf, norm_ppf_fast, uniform_block


def test_uniforms_are_reproducible_open_and_uniform():
    a = uniform_block(42, np.arange(2000), 50)
    b = uniform_block(42, np.arange(2000), 50)
    assert np.array_equal(a, b)
    assert (a > 0).all() and (a < 1).all()
    assert abs(a.mean() - 0.5) < 0.005
    assert abs(a.var() - 1 / 12) < 0.002
    c = uniform_block(43, np.arange(2000), 50)
    assert abs(np.corrcoef(a.ravel(), c.ravel())[0, 1]) < 0.01


def test_norm_ppf_accuracy():
    p = np.concatenate([np.geomspace(1e-12, 0.02, 200), np.linspace(0.02, 0.98, 400), 1 - np.geomspace(1e-12, 0.02, 200)])
    exact = norm.ppf(p)
    fine = np.array([norm_ppf(x) for x in p])
    fast = np.array([norm_ppf_fast(x) for x in p])
    assert np.max(np.abs(fine - exact)) < 1e-9
    assert np.max(np.abs(fast - exact) / np.maximum(np.abs(exact), 1)) < 1e-8


def test_geodesy():
    # Miami -> New York ~ 1757 km
    d = haversine_km(25.7617, -80.1918, 40.7128, -74.0060)
    assert 1740 < d < 1775
    lat, lon = destination(25.0, -80.0, 90.0, 100.0)
    assert abs(haversine_km(25.0, -80.0, lat, lon) - 100.0) < 1e-6


def test_point_in_polygon():
    px, py = [0, 1, 1, 0], [0, 0, 1, 1]
    inside = points_in_polygon([0.5, 1.5, 0.2], [0.5, 0.5, 0.9], px, py)
    assert inside.tolist() == [True, False, True]
