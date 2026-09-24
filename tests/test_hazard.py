import math

import numpy as np

from catforge.data.coast import REGIONS
from catforge.hazard import Sites, compute_pairs, hazard_curve, single_rupture, single_track
from catforge.hazard.earthquake import gr_bin_rates, ln_pga_median
from catforge.hazard.tropical_cyclone import _gust, intensity_rate_multiplier


def test_tc_catalog_rates_and_intensity(small_model):
    cat = small_model.catalogs["TC"]
    ev = cat.events
    hu = ev[ev.storm_class == "HU"]
    total = sum(r[0] for r in REGIONS.values())
    # importance-sampled rates are unbiased for the regional climatology (Monte Carlo tolerance)
    assert abs(hu.rate.sum() / total - 1) < 0.15
    share_major = hu[hu.category >= 3].rate.sum() / hu.rate.sum()
    assert 0.2 < share_major < 0.4
    assert (hu.dp_hpa > 10).all() and (ev.dp_hpa > 0).all() and (ev.rmax_km >= 8).all()
    assert cat.geometry["ptr"][-1] == cat.geometry["lat"].size


def test_holland_profile_peaks_near_rmax():
    dp, rm, b = 6000.0, 30.0, 1.4
    lat0, lon0 = 27.0, -80.0
    r = np.linspace(2, 300, 300)
    v = np.array([_gust(lat0 + ri / 110.574, lon0, lat0, lon0, dp, rm, b, 0.0, 0.0, 1.0) for ri in r])
    assert abs(r[np.argmax(v)] - rm) < 6
    assert v[-1] < 0.35 * v.max()
    # translation asymmetry: right-of-track (east for a northbound storm) is stronger
    right = _gust(lat0, lon0 + 30 / 99.0, lat0, lon0, dp, rm, b, 0.0, 6.0, 1.0)
    left = _gust(lat0, lon0 - 30 / 99.0, lat0, lon0, dp, rm, b, 0.0, 6.0, 1.0)
    assert right > left


def test_single_track_footprint_decays_inland():
    cat = single_track(25.77, -80.13, 285, 64, 25, 5.5)
    sites = Sites.plain([25.77, 25.8, 26.2, 27.5], [-80.2, -80.6, -81.5, -81.8])
    p = compute_pairs(cat, sites, min_intensity=1.0)
    g = np.exp(p.log_i)[np.argsort(p.site)]
    assert g[0] > 60 and g[0] > g[2] > g[3]


def test_gr_rates_and_gmpe():
    m, r = gr_bin_rates(5.0, 7.0, 1.0, 1.0, 0.1)
    assert abs(r.sum() - 1.0) < 1e-12 and (np.diff(r) < 0).all()
    assert ln_pga_median(7.0, 10.0, 4.5, 760.0, 1.0) > ln_pga_median(7.0, 50.0, 4.5, 760.0, 1.0)
    assert ln_pga_median(7.5, 20.0, 4.5, 760.0, 1.0) > ln_pga_median(6.0, 20.0, 4.5, 760.0, 1.0)
    pga = math.exp(ln_pga_median(7.0, 10.0, 4.5, 400.0, 1.0))
    assert 0.15 < pga < 0.5  # plausible near-field median on stiff soil


def test_eq_single_rupture_and_hazard_curve(small_model):
    cat = single_rupture(34.2, -118.5, 6.7, strike=120)
    p = compute_pairs(cat, Sites.plain([34.2, 34.6, 35.5], [-118.5, -118.5, -118.5]))
    assert p.n_pairs >= 2
    hc = hazard_curve(small_model.catalogs["EQ"], 34.05, -118.24)
    aep = np.array(hc["annual_exceedance_prob"])
    assert (np.diff(aep) <= 1e-12).all() and aep[0] > 0.01


def test_intensity_reweighting_is_normalised(small_model):
    ev = small_model.catalogs["TC"].events
    base = ev.rate.to_numpy()
    for s in (0.9, 1.1):
        w = intensity_rate_multiplier(ev, s)
        hu = (ev.storm_class == "HU").to_numpy()
        # E_f[f'/f] = 1: total hurricane rate is preserved up to Monte Carlo error
        assert abs((base * w)[hu].sum() / base[hu].sum() - 1) < 0.1
    up = intensity_rate_multiplier(ev, 1.1)
    strong = (ev.vmax > 65).to_numpy()
    assert up[strong].mean() > 1.0
