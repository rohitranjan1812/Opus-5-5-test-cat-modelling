"""Synthetic multi-peril US property portfolio generator (personal + commercial lines)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.cities import CITIES
from ..data.coast import COASTLINE
from ..geo import KM_PER_DEG_LAT, KM_PER_DEG_LON_EQ
from .portfolio import Portfolio

FL = {"FL"}
GULF = {"TX", "LA", "MS", "AL"}
WEST = {"CA", "WA", "OR", "UT"}
CENTRAL = {"TN", "MO"}
VALUE_FACTOR = {"CA": 1.7, "NY": 1.5, "MA": 1.4, "DC": 1.4, "WA": 1.3, "NJ": 1.3, "CT": 1.2, "RI": 1.1, "FL": 1.05,
                "OR": 1.1, "VA": 1.0, "MD": 1.1, "PA": 1.0, "NC": 0.95, "SC": 0.9, "GA": 0.9, "TX": 0.9,
                "LA": 0.8, "MS": 0.7, "AL": 0.75, "TN": 0.8, "MO": 0.8, "UT": 1.0, "ME": 0.9}


def _personal_construction(state: str, rng) -> str:
    if state in FL:
        opts, p = ["MASONRY", "WOOD", "MOBILE_HOME"], [0.62, 0.31, 0.07]
    elif state in GULF:
        opts, p = ["WOOD", "MASONRY", "MOBILE_HOME"], [0.74, 0.16, 0.10]
    elif state in WEST:
        opts, p = ["WOOD", "MASONRY", "URM", "MOBILE_HOME"], [0.87, 0.05, 0.03, 0.05]
    elif state in CENTRAL:
        opts, p = ["WOOD", "MASONRY", "URM", "MOBILE_HOME"], [0.72, 0.13, 0.09, 0.06]
    else:
        opts, p = ["WOOD", "MASONRY", "MOBILE_HOME"], [0.80, 0.15, 0.05]
    return opts[rng.choice(len(opts), p=p)]


def _dist_to_coast_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    cl = np.array([(c[1], c[0]) for c in COASTLINE])
    out = np.full(lat.shape, np.inf)
    for i in range(len(cl) - 1):
        la0 = 0.5 * (cl[i, 0] + cl[i + 1, 0])
        kx = KM_PER_DEG_LON_EQ * np.cos(np.radians(la0))
        px, py = (lon - cl[i, 1]) * kx, (lat - cl[i, 0]) * KM_PER_DEG_LAT
        bx, by = (cl[i + 1, 1] - cl[i, 1]) * kx, (cl[i + 1, 0] - cl[i, 0]) * KM_PER_DEG_LAT
        l2 = bx * bx + by * by
        t = np.clip((px * bx + py * by) / max(l2, 1e-12), 0, 1)
        out = np.minimum(out, np.hypot(px - t * bx, py - t * by))
    return out


def generate_portfolio(n_locations: int = 5000, seed: int = 11, states: list[str] | None = None,
                       commercial_share: float = 0.18, name: str | None = None) -> Portfolio:
    """Generate a realistic synthetic portfolio.

    Locations are drawn around insured-value hubs (see ``data/cities.py``) with bivariate normal
    scatter; construction/occupancy/year-built mixes are state-dependent; commercial accounts are
    multi-location (and may span states); deductibles follow typical personal-lines practice
    (percentage hurricane / earthquake deductibles).
    """
    rng = np.random.default_rng(seed)
    cities = [c for c in CITIES if states is None or c[1] in set(states)]
    if not cities:
        raise ValueError(f"no exposure hubs in states {states}")
    w = np.array([c[4] for c in cities])
    ci = rng.choice(len(cities), size=n_locations, p=w / w.sum())
    clat = np.array([cities[i][2] for i in ci])
    clon = np.array([cities[i][3] for i in ci])
    spread = np.array([cities[i][5] for i in ci]) * np.exp(0.35 * rng.standard_normal(n_locations))
    r = spread * np.sqrt(rng.chisquare(2, n_locations) / 2)
    th = rng.uniform(0, 2 * np.pi, n_locations)
    lat = clat + r * np.cos(th) / KM_PER_DEG_LAT
    lon = clon + r * np.sin(th) / (KM_PER_DEG_LON_EQ * np.cos(np.radians(clat)))
    state = np.array([cities[i][1] for i in ci])
    dcoast = _dist_to_coast_km(lat, lon)
    rel_r = r / np.maximum(spread, 1e-6)

    is_comm = rng.random(n_locations) < commercial_share
    lob = np.where(is_comm, "commercial", "personal")
    construction, occupancy, stories, roof, shutters = [], [], [], [], []
    yb_band = rng.choice(5, size=n_locations, p=[0.10, 0.28, 0.22, 0.10, 0.30])
    lo = np.array([1900, 1950, 1975, 1995, 2002])[yb_band]
    hi = np.array([1949, 1974, 1994, 2001, 2024])[yb_band]
    year = rng.integers(lo, hi + 1)
    for i in range(n_locations):
        st = state[i]
        if is_comm[i]:
            occ = "COMMERCIAL" if rng.random() < 0.75 else "INDUSTRIAL"
            con = ["RC", "STEEL", "MASONRY", "LIGHT_METAL", "URM"][rng.choice(5, p=[0.34, 0.30, 0.20, 0.11, 0.05])]
            if con == "URM":
                year[i] = min(year[i], 1949)
            u = rng.random()
            sto = int(rng.integers(1, 4) if u < 0.6 else (rng.integers(4, 11) if u < 0.9 else rng.integers(11, 41)))
            rf = "flat" if rng.random() < 0.8 else "gable"
            sh = int(rng.random() < 0.15)
        else:
            occ = "RES_SF" if rng.random() < 0.85 else "RES_MF"
            con = _personal_construction(st, rng)
            sto = int(rng.integers(1, 3)) if occ == "RES_SF" else int(rng.integers(2, 7))
            rf = ["hip", "gable", "flat"][rng.choice(3, p=[0.55, 0.40, 0.05] if st in FL else [0.30, 0.62, 0.08])]
            p_sh = (0.65 if year[i] >= 2002 else 0.30) if (st in FL or dcoast[i] < 15) else 0.04
            sh = int(rng.random() < p_sh)
        construction.append(con)
        occupancy.append(occ)
        stories.append(sto)
        roof.append(rf)
        shutters.append(sh)

    vf = np.array([VALUE_FACTOR.get(s, 1.0) for s in state])
    bldg = np.where(is_comm,
                    np.minimum(np.exp(np.log(4.0e6) + 1.0 * rng.standard_normal(n_locations)), 4.0e8),
                    np.exp(np.log(3.2e5) + 0.5 * rng.standard_normal(n_locations)) * vf)
    bldg = np.round(bldg, -3)
    cont = np.round(bldg * np.where(is_comm, rng.uniform(0.5, 1.0, n_locations), 0.5), -3)
    bi = np.round(bldg * np.where(is_comm, rng.uniform(0.25, 0.6, n_locations), 0.2), -3)

    terrain = np.where(dcoast < 3.0, "coastal",
                       np.where(rel_r < 0.4, "urban", np.where(rng.random(n_locations) < 0.12, "open", "suburban")))
    vs30 = np.clip(np.exp(np.log(420.0) + 0.28 * rng.standard_normal(n_locations)), 180, 1200).round(0)

    # deductibles
    pers_tc = rng.choice([0.01, 0.02, 0.05, 1000.0, 2500.0], size=n_locations, p=[0.25, 0.35, 0.10, 0.15, 0.15])
    pers_tc = np.where(np.isin(state, list(FL)), rng.choice([0.02, 0.05, 2500.0], size=n_locations,
                                                            p=[0.65, 0.2, 0.15]), pers_tc)
    ded_tc = np.where(is_comm, rng.choice([0.02, 0.03, 0.05], size=n_locations), pers_tc)
    ded_eq = np.where(is_comm, 0.05, rng.choice([0.10, 0.15], size=n_locations, p=[0.6, 0.4]))

    loc_id = np.array([f"L{i + 1:06d}" for i in range(n_locations)], dtype=object)
    acc_id = np.array([f"P-{i + 1:06d}" for i in range(n_locations)], dtype=object)
    acc_ded = np.zeros(n_locations)
    acc_limit = np.zeros(n_locations)
    acc_share = np.ones(n_locations)
    # commercial accounts: group commercial locations into multi-location accounts
    comm_idx = np.nonzero(is_comm)[0]
    rng.shuffle(comm_idx)
    pos, a = 0, 0
    while pos < comm_idx.size:
        size = int(min(rng.geometric(0.18), 40))
        members = comm_idx[pos:pos + size]
        pos += size
        a += 1
        acc_id[members] = f"C-{a:05d}"
        acc_tiv = float((bldg[members] + cont[members] + bi[members]).sum())
        acc_ded[members] = float(rng.choice([25e3, 50e3, 100e3, 250e3]))
        acc_limit[members] = round(acc_tiv * float(rng.choice([0.25, 0.4, 0.6, 1.0])), -3)
        acc_share[members] = float(rng.choice([1.0, 1.0, 0.5, 0.25]))

    df = pd.DataFrame({
        "loc_id": loc_id, "acc_id": acc_id, "lat": lat.round(5), "lon": lon.round(5), "state": state, "lob": lob,
        "construction": construction, "occupancy": occupancy, "year_built": year, "stories": stories,
        "tiv_building": bldg, "tiv_contents": cont, "tiv_bi": bi, "ded_tc": ded_tc, "ded_eq": ded_eq,
        "loc_limit": 0.0, "terrain": terrain, "vs30": vs30, "roof_shape": roof, "shutters": shutters,
        "acc_ded": acc_ded, "acc_limit": acc_limit, "acc_attach": 0.0, "acc_layer_limit": 0.0, "acc_share": acc_share,
    })
    return Portfolio.from_frame(df, name=name or f"Synthetic US book ({n_locations:,} locations)",
                                meta={"source": "synthetic", "seed": seed, "states": states,
                                      "commercial_share": commercial_share})
