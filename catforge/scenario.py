"""Deterministic (single-event) scenarios: historical analogs and user-defined events.

The scenario event is run through the *same* engine as the stochastic analysis — hazard
uncertainty, vulnerability uncertainty, copula dependence and financial terms — using N
independent realisations, so the output is a full loss distribution, not a point estimate.
Analog parameters are approximate public values chosen to reproduce the storm/earthquake
character at today's exposure; they are not reconstructions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data.seismic_sources import FAULTS
from .engine.model import AnalysisConfig, CatModel, build_context
from .exposure.portfolio import Portfolio
from .hazard.earthquake import single_rupture
from .hazard.footprint import event_footprint_grid
from .hazard.tropical_cyclone import single_track, track_of


def _fault(name):
    return next(f for f in FAULTS if f["name"] == name)


ANALOGS: dict[str, dict] = {
    "andrew_1992": {"label": "Andrew 1992 (analog) — Cat 5, Homestead FL", "peril": "TC",
                    "params": {"landfall_lat": 25.52, "landfall_lon": -80.33, "heading": 275, "vmax": 72, "rmax_km": 17,
                               "vt": 8.0}},
    "katrina_2005": {"label": "Katrina 2005 (analog) — Cat 3, SE Louisiana", "peril": "TC",
                     "params": {"landfall_lat": 29.30, "landfall_lon": -89.60, "heading": 5, "vmax": 56, "rmax_km": 45,
                                "vt": 5.5}},
    "ian_2022": {"label": "Ian 2022 (analog) — Cat 4, SW Florida", "peril": "TC",
                 "params": {"landfall_lat": 26.70, "landfall_lon": -82.25, "heading": 30, "vmax": 67, "rmax_km": 30,
                            "vt": 3.5}},
    "michael_2018": {"label": "Michael 2018 (analog) — Cat 5, Florida Panhandle", "peril": "TC",
                     "params": {"landfall_lat": 29.94, "landfall_lon": -85.42, "heading": 20, "vmax": 72, "rmax_km": 20,
                                "vt": 6.5}},
    "hugo_1989": {"label": "Hugo 1989 (analog) — Cat 4, Charleston SC", "peril": "TC",
                  "params": {"landfall_lat": 32.80, "landfall_lon": -79.85, "heading": 320, "vmax": 61, "rmax_km": 40,
                             "vt": 11.0}},
    "harvey_2017": {"label": "Harvey 2017 wind (analog) — Cat 4, Rockport TX", "peril": "TC",
                    "params": {"landfall_lat": 28.00, "landfall_lon": -97.05, "heading": 320, "vmax": 58, "rmax_km": 25,
                               "vt": 3.0}},
    "sandy_2012": {"label": "Sandy 2012 (analog) — large post-tropical, New Jersey", "peril": "TC",
                   "params": {"landfall_lat": 39.40, "landfall_lon": -74.40, "heading": 290, "vmax": 36, "rmax_km": 150,
                              "vt": 12.0}},
    "new_england_1938": {"label": "Great New England 1938 (analog) — Cat 3, Long Island", "peril": "TC",
                         "params": {"landfall_lat": 40.75, "landfall_lon": -72.75, "heading": 355, "vmax": 54,
                                    "rmax_km": 60, "vt": 22.0}},
    "miami_direct_hit": {"label": "Miami Cat 4 direct hit (hypothetical)", "peril": "TC",
                         "params": {"landfall_lat": 25.77, "landfall_lon": -80.13, "heading": 285, "vmax": 64,
                                    "rmax_km": 25, "vt": 5.5}},
    "northridge_1994": {"label": "Northridge 1994 (analog) — M6.7 blind thrust", "peril": "EQ",
                        "params": {"lat": 34.28, "lon": -118.56, "mag": 6.7, "strike": 122, "dip": 40, "ztor": 5.0,
                                   "zbot": 21.0, "width_km": 21.0, "depth_h": 6.0, "mech": "RV", "hypo_depth_km": 17.5,
                                   "hypo_along": 0.55}},
    "loma_prieta_1989": {"label": "Loma Prieta 1989 (analog) — M6.9", "peril": "EQ",
                         "params": {"lat": 37.04, "lon": -121.88, "mag": 6.9, "strike": 128, "dip": 70, "ztor": 3.0,
                                    "zbot": 18.0, "depth_h": 8.0, "mech": "RV", "hypo_depth_km": 17.0,
                                    "hypo_along": 0.5}},
    "sf_1906": {"label": "San Francisco 1906 (analog) — M7.9 San Andreas North", "peril": "EQ",
                "params": {"mag": 7.9, "trace": _fault("San Andreas (North)")["trace"], "depth_h": 4.5,
                           "lat": 37.7, "lon": -122.5, "zbot": 12.0, "hypo_along": 0.2, "hypo_depth_km": 8.0}},
    "shakeout_m78": {"label": "ShakeOut M7.8 southern San Andreas (scenario)", "peril": "EQ",
                     "params": {"mag": 7.8, "trace": _fault("San Andreas (South)")["trace"][:5], "depth_h": 4.5,
                                "lat": 34.2, "lon": -117.4, "hypo_along": 0.02, "hypo_depth_km": 10.0}},
    "haywired_m70": {"label": "HayWired M7.0 Hayward fault (scenario)", "peril": "EQ",
                     "params": {"mag": 7.0, "trace": _fault("Hayward-Rodgers Creek")["trace"][:3], "depth_h": 4.5,
                                "lat": 37.75, "lon": -122.1}},
    "cascadia_m90": {"label": "Cascadia M9.0 full-margin rupture (scenario)", "peril": "EQ",
                     "params": {"mag": 9.0, "trace": _fault("Cascadia Subduction Zone")["trace"], "depth_h": 20.0,
                                "lat": 45.0, "lon": -124.5, "region": "Pacific Northwest", "dip": 11.0, "ztor": 5.0,
                                "zbot": 30.0, "mech": "SUB", "hypo_along": 0.15, "hypo_depth_km": 20.0}},
    "new_madrid_m75": {"label": "New Madrid M7.5 (scenario)", "peril": "EQ",
                       "params": {"mag": 7.5, "trace": _fault("New Madrid")["trace"], "depth_h": 6.0, "lat": 36.3,
                                  "lon": -89.6, "region": "Central US"}},
    "charleston_1886": {"label": "Charleston 1886 (analog) — M7.0", "peril": "EQ",
                        "params": {"lat": 32.90, "lon": -80.20, "mag": 7.0, "strike": 40, "depth_h": 6.0,
                                   "region": "South Carolina"}},
}


def build_event(peril: str, params: dict):
    p = dict(params)
    if peril == "TC":
        return single_track(p["landfall_lat"], p["landfall_lon"], p.get("heading", 0.0), p["vmax"],
                            p.get("rmax_km", 35.0), p.get("vt", 6.0), p.get("holland_b"), p.get("omega", 0.0),
                            name=p.get("name", "scenario"))
    if peril == "EQ":
        return single_rupture(p.get("lat", 0.0), p.get("lon", 0.0), p["mag"], p.get("strike", 0.0), p.get("depth_h", 6.0),
                              p.get("length_km"), p.get("region", "scenario"), trace=p.get("trace"),
                              name=p.get("name", "scenario"), dip=float(p.get("dip", 90.0)),
                              ztor=float(p.get("ztor", 0.0)), width_km=p.get("width_km"),
                              zbot=float(p.get("zbot", 15.0)), mech=p.get("mech", "SS"),
                              hypo_depth_km=p.get("hypo_depth_km"), hypo_along=p.get("hypo_along"))
    raise ValueError(f"unknown peril {peril}")


def run_scenario(portfolio: Portfolio, peril: str, params: dict, n_samples: int = 1000, seed: int = 99,
                 base_model: CatModel | None = None, with_footprint: bool = True) -> dict:
    cat = build_event(peril, params)
    if base_model is not None and peril in base_model.catalogs:
        cat.uncertainty = base_model.catalogs[peril].uncertainty
    model = CatModel({peril: cat})
    cfg = AnalysisConfig(name="scenario", perils=[peril], seed=seed)
    ctx = build_context(model, portfolio, cfg)
    n_aff = int(np.diff(ctx.ev_ptr)[0])
    S = int(n_samples)
    occ_event = np.zeros(S, np.int64)
    keys = np.arange(S, dtype=np.uint64) + np.uint64(7 << 40)
    gu, gross, pr, _, loc_g, loc_gu, _, _ = ctx.kernel(occ_event, keys, np.full(S, 1.0 / S), want_loc=True)
    L = portfolio.locations
    q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    tiv = portfolio.tiv.sum(axis=1)
    loc_df = pd.DataFrame({"loc_id": L["loc_id"], "acc_id": L["acc_id"], "state": L["state"], "lat": L["lat"],
                           "lon": L["lon"], "mean_gross": loc_g, "mean_gu": loc_gu, "tiv": tiv})
    hit = loc_df[loc_df["mean_gu"] > 0]
    by_state = hit.groupby("state").agg(n=("loc_id", "size"), gross=("mean_gross", "sum"), gu=("mean_gu", "sum"),
                                        tiv=("tiv", "sum")).sort_values("gross", ascending=False)
    out = {
        "peril": peril, "params": params, "n_samples": S, "n_locations_affected": n_aff,
        "tiv_affected": float(tiv[ctx.pair_loc[: n_aff]].sum()) if n_aff else 0.0,
        "gross": {"mean": float(gross.mean()), "sd": float(gross.std()),
                  **{f"p{int(x * 100)}": float(np.quantile(gross, x)) for x in q}},
        "gu": {"mean": float(gu.mean()), "sd": float(gu.std()), **{f"p{int(x * 100)}": float(np.quantile(gu, x)) for x in q}},
        "histogram": _hist(gross),
        "by_state": [{"state": k, **{c: float(v) for c, v in r.items()}} for k, r in by_state.iterrows()],
        "top_locations": hit.sort_values("mean_gross", ascending=False).head(25).to_dict(orient="records"),
        "locations": {"lat": hit["lat"].round(4).tolist(), "lon": hit["lon"].round(4).tolist(),
                      "mean_gross": hit["mean_gross"].round(0).tolist(),
                      "damage_ratio": (hit["mean_gu"] / hit["tiv"].clip(lower=1)).round(4).tolist()},
        "event": cat.events.iloc[0].to_dict(),
    }
    if peril == "TC":
        out["geometry"] = {"type": "track", **track_of(cat, 0)}
    else:
        g = cat.geometry
        out["geometry"] = {"type": "rupture", "lat": g["lat"].tolist(), "lon": g["lon"].tolist()}
    if with_footprint:
        out["footprint"] = event_footprint_grid(cat, 0, res_deg=0.05 if peril == "TC" else 0.04)
    return out


def _hist(x: np.ndarray, bins: int = 40) -> dict:
    if x.size == 0 or x.max() <= 0:
        return {"edges": [0, 1], "counts": [int(x.size)]}
    c, e = np.histogram(x, bins=bins)
    return {"edges": e.tolist(), "counts": c.tolist()}
