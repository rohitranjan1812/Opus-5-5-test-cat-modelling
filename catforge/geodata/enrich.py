"""Exposure enrichment: building-scale ground elevation + mapped-footprint attributes + data-quality report.

For each location in scope:

- ``ground_elev_m``: bilinear ground elevation from USGS 3DEP-based terrain tiles (≈10 m). It replaces
  the 2′ ETOPO1 cell value, which smooths coastal cities upward by metres and so understates surge depth.
- The OSM footprint match (inside / snapped ≤ ``max_snap_m``), with footprint area and mapped height.
- ``stories``: re-derived from the mapped height only when that height is informative. OpenMapTiles
  reports 5 m when a building has no height or levels, so only taller values are used.
- ``building_height_m`` and ``floor_area_m2`` = footprint area × storeys.

The source portfolio is never modified: a new portfolio is returned, with the report and per-location
QA in ``meta``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ..exposure.portfolio import Portfolio
from ..physics.dem import elevation
from .buildings import match_footprints
from .fetch import TileFetcher
from .terrain import ground_elevation

DEFAULT_HEIGHT_M = 5.0  # OpenMapTiles render_height when OSM has no height / building:levels
STOREY_M = 3.3
SURGE_GROUND_FLOOR = 1.0  # the coarse-DEM ground assumed by the surge model when no measurement exists


def coastal_mask(lat, lon, max_km: float = 30.0, max_elev_m: float = 20.0) -> np.ndarray:
    from ..exposure.synthetic import _dist_to_coast_km

    z = elevation(lat, lon, fill=np.inf)
    return (_dist_to_coast_km(np.asarray(lat), np.asarray(lon)) < max_km) & (z < max_elev_m)


def _stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}


def enrich_portfolio(pf: Portfolio, scope: str = "coastal", with_elevation: bool = True, with_footprints: bool = True,
                     fetcher: TileFetcher | None = None, max_snap_m: float = 35.0,
                     progress: Callable[[float, str], None] | None = None) -> tuple[Portfolio, dict]:
    t0 = time.time()
    fetcher = fetcher or TileFetcher()
    L = pf.locations.copy()
    for col in ("ground_elev_m", "building_height_m", "floor_area_m2", "first_floor_height_m"):
        if col not in L:
            L[col] = np.nan
    lat, lon = L["lat"].to_numpy(float), L["lon"].to_numpy(float)
    sel = coastal_mask(lat, lon) if scope == "coastal" else np.ones(lat.size, bool)
    idx = np.nonzero(sel)[0]
    report: dict = {"scope": scope, "n_locations": int(lat.size), "n_scope": int(idx.size)}
    quality: dict = {"loc_id": L["loc_id"].to_numpy()[idx].tolist()}
    say = progress or (lambda f, m: None)

    etopo = elevation(lat[idx], lon[idx], fill=np.nan)
    quality["etopo_m"] = np.round(etopo, 2).tolist()
    if with_elevation and idx.size:
        say(0.02, f"Ground elevation for {idx.size:,} locations (USGS 3DEP terrain tiles)")
        g, info = ground_elevation(lat[idx], lon[idx], fetcher,
                                   progress=lambda k, n: say(0.02 + 0.45 * k / n, f"Terrain tiles {k:,}/{n:,}"))
        ok = np.isfinite(g)
        col = L.columns.get_loc("ground_elev_m")
        L.iloc[idx[ok], col] = np.round(g[ok], 2)
        surge_old = np.maximum(np.nan_to_num(etopo, nan=SURGE_GROUND_FLOOR), SURGE_GROUND_FLOOR)
        diff = surge_old - g
        report["elevation"] = {
            **info, "n": int(ok.sum()),
            "coarse_minus_measured_m": _stats(diff),
            "share_lower_by_1m": float(np.mean(diff[ok] > 1.0)) if ok.any() else None,
            "below_1m": int(np.sum(g[ok] < 1.0)), "below_msl": int(np.sum(g[ok] < 0.0)),
        }
        quality["ground_elev_m"] = np.round(g, 2).tolist()

    stories = L["stories"].to_numpy(int).copy()
    if with_footprints and idx.size:
        say(0.5, f"Matching {idx.size:,} locations to mapped building footprints (OpenStreetMap)")
        m, info = match_footprints(lat[idx], lon[idx], fetcher, max_snap_m=max_snap_m,
                                   progress=lambda k, n: say(0.5 + 0.45 * k / n, f"Building tiles {k:,}/{n:,}"))
        kinds = np.array([r["match"] if r else "none" for r in m])
        snap = np.array([r["snap_m"] if r else np.nan for r in m])
        area = np.array([r["area_m2"] if r else np.nan for r in m])
        height = np.array([r["height_m"] - r["min_height_m"] if r else np.nan for r in m])
        informative = np.isfinite(height) & (height > DEFAULT_HEIGHT_M + 0.5)
        est = np.where(informative, np.clip(np.rint(height / STOREY_M), 1, 120), 0).astype(int)
        before = stories[idx].copy()
        after = np.where(informative, est, before)
        stories[idx] = after
        L["stories"] = stories
        hcol, acol = L.columns.get_loc("building_height_m"), L.columns.get_loc("floor_area_m2")
        L.iloc[idx[informative], hcol] = np.round(height[informative], 1)
        matched = kinds != "none"
        L.iloc[idx[matched], acol] = np.round(area[matched] * after[matched], 0)
        changed = after != before
        report["footprints"] = {
            **info, "matched": int(matched.sum()), "inside": int(np.sum(kinds == "inside")),
            "snapped": int(np.sum(kinds == "snapped")), "none": int(np.sum(kinds == "none")),
            "match_rate": float(matched.mean()) if idx.size else None,
            "median_snap_m": float(np.nanmedian(snap[kinds == "snapped"])) if np.any(kinds == "snapped") else None,
            "footprint_area_m2": _stats(area), "informative_height": int(informative.sum()),
            "stories_changed": int(changed.sum()), "stories_up": int(np.sum(after > before)),
            "stories_down": int(np.sum(after < before)),
        }
        quality.update({"fp_match": kinds.tolist(), "fp_snap_m": np.round(snap, 1).tolist(),
                        "fp_area_m2": np.round(area, 0).tolist(), "fp_height_m": np.round(height, 1).tolist(),
                        "stories_before": before.tolist(), "stories_after": after.tolist()})
        # geocodes sitting on water with no building nearby
        g_arr = np.asarray(quality.get("ground_elev_m", [np.nan] * idx.size), float)
        offshore = (kinds == "none") & np.isfinite(g_arr) & (g_arr <= 0.1)
        report["flags"] = {"possible_offshore": int(offshore.sum()),
                           "examples": L["loc_id"].to_numpy()[idx[offshore]][:25].tolist()}

    report["fetch"] = dict(fetcher.stats)
    report["elapsed_s"] = round(time.time() - t0, 1)
    say(1.0, "Enrichment complete")
    meta = {**pf.meta, "enriched_from": pf.id, "enrichment": report, "quality": quality}
    return Portfolio(name=f"{pf.name} (enriched)", locations=L.reset_index(drop=True), meta=meta), report
