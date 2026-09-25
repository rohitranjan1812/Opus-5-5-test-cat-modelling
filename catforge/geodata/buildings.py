"""Server-side footprint matching against OpenStreetMap buildings (OpenMapTiles vector tiles).

Same rule as the 3-D view: a location inside a footprint matches it (smallest if several), otherwise
the nearest footprint edge within ``max_snap_m``; the result carries the footprint's area and height.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import numpy as np

from .fetch import TileFetcher
from .mvt import decode_polygons
from .terrain import tile_frac

OSM_TILEJSON = "https://tiles.openfreemap.org/planet"
ATTRIBUTION = "© OpenStreetMap contributors, © OpenMapTiles, OpenFreeMap"
M_LAT = 110_574.0


def osm_tile_template(fetcher: TileFetcher, tilejson: str = OSM_TILEJSON) -> str | None:
    """Current tile URL template (the TileJSON is re-read live so weekly planet builds are picked up)."""
    data = fetcher.get(tilejson, cache=False)
    if not data:
        return None
    try:
        return json.loads(data)["tiles"][0]
    except (ValueError, KeyError, IndexError):
        return None


def _local(lon, lat, lon0, lat0):
    return (np.asarray(lon) - lon0) * 111_320.0 * math.cos(math.radians(lat0)), (np.asarray(lat) - lat0) * M_LAT


def _pip(x, y) -> bool:
    """Is the origin inside the closed ring (x, y)?"""
    inside = False
    for a in range(len(x) - 1):
        if (y[a] > 0) != (y[a + 1] > 0) and 0 < (x[a + 1] - x[a]) * (0 - y[a]) / (y[a + 1] - y[a]) + x[a]:
            inside = not inside
    return inside


def _edge_dist(x, y) -> float:
    ax, ay, bx, by = x[:-1], y[:-1], x[1:], y[1:]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    u = np.clip(np.where(L2 > 0, -(ax * dx + ay * dy) / np.where(L2 > 0, L2, 1), 0), 0, 1)
    return float(np.min(np.hypot(ax + u * dx, ay + u * dy)))


def match_footprints(lat, lon, fetcher: TileFetcher, z: int = 14, max_snap_m: float = 35.0,
                     progress: Callable[[int, int], None] | None = None, template: str | None = None) -> tuple[list[dict | None], dict]:
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    template = template or osm_tile_template(fetcher)
    if not template:
        return [None] * lat.size, {"error": "OSM vector tiles unavailable", "tiles": 0}
    xf, yf = tile_frac(lat, lon, z)
    tx, ty = np.floor(xf).astype(int), np.floor(yf).astype(int)
    keys = sorted(set(zip(tx.tolist(), ty.tolist())))
    urls = {k: template.replace("{z}", str(z)).replace("{x}", str(k[0])).replace("{y}", str(k[1])) for k in keys}
    blobs = fetcher.get_many(urls.values(), progress)
    out: list[dict | None] = [None] * lat.size
    for k in keys:
        data = blobs.get(urls[k])
        idx = np.nonzero((tx == k[0]) & (ty == k[1]))[0]
        if not data:
            continue
        polys = decode_polygons(data, z, k[0], k[1], near=[(lon[i], lat[i]) for i in idx], radius_m=max_snap_m + 120)
        for i in idx:
            best = None
            for p in polys:
                x, y = _local(p.lon, p.lat, lon[i], lat[i])
                if x.min() - max_snap_m > 0 or x.max() + max_snap_m < 0 or y.min() - max_snap_m > 0 or y.max() + max_snap_m < 0:
                    continue
                area = 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))
                if _pip(x, y):
                    if best is None or best["match"] != "inside" or area < best["area_m2"]:
                        best = {"match": "inside", "snap_m": 0.0, "area_m2": area, "props": p.props}
                elif best is None or best["match"] != "inside":
                    d = _edge_dist(x, y)
                    if d <= max_snap_m and (best is None or d < best["snap_m"]):
                        best = {"match": "snapped", "snap_m": d, "area_m2": area, "props": p.props}
            if best:
                pr = best.pop("props")
                h = pr.get("render_height")
                best["height_m"] = float(h) if isinstance(h, (int, float)) else float("nan")
                best["min_height_m"] = float(pr.get("render_min_height") or 0.0)
                out[i] = best
    return out, {"source": ATTRIBUTION, "zoom": z, "tiles": len(keys), "tiles_missing": sum(1 for u in urls.values() if not blobs.get(u))}
