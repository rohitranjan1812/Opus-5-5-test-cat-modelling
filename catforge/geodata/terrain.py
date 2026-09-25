"""Building-scale ground elevation.

- **Terrain Tiles** (AWS Open Data, Tilezen "terrarium" encoding): in the US built from USGS 3DEP/NED
  (≈10 m); at z14 a pixel is ≈9.5 m at 26°N. Keyless, fast, cached — used for bulk enrichment.
- **USGS 3DEP Elevation Point Query Service**: the best available DEM at a point (often 1 m lidar),
  ≈5 s per query — used for single-point lookups.

Heights are orthometric (≈NAVD88 / mean sea level), the same frame as the surge model's water levels
to within the local MSL–NAVD88 offset (≈0.1–0.3 m on the Gulf and Atlantic coasts).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .fetch import TileFetcher
from .png import decode_png

TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
EPQS = "https://epqs.nationalmap.gov/v1/json"
ATTRIBUTION = "Terrain Tiles (Mapzen/Tilezen, AWS Open Data): USGS 3DEP/NED, SRTM, GMTED2010, ETOPO1"


def tile_frac(lat, lon, z: int):
    """Global tile coordinates (fractional) of lon/lat at zoom z."""
    n = 2.0 ** z
    lat = np.clip(np.asarray(lat, float), -85.0511, 85.0511)
    x = (np.asarray(lon, float) + 180.0) / 360.0 * n
    r = np.radians(lat)
    y = (1.0 - np.log(np.tan(r) + 1.0 / np.cos(r)) / math.pi) / 2.0 * n
    return x, y


def terrarium_to_m(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


def ground_elevation(lat, lon, fetcher: TileFetcher, z: int = 14, url: str = TERRARIUM,
                     progress: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, dict]:
    """Bilinear ground elevation (m) at each point from terrarium tiles; NaN where no tile is available."""
    xf, yf = tile_frac(lat, lon, z)
    tx, ty = np.floor(xf).astype(int), np.floor(yf).astype(int)
    keys = sorted(set(zip(tx.tolist(), ty.tolist())))
    urls = {k: url.format(z=z, x=k[0], y=k[1]) for k in keys}
    blobs = fetcher.get_many(urls.values(), progress)
    out = np.full(xf.shape, np.nan)
    bad_tiles = 0
    for k in keys:
        data = blobs.get(urls[k])
        if not data:
            bad_tiles += 1
            continue
        try:
            img = decode_png(data)
        except ValueError:
            bad_tiles += 1
            continue
        e = terrarium_to_m(img[..., :3])
        h, w = e.shape
        m = (tx == k[0]) & (ty == k[1])
        px = np.clip((xf[m] - k[0]) * w - 0.5, 0, w - 1.0001)
        py = np.clip((yf[m] - k[1]) * h - 0.5, 0, h - 1.0001)
        i, j = np.floor(py).astype(int), np.floor(px).astype(int)
        fy, fx = py - i, px - j
        out[m] = ((1 - fy) * ((1 - fx) * e[i, j] + fx * e[i, j + 1]) + fy * ((1 - fx) * e[i + 1, j] + fx * e[i + 1, j + 1]))
    res_m = 40_075_016.7 * np.cos(np.radians(np.nanmean(np.asarray(lat, float)))) / (256 * 2 ** z) if np.size(lat) else float("nan")
    return out, {"source": ATTRIBUTION, "zoom": z, "tiles": len(keys), "tiles_missing": bad_tiles, "pixel_m": float(res_m)}


def point_elevation_3dep(lat: float, lon: float, fetcher: TileFetcher) -> dict | None:
    """USGS 3DEP point query (best available resolution); None when unavailable."""
    try:
        r = fetcher.client.get(EPQS, params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326, "includeDate": "false"})
        r.raise_for_status()
        j = r.json()
        v = float(j["value"])
    except Exception:  # service/network errors surface as "no value"
        return None
    if v < -1000:
        return None
    return {"elevation_m": v, "resolution_m": j.get("resolution"), "source": "USGS 3DEP Elevation Point Query Service"}
