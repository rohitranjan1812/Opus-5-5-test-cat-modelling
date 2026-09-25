"""Topography / bathymetry from NOAA ETOPO1 (1 arc-minute global relief, public domain),
resampled to 2 arc-minutes over the Gulf/Atlantic ("east") and Western US ("west") domains.

Provides bilinear elevation sampling, regional sub-grids for the surge solver, and Terrarium-
encoded PNG terrain tiles (for 3-D map terrain) with a dependency-free PNG encoder.
"""

from __future__ import annotations

import math
import struct
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "data" / "etopo1_2min.npz"


@lru_cache(maxsize=1)
def _grids() -> dict:
    d = np.load(_PATH)
    out = {}
    for name in ("east", "west"):
        lat0, lon0, dlat, dlon, ny, nx = d[f"{name}_meta"]
        out[name] = {"z": d[f"{name}_z"].astype(np.float32), "lat0": float(lat0), "lon0": float(lon0),
                     "dlat": float(dlat), "dlon": float(dlon), "ny": int(ny), "nx": int(nx)}
    return out


def _covers(g, lat, lon):
    return ((lat >= g["lat0"]) & (lat <= g["lat0"] + (g["ny"] - 1) * g["dlat"]) &
            (lon >= g["lon0"]) & (lon <= g["lon0"] + (g["nx"] - 1) * g["dlon"]))


def _bilinear(g, lat, lon):
    y = np.clip((lat - g["lat0"]) / g["dlat"], 0, g["ny"] - 1.000001)
    x = np.clip((lon - g["lon0"]) / g["dlon"], 0, g["nx"] - 1.000001)
    i = np.floor(y).astype(np.int64)
    j = np.floor(x).astype(np.int64)
    fy = y - i
    fx = x - j
    z = g["z"]
    return ((1 - fy) * ((1 - fx) * z[i, j] + fx * z[i, j + 1]) + fy * ((1 - fx) * z[i + 1, j] + fx * z[i + 1, j + 1]))


def elevation(lat, lon, fill: float = np.nan) -> np.ndarray:
    """Elevation (m, + up) by bilinear interpolation; ``fill`` outside the covered domains."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    out = np.full(np.broadcast(lat, lon).shape, fill, dtype=float)
    lat, lon = np.broadcast_arrays(lat, lon)
    for g in _grids().values():
        m = _covers(g, lat, lon)
        if m.any():
            out[m] = _bilinear(g, lat[m], lon[m])
    return out


def subgrid(lat_min: float, lat_max: float, lon_min: float, lon_max: float, stride: int = 1):
    """Native-resolution sub-grid (lat axis, lon axis, z[ny, nx]) of the domain containing the box."""
    for g in _grids().values():
        la1 = g["lat0"] + (g["ny"] - 1) * g["dlat"]
        lo1 = g["lon0"] + (g["nx"] - 1) * g["dlon"]
        if lat_min < la1 and lat_max > g["lat0"] and lon_min < lo1 and lon_max > g["lon0"]:
            i0 = max(int(math.floor((lat_min - g["lat0"]) / g["dlat"])), 0)
            i1 = min(int(math.ceil((lat_max - g["lat0"]) / g["dlat"])), g["ny"] - 1)
            j0 = max(int(math.floor((lon_min - g["lon0"]) / g["dlon"])), 0)
            j1 = min(int(math.ceil((lon_max - g["lon0"]) / g["dlon"])), g["nx"] - 1)
            z = g["z"][i0:i1 + 1:stride, j0:j1 + 1:stride]
            lat = g["lat0"] + g["dlat"] * np.arange(i0, i1 + 1, stride)
            lon = g["lon0"] + g["dlon"] * np.arange(j0, j1 + 1, stride)
            return lat, lon, z
    raise ValueError("box outside DEM coverage")


# ---------------------------------------------------------------------------- terrain tiles
def _png_rgb(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].astype(np.uint8).tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def tile_bounds(z: int, x: int, y: int):
    n = 2 ** z
    lon0 = x / n * 360.0 - 180.0
    lon1 = (x + 1) / n * 360.0 - 180.0
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat0, lat1, lon0, lon1


@lru_cache(maxsize=512)
def terrarium_tile(z: int, x: int, y: int, size: int = 256) -> bytes:
    """Terrarium-encoded elevation tile: elev = R·256 + G + B/256 − 32768."""
    n = 2 ** z
    px = (np.arange(size) + 0.5) / size
    lon = (x + px) / n * 360.0 - 180.0
    merc = math.pi * (1 - 2 * (y + px) / n)
    lat = np.degrees(np.arctan(np.sinh(merc)))
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    e = elevation(LA, LO, fill=-50.0)
    v = np.clip(e + 32768.0, 0, 65535.99)
    r = np.floor(v / 256.0)
    g = np.floor(v - r * 256.0)
    b = np.floor((v - r * 256.0 - g) * 256.0)
    return _png_rgb(np.stack([r, g, b], axis=-1))
