"""Geodesy and computational-geometry helpers (numpy + numba)."""

from __future__ import annotations

import math

import numba as nb
import numpy as np

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def destination(lat, lon, bearing_deg, dist_km):
    """Point reached travelling ``dist_km`` from (lat, lon) on initial ``bearing_deg`` (great circle)."""
    phi1 = np.radians(lat)
    lam1 = np.radians(lon)
    th = np.radians(bearing_deg)
    d = np.asarray(dist_km) / EARTH_RADIUS_KM
    sin_phi2 = np.sin(phi1) * np.cos(d) + np.cos(phi1) * np.sin(d) * np.cos(th)
    phi2 = np.arcsin(np.clip(sin_phi2, -1, 1))
    lam2 = lam1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(phi1), np.cos(d) - np.sin(phi1) * sin_phi2)
    return np.degrees(phi2), (np.degrees(lam2) + 540.0) % 360.0 - 180.0


def initial_bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing (degrees clockwise from north) from point 1 to point 2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def wrap180(angle_deg):
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def points_in_polygon(lon, lat, poly_lon, poly_lat) -> np.ndarray:
    """Even-odd ray casting, vectorised over points. Polygon need not be closed."""
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    return _pip(lon, lat, np.asarray(poly_lon, float), np.asarray(poly_lat, float))


@nb.njit(cache=True)
def _pip(x, y, px, py):
    n = px.shape[0]
    out = np.zeros(x.shape[0], dtype=np.bool_)
    for k in range(x.shape[0]):
        inside = False
        j = n - 1
        for i in range(n):
            if ((py[i] > y[k]) != (py[j] > y[k])) and \
                    (x[k] < (px[j] - px[i]) * (y[k] - py[i]) / (py[j] - py[i] + 1e-300) + px[i]):
                inside = not inside
            j = i
        out[k] = inside
    return out


@nb.njit(inline="always", cache=True)
def local_xy_km(lat, lon, lat0, lon0):
    """Equirectangular projection about (lat0, lon0). Accurate to <0.5% within ~500 km."""
    x = (lon - lon0) * KM_PER_DEG_LON_EQ * math.cos(0.5 * (lat + lat0) * DEG2RAD)
    y = (lat - lat0) * KM_PER_DEG_LAT
    return x, y


@nb.njit(inline="always", cache=True)
def point_segment_dist(px, py, ax, ay, bx, by):
    """Planar distance from P to segment AB."""
    abx = bx - ax
    aby = by - ay
    l2 = abx * abx + aby * aby
    if l2 <= 0.0:
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy)
    t = ((px - ax) * abx + (py - ay) * aby) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    dx = px - (ax + t * abx)
    dy = py - (ay + t * aby)
    return math.sqrt(dx * dx + dy * dy)


class Polyline:
    """Arc-length parametrised polyline in (lon, lat) with per-vertex attributes."""

    def __init__(self, lon: np.ndarray, lat: np.ndarray):
        self.lon = np.asarray(lon, float)
        self.lat = np.asarray(lat, float)
        seg = haversine_km(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])
        self.seg_km = seg
        self.cum_km = np.concatenate([[0.0], np.cumsum(seg)])
        self.length_km = float(self.cum_km[-1])
        self.seg_bearing = initial_bearing(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])

    def locate(self, s_km):
        """Return (segment index, fraction along segment) for arc-length positions."""
        s = np.clip(np.asarray(s_km, float), 0.0, self.length_km - 1e-9)
        i = np.searchsorted(self.cum_km, s, side="right") - 1
        i = np.clip(i, 0, len(self.seg_km) - 1)
        f = (s - self.cum_km[i]) / np.maximum(self.seg_km[i], 1e-12)
        return i, f

    def point_at(self, s_km):
        i, f = self.locate(s_km)
        lat = self.lat[i] + f * (self.lat[i + 1] - self.lat[i])
        lon = self.lon[i] + f * (self.lon[i + 1] - self.lon[i])
        return lat, lon

    def bearing_at(self, s_km):
        i, _ = self.locate(s_km)
        return self.seg_bearing[i]


def grid_cell_id(lat, lon, res_deg: float = 0.25) -> np.ndarray:
    """Integer id of the regular lat/lon cell containing each point (used for spatial correlation)."""
    iy = np.floor((np.asarray(lat) + 90.0) / res_deg).astype(np.int64)
    ix = np.floor((np.asarray(lon) + 180.0) / res_deg).astype(np.int64)
    return iy * 100000 + ix


class SpatialGrid:
    """Bucket index of points on a regular lat/lon grid for fast radius queries inside numba kernels."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, cell_deg: float = 0.5):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        self.cell_deg = cell_deg
        self.lat0 = float(np.floor(lat.min() / cell_deg) * cell_deg) if lat.size else 0.0
        self.lon0 = float(np.floor(lon.min() / cell_deg) * cell_deg) if lon.size else 0.0
        self.ny = int(np.floor((lat.max() - self.lat0) / cell_deg)) + 1 if lat.size else 1
        self.nx = int(np.floor((lon.max() - self.lon0) / cell_deg)) + 1 if lon.size else 1
        iy = np.floor((lat - self.lat0) / cell_deg).astype(np.int64)
        ix = np.floor((lon - self.lon0) / cell_deg).astype(np.int64)
        cid = iy * self.nx + ix
        order = np.argsort(cid, kind="stable")
        counts = np.bincount(cid, minlength=self.ny * self.nx)
        self.cell_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.cell_items = order.astype(np.int64)

    def as_tuple(self):
        return (self.lat0, self.lon0, self.cell_deg, self.ny, self.nx, self.cell_ptr, self.cell_items)
