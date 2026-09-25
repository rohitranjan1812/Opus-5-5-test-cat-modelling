"""Minimal Mapbox Vector Tile (v2) reader for building footprints — no protobuf dependency.

Only what exposure matching needs: polygon exterior rings of one layer (default ``building``,
OpenMapTiles schema) and numeric properties such as ``render_height``. Geometry is decoded only for
features whose first vertex lies near a point of interest, which keeps dense city tiles cheap.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if x < 0x80:
            return r, i
        s += 7


def _fields(b: bytes, start: int = 0, end: int | None = None):
    """Iterate (field_number, wire_type, value) — value is an int or a (start, end) slice for bytes."""
    i, end = start, len(b) if end is None else end
    while i < end:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield f, wt, v
        elif wt == 2:
            n, i = _varint(b, i)
            yield f, wt, (i, i + n)
            i += n
        elif wt == 1:
            yield f, wt, (i, i + 8)
            i += 8
        elif wt == 5:
            yield f, wt, (i, i + 4)
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")


def _packed(b: bytes, s: tuple[int, int]) -> list[int]:
    out, i = [], s[0]
    while i < s[1]:
        v, i = _varint(b, i)
        out.append(v)
    return out


def _zz(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _value(b: bytes, s: tuple[int, int]):
    for f, _wt, v in _fields(b, *s):
        if f == 1:
            return b[v[0]:v[1]].decode("utf-8", "replace")
        if f == 2:
            return struct.unpack("<f", b[v[0]:v[1]])[0]
        if f == 3:
            return struct.unpack("<d", b[v[0]:v[1]])[0]
        if f in (4, 5):
            return v if v < (1 << 63) else v - (1 << 64)
        if f == 6:
            return _zz(v)
        if f == 7:
            return bool(v)
    return None


@dataclass
class Footprint:
    lon: np.ndarray  # exterior ring (closed)
    lat: np.ndarray
    props: dict


def _rings(cmds: list[int]) -> list[list[tuple[int, int]]]:
    rings, cur, x, y, i = [], [], 0, 0, 0
    while i < len(cmds):
        cid, cnt = cmds[i] & 7, cmds[i] >> 3
        i += 1
        if cid in (1, 2):
            for _ in range(cnt):
                x += _zz(cmds[i])
                y += _zz(cmds[i + 1])
                i += 2
                if cid == 1 and cur:
                    rings.append(cur)
                    cur = []
                cur.append((x, y))
        elif cid == 7:
            if cur:
                cur.append(cur[0])
                rings.append(cur)
                cur = []
    if cur:
        rings.append(cur)
    return rings


def decode_polygons(data: bytes, z: int, tx: int, ty: int, layer: str = "building",
                    near: list[tuple[float, float]] | None = None, radius_m: float = 150.0,
                    props: tuple[str, ...] = ("render_height", "render_min_height")) -> list[Footprint]:
    """Exterior rings (lon/lat) of the polygons in ``layer``; with ``near`` only those close to those points."""
    if data[:2] == b"\x1f\x8b":  # gzip-wrapped tile
        import gzip

        data = gzip.decompress(data)
    b = data
    n = 2 ** z
    out: list[Footprint] = []
    for f, wt, v in _fields(b):
        if f != 3 or wt != 2:
            continue
        name, extent, keys, values, feats = None, 4096, [], [], []
        for lf, _lwt, lv in _fields(b, *v):
            if lf == 1:
                name = b[lv[0]:lv[1]].decode()
            elif lf == 2:
                feats.append(lv)
            elif lf == 3:
                keys.append(b[lv[0]:lv[1]].decode())
            elif lf == 4:
                values.append(lv)
            elif lf == 5:
                extent = lv
        if name != layer:
            continue
        want = {k: i for i, k in enumerate(keys) if k in props}
        # points of interest in tile units, with a squared radius
        m_per_unit = 40_075_016.7 * math.cos(math.radians(near[0][1])) / (n * extent) if near else 1.0
        pts = []
        for lon, lat in near or []:
            px = ((lon + 180.0) / 360.0 * n - tx) * extent
            py = ((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n - ty) * extent
            pts.append((px, py))
        r2 = (radius_m / m_per_unit) ** 2
        for fs in feats:
            tags, geom, gtype = None, None, 0
            for ff, _fwt, fv in _fields(b, *fs):
                if ff == 2:
                    tags = fv
                elif ff == 3:
                    gtype = fv
                elif ff == 4:
                    geom = fv
            if gtype != 3 or geom is None:
                continue
            p = {}
            if tags is not None and want:
                t = _packed(b, tags)
                for k in range(0, len(t) - 1, 2):
                    if t[k] in want.values():
                        p[keys[t[k]]] = _value(b, values[t[k + 1]])
            # producers merge same-attribute buildings into one MultiPolygon feature, so the proximity
            # test is per ring (bounding box vs the points of interest), not per feature
            for ring in _rings(_packed(b, geom)):
                if len(ring) < 4:
                    continue
                xs = np.array([q[0] for q in ring], float)
                ys = np.array([q[1] for q in ring], float)
                if pts:
                    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
                    if min(max(x0 - qx, 0, qx - x1) ** 2 + max(y0 - qy, 0, qy - y1) ** 2 for qx, qy in pts) > r2:
                        continue
                area2 = float(np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1]))
                if area2 <= 0:  # MVT: exterior rings have positive area in tile coordinates (y down)
                    continue
                lon = (tx + xs / extent) / n * 360.0 - 180.0
                lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (ty + ys / extent) / n))))
                out.append(Footprint(lon, lat, p))
    return out


def encode_polygon_tile(polys: list[list[tuple[int, int]]], props: list[dict], layer: str = "building",
                        extent: int = 4096) -> bytes:
    """Tiny MVT encoder (tests/fixtures): polygons in tile units with numeric properties."""
    def vint(v):
        out = bytearray()
        while True:
            x = v & 0x7F
            v >>= 7
            out.append(x | (0x80 if v else 0))
            if not v:
                return bytes(out)

    def field(num, wt, payload):
        if wt == 0:
            return vint(num << 3) + vint(payload)
        return vint((num << 3) | 2) + vint(len(payload)) + payload

    def zz(n):
        return (n << 1) ^ (n >> 63)

    keys = sorted({k for p in props for k in p})
    vals: list[float] = []
    feats = b""
    for ring, p in zip(polys, props):
        cmds, x, y = [], 0, 0
        cmds.append((1 & 7) | (1 << 3))
        cmds += [zz(ring[0][0] - x), zz(ring[0][1] - y)]
        x, y = ring[0]
        cmds.append((2 & 7) | ((len(ring) - 1) << 3))
        for qx, qy in ring[1:]:
            cmds += [zz(qx - x), zz(qy - y)]
            x, y = qx, qy
        cmds.append(7 | (1 << 3))
        tags = []
        for k, v in p.items():
            vals.append(float(v))
            tags += [keys.index(k), len(vals) - 1]
        feat = field(2, 2, b"".join(vint(t) for t in tags)) + field(3, 0, 3) + field(4, 2, b"".join(vint(c) for c in cmds))
        feats += field(2, 2, feat)
    lay = field(15, 0, 2) + field(1, 2, layer.encode()) + feats
    lay += b"".join(field(3, 2, k.encode()) for k in keys)
    lay += b"".join(field(4, 2, bytes([0x19]) + struct.pack("<d", v)) for v in vals)
    lay += field(5, 0, extent)
    return field(3, 2, lay)
