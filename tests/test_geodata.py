import json
import math
import struct
import zlib

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from catforge.exposure.portfolio import Portfolio
from catforge.geodata.buildings import match_footprints
from catforge.geodata.enrich import enrich_portfolio
from catforge.geodata.fetch import TileFetcher
from catforge.geodata.mvt import decode_polygons, encode_polygon_tile
from catforge.geodata.png import decode_png
from catforge.geodata.terrain import ground_elevation, tile_frac
from catforge.physics.dem import _png_rgb
from catforge.vulnerability.damage import surge_damage_ratio

Z = 14
# three buildings: A has a mapped 12 m footprint around it, B has one ~30 m away, C has none nearby
SITES = {"A": (26.5812, -81.9486), "B": (26.5790, -81.9440), "C": (26.5700, -81.9300)}


def _png_filtered(img: np.ndarray) -> bytes:
    """PNG encoder that cycles through all five scanline filters (to exercise the decoder)."""
    h, w, ch = img.shape
    rows, prev = [], np.zeros(w * ch, np.int32)
    for y in range(h):
        cur = img[y].reshape(-1).astype(np.int32)
        left = np.concatenate([np.zeros(ch, np.int32), cur[:-ch]])
        upleft = np.concatenate([np.zeros(ch, np.int32), prev[:-ch]])
        f = y % 5
        if f == 0:
            r = cur
        elif f == 1:
            r = cur - left
        elif f == 2:
            r = cur - prev
        elif f == 3:
            r = cur - (left + prev) // 2
        else:
            pa, pb, pc = np.abs(prev - upleft), np.abs(left - upleft), np.abs(left + prev - 2 * upleft)
            pred = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, upleft))
            r = cur - pred
        rows.append(bytes([f]) + (r & 255).astype(np.uint8).tobytes())
        prev = cur

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ctype = {1: 0, 3: 2, 4: 6}[ch]
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def test_png_decoder_all_filters():
    rng = np.random.default_rng(0)
    for ch in (1, 3, 4):
        img = rng.integers(0, 256, (37, 29, ch), dtype=np.uint8)
        assert np.array_equal(decode_png(_png_filtered(img)), img)


def _terrarium(elev: np.ndarray) -> bytes:
    v = elev + 32768.0
    rgb = np.stack([np.floor(v / 256), np.floor(v) % 256, np.floor((v - np.floor(v)) * 256)], axis=-1)
    return _png_rgb(rgb.astype(np.uint8))


def _unit(lat, lon, tx, ty, extent=4096):
    x, y = tile_frac(lat, lon, Z)
    return float((x - tx) * extent), float((y - ty) * extent)


def _mock_upstream(seen: list):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        path = req.url.path
        if req.url.host == "tiles.openfreemap.org" and path == "/planet":
            return httpx.Response(200, json={"tiles": ["https://tiles.openfreemap.org/planet/test/{z}/{x}/{y}.pbf"]})
        parts = path.removesuffix(".png").removesuffix(".pbf").split("/")
        tx, ty = int(parts[-2]), int(parts[-1])
        if path.endswith(".png"):  # ground rises 2 cm per pixel row from 1 m
            e = 1.0 + 0.02 * np.repeat(np.arange(256.0)[:, None], 256, axis=1)
            return httpx.Response(200, content=_terrarium(e), headers={"content-type": "image/png"})
        polys, props = [], []
        for name, (la, lo) in SITES.items():
            ux, uy = _unit(la, lo, tx, ty)
            if not (0 <= ux < 4096 and 0 <= uy < 4096) or name == "C":
                continue
            cx = ux if name == "A" else ux + 80  # B: footprint edge ~29 m east of the geocode (0.53 m per unit)
            d = 25
            ring = [(round(cx - d), round(uy - d)), (round(cx + d), round(uy - d)), (round(cx + d), round(uy + d)),
                    (round(cx - d), round(uy + d)), (round(cx - d), round(uy - d))]
            polys.append(ring)
            props.append({"render_height": 12.0 if name == "A" else 5.0, "render_min_height": 0.0})
        return httpx.Response(200, content=encode_polygon_tile(polys, props), headers={"content-type": "application/x-protobuf"})
    return handler


@pytest.fixture()
def fetcher(tmp_path):
    seen: list = []
    f = TileFetcher(cache_dir=tmp_path, transport=httpx.MockTransport(_mock_upstream(seen)))
    f.seen = seen
    return f


def test_mvt_roundtrip_and_proximity():
    ring = [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]
    far = [(3000, 3000), (3100, 3000), (3100, 3100), (3000, 3100), (3000, 3000)]
    tile = encode_polygon_tile([ring, far], [{"render_height": 9.0}, {"render_height": 5.0}])
    polys = decode_polygons(tile, Z, 4460, 6952)
    assert len(polys) == 2 and polys[0].props["render_height"] == 9.0
    n = 2 ** Z
    assert polys[0].lon[0] == pytest.approx((4460 + 100 / 4096) / n * 360 - 180)
    near_lon, near_lat = polys[0].lon[:4].mean(), polys[0].lat[:4].mean()
    assert len(decode_polygons(tile, Z, 4460, 6952, near=[(near_lon, near_lat)], radius_m=50)) == 1


def test_ground_elevation_bilinear(fetcher):
    la, lo = SITES["A"]
    g, info = ground_elevation([la], [lo], fetcher)
    _, y = tile_frac(la, lo, Z)
    py = min(max((y - math.floor(y)) * 256 - 0.5, 0), 255)
    assert g[0] == pytest.approx(1.0 + 0.02 * py, abs=0.01) and info["tiles"] == 1


def test_footprint_matching(fetcher):
    lat = np.array([v[0] for v in SITES.values()])
    lon = np.array([v[1] for v in SITES.values()])
    m, _ = match_footprints(lat, lon, fetcher)
    assert m[0]["match"] == "inside" and m[0]["height_m"] == 12.0 and 200 < m[0]["area_m2"] < 900
    assert m[1]["match"] == "snapped" and 15 < m[1]["snap_m"] < 35
    assert m[2] is None


def _portfolio():
    df = pd.DataFrame({"loc_id": list(SITES), "lat": [v[0] for v in SITES.values()], "lon": [v[1] for v in SITES.values()],
                       "tiv_building": [3e5, 4e5, 5e5], "stories": [1, 1, 2], "construction": "WOOD"})
    return Portfolio.from_frame(df, name="t")


def test_enrich_portfolio_and_offline_replay(fetcher, tmp_path):
    pf = _portfolio()
    new, rep = enrich_portfolio(pf, scope="all", fetcher=fetcher)
    L = new.locations
    assert np.isfinite(L["ground_elev_m"]).all() and (L["ground_elev_m"] > 1).all()
    assert list(L["stories"]) == [4, 1, 2]  # A: 12 m mapped → 4 storeys; B's 5 m is the "unknown" default
    assert L.loc[0, "building_height_m"] == 12.0 and L.loc[0, "floor_area_m2"] > 0
    assert rep["footprints"]["inside"] == 1 and rep["footprints"]["snapped"] == 1 and rep["footprints"]["none"] == 1
    assert list(pf.locations["stories"]) == [1, 1, 2] and "ground_elev_m" not in pf.locations or pf.locations["ground_elev_m"].isna().all()
    assert new.meta["enriched_from"] == pf.id and new.meta["quality"]["fp_match"] == ["inside", "snapped", "none"]
    # cached tiles replay offline (the TileJSON is re-read live, so offline falls back to its cached copy)
    off = TileFetcher(cache_dir=tmp_path, offline=True)
    again, rep2 = enrich_portfolio(pf, scope="all", fetcher=off)
    assert np.allclose(again.locations["ground_elev_m"], L["ground_elev_m"]) and rep2["fetch"]["requests"] == 0


def test_measured_ground_and_first_floor_drive_surge_damage():
    d = surge_damage_ratio(np.array([1.5, 1.5]), np.array(["WOOD", "WOOD"]), np.array(["RES_SF", "RES_SF"]),
                           np.array([1, 1]), np.array([1980, 1980]), first_floor_m=np.array([np.nan, 1.4]))
    assert d[0] > d[1] > 0  # an elevated first floor (elevation certificate) cuts damage at the same depth
    from catforge.physics.develop import _site_ground

    assert _site_ground(np.array([25.7617]), np.array([-80.1918]), np.array([0.5]))[0] == 0.5
    assert _site_ground(np.array([25.7617]), np.array([-80.1918]), np.array([np.nan]))[0] > 3  # 2′ DEM smooths Miami up


def test_enrichment_api(small_model, tmp_path):
    from catforge.api import create_app

    seen: list = []
    app = create_app(model=small_model, data_dir=str(tmp_path), demo=False, geodata_transport=httpx.MockTransport(_mock_upstream(seen)))
    with TestClient(app) as c:
        csv = _portfolio().to_csv()
        pid = c.post("/api/portfolios/upload", files={"file": ("b.csv", csv, "text/csv")}).json()["id"]
        assert c.get(f"/api/portfolios/{pid}/quality").status_code == 404
        r = c.post(f"/api/portfolios/{pid}/enrich", params={"wait": True}, json={"scope": "all"}).json()["result"]
        new = r["portfolio"]["id"]
        assert r["report"]["footprints"]["match_rate"] == pytest.approx(2 / 3) and new != pid
        q = c.get(f"/api/portfolios/{new}/quality").json()
        assert q["enriched_from"] == pid and q["locations"]["fp_match"][0] == "inside"
        e = c.get("/api/geodata/elevation", params={"lat": SITES["A"][0], "lon": SITES["A"][1]}).json()
        assert e["terrain_tiles_m"] > 1 and "etopo1_2min_m" in e
        assert json.dumps(q)  # JSON-clean (no NaN)
