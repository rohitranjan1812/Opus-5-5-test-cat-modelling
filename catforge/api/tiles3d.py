"""Google Photorealistic 3D Tiles — server-side pass-through so the API key never reaches the browser.

The browser loads ``/v1/3dtiles/root.json`` from this server. Google's tileset references children by
absolute paths (``/v1/3dtiles/datasets/...?session=...``), which resolve back to this same origin and
route through here too. Responses are relayed as-is: no caching or storage beyond what Google's own
HTTP cache headers allow, which keeps within the Map Tiles API policies. The page must display the
tileset's attributions.

Configure with ``GOOGLE_MAPS_API_KEY`` (or ``CATFORGE_GOOGLE_MAPS_API_KEY``). Use a key restricted to the
**Map Tiles API**; this is a server-side key, so use an API/IP restriction, not an HTTP-referrer one.
"""

from __future__ import annotations

import os
import re

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

UPSTREAM = "https://tile.googleapis.com"
# only the 3D Tiles tree: root.json and datasets/... files (no traversal to other Google APIs)
_SAFE_PATH = re.compile(r"^(root\.json|datasets/[A-Za-z0-9_\-]+(/[A-Za-z0-9_\-=]+)*(\.[A-Za-z0-9]+)?)$")
_PASS_HEADERS = ("cache-control", "expires", "etag", "last-modified")


def google_key() -> str | None:
    return os.environ.get("CATFORGE_GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_MAPS_API_KEY") or None


def register(app: FastAPI, key: str | None = None, upstream: str = UPSTREAM, transport: httpx.BaseTransport | None = None):
    key = key if key is not None else google_key()
    client = httpx.Client(base_url=upstream, timeout=30.0, transport=transport, follow_redirects=False)

    @app.get("/api/integrations", tags=["meta"])
    def integrations():
        """External map integrations available to the 3-D development view (never exposes secrets)."""
        return {
            "google_3d_tiles": {
                "available": bool(key), "mode": "server-proxy" if key else "none", "root": "/v1/3dtiles/root.json",
                "setup": None if key else "Set GOOGLE_MAPS_API_KEY on the server (Map Tiles API enabled), or paste a "
                                          "browser key in the 3-D view (stored only in that browser).",
                "attribution": "Google",
            },
            "osm_vector_tiles": {"tilejson": "https://tiles.openfreemap.org/planet",
                                 "attribution": "© OpenMapTiles © OpenStreetMap contributors"},
        }

    @app.get("/v1/3dtiles/{path:path}", include_in_schema=False)
    def google_3dtiles(path: str, request: Request):
        if not key:
            raise HTTPException(503, "Google Photorealistic 3D Tiles are not configured on this server "
                                     "(set GOOGLE_MAPS_API_KEY with the Map Tiles API enabled)")
        if not _SAFE_PATH.match(path):
            raise HTTPException(404, "not a 3D Tiles resource")
        params = [(k, v) for k, v in request.query_params.multi_items() if k.lower() != "key"]
        try:
            r = client.get(f"/v1/3dtiles/{path}", params=params, headers={"X-Goog-Api-Key": key})
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"upstream error: {type(exc).__name__}") from exc
        headers = {h: r.headers[h] for h in _PASS_HEADERS if h in r.headers}
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"), headers=headers)

    return app
