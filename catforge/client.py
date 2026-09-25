"""Python SDK for the CatForge REST API.

    from catforge.client import CatForgeClient
    cf = CatForgeClient("http://localhost:8000")
    pf = cf.create_synthetic_portfolio(n_locations=10000, states=["FL", "TX"])
    an = cf.run_analysis(pf["id"], n_years=50000, reinsurance={"contracts": [...]})
    cf.ep(an["id"], basis="net")
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx


class CatForgeError(RuntimeError):
    pass


class CatForgeClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0,
                 client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self._http = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------ plumbing
    def _req(self, method: str, path: str, **kw):
        r = self._http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text
            raise CatForgeError(f"{method} {path} → {r.status_code}: {detail}")
        ctype = r.headers.get("content-type", "")
        return r.json() if "json" in ctype else r.text

    def get(self, path, **params):
        return self._req("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path, json=None, **params):
        return self._req("POST", path, json=json, params={k: v for k, v in params.items() if v is not None})

    def wait(self, job: dict, poll: float = 0.5, timeout: float = 3600.0, on_progress=None) -> dict:
        t0 = time.time()
        jid = job["id"]
        while True:
            j = self.get(f"/api/jobs/{jid}")
            if on_progress:
                on_progress(j)
            if j["status"] == "done":
                return j
            if j["status"] == "error":
                raise CatForgeError(j.get("error"))
            if time.time() - t0 > timeout:
                raise TimeoutError(f"job {jid} still {j['status']}")
            time.sleep(poll)

    # ------------------------------------------------------------------ meta / hazard
    def health(self):
        return self.get("/api/health")

    def meta(self):
        return self.get("/api/meta")

    def catalogs(self):
        return self.get("/api/catalogs")

    def events(self, peril: str, limit: int = 100, sort: str = "rate", order: str = "desc", q: str | None = None):
        return self.get(f"/api/catalogs/{peril}/events", limit=limit, sort=sort, order=order, q=q)

    def hazard_curve(self, peril: str, lat: float, lon: float, terrain: str = "open", vs30: float = 400.0):
        return self.get("/api/hazard/curve", peril=peril, lat=lat, lon=lon, terrain=terrain, vs30=vs30)

    def hazard_map(self, peril: str, rp: float = 100.0, res_deg: float = 0.25):
        return self.get("/api/hazard/map", peril=peril, rp=rp, res_deg=res_deg)

    def vulnerability_curve(self, **params):
        return self.get("/api/vulnerability/curve", **params)

    # ------------------------------------------------------------------ exposure
    def portfolios(self):
        return self.get("/api/portfolios")

    def create_synthetic_portfolio(self, **params):
        return self.post("/api/portfolios/synthetic", json=params)

    def upload_portfolio(self, path: str | Path, name: str | None = None):
        p = Path(path)
        with open(p, "rb") as fh:
            r = self._http.post("/api/portfolios/upload", files={"file": (p.name, fh, "text/csv")},
                                data={"name": name or p.stem})
        if r.status_code >= 400:
            raise CatForgeError(r.text)
        return r.json()

    def portfolio(self, pid: str):
        return self.get(f"/api/portfolios/{pid}")

    # ------------------------------------------------------------------ analyses
    def run_analysis(self, portfolio_id: str, wait: bool = True, on_progress=None, **config) -> dict:
        job = self.post("/api/analyses", json={"portfolio_id": portfolio_id, "config": config})
        if not wait:
            return job
        done = self.wait(job, on_progress=on_progress)
        return self.analysis(done["result_id"])

    def analyses(self):
        return self.get("/api/analyses")

    def analysis(self, aid: str):
        return self.get(f"/api/analyses/{aid}")

    def ep(self, aid: str, basis: str = "gross", peril: str = "all"):
        return self.get(f"/api/analyses/{aid}/ep", basis=basis, peril=peril)

    def elt(self, aid: str, limit: int = 200, peril: str | None = None):
        return self.get(f"/api/analyses/{aid}/elt", limit=limit, peril=peril)

    def allocation(self, aid: str, dimension: str = "state", top: int | None = None):
        return self.get(f"/api/analyses/{aid}/allocation", dimension=dimension, top=top)

    def insights(self, aid: str):
        return self.get(f"/api/analyses/{aid}/insights")

    def evaluate_reinsurance(self, aid: str, program: dict):
        return self.post(f"/api/analyses/{aid}/reinsurance", json={"program": program})

    def optimise_reinsurance(self, aid: str, **params):
        return self.post(f"/api/analyses/{aid}/reinsurance/optimize", json=params)

    def climate(self, aid: str, tc_frequency=1.0, tc_intensity=1.0, eq_frequency=1.0):
        return self.post(f"/api/analyses/{aid}/climate",
                         json={"tc_frequency": tc_frequency, "tc_intensity": tc_intensity, "eq_frequency": eq_frequency})

    def sensitivity(self, aid: str, spread: float = 0.2):
        return self.wait(self.post(f"/api/analyses/{aid}/sensitivity", spread=spread))["result"]

    def mitigation(self, aid: str, preset: str | None = None, filter: dict | None = None, changes: dict | None = None):
        job = self.post(f"/api/analyses/{aid}/mitigation", json={"preset": preset, "filter": filter, "changes": changes})
        return self.wait(job)["result"]

    def marginal(self, aid: str, acc_ids: list[str]):
        return self.wait(self.post(f"/api/analyses/{aid}/marginal", json={"acc_ids": acc_ids}))["result"]

    def scenario(self, portfolio_id: str, analog: str | None = None, peril: str | None = None,
                 params: dict | None = None, n_samples: int = 1000):
        out = self.post("/api/scenarios/run", json={"portfolio_id": portfolio_id, "analog": analog, "peril": peril,
                                                    "params": params, "n_samples": n_samples})
        return out["result"]

    # ------------------------------------------------------------------ event development (4-D)
    def develop(self, portfolio_id: str | None = None, analog: str | None = None, peril: str | None = None,
                event_id: int | None = None, params: dict | None = None, seed: int = 1, surge: bool = True) -> dict:
        """Time-resolved realization of one event (surge/rain frames or rupture/wave grids, per-building damage)."""
        job = self.post("/api/develop", json={"portfolio_id": portfolio_id, "analog": analog, "peril": peril,
                                              "event_id": event_id, "params": params, "seed": seed, "surge": surge})
        return self.wait(job)["result"]

    def seismogram(self, lat: float, lon: float, analog: str | None = None, event_id: int | None = None,
                   params: dict | None = None, vs30: float = 400.0, seed: int = 1) -> dict:
        """Stochastic finite-fault seismogram + 5 %-damped PSA at a site, with the GMPE median ±1σ for comparison."""
        return self.post("/api/develop/seismogram", json={"analog": analog, "peril": "EQ", "event_id": event_id,
                                                           "params": params, "lat": lat, "lon": lon, "vs30": vs30,
                                                           "seed": seed})

    @staticmethod
    def decode(arr: dict):
        """Decode a base64 grid from a development payload into a float numpy array (NaN for gaps)."""
        import base64

        import numpy as np

        raw = np.frombuffer(base64.b64decode(arr["b64"]), {"int16": "<i2", "uint8": "u1", "float32": "<f4"}[arr["dtype"]])
        out = raw.astype(float).reshape(arr["shape"])
        if arr.get("nan") is not None:
            out[raw.reshape(arr["shape"]) == arr["nan"]] = np.nan
        return out / arr["scale"]

    # ------------------------------------------------------------------ exposure enrichment
    def enrich(self, portfolio_id: str, scope: str = "coastal", elevation: bool = True, footprints: bool = True,
               max_snap_m: float = 35.0, on_progress=None) -> dict:
        """Building-scale ground elevation + mapped-footprint attributes; returns the new portfolio and the report."""
        job = self.post(f"/api/portfolios/{portfolio_id}/enrich", json={"scope": scope, "elevation": elevation,
                                                                       "footprints": footprints, "max_snap_m": max_snap_m})
        return self.wait(job, on_progress=on_progress)["result"]

    def quality(self, portfolio_id: str, limit: int = 500) -> dict:
        return self.get(f"/api/portfolios/{portfolio_id}/quality", limit=limit)

    def elevation(self, lat: float, lon: float, lidar: bool = False) -> dict:
        return self.get("/api/geodata/elevation", lat=lat, lon=lon, lidar=str(lidar).lower())
