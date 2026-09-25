"""CatForge REST API (FastAPI).  Interactive OpenAPI docs at /docs.

Heavy computations run as background jobs (``/api/jobs/{id}``) — or synchronously with ``?wait=true``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..analytics import allocation as alloc
from ..analytics import sensitivity as sens
from ..analytics.ep import quantile_at_rp
from ..config import (
    CONSTRUCTION_CLASSES,
    OCCUPANCIES,
    PERIL_INDEX,
    PERIL_NAMES,
    PERILS,
    ROOF_SHAPES,
    TERRAINS,
)
from ..engine.model import AnalysisConfig, AnalysisResult, CatModel, compute_ep
from ..exposure.portfolio import SCHEMA, Portfolio
from ..exposure.synthetic import generate_portfolio
from ..financial.reinsurance import Program, apply_program, layer_metrics, optimise_cat_xl
from ..hazard.earthquake import generate_eq_catalog, rupture_of
from ..hazard.footprint import event_footprint_grid, hazard_curve, hazard_map
from ..hazard.tropical_cyclone import TERRAIN_GUST_FACTOR, generate_tc_catalog, track_of
from ..scenario import ANALOGS, run_scenario
from ..vulnerability.damage import vulnerability_curve
from .schemas import (
    AnalysisRequest,
    CatalogRebuildRequest,
    ClimateRequest,
    DevelopRequest,
    MarginalRequest,
    MitigationRequest,
    OptimizeRequest,
    ReinsuranceRequest,
    ScenarioRequest,
    SeismogramRequest,
    SyntheticPortfolioRequest,
)
from .store import Job, Store

log = logging.getLogger("catforge")


def clean(o):
    """Make numpy / pandas objects JSON-safe (NaN/inf → null)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict(orient="records"))
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def J(o, status: int = 200) -> JSONResponse:
    return JSONResponse(content=clean(o), status_code=status)


def create_app(model: CatModel | None = None, data_dir: str | None = None, demo: bool | None = None,
               demo_locations: int | None = None, demo_years: int | None = None,
               google_maps_key: str | None = None, google_transport=None) -> FastAPI:
    data_dir = data_dir if data_dir is not None else os.environ.get("CATFORGE_DATA_DIR")
    demo = demo if demo is not None else os.environ.get("CATFORGE_DEMO", "1") != "0"
    demo_locations = demo_locations or int(os.environ.get("CATFORGE_DEMO_LOCATIONS", "5000"))
    demo_years = demo_years or int(os.environ.get("CATFORGE_DEMO_YEARS", "20000"))
    store = Store(model or CatModel.default(), data_dir=data_dir or None)

    app = FastAPI(title="CatForge API", version=__version__,
                  description="Multi-peril catastrophe modelling platform: hazard, vulnerability, financial, "
                              "Monte Carlo loss engine, analytics, reinsurance and insights.")
    app.state.store = store
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # ------------------------------------------------------------------ helpers
    def get_pf(pid: str) -> Portfolio:
        pf = store.portfolios.get(pid)
        if pf is None:
            raise HTTPException(404, f"portfolio {pid} not found")
        return pf

    def get_an(aid: str) -> AnalysisResult:
        res = store.analyses.get(aid)
        if res is None:
            raise HTTPException(404, f"analysis {aid} not found")
        if res.ctx is None:
            raise HTTPException(409, "analysis context unavailable")
        return res

    def job_or_wait(job: Job, wait: bool, timeout: float = 600.0):
        if not wait:
            return J(job.public(), 202)
        t0 = time.time()
        while job.status in ("queued", "running") and time.time() - t0 < timeout:
            time.sleep(0.05)
        if job.status == "error":
            raise HTTPException(500, job.error)
        return J({"job": job.public(), "result": job.result})

    def analysis_brief(r: AnalysisResult) -> dict:
        s = r.summary
        rp = {row["rp"]: row for row in s.get("rp_table", [])}
        return {"id": r.id, "name": r.name, "portfolio_id": r.portfolio_id, "portfolio_name": s.get("portfolio_name"),
                "created": r.created, "n_years": r.n_years, "perils": r.config.perils, "aal": s.get("aal"),
                "aep_100": rp.get(100, {}).get("gross_aep"), "aep_250": rp.get(250, {}).get("gross_aep"),
                "has_reinsurance": bool(r.reinsurance), "total_s": r.timings.get("total_s")}

    # ------------------------------------------------------------------ meta
    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok", "version": __version__, "portfolios": len(store.portfolios),
                "analyses": len(store.analyses), "catalogs": {p: c.n_events for p, c in store.model.catalogs.items()}}

    @app.get("/api/meta", tags=["meta"])
    def meta():
        return J({
            "version": __version__, "perils": [{"code": p, "name": PERIL_NAMES[p]} for p in PERILS],
            "construction_classes": CONSTRUCTION_CLASSES, "occupancies": OCCUPANCIES, "terrains": list(TERRAINS),
            "terrain_gust_factor": TERRAIN_GUST_FACTOR, "roof_shapes": list(ROOF_SHAPES),
            "dimensions": list(alloc.DIMENSIONS), "mitigation_presets": {k: v["label"] for k, v in sens.MITIGATION_PRESETS.items()},
            "analogs": {k: {"label": v["label"], "peril": v["peril"]} for k, v in ANALOGS.items()},
            "default_config": AnalysisConfig().to_dict(),
            "exposure_schema": {k: {"type": t, "default": d, "required": k in ("lat", "lon", "tiv_building")}
                                for k, (t, d) in SCHEMA.items()},
        })

    # ------------------------------------------------------------------ catalogs & hazard
    @app.get("/api/catalogs", tags=["hazard"])
    def catalogs():
        return J([c.summary() for c in store.model.catalogs.values()])

    def get_cat(peril: str):
        cat = store.model.catalogs.get(peril.upper())
        if cat is None:
            raise HTTPException(404, f"no catalog for peril {peril}")
        return cat

    def event_index(cat, event_id: int) -> int:
        idx = np.nonzero(cat.events["event_id"].to_numpy() == event_id)[0]
        if idx.size == 0:
            raise HTTPException(404, f"event {event_id} not found")
        return int(idx[0])

    @app.get("/api/catalogs/{peril}/stats", tags=["hazard"])
    def catalog_stats(peril: str):
        cat = get_cat(peril)
        ev = cat.events
        if cat.peril == "TC":
            by_region = ev.groupby(["region", "category"]).rate.sum().unstack(fill_value=0.0)
            return J({"peril": "TC", "total_rate": cat.total_rate,
                      "by_region": [{"region": r, **{f"cat{c}": float(v) for c, v in row.items()}}
                                    for r, row in by_region.iterrows()],
                      "by_category": ev.groupby("category").rate.sum().to_dict(),
                      "vmax_hist": np.histogram(ev["vmax"], bins=30, weights=ev["rate"])})
        mbins = np.arange(5.0, 9.3, 0.1)
        ev_m = ev["mag"].to_numpy()
        cum = [float(ev.rate[ev_m >= m].sum()) for m in mbins]
        by_src = ev.groupby("source").agg(rate=("rate", "sum"), mmax=("mag", "max"), n=("event_id", "size"))
        return J({"peril": "EQ", "total_rate": cat.total_rate, "mfd": {"m": mbins.round(2), "rate_ge": cum},
                  "by_source": [{"source": k, **r.to_dict()} for k, r in by_src.sort_values("rate", ascending=False).iterrows()]})

    @app.get("/api/catalogs/{peril}/events", tags=["hazard"])
    def catalog_events(peril: str, limit: int = Query(100, le=5000), offset: int = 0, sort: str = "rate",
                       order: str = "desc", q: str | None = None):
        cat = get_cat(peril)
        ev = cat.events
        if q:
            mask = ev.astype(str).apply(lambda c: c.str.contains(q, case=False, regex=False)).any(axis=1)
            ev = ev[mask]
        if sort in ev.columns:
            ev = ev.sort_values(sort, ascending=(order == "asc"))
        page = ev.iloc[offset: offset + limit]
        return J({"total": int(len(ev)), "offset": offset, "rows": page})

    @app.get("/api/catalogs/{peril}/events/{event_id}", tags=["hazard"])
    def catalog_event(peril: str, event_id: int):
        cat = get_cat(peril)
        i = event_index(cat, event_id)
        geom = track_of(cat, i) if cat.peril == "TC" else rupture_of(cat, i)
        return J({"event": cat.events.iloc[i].to_dict(), "geometry": geom})

    @app.get("/api/catalogs/{peril}/events/{event_id}/footprint", tags=["hazard"])
    def catalog_event_footprint(peril: str, event_id: int, res_deg: float = Query(0.05, ge=0.01, le=0.5)):
        cat = get_cat(peril)
        i = event_index(cat, event_id)
        return J(store.cached(("fp", cat.peril, event_id, res_deg), lambda: event_footprint_grid(cat, i, res_deg)))

    @app.get("/api/catalogs/{peril}/geometry", tags=["hazard"])
    def catalog_geometry(peril: str, limit: int = Query(300, le=3000), min_mag: float = 6.5, seed: int = 0):
        """A representative subset of tracks / ruptures for map display."""
        cat = get_cat(peril)
        ev = cat.events
        rng = np.random.default_rng(seed)
        if cat.peril == "TC":
            strong = ev.sort_values("vmax", ascending=False).head(limit // 2).index.to_numpy()
            rest = np.setdiff1d(np.arange(len(ev)), strong)
            pick = np.concatenate([strong, rng.choice(rest, size=min(limit - strong.size, rest.size), replace=False)])
            items = []
            for i in pick:
                t = track_of(cat, int(i))
                items.append({"event_id": int(ev.event_id.iloc[i]), "category": int(ev.category.iloc[i]),
                              "vmax": float(ev.vmax.iloc[i]), "lat": t["lat"][::2], "lon": t["lon"][::2]})
            return J({"peril": "TC", "items": items})
        sel = ev[(ev.mag >= min_mag)]
        if len(sel) > limit:
            sel = sel.sample(limit, random_state=seed)
        items = [{"event_id": int(r.event_id), "mag": float(r.mag), "source": r.source,
                  **rupture_of(cat, int(i))} for i, r in sel.iterrows()]
        return J({"peril": "EQ", "items": items})

    @app.post("/api/catalogs/{peril}/rebuild", tags=["hazard"])
    def catalog_rebuild(peril: str, req: CatalogRebuildRequest, wait: bool = False):
        peril = peril.upper()
        if peril not in PERILS:
            raise HTTPException(404, f"unknown peril {peril}")

        def work(job: Job):
            job.message = f"Generating {peril} catalog"
            if peril == "TC":
                m = store.model.catalogs["TC"].meta
                cat = generate_tc_catalog(req.n_hurricanes or m.get("n_hurricanes", 4000),
                                          req.n_tropical_storms if req.n_tropical_storms is not None else
                                          m.get("n_tropical_storms", 800), seed=req.seed or m.get("seed", 2024))
            else:
                cat = generate_eq_catalog(seed=req.seed or 7, area_position_density=req.area_position_density or 1.0)
            store.model.catalogs[peril] = cat
            store.model.invalidate(peril)
            with store.lock:
                store.cache = {k: v for k, v in store.cache.items() if peril not in k}
            return cat.summary()

        return job_or_wait(store.submit("catalog", work), wait)

    @app.get("/api/hazard/curve", tags=["hazard"])
    def api_hazard_curve(peril: str, lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180),
                         terrain: str = "open", vs30: float = Query(400.0, ge=150, le=1500)):
        cat = get_cat(peril)
        return J(hazard_curve(cat, lat, lon, terrain_k=TERRAIN_GUST_FACTOR.get(terrain, 1.10), vs30=vs30))

    @app.get("/api/hazard/map", tags=["hazard"])
    def api_hazard_map(peril: str, rp: float = Query(100.0, ge=2, le=10000), res_deg: float = Query(0.25, ge=0.1, le=1.0)):
        cat = get_cat(peril)
        return J(store.cached(("hazmap", cat.peril, rp, res_deg), lambda: hazard_map(cat, rp, res_deg)))

    @app.get("/api/vulnerability/curve", tags=["vulnerability"])
    def api_vuln_curve(peril: str = "TC", construction: str = "WOOD", occupancy: str = "RES_SF", year_built: int = 1990,
                       stories: int = 1, roof_shape: str = "unknown", shutters: int = 0, with_hazard_uncertainty: bool = True):
        peril = peril.upper()
        if construction not in CONSTRUCTION_CLASSES:
            raise HTTPException(422, f"unknown construction {construction}")
        sw = store.model.catalogs[peril].uncertainty.sigma_within if with_hazard_uncertainty else 0.0
        return J(vulnerability_curve(peril, construction, occupancy, year_built, stories, roof_shape, shutters, sw))

    # ------------------------------------------------------------------ portfolios
    @app.get("/api/portfolios", tags=["exposure"])
    def list_portfolios():
        return J([{"id": p.id, "name": p.name, "created": p.created, "n_locations": p.n, "tiv": p.tiv_total,
                   "source": p.meta.get("source", "upload")} for p in store.portfolios.values()])

    @app.post("/api/portfolios/synthetic", tags=["exposure"])
    def create_synthetic(req: SyntheticPortfolioRequest):
        try:
            pf = generate_portfolio(req.n_locations, req.seed, req.states, req.commercial_share, req.name)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        store.add_portfolio(pf)
        return J(pf.summary(), 201)

    @app.post("/api/portfolios/upload", tags=["exposure"])
    async def upload_portfolio(file: UploadFile = File(...), name: str = Form("uploaded portfolio")):
        raw = await file.read()
        try:
            pf = Portfolio.from_csv(raw, name=name)
        except Exception as exc:
            raise HTTPException(422, f"invalid portfolio: {exc}") from exc
        pf.meta["source"] = "upload"
        pf.meta["filename"] = file.filename
        store.add_portfolio(pf)
        return J(pf.summary(), 201)

    @app.get("/api/portfolios/{pid}", tags=["exposure"])
    def get_portfolio(pid: str):
        return J(get_pf(pid).summary())

    @app.get("/api/portfolios/{pid}/locations", tags=["exposure"])
    def portfolio_locations(pid: str, format: str = Query("columns", pattern="^(columns|records|csv)$"),
                            limit: int = Query(250000, le=1_000_000), offset: int = 0):
        pf = get_pf(pid)
        L = pf.locations.iloc[offset: offset + limit]
        if format == "csv":
            return Response(L.to_csv(index=False), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{pid}.csv"'})
        if format == "records":
            return J(L)
        cols = ["loc_id", "acc_id", "lat", "lon", "state", "construction", "occupancy", "lob", "year_built", "stories",
                "terrain"]
        out = {c: L[c].tolist() for c in cols}
        out["tiv"] = L[["tiv_building", "tiv_contents", "tiv_bi"]].sum(axis=1).round(0).tolist()
        return J(out)

    @app.delete("/api/portfolios/{pid}", tags=["exposure"])
    def delete_portfolio(pid: str):
        get_pf(pid)
        store.delete_portfolio(pid)
        return {"deleted": pid}

    # ------------------------------------------------------------------ analyses
    @app.post("/api/analyses", tags=["analysis"])
    def create_analysis(req: AnalysisRequest, wait: bool = False):
        pf = get_pf(req.portfolio_id)
        cfg = AnalysisConfig.from_dict(req.config.model_dump(exclude_none=True))

        def work(job: Job):
            def progress(f, msg=""):
                job.progress, job.message = f, msg

            res = store.model.run(pf, cfg, progress=progress)
            store.add_analysis(res)
            job.result_id = res.id
            return analysis_brief(res)

        return job_or_wait(store.submit("analysis", work), wait)

    @app.get("/api/analyses", tags=["analysis"])
    def list_analyses():
        return J(sorted([analysis_brief(r) for r in store.analyses.values()], key=lambda d: d["created"], reverse=True))

    @app.get("/api/analyses/{aid}", tags=["analysis"])
    def get_analysis(aid: str):
        r = get_an(aid)
        return J({**r.summary, "insights": r.insights})

    @app.delete("/api/analyses/{aid}", tags=["analysis"])
    def delete_analysis(aid: str):
        get_an(aid)
        store.delete_analysis(aid)
        return {"deleted": aid}

    @app.get("/api/analyses/{aid}/ep", tags=["results"])
    def analysis_ep(aid: str, basis: str = Query("gross", pattern="^(gu|gross|net)$"), peril: str = "all"):
        r = get_an(aid)
        if basis not in r.ep:
            raise HTTPException(404, f"basis {basis} not available (no net without reinsurance/per-risk)")
        if peril not in r.ep[basis]:
            raise HTTPException(404, f"peril {peril} not in analysis")
        return J({"basis": basis, "peril": peril, **r.ep[basis][peril]})

    @app.get("/api/analyses/{aid}/ep/all", tags=["results"])
    def analysis_ep_all(aid: str):
        r = get_an(aid)
        return J({b: {p: {"aep_curve": v["aep_curve"], "oep_curve": v["oep_curve"], "stats": v["stats"]}
                      for p, v in d.items()} for b, d in r.ep.items()})

    @app.get("/api/analyses/{aid}/analytic", tags=["results"])
    def analysis_analytic(aid: str):
        return J(get_an(aid).analytic)

    @app.get("/api/analyses/{aid}/elt", tags=["results"])
    def analysis_elt(aid: str, limit: int = Query(200, le=100000), offset: int = 0, peril: str | None = None,
                     format: str = Query("json", pattern="^(json|csv)$")):
        r = get_an(aid)
        elt = r.elt if not peril else r.elt[r.elt.peril == peril.upper()]
        if format == "csv":
            return Response(elt.to_csv(index=False), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{aid}_elt.csv"'})
        return J({"total": int(len(elt)), "rows": elt.iloc[offset: offset + limit]})

    def annual_frame(r: AnalysisResult) -> pd.DataFrame:
        y = r.ylt
        df = pd.DataFrame({"year": np.arange(r.n_years), "gu": r.annual("gu"), "gross": r.annual("gross"),
                           "net": r.annual("net"), "max_occurrence": r.annual_max("gross"),
                           "n_occurrences": np.bincount(y.year, minlength=r.n_years)})
        for p in r.config.perils:
            df[f"gross_{p}"] = r.annual("gross", p)
        if "TC" in r.config.perils and r.ctx.freq.get("TC") and r.ctx.freq["TC"].regimes:
            names = [g.name for g in r.ctx.freq["TC"].regimes]
            df["tc_regime"] = [names[i] if i >= 0 else "" for i in y.regime[PERIL_INDEX["TC"]]]
        return df

    @app.get("/api/analyses/{aid}/ylt", tags=["results"])
    def analysis_ylt(aid: str, limit: int = Query(100, le=1_000_000), format: str = Query("json", pattern="^(json|csv|occurrences)$")):
        r = get_an(aid)
        if format == "occurrences":
            y = r.ylt
            ev = r.events
            df = pd.DataFrame({"year": y.year, "day": y.day.round(1), "peril": np.array(PERILS)[y.peril],
                               "event_uid": ev["event_uid"].to_numpy()[y.event], "gu": r.occ_gu, "gross": r.occ_gross,
                               "net": r.occ_net})
            return Response(df.to_csv(index=False), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{aid}_occurrences.csv"'})
        df = annual_frame(r)
        if format == "csv":
            return Response(df.to_csv(index=False), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{aid}_ylt.csv"'})
        top = df.sort_values("gross", ascending=False).head(limit)
        hist_edges = np.quantile(df["gross"], np.linspace(0, 1, 2))
        return J({"n_years": r.n_years, "top_years": top, "p_zero": float((df["gross"] <= 0).mean()),
                  "histogram": _log_hist(df["gross"].to_numpy()), "range": hist_edges})

    @app.get("/api/analyses/{aid}/locations", tags=["results"])
    def analysis_locations(aid: str, format: str = Query("columns", pattern="^(columns|csv)$")):
        r = get_an(aid)
        df = alloc.location_frame(r)
        if format == "csv":
            return Response(df.to_csv(index=False), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{aid}_locations.csv"'})
        cols = ["loc_id", "acc_id", "lat", "lon", "state", "construction", "occupancy", "lob"]
        out = {c: df[c].tolist() for c in cols}
        for c in ("tiv", "aal", "aal_gu", "cotvar", "loss_cost_pm"):
            out[c] = df[c].round(2).tolist()
        return J(out)

    @app.get("/api/analyses/{aid}/allocation", tags=["results"])
    def analysis_allocation(aid: str, dimension: str = "state", top: int | None = Query(None, le=5000)):
        try:
            return J(alloc.allocate(get_an(aid), dimension, top))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/analyses/{aid}/segments", tags=["results"])
    def analysis_segments(aid: str):
        return J(alloc.segment_ep(get_an(aid)))

    @app.get("/api/analyses/{aid}/insights", tags=["results"])
    def analysis_insights(aid: str):
        return J(get_an(aid).insights)

    @app.get("/api/analyses/{aid}/enso", tags=["what-if"])
    def analysis_enso(aid: str):
        return J(sens.enso_conditional(get_an(aid)))

    @app.post("/api/analyses/{aid}/climate", tags=["what-if"])
    def analysis_climate(aid: str, req: ClimateRequest):
        return J(sens.climate_scenario(get_an(aid), req.tc_frequency, req.tc_intensity, req.eq_frequency))

    @app.post("/api/analyses/{aid}/sensitivity", tags=["what-if"])
    def analysis_sensitivity(aid: str, spread: float = Query(0.2, gt=0, lt=0.9), wait: bool = False):
        r = get_an(aid)

        def work(job: Job):
            job.message = "Re-running kernel with common random numbers"
            return store.cached(("tornado", aid, spread), lambda: sens.tornado(r, spread))

        return job_or_wait(store.submit("sensitivity", work), wait)

    @app.post("/api/analyses/{aid}/mitigation", tags=["what-if"])
    def analysis_mitigation(aid: str, req: MitigationRequest, wait: bool = False):
        r = get_an(aid)
        if req.preset and req.preset not in sens.MITIGATION_PRESETS:
            raise HTTPException(422, f"unknown preset {req.preset}")

        def work(job: Job):
            job.message = "Rebuilding vulnerability and re-running"
            return sens.mitigation(r, req.preset, req.filter, req.changes)

        return job_or_wait(store.submit("mitigation", work), wait)

    @app.post("/api/analyses/{aid}/marginal", tags=["what-if"])
    def analysis_marginal(aid: str, req: MarginalRequest, wait: bool = False):
        r = get_an(aid)
        return job_or_wait(store.submit("marginal", lambda job: sens.marginal_accounts(r, req.acc_ids)), wait)

    @app.post("/api/analyses/{aid}/reinsurance", tags=["reinsurance"])
    def analysis_reinsurance(aid: str, req: ReinsuranceRequest):
        """Evaluate any programme instantly against the stored occurrence losses (no re-simulation)."""
        r = get_an(aid)
        prog = Program.from_dict(req.program.model_dump())
        subject = r.occ_gross - r.occ_pr
        applied = apply_program(prog, r.ylt.year, r.ylt.peril, subject, r.n_years)
        gross_annual = r.annual("gross")
        metrics = layer_metrics(prog, applied, gross_annual, r.n_years)
        net_ann = applied["net_annual"]
        net_max = np.zeros(r.n_years)
        np.maximum.at(net_max, r.ylt.year, applied["net_occ"])
        from ..analytics.ep import ep_curve, ep_table

        return J({"metrics": metrics, "net_aep": ep_table(net_ann, n_boot=0), "net_oep": ep_table(net_max, n_boot=0),
                  "net_aep_curve": ep_curve(net_ann), "gross_aep_curve": r.ep["gross"]["all"]["aep_curve"],
                  "gross_aep": r.ep["gross"]["all"]["aep"]})

    @app.post("/api/analyses/{aid}/reinsurance/apply", tags=["reinsurance"])
    def analysis_reinsurance_apply(aid: str, req: ReinsuranceRequest):
        """Persist a programme on the analysis (recomputes net EP and insights)."""
        r = get_an(aid)
        prog = Program.from_dict(req.program.model_dump())
        subject = r.occ_gross - r.occ_pr
        applied = apply_program(prog, r.ylt.year, r.ylt.peril, subject, r.n_years)
        r.occ_net = applied["net_occ"]
        r.config.reinsurance = prog.to_dict()
        r.reinsurance = {"program": prog.to_dict(),
                         "metrics": layer_metrics(prog, applied, r.annual("gross"), r.n_years),
                         "annual_recoveries": {c["contract"].name: c["annual_recovery"] for c in applied["contracts"]}}
        from ..analytics.insights import generate_insights
        from ..engine.model import build_summary

        r.ep = compute_ep(r)
        r.summary = build_summary(r)
        r.insights = generate_insights(r)
        store.add_analysis(r)
        return J({**r.summary, "insights": r.insights})

    @app.post("/api/analyses/{aid}/reinsurance/optimize", tags=["reinsurance"])
    def analysis_optimize(aid: str, req: OptimizeRequest):
        r = get_an(aid)
        oep = np.sort(r.annual_max("gross"))
        subject = r.occ_gross - r.occ_pr
        return J(optimise_cat_xl(r.ylt.year, r.ylt.peril, subject, r.n_years, lambda rp: quantile_at_rp(oep, rp),
                                 tuple(req.attach_rps), tuple(req.exhaust_rps), req.reinstatements, req.pricing_load,
                                 req.pricing_method))

    # ------------------------------------------------------------------ scenarios
    @app.get("/api/scenarios/analogs", tags=["scenarios"])
    def analogs():
        return J(ANALOGS)

    @app.post("/api/scenarios/run", tags=["scenarios"])
    def scenario_run(req: ScenarioRequest, wait: bool = True):
        pf = get_pf(req.portfolio_id)
        if req.analog:
            if req.analog not in ANALOGS:
                raise HTTPException(422, f"unknown analog {req.analog}")
            a = ANALOGS[req.analog]
            peril, params = a["peril"], {**a["params"], "name": a["label"]}
        else:
            if not req.peril or not req.params:
                raise HTTPException(422, "provide either analog or peril+params")
            peril, params = req.peril, req.params

        def work(job: Job):
            job.message = "Running scenario"
            return run_scenario(pf, peril, params, req.n_samples, req.seed, base_model=store.model)

        return job_or_wait(store.submit("scenario", work), wait)

    # ------------------------------------------------------------------ event development (4-D)
    @app.post("/api/develop", tags=["development"])
    def develop_event(req: DevelopRequest, wait: bool = False):
        """Time-resolved physics of one event realization (wind/surge/rain or rupture/waves) + building damage."""
        from ..physics.develop import develop

        pf = get_pf(req.portfolio_id) if req.portfolio_id else None
        if req.analog and req.analog not in ANALOGS:
            raise HTTPException(422, f"unknown analog {req.analog}")

        def work(job: Job):
            job.message = "Simulating event development (surge model may take ~10 s)"
            key = ("develop", req.portfolio_id, req.analog, req.peril, req.event_id, str(req.params), req.seed, req.surge)
            return store.cached(key, lambda: clean(develop(store.model, pf, req.peril, req.analog, req.params,
                                                           req.event_id, req.seed, req.surge)))

        return job_or_wait(store.submit("develop", work), wait)

    @app.post("/api/develop/seismogram", tags=["development"])
    def develop_seismogram(req: SeismogramRequest):
        """Stochastic finite-fault (EXSIM-style) accelerogram, velocity, FAS and PSA at a site."""
        from ..physics.develop import resolve_event, seismogram

        try:
            peril, cat, _ = resolve_event(store.model, req.peril, req.analog, req.params, req.event_id)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if peril != "EQ":
            raise HTTPException(422, "seismograms are available for earthquake events")
        cat.uncertainty = store.model.catalogs["EQ"].uncertainty
        return J(seismogram(cat, req.lat, req.lon, req.vs30, req.seed))

    @app.get("/api/terrain/{z}/{x}/{y}.png", tags=["development"], include_in_schema=False)
    def terrain_tile(z: int, x: int, y: int):
        from ..physics.dem import terrarium_tile

        if z < 0 or z > 14 or not (0 <= x < 2 ** z and 0 <= y < 2 ** z):
            raise HTTPException(404, "tile out of range")
        return Response(terrarium_tile(z, x, y), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})

    # Google Photorealistic 3D Tiles pass-through + integrations discovery
    from . import tiles3d

    tiles3d.register(app, key=google_maps_key, transport=google_transport)

    # ------------------------------------------------------------------ jobs
    @app.get("/api/jobs", tags=["jobs"])
    def list_jobs():
        return J(sorted([j.public() for j in store.jobs.values()], key=lambda d: -d["created"])[:100])

    @app.get("/api/jobs/{jid}", tags=["jobs"])
    def get_job(jid: str, include_result: bool = True):
        job = store.jobs.get(jid)
        if job is None:
            raise HTTPException(404, f"job {jid} not found")
        out = job.public()
        if include_result and job.status == "done":
            out["result"] = job.result
        return J(out)

    # ------------------------------------------------------------------ demo bootstrap & static UI
    if demo and not store.portfolios:
        def bootstrap(job: Job):
            job.message = "Generating demo portfolio"
            pf = generate_portfolio(demo_locations, seed=11, name=f"Demo US book ({demo_locations:,} locations)")
            store.add_portfolio(pf)

            def progress(f, msg=""):
                job.progress, job.message = f, msg

            cfg = AnalysisConfig(name="Demo analysis (gross + cat XL tower)", n_years=demo_years, reinsurance={
                "contracts": [{"name": "Cat XL 150m xs 100m", "type": "cat_xl", "attachment": 1.0e8, "limit": 1.5e8,
                               "reinstatements": 1},
                              {"name": "Cat XL 250m xs 250m", "type": "cat_xl", "attachment": 2.5e8, "limit": 2.5e8,
                               "reinstatements": 1}]})
            res = store.model.run(pf, cfg, progress=progress)
            store.add_analysis(res)
            job.result_id = res.id
            job.message = "Warming hazard maps"
            for peril, rp in (("TC", 100.0), ("EQ", 475.0)):
                cat = store.model.catalogs[peril]
                store.cached(("hazmap", peril, rp, 0.25), lambda cat=cat, rp=rp: hazard_map(cat, rp, 0.25))
            return analysis_brief(res)

        app.state.bootstrap_job = store.submit("bootstrap", bootstrap)

    dist = Path(os.environ.get("CATFORGE_UI_DIST", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
    if dist.is_dir() and (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")
    else:
        @app.get("/", include_in_schema=False)
        def root():
            return {"name": "CatForge API", "docs": "/docs", "ui": "frontend not built (cd frontend && npm run build)"}

    return app


def _log_hist(x: np.ndarray, bins: int = 40) -> dict:
    pos = x[x > 0]
    if pos.size == 0:
        return {"edges": [], "counts": [], "zero": int(x.size)}
    e = np.geomspace(max(pos.min(), 1.0), pos.max() * 1.0001, bins + 1)
    c, _ = np.histogram(pos, bins=e)
    return {"edges": e.tolist(), "counts": c.tolist(), "zero": int((x <= 0).sum())}

