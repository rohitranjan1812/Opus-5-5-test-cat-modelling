import pytest
from fastapi.testclient import TestClient

from catforge.api import create_app
from catforge.client import CatForgeClient


@pytest.fixture(scope="module")
def client(small_model, tmp_path_factory):
    app = create_app(model=small_model, data_dir=str(tmp_path_factory.mktemp("data")), demo=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ids(client):
    pf = client.post("/api/portfolios/synthetic", json={"n_locations": 300, "seed": 3, "states": ["FL", "CA", "LA"]})
    assert pf.status_code == 201, pf.text
    pid = pf.json()["id"]
    r = client.post("/api/analyses", params={"wait": True}, json={"portfolio_id": pid, "config": {
        "n_years": 2000, "elt_samples": 8, "reinsurance": {"contracts": [
            {"name": "XL", "attachment": 2e6, "limit": 1e7, "reinstatements": 1}]}}})
    assert r.status_code == 200, r.text
    return {"pid": pid, "aid": r.json()["result"]["id"]}


def test_meta_and_hazard(client):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert "WOOD" in meta["construction_classes"] and "andrew_1992" in meta["analogs"]
    cats = client.get("/api/catalogs").json()
    assert {c["peril"] for c in cats} == {"TC", "EQ"}
    ev = client.get("/api/catalogs/TC/events", params={"limit": 5, "sort": "vmax"}).json()
    assert len(ev["rows"]) == 5 and ev["rows"][0]["vmax"] >= ev["rows"][-1]["vmax"]
    eid = ev["rows"][0]["event_id"]
    d = client.get(f"/api/catalogs/TC/events/{eid}").json()
    assert len(d["geometry"]["lat"]) > 10
    fp = client.get(f"/api/catalogs/TC/events/{eid}/footprint", params={"res_deg": 0.1}).json()
    assert fp["ny"] > 0 and max(max(r) for r in fp["values"]) > 30
    hc = client.get("/api/hazard/curve", params={"peril": "EQ", "lat": 34.05, "lon": -118.24}).json()
    assert len(hc["intensity"]) == len(hc["annual_exceedance_prob"])
    vc = client.get("/api/vulnerability/curve", params={"peril": "TC", "construction": "MOBILE_HOME"}).json()
    assert vc["mean"][-1] > 0.5
    assert client.get("/api/catalogs/EQ/stats").json()["mfd"]["rate_ge"][0] > 0
    assert client.get("/api/catalogs/XX/events").status_code == 404


def test_portfolio_roundtrip(client, ids):
    s = client.get(f"/api/portfolios/{ids['pid']}").json()
    assert s["n_locations"] == 300 and s["tiv_total"] > 0
    cols = client.get(f"/api/portfolios/{ids['pid']}/locations").json()
    assert len(cols["lat"]) == 300
    csv = client.get(f"/api/portfolios/{ids['pid']}/locations", params={"format": "csv"}).text
    up = client.post("/api/portfolios/upload", files={"file": ("book.csv", csv, "text/csv")}, data={"name": "rt"})
    assert up.status_code == 201 and up.json()["n_locations"] == 300
    bad = client.post("/api/portfolios/upload", files={"file": ("bad.csv", "a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 422


def test_analysis_results(client, ids):
    aid = ids["aid"]
    s = client.get(f"/api/analyses/{aid}").json()
    assert s["aal"]["gross"] > 0 and s["rp_table"] and s["insights"]
    assert s["reinsurance"]["metrics"]["contracts"][0]["name"] == "XL"
    ep = client.get(f"/api/analyses/{aid}/ep", params={"basis": "net"}).json()
    assert ep["aep"] and ep["aep_curve"]["loss"]
    assert client.get(f"/api/analyses/{aid}/elt", params={"limit": 3}).json()["rows"]
    assert "event_uid" in client.get(f"/api/analyses/{aid}/elt", params={"format": "csv"}).text
    y = client.get(f"/api/analyses/{aid}/ylt").json()
    assert y["n_years"] == 2000 and y["top_years"]
    loc = client.get(f"/api/analyses/{aid}/locations").json()
    assert len(loc["aal"]) == 300
    al = client.get(f"/api/analyses/{aid}/allocation", params={"dimension": "construction"}).json()
    assert abs(sum(r["cotvar_share"] for r in al["rows"]) - 1) < 1e-6
    assert client.get(f"/api/analyses/{aid}/allocation", params={"dimension": "nope"}).status_code == 422
    assert client.get(f"/api/analyses/{aid}/segments").json()["rows"]
    assert len(client.get(f"/api/analyses/{aid}/enso").json()["regimes"]) == 3
    c = client.post(f"/api/analyses/{aid}/climate", json={"tc_frequency": 1.2}).json()
    assert c["scenario"]["aal"] > c["base"]["aal"]
    assert client.get("/api/analyses").json()[0]["id"]


def test_reinsurance_and_whatifs(client, ids):
    aid = ids["aid"]
    prog = {"contracts": [{"name": "QS", "type": "quota_share", "stage": 1, "cession": 0.25},
                          {"name": "XL", "stage": 2, "attachment": 1e6, "limit": 5e6, "reinstatements": 2}]}
    r = client.post(f"/api/analyses/{aid}/reinsurance", json={"program": prog}).json()
    assert r["metrics"]["net"]["aal"] < r["metrics"]["gross"]["aal"]
    opt = client.post(f"/api/analyses/{aid}/reinsurance/optimize", json={}).json()
    assert opt["points"] and any(p["efficient"] for p in opt["points"])
    t = client.post(f"/api/analyses/{aid}/sensitivity", params={"wait": True}).json()["result"]
    assert t["rows"]
    m = client.post(f"/api/analyses/{aid}/mitigation", params={"wait": True}, json={"preset": "urm_retrofit"}).json()
    assert "result" in m
    loc = client.get(f"/api/analyses/{aid}/locations").json()
    acc = loc["acc_id"][0]
    mg = client.post(f"/api/analyses/{aid}/marginal", params={"wait": True}, json={"acc_ids": [acc]}).json()["result"]
    assert mg["n_locations"] >= 1 and mg["marginal"]["aal"] >= -1e-6
    ap = client.post(f"/api/analyses/{aid}/reinsurance/apply", json={"program": prog}).json()
    assert ap["reinsurance"]["program"]["contracts"][0]["name"] == "QS"


def test_scenario_and_jobs(client, ids):
    s = client.post("/api/scenarios/run", json={"portfolio_id": ids["pid"], "analog": "northridge_1994",
                                                "n_samples": 100}).json()["result"]
    assert s["peril"] == "EQ" and "footprint" in s
    custom = client.post("/api/scenarios/run", json={"portfolio_id": ids["pid"], "peril": "TC", "n_samples": 50,
                                                     "params": {"landfall_lat": 27.9, "landfall_lon": -82.8,
                                                                "heading": 60, "vmax": 60, "rmax_km": 30, "vt": 6}})
    assert custom.status_code == 200
    jobs = client.get("/api/jobs").json()
    assert jobs and all(j["status"] in ("done", "error", "running", "queued") for j in jobs)


def test_sdk_client(client, ids):
    cf = CatForgeClient(client=client)
    assert cf.health()["status"] == "ok"
    an = cf.run_analysis(ids["pid"], n_years=500, elt_samples=4, perils=["TC"])
    assert an["aal"]["gross"] >= 0 and an["config"]["perils"] == ["TC"]
    assert cf.allocation(an["id"], "state")["rows"]


def test_event_development_endpoints(client, ids):
    tc = client.post("/api/develop", params={"wait": True}, json={"portfolio_id": ids["pid"], "analog": "andrew_1992",
                                                                  "seed": 2, "surge": False}).json()["result"]
    assert tc["peril"] == "TC" and len(tc["track"]["t"]) > 10 and tc["probes"]
    assert tc["rain"]["max_mm"] > 0 and tc["loss"]["gu"][-1] >= tc["loss"]["gu"][0]
    eq = client.post("/api/develop", params={"wait": True}, json={"portfolio_id": ids["pid"], "analog": "northridge_1994",
                                                                  "seed": 2}).json()["result"]
    assert eq["peril"] == "EQ" and eq["fault"]["nL"] >= 2 and eq["grid"]["t_s"]["b64"]
    sg = client.post("/api/develop/seismogram", json={"analog": "northridge_1994", "lat": 34.2, "lon": -118.4}).json()
    assert sg["pga_g"] > 0 and len(sg["acc_g"]) == len(sg["vel_cms"]) and sg["t_p"] < sg["t_s"]
    assert client.post("/api/develop/seismogram", json={"analog": "andrew_1992", "lat": 25, "lon": -80}).status_code == 422
    tile = client.get("/api/terrain/7/34/53.png")
    assert tile.status_code == 200 and tile.content[:4] == b"\x89PNG"
