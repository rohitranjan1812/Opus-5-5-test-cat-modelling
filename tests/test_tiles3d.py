import httpx
import pytest
from fastapi.testclient import TestClient

from catforge.api import create_app

SECRET = "test-key-123"


@pytest.fixture(scope="module")
def seen():
    return []


@pytest.fixture(scope="module")
def client(small_model, tmp_path_factory, seen):
    def upstream(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/v1/3dtiles/root.json":
            return httpx.Response(200, json={"root": {"children": [{"content": {"uri": "/v1/3dtiles/datasets/CgA/files/a.json?session=S1"}}]}},
                                  headers={"cache-control": "private, max-age=3600"})
        return httpx.Response(200, content=b"glTF-bytes", headers={"content-type": "model/gltf-binary"})

    app = create_app(model=small_model, data_dir=str(tmp_path_factory.mktemp("d")), demo=False,
                     google_maps_key=SECRET, google_transport=httpx.MockTransport(upstream))
    with TestClient(app) as c:
        yield c


def test_integrations_never_expose_the_key(client):
    r = client.get("/api/integrations")
    assert r.status_code == 200 and SECRET not in r.text
    g = r.json()["google_3d_tiles"]
    assert g["available"] and g["mode"] == "server-proxy" and g["root"] == "/v1/3dtiles/root.json"


def test_proxy_injects_key_and_forwards_session(client, seen):
    r = client.get("/v1/3dtiles/root.json")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, max-age=3600"
    assert seen[-1].headers["x-goog-api-key"] == SECRET
    r = client.get("/v1/3dtiles/datasets/CgA/files/a.json", params={"session": "S1", "key": "attacker"})
    assert r.status_code == 200 and r.content == b"glTF-bytes" and r.headers["content-type"] == "model/gltf-binary"
    assert seen[-1].url.params.get("session") == "S1" and "key" not in seen[-1].url.params
    assert SECRET not in str(seen[-1].url)


@pytest.mark.parametrize("path", ["../maps/api/geocode/json", "datasets/../../v1/other", "datasets/a//b", "elevation.json"])
def test_proxy_rejects_paths_outside_the_tileset(client, path, seen):
    n = len(seen)
    assert client.get(f"/v1/3dtiles/{path}").status_code == 404
    assert len(seen) == n  # nothing reached upstream


def test_proxy_without_key(small_model, tmp_path):
    app = create_app(model=small_model, data_dir=str(tmp_path), demo=False, google_maps_key="")
    with TestClient(app) as c:
        assert c.get("/api/integrations").json()["google_3d_tiles"]["available"] is False
        assert c.get("/v1/3dtiles/root.json").status_code == 503
