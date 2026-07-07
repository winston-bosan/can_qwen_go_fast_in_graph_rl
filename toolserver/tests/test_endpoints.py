"""Integration tests against a *live* tool server (and its backing services).

Skips cleanly (with a message) when the server is not reachable or reports
unhealthy backends. Designed to pass against the smoke-test index (only part
of the corpus embedded in Qdrant).

Run:  .venv/bin/pytest toolserver/tests -v
"""

from __future__ import annotations

import os

import httpx
import pytest

BASE = os.environ.get("ECS_TOOLSERVER_URL", "http://localhost:7801")
HARVARD = "Q13371"


def _health() -> dict | None:
    try:
        r = httpx.get(f"{BASE}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


_h = _health()
if _h is None:
    pytest.skip(
        f"toolserver not reachable at {BASE} — start it with scripts/up.sh",
        allow_module_level=True,
    )
if not _h.get("ok"):
    pytest.skip(
        f"toolserver backends unhealthy: {_h} — check docker compose / ingest",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE, timeout=120) as c:
        yield c


def test_vector_search_shape(client):
    r = client.post("/vector_search", json={"query": "Ivy League university in Cambridge, Massachusetts", "k": 5})
    assert r.status_code == 200
    hits = r.json()
    assert 0 < len(hits) <= 5
    for h in hits:
        assert set(h) == {"qid", "title", "score", "snippet"}
        assert h["qid"].startswith("Q")
        assert isinstance(h["score"], float)
        assert len(h["snippet"]) <= 200
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_vector_search_k_bound(client):
    r = client.post("/vector_search", json={"query": "x", "k": 51})
    assert r.status_code == 422


def test_get_entity_harvard(client):
    r = client.post("/get_entity", json={"qid": HARVARD})
    assert r.status_code == 200
    e = r.json()
    assert e["qid"] == HARVARD
    assert "harvard" in e["title"].lower()
    assert e["abstract"] and "university" in e["abstract"].lower()
    assert any("harvard" in a.lower() for a in e["aliases"])
    assert e["degree_in"] > 100  # Harvard is a hub
    assert e["degree_out"] > 0


def test_get_entity_invalid_qid(client):
    assert client.post("/get_entity", json={"qid": "banana"}).status_code == 422
    assert client.post("/get_entity", json={"qid": "Q999999999999"}).status_code == 404


def test_get_neighbors_pagination(client):
    r = client.post(
        "/get_neighbors",
        json={"qid": HARVARD, "direction": "both", "limit": 10, "offset": 0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 100
    assert len(body["edges"]) == 10
    e = body["edges"][0]
    assert set(e) == {"src", "rel", "rel_label", "dst", "dst_title"}
    assert e["rel"].startswith("P")

    r2 = client.post(
        "/get_neighbors",
        json={"qid": HARVARD, "direction": "both", "limit": 10, "offset": 10},
    )
    page2 = r2.json()["edges"]
    assert page2 != body["edges"]


def test_get_neighbors_relation_and_direction_filter(client):
    # people educated at Harvard: incoming P69 edges
    r = client.post(
        "/get_neighbors",
        json={"qid": HARVARD, "relation": "P69", "direction": "in", "limit": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    for e in body["edges"]:
        assert e["rel"] == "P69"
        assert e["dst"] == HARVARD  # incoming edges point at Harvard

    both = client.post(
        "/get_neighbors", json={"qid": HARVARD, "relation": "P69", "direction": "both"}
    ).json()["total"]
    assert both >= body["total"]


def test_find_paths(client):
    # pick a neighbor of Harvard, then ask for paths between them
    nb = client.post(
        "/get_neighbors", json={"qid": HARVARD, "direction": "out", "limit": 1}
    ).json()["edges"][0]
    other = nb["dst"]
    r = client.post(
        "/find_paths",
        json={"src_qid": HARVARD, "dst_qid": other, "max_hops": 2, "limit": 5},
    )
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert paths, f"expected a path {HARVARD} -> {other}"
    p = paths[0]
    assert p["length"] == 1  # direct neighbors
    assert {p["nodes"][0]["qid"], p["nodes"][-1]["qid"]} == {HARVARD, other}
    assert p["edges"][0]["rel"].startswith("P")
    assert len(p["nodes"]) == p["length"] + 1


def test_find_paths_same_entity_rejected(client):
    r = client.post(
        "/find_paths", json={"src_qid": HARVARD, "dst_qid": HARVARD}
    )
    assert r.status_code == 422
