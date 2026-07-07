"""FastAPI tool server (DESIGN.md) — four endpoints on :7801.

  POST /vector_search {query, k<=50}            -> [{qid, title, score, snippet}]
  POST /get_entity {qid}                        -> {qid, title, abstract, aliases,
                                                    degree_in, degree_out}
  POST /get_neighbors {qid, relation?, direction, limit<=100, offset}
                                                -> {total, edges: [...]}   (hub-safe)
  POST /find_paths {src_qid, dst_qid, max_hops<=4, limit<=20} -> {paths: [...]}

Plus GET /health for liveness. Run:
  .venv/bin/uvicorn toolserver.app:app --host 0.0.0.0 --port 7801
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecs import config  # noqa: E402

SIDECAR = os.path.join(config.DATA_DIR, "sidecar.db")
_QID_RE = re.compile(r"^Q\d+$")
_PID_RE = re.compile(r"^P\d+$")
SNIPPET_CHARS = 200


# ---------------------------------------------------------------- resources
class Resources:
    driver = None
    qdrant = None
    embedder = None

    def sidecar(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con


res = Resources()


@asynccontextmanager
async def lifespan(_: FastAPI):
    from neo4j import GraphDatabase
    from qdrant_client import QdrantClient

    res.driver = GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH)
    res.qdrant = QdrantClient(url=config.QDRANT_URL, timeout=60)
    # embedder is loaded lazily on the first /vector_search call
    yield
    res.driver.close()
    res.qdrant.close()


app = FastAPI(title="ecs toolserver", lifespan=lifespan)


def _check_qid(qid: str) -> str:
    if not _QID_RE.match(qid or ""):
        raise HTTPException(422, f"invalid qid: {qid!r}")
    return qid


# ---------------------------------------------------------------- schemas
class VectorSearchIn(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=50)


class GetEntityIn(BaseModel):
    qid: str


class GetNeighborsIn(BaseModel):
    qid: str
    relation: str | None = None
    direction: Literal["out", "in", "both"] = "both"
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class FindPathsIn(BaseModel):
    src_qid: str
    dst_qid: str
    max_hops: int = Field(default=3, ge=1, le=4)
    limit: int = Field(default=10, ge=1, le=20)


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    out = {"sidecar": os.path.exists(SIDECAR)}
    try:
        res.driver.verify_connectivity()
        out["neo4j"] = True
    except Exception:  # noqa: BLE001
        out["neo4j"] = False
    try:
        out["qdrant_points"] = res.qdrant.get_collection(
            config.QDRANT_COLLECTION
        ).points_count
        out["qdrant"] = True
    except Exception:  # noqa: BLE001
        out["qdrant"] = False
    out["ok"] = bool(out["sidecar"] and out["neo4j"] and out["qdrant"])
    return out


@app.post("/vector_search")
def vector_search(body: VectorSearchIn) -> list[dict]:
    if res.embedder is None:
        from ecs.embedder import Embedder

        res.embedder = Embedder()
    vec = res.embedder.embed_query(body.query)
    hits = res.qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vec.tolist(),
        limit=body.k,
        with_payload=True,
    ).points
    con = res.sidecar()
    try:
        out = []
        for h in hits:
            qid = h.payload.get("qid", f"Q{h.id}")
            row = con.execute(
                "SELECT abstract FROM entity WHERE qid=?", (qid,)
            ).fetchone()
            abstract = (row["abstract"] if row else None) or ""
            out.append(
                {
                    "qid": qid,
                    "title": h.payload.get("title"),
                    "score": round(float(h.score), 6),
                    "snippet": abstract[:SNIPPET_CHARS],
                }
            )
        return out
    finally:
        con.close()


@app.post("/get_entity")
def get_entity(body: GetEntityIn) -> dict:
    qid = _check_qid(body.qid)
    con = res.sidecar()
    try:
        row = con.execute(
            "SELECT title, abstract FROM entity WHERE qid=?", (qid,)
        ).fetchone()
        aliases = [
            r["alias"]
            for r in con.execute(
                "SELECT alias FROM alias WHERE qid=? LIMIT 50", (qid,)
            )
        ]
    finally:
        con.close()
    with res.driver.session() as s:
        rec = s.run(
            "MATCH (n:Entity {qid:$qid}) "
            "RETURN COUNT { (n)-[]->() } AS dout, COUNT { (n)<-[]-() } AS din",
            qid=qid,
        ).single()
    if row is None and rec is None:
        raise HTTPException(404, f"unknown entity {qid}")
    return {
        "qid": qid,
        "title": row["title"] if row else qid,
        "abstract": row["abstract"] if row else None,
        "aliases": aliases,
        "degree_in": rec["din"] if rec else 0,
        "degree_out": rec["dout"] if rec else 0,
    }


@app.post("/get_neighbors")
def get_neighbors(body: GetNeighborsIn) -> dict:
    qid = _check_qid(body.qid)
    if body.relation is not None and not _PID_RE.match(body.relation):
        raise HTTPException(422, f"invalid relation (want P-id): {body.relation!r}")
    pattern = {
        "out": "(n)-[r]->(m:Entity)",
        "in": "(n)<-[r]-(m:Entity)",
        "both": "(n)-[r]-(m:Entity)",
    }[body.direction]
    where = "WHERE type(r) = $relation " if body.relation else ""
    params = {"qid": qid, "relation": body.relation,
              "offset": body.offset, "limit": body.limit}
    with res.driver.session() as s:
        total = s.run(
            f"MATCH (n:Entity {{qid:$qid}}) MATCH {pattern} {where}"
            "RETURN count(r) AS c",
            **params,
        ).single()["c"]
        edges = s.run(
            f"MATCH (n:Entity {{qid:$qid}}) MATCH {pattern} {where}"
            "RETURN startNode(r).qid AS src, type(r) AS rel, r.label AS rel_label, "
            "endNode(r).qid AS dst, endNode(r).title AS dst_title "
            "ORDER BY rel, dst SKIP $offset LIMIT $limit",
            **params,
        ).data()
    return {"total": total, "edges": edges}


@app.post("/find_paths")
def find_paths(body: FindPathsIn) -> dict:
    src, dst = _check_qid(body.src_qid), _check_qid(body.dst_qid)
    if src == dst:
        raise HTTPException(422, "src_qid and dst_qid must differ")
    # allShortestPaths = hub-safe bidirectional BFS; returns all shortest
    # undirected paths up to max_hops (see toolserver/README.md).
    from neo4j import Query

    query = Query(
        "MATCH (a:Entity {qid:$src}), (b:Entity {qid:$dst}) "
        f"MATCH p = allShortestPaths((a)-[*..{int(body.max_hops)}]-(b)) "
        "RETURN p LIMIT $limit",
        timeout=30,
    )
    paths = []
    with res.driver.session() as s:
        result = s.run(query, src=src, dst=dst, limit=body.limit)
        for rec in result:
            p = rec["p"]
            nodes = [{"qid": n["qid"], "title": n["title"]} for n in p.nodes]
            edges = [
                {
                    "src": r.start_node["qid"],
                    "rel": r.type,
                    "rel_label": r.get("label"),
                    "dst": r.end_node["qid"],
                }
                for r in p.relationships
            ]
            paths.append({"length": len(edges), "nodes": nodes, "edges": edges})
    return {"paths": paths}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7801)
