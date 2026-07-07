"""Print stack status: services up?, neo4j triple/entity counts, qdrant points,
sidecar row counts, toolserver health.

Usage: .venv/bin/python scripts/status.py
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecs import config  # noqa: E402


def main() -> None:
    # neo4j
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH)
        driver.verify_connectivity()
        with driver.session() as s:
            ents = s.run(
                "MATCH (n:Entity) RETURN count(n) AS c"
            ).single()["c"]
            trips = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        driver.close()
        print(f"neo4j     UP    entities={ents:,} triples={trips:,}")
    except Exception as e:  # noqa: BLE001
        print(f"neo4j     DOWN  ({type(e).__name__}: {e})")

    # qdrant
    try:
        import httpx

        r = httpx.get(
            f"{config.QDRANT_URL}/collections/{config.QDRANT_COLLECTION}", timeout=10
        )
        if r.status_code == 200:
            pts = r.json()["result"]["points_count"]
            print(f"qdrant    UP    {config.QDRANT_COLLECTION} points={pts:,}")
        else:
            httpx.get(f"{config.QDRANT_URL}/collections", timeout=10).raise_for_status()
            print(f"qdrant    UP    (collection {config.QDRANT_COLLECTION} missing)")
    except Exception as e:  # noqa: BLE001
        print(f"qdrant    DOWN  ({type(e).__name__}: {e})")

    # sidecar
    db = os.path.join(config.DATA_DIR, "sidecar.db")
    if os.path.exists(db):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        ent = con.execute("SELECT COUNT(*) FROM entity").fetchone()[0]
        ali = con.execute("SELECT COUNT(*) FROM alias").fetchone()[0]
        rel = con.execute("SELECT COUNT(*) FROM relation").fetchone()[0]
        con.close()
        print(f"sidecar   OK    entities={ent:,} aliases={ali:,} relations={rel:,}")
    else:
        print(f"sidecar   MISSING ({db})")

    # toolserver
    try:
        import httpx

        h = httpx.get(f"{config.TOOLSERVER_URL}/health", timeout=10).json()
        state = "UP" if h.get("ok") else "DEGRADED"
        print(f"toolserver {state}  {h}")
    except Exception as e:  # noqa: BLE001
        print(f"toolserver DOWN ({type(e).__name__})")


if __name__ == "__main__":
    main()
