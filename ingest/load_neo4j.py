"""Bulk-load Wikidata5M triples into Neo4j via `neo4j-admin database import full`.

Procedure (see ingest/README.md for details):
  1. Generate CSVs into data/neo4j/import/ (mounted at /import in the container):
       entities.csv  header  qid:ID,title
       triples.csv   header  :START_ID,:TYPE,:END_ID,label
  2. Stop the running neo4j service (bulk import requires the DB offline).
  3. Run the import in a one-off container with --overwrite-destination
     (destroys any existing `neo4j` database), then chown the store files
     back to the neo4j user.
  4. Start the service, wait for bolt, CREATE INDEX on :Entity(qid), verify.

Node set = every qid in the sidecar entity table UNION every qid appearing in
the train triples (title falls back to the qid itself for alias-less entities).

Usage:
  .venv/bin/python ingest/load_neo4j.py            # full pipeline
  .venv/bin/python ingest/load_neo4j.py --csv-only # just (re)generate CSVs
  .venv/bin/python ingest/load_neo4j.py --verify   # counts + spot query only
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecs import config  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WD5M = os.path.join(config.DATA_DIR, "wikidata5m")
TRIPLES = os.path.join(WD5M, "wikidata5m_transductive_train.txt")
SIDECAR = os.path.join(config.DATA_DIR, "sidecar.db")
IMPORT_DIR = os.path.join(config.DATA_DIR, "neo4j", "import")
NODES_CSV = os.path.join(IMPORT_DIR, "entities.csv")
RELS_CSV = os.path.join(IMPORT_DIR, "triples.csv")


def compose(*args: str) -> None:
    cmd = ["docker", "compose", *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def ensure_import_dir_writable() -> None:
    """data/neo4j/import is created root-owned by the container; chown it to
    the current user via a root shell inside the (running or one-off) container."""
    if os.path.isdir(IMPORT_DIR) and os.access(IMPORT_DIR, os.W_OK):
        return
    uid, gid = os.getuid(), os.getgid()
    cmd = f"mkdir -p /import && chown {uid}:{gid} /import"
    try:
        subprocess.run(
            ["docker", "exec", "-u", "root", "ecs-neo4j", "bash", "-c", cmd],
            cwd=REPO_ROOT, check=True,
        )
    except subprocess.CalledProcessError:
        compose("run", "--rm", "--no-deps", "--entrypoint", "bash", "neo4j", "-c", cmd)
    if not os.access(IMPORT_DIR, os.W_OK):
        sys.exit(f"cannot make {IMPORT_DIR} writable")


def generate_csvs() -> None:
    ensure_import_dir_writable()
    os.makedirs(IMPORT_DIR, exist_ok=True)
    con = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    rel_label = dict(con.execute("SELECT pid, label FROM relation"))

    print("writing", RELS_CSV)
    t0 = time.time()
    seen: set[str] = set()
    n_rels = 0
    with open(TRIPLES, encoding="utf-8") as fin, open(
        RELS_CSV + ".part", "w", newline="", encoding="utf-8"
    ) as fout:
        w = csv.writer(fout)
        w.writerow([":START_ID", ":TYPE", ":END_ID", "label"])
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            src, pid, dst = parts
            w.writerow([src, pid, dst, rel_label.get(pid, pid)])
            seen.add(src)
            seen.add(dst)
            n_rels += 1
    os.replace(RELS_CSV + ".part", RELS_CSV)
    print(f"  {n_rels:,} triples, {len(seen):,} distinct qids ({time.time()-t0:.0f}s)")

    print("writing", NODES_CSV)
    n_nodes = 0
    with open(NODES_CSV + ".part", "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["qid:ID", "title"])
        for qid, title in con.execute("SELECT qid, title FROM entity"):
            w.writerow([qid, title if title else qid])
            seen.discard(qid)
            n_nodes += 1
        for qid in sorted(seen):  # triple qids missing from sidecar
            w.writerow([qid, qid])
            n_nodes += 1
    os.replace(NODES_CSV + ".part", NODES_CSV)
    con.close()
    print(f"  {n_nodes:,} nodes ({len(seen):,} without sidecar entry)")


def run_import() -> None:
    compose("stop", "neo4j")
    # One-off root container: import (destroying any existing db), then hand
    # the store files back to the neo4j user the server runs as.
    compose(
        "run", "--rm", "--no-deps", "--entrypoint", "bash", "neo4j", "-c",
        "neo4j-admin database import full neo4j"
        " --nodes=Entity=/import/entities.csv"
        " --relationships=/import/triples.csv"
        " --overwrite-destination --verbose"
        " && chown -R neo4j:neo4j /data",
    )
    compose("start", "neo4j")


def wait_for_bolt(timeout: float = 300.0):
    from neo4j import GraphDatabase

    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            driver = GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH)
            driver.verify_connectivity()
            return driver
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3)
    raise RuntimeError(f"neo4j did not come up: {last_err}")


def create_index_and_verify() -> None:
    driver = wait_for_bolt()
    with driver.session() as s:
        s.run(
            "CREATE INDEX entity_qid IF NOT EXISTS FOR (n:Entity) ON (n.qid)"
        ).consume()
        s.run("CALL db.awaitIndexes(600)").consume()
        nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"nodes: {nodes:,}  relationships: {rels:,}")
        rec = s.run(
            "MATCH (h:Entity {qid:'Q13371'}) RETURN h.title AS title"
        ).single()
        print("Q13371 title:", rec["title"] if rec else "NOT FOUND")
        rows = s.run(
            "MATCH (h:Entity {qid:'Q13371'})-[r]-(n:Entity) "
            "RETURN type(r) AS rel, r.label AS label, n.qid AS qid, "
            "n.title AS title LIMIT 10"
        ).data()
        for row in rows:
            print(f"  {row['rel']} ({row['label']}) -- {row['qid']} {row['title']}")
        deg = s.run(
            "MATCH (h:Entity {qid:'Q13371'}) "
            "RETURN COUNT { (h)-[]->() } AS dout, COUNT { (h)<-[]-() } AS din"
        ).single()
        print(f"Q13371 degree out={deg['dout']:,} in={deg['din']:,}")
    driver.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--verify", action="store_true", help="skip csv+import")
    args = ap.parse_args()
    if not args.verify:
        if not os.path.exists(SIDECAR):
            sys.exit("sidecar.db missing — run ingest/build_sidecar.py first")
        if not os.path.exists(TRIPLES):
            sys.exit("triples missing — run ingest/download.py first")
        generate_csvs()
        if args.csv_only:
            return
        run_import()
    create_index_and_verify()


if __name__ == "__main__":
    main()
