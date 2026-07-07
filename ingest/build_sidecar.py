"""Build data/sidecar.db — sqlite lookup for entity text/aliases and relation labels.

Tables:
  entity(qid TEXT PRIMARY KEY, title TEXT, abstract TEXT)   -- title = first alias
  alias(qid TEXT, alias TEXT)                               -- indexed on alias
  relation(pid TEXT PRIMARY KEY, label TEXT)                -- label = first alias

Idempotent: builds into data/sidecar.db.part, then atomically renames.

Usage:
  .venv/bin/python ingest/build_sidecar.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecs import config  # noqa: E402

WD5M = os.path.join(config.DATA_DIR, "wikidata5m")
DB_PATH = os.path.join(config.DATA_DIR, "sidecar.db")

ENTITY_ALIAS = os.path.join(WD5M, "wikidata5m_entity.txt")
RELATION_ALIAS = os.path.join(WD5M, "wikidata5m_relation.txt")
TEXT = os.path.join(WD5M, "wikidata5m_text.txt")


def iter_tsv(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line.split("\t")


def main() -> None:
    for p in (ENTITY_ALIAS, RELATION_ALIAS, TEXT):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — run ingest/download.py first")

    part = DB_PATH + ".part"
    if os.path.exists(part):
        os.remove(part)
    con = sqlite3.connect(part)
    con.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE entity(qid TEXT PRIMARY KEY, title TEXT, abstract TEXT);
        CREATE TABLE alias(qid TEXT, alias TEXT);
        CREATE TABLE relation(pid TEXT PRIMARY KEY, label TEXT);
        """
    )

    t0 = time.time()

    # entity aliases -> title (first alias) + alias table
    n_ent = n_alias = 0
    titles: list[tuple[str, str]] = []
    alias_rows: list[tuple[str, str]] = []
    for parts in iter_tsv(ENTITY_ALIAS):
        qid, aliases = parts[0], [a for a in parts[1:] if a]
        if not aliases:
            continue
        titles.append((qid, aliases[0]))
        alias_rows.extend((qid, a) for a in aliases)
        n_ent += 1
        n_alias += len(aliases)
        if len(alias_rows) >= 500_000:
            con.executemany("INSERT OR IGNORE INTO entity VALUES(?,?,NULL)", titles)
            con.executemany("INSERT INTO alias VALUES(?,?)", alias_rows)
            titles.clear()
            alias_rows.clear()
    con.executemany("INSERT OR IGNORE INTO entity VALUES(?,?,NULL)", titles)
    con.executemany("INSERT INTO alias VALUES(?,?)", alias_rows)
    con.commit()
    print(f"entities: {n_ent:,}  aliases: {n_alias:,}  ({time.time()-t0:.0f}s)")

    # abstracts; entities present in text but missing from the alias file are
    # inserted with title = qid so they still get embedded / looked up.
    upsert = (
        "INSERT INTO entity VALUES(?,?,?) "
        "ON CONFLICT(qid) DO UPDATE SET abstract=excluded.abstract"
    )
    n_text = 0
    batch: list[tuple[str, str, str]] = []
    for parts in iter_tsv(TEXT):
        if len(parts) < 2:
            continue
        qid, abstract = parts[0], "\t".join(parts[1:])
        batch.append((qid, qid, abstract))
        n_text += 1
        if len(batch) >= 200_000:
            con.executemany(upsert, batch)
            batch.clear()
    con.executemany(upsert, batch)
    cur = con.execute("SELECT COUNT(*) FROM entity WHERE abstract IS NOT NULL")
    print(f"text entries: {n_text:,}  entities with abstract: {cur.fetchone()[0]:,}")
    con.commit()

    # relations
    rel_rows = []
    for parts in iter_tsv(RELATION_ALIAS):
        pid, aliases = parts[0], [a for a in parts[1:] if a]
        if aliases:
            rel_rows.append((pid, aliases[0]))
    con.executemany("INSERT OR IGNORE INTO relation VALUES(?,?)", rel_rows)
    con.commit()
    print(f"relations: {len(rel_rows):,}")

    print("indexing alias(alias), alias(qid) ...")
    con.execute("CREATE INDEX idx_alias_alias ON alias(alias)")
    con.execute("CREATE INDEX idx_alias_qid ON alias(qid)")
    con.commit()
    con.execute("PRAGMA optimize")
    con.close()
    os.replace(part, DB_PATH)
    print(f"done -> {DB_PATH} ({os.path.getsize(DB_PATH)/1e9:.2f} GB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
