"""Build data/sidecar.db — sqlite lookup for entity text/aliases and relation labels.

Tables:
  entity(qid TEXT PRIMARY KEY, title TEXT, abstract TEXT)
  alias(qid TEXT, alias TEXT)                               -- indexed on alias
  relation(pid TEXT PRIMARY KEY, label TEXT)                -- label = first alias

Title rule (DESIGN.md "Entity titles"): the Wikidata5M alias file is unordered,
so "first alias" is noise. Title = the LONGEST alias that case-insensitively
prefix-matches the entity's abstract opening (leading whitespace/quotes
stripped, whitespace normalized on both sides); fallback = first alias;
final fallback = QID. Entities with no abstract keep the alias/QID fallback.

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

_LEADING_QUOTES = "\"'“”‘’«»`"


def _norm(s: str) -> str:
    """Whitespace-normalize + casefold for comparison."""
    return " ".join(s.split()).casefold()


def _abstract_opening(abstract: str) -> str:
    """Normalized abstract opening: leading whitespace/quotes stripped."""
    return _norm(abstract.lstrip().lstrip(_LEADING_QUOTES).lstrip())


def prefix_title(aliases: list[str], abstract: str) -> str | None:
    """The longest alias that case-insensitively prefix-matches the abstract
    opening, or None if no alias matches."""
    opening = _abstract_opening(abstract)
    best: str | None = None
    best_len = 0
    for a in aliases:
        a_norm = _norm(a)
        if a_norm and len(a_norm) > best_len and opening.startswith(a_norm):
            best, best_len = a, len(a_norm)
    return best


def derive_title(aliases: list[str] | None, abstract: str | None, qid: str) -> str:
    """Title rule per DESIGN.md: abstract-prefix match > first alias > QID."""
    if not aliases:
        return qid
    if abstract:
        best = prefix_title(aliases, abstract)
        if best is not None:
            return best
    return aliases[0]


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

    # pass 1: alias table + in-memory alias map (needed for title derivation)
    alias_map: dict[str, list[str]] = {}
    alias_rows: list[tuple[str, str]] = []
    n_alias = 0
    for parts in iter_tsv(ENTITY_ALIAS):
        qid, aliases = parts[0], [a for a in parts[1:] if a]
        if not aliases:
            continue
        alias_map.setdefault(qid, aliases)
        alias_rows.extend((qid, a) for a in aliases)
        n_alias += len(aliases)
        if len(alias_rows) >= 500_000:
            con.executemany("INSERT INTO alias VALUES(?,?)", alias_rows)
            alias_rows.clear()
    con.executemany("INSERT INTO alias VALUES(?,?)", alias_rows)
    con.commit()
    print(
        f"alias entities: {len(alias_map):,}  aliases: {n_alias:,}  "
        f"({time.time()-t0:.0f}s)"
    )

    # pass 2: abstracts -> entity rows with derived titles
    n_text = n_prefix = 0
    seen: set[str] = set()
    batch: list[tuple[str, str, str]] = []
    for parts in iter_tsv(TEXT):
        if len(parts) < 2:
            continue
        qid, abstract = parts[0], "\t".join(parts[1:])
        aliases = alias_map.get(qid)
        if aliases and prefix_title(aliases, abstract) is not None:
            n_prefix += 1
        title = derive_title(aliases, abstract, qid)
        seen.add(qid)
        batch.append((qid, title, abstract))
        n_text += 1
        if len(batch) >= 200_000:
            con.executemany("INSERT OR REPLACE INTO entity VALUES(?,?,?)", batch)
            batch.clear()
    con.executemany("INSERT OR REPLACE INTO entity VALUES(?,?,?)", batch)
    con.commit()
    print(
        f"text entries: {n_text:,}  titles via abstract-prefix rule: {n_prefix:,} "
        f"({100*n_prefix/max(n_text,1):.1f}%)  ({time.time()-t0:.0f}s)"
    )

    # pass 3: alias-only entities (no abstract) keep first-alias titles
    rows = [
        (qid, aliases[0], None)
        for qid, aliases in alias_map.items()
        if qid not in seen
    ]
    con.executemany("INSERT OR IGNORE INTO entity VALUES(?,?,?)", rows)
    con.commit()
    print(f"alias-only entities (no abstract): {len(rows):,}")
    del alias_map

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
