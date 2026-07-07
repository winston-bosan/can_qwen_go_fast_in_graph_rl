"""Read-only adapter for the sqlite sidecar built by ingest/build_sidecar.py.

Schema (owned by workstream A — see ingest/build_sidecar.py):

    entity(qid TEXT PRIMARY KEY, title TEXT, abstract TEXT)
    alias(qid TEXT, alias TEXT)      -- indexed on alias
    relation(pid TEXT PRIMARY KEY, label TEXT)

Degrades gracefully: `open_sidecar()` returns None (with a clear message via
`available()`) when the DB has not been built yet. Pure consumers should
accept any object satisfying `SidecarLike` so tests can use `DictSidecar`.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Protocol

from ecs import config

DB_PATH = os.path.join(config.DATA_DIR, "sidecar.db")


class SidecarLike(Protocol):
    def title(self, qid: str) -> str | None: ...

    def abstract(self, qid: str) -> str | None: ...

    def qids_for_title(self, title: str) -> list[str]: ...

    def relation_label(self, pid: str) -> str | None: ...


def available(path: str = DB_PATH) -> bool:
    return os.path.exists(path)


def open_sidecar(path: str = DB_PATH) -> "SqliteSidecar | None":
    """Open the sidecar if it exists; None otherwise (caller prints skip msg)."""
    if not available(path):
        return None
    return SqliteSidecar(path)


class SqliteSidecar:
    """Thin read-only wrapper over data/sidecar.db."""

    def __init__(self, path: str = DB_PATH):
        # read-only URI so we never contend with the ingest writer
        self._con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def title(self, qid: str) -> str | None:
        row = self._con.execute(
            "SELECT title FROM entity WHERE qid = ?", (qid,)
        ).fetchone()
        return row[0] if row else None

    def abstract(self, qid: str) -> str | None:
        row = self._con.execute(
            "SELECT abstract FROM entity WHERE qid = ?", (qid,)
        ).fetchone()
        return row[0] if row else None

    def qids_for_title(self, title: str) -> list[str]:
        """QIDs whose alias exactly matches `title` (a few case variants).

        Uses the indexed alias table; falls back to the entity.title column.
        Deterministic order: ascending numeric QID.
        """
        variants = {title, title.lower(), title.capitalize()}
        qids: set[str] = set()
        for v in variants:
            for (qid,) in self._con.execute(
                "SELECT DISTINCT qid FROM alias WHERE alias = ?", (v,)
            ):
                qids.add(qid)
        if not qids:
            for v in variants:
                for (qid,) in self._con.execute(
                    "SELECT qid FROM entity WHERE title = ?", (v,)
                ):
                    qids.add(qid)
        return sorted(qids, key=_qid_sort_key)

    def relation_label(self, pid: str) -> str | None:
        row = self._con.execute(
            "SELECT label FROM relation WHERE pid = ?", (pid,)
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._con.close()


class DictSidecar:
    """In-memory SidecarLike for unit tests."""

    def __init__(
        self,
        titles: dict[str, str] | None = None,
        abstracts: dict[str, str] | None = None,
        aliases: dict[str, list[str]] | None = None,  # alias -> [qids]
        relations: dict[str, str] | None = None,
    ):
        self._titles = titles or {}
        self._abstracts = abstracts or {}
        self._aliases = aliases or {}
        self._relations = relations or {}

    def title(self, qid: str) -> str | None:
        return self._titles.get(qid)

    def abstract(self, qid: str) -> str | None:
        return self._abstracts.get(qid)

    def qids_for_title(self, title: str) -> list[str]:
        for v in (title, title.lower(), title.capitalize()):
            if v in self._aliases:
                return sorted(self._aliases[v], key=_qid_sort_key)
        hits = [q for q, t in self._titles.items() if t == title]
        return sorted(hits, key=_qid_sort_key)

    def relation_label(self, pid: str) -> str | None:
        return self._relations.get(pid)


def _qid_sort_key(qid: str) -> tuple[int, str]:
    try:
        return (int(qid[1:]), qid)
    except ValueError:
        return (1 << 62, qid)
