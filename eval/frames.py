"""FRAMES (google/frames-benchmark) -> question JSONL with QID golden sets.

FRAMES rows carry a multi-hop question (`Prompt`), a free-text `Answer`, and
`wiki_links` — the Wikipedia articles a human needs to reason over. We map
article titles -> Wikidata QIDs via the sidecar alias table (fallback:
entity.title match) and emit the shared JSONL schema with source="frames".

CAVEAT (documented contract decision): FRAMES gives *evidence articles*, not
a separated answer/bridge split — the linked articles are closer to
bridge+answer combined. We therefore put ALL mapped article entities in
`answer_qids` and leave `bridge_qids` empty. Scores on the frames slice are
comparable across models but not directly against the synthetic slices.

`difficulty` is a proxy: the number of linked wiki articles (FRAMES does not
expose a hop count).

Degrades gracefully:
  * `datasets` missing / download failing -> clear message
  * sidecar missing -> download+parse still works; mapping is lazy (rows are
    parsed and counted, records are emitted only for rows with >=1 mapped
    QID once the sidecar exists)

Usage:
  .venv/bin/python -m eval.frames --out data/questions/frames.jsonl
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import unquote

from ecs import config
from questiongen import sidecar as sidecar_mod
from questiongen.schema import QuestionRecord, append_records

HF_DATASET = "google/frames-benchmark"
DEFAULT_OUT = os.path.join(config.DATA_DIR, "questions", "frames.jsonl")

_URL_RE = re.compile(r"https?://[^\s'\",\]]+")


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------


@dataclass
class FramesRow:
    idx: int
    question: str
    answer_text: str
    wiki_links: list[str] = field(default_factory=list)

    @property
    def titles(self) -> list[str]:
        return [t for t in (title_from_url(u) for u in self.wiki_links) if t]


def parse_wiki_links(raw) -> list[str]:
    """FRAMES stores wiki_links as a stringified python list; be tolerant."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(u).strip() for u in raw if str(u).strip()]
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(u).strip() for u in parsed if str(u).strip()]
    except (ValueError, SyntaxError):
        pass
    return _URL_RE.findall(text)


def title_from_url(url: str) -> str | None:
    """'https://en.wikipedia.org/wiki/Alan_Turing#Legacy' -> 'Alan Turing'."""
    if "/wiki/" not in url:
        return None
    tail = url.split("/wiki/", 1)[1]
    tail = tail.split("#", 1)[0].split("?", 1)[0]
    title = unquote(tail).replace("_", " ").strip()
    return title or None


def _first(d: dict, keys: Iterable[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def row_from_raw(idx: int, raw: dict) -> FramesRow:
    return FramesRow(
        idx=idx,
        question=str(_first(raw, ("Prompt", "prompt", "question"), "")).strip(),
        answer_text=str(_first(raw, ("Answer", "answer"), "")).strip(),
        wiki_links=parse_wiki_links(
            _first(raw, ("wiki_links", "wikipedia_links", "links"))
        ),
    )


def map_titles_to_qids(
    titles: list[str], side: sidecar_mod.SidecarLike
) -> list[str]:
    """Map article titles to QIDs (dedup, order-preserving).

    Ambiguous aliases resolve to the lowest-numbered QID (deterministic;
    documented approximation). Unmapped titles are dropped.
    """
    out: list[str] = []
    seen: set[str] = set()
    for title in titles:
        qids = side.qids_for_title(title)
        if not qids:
            continue
        qid = qids[0]
        if qid not in seen:
            seen.add(qid)
            out.append(qid)
    return out


def row_to_record(
    row: FramesRow, side: sidecar_mod.SidecarLike
) -> QuestionRecord | None:
    """None when the row has no question or no mappable article."""
    if not row.question:
        return None
    qids = map_titles_to_qids(row.titles, side)
    if not qids:
        return None
    return QuestionRecord(
        id=f"frames-{row.idx}",
        question=row.question,
        answer_qids=qids,  # evidence articles: bridge+answer combined (see module docstring)
        bridge_qids=[],
        source="frames",
        cypher=None,
        difficulty=max(1, len(row.wiki_links)),  # proxy, not a true hop count
    )


# ---------------------------------------------------------------------------
# Download (guarded)
# ---------------------------------------------------------------------------


def load_frames_rows(split: str = "test") -> list[FramesRow]:
    """Download+parse FRAMES from Hugging Face. Raises RuntimeError with a
    clear message when `datasets` is missing or the download fails."""
    try:
        from datasets import load_dataset  # lazy import
    except ImportError as exc:
        raise RuntimeError(
            "the `datasets` package is not installed — `pip install datasets` "
            "to fetch FRAMES."
        ) from exc
    try:
        ds = load_dataset(HF_DATASET, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"could not download {HF_DATASET} (split={split!r}): {exc}"
        ) from exc
    return [row_from_raw(i, dict(raw)) for i, raw in enumerate(ds)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    try:
        rows = load_frames_rows(args.split)
    except RuntimeError as exc:
        print(f"SKIP: {exc}")
        return 0
    if args.limit:
        rows = rows[: args.limit]
    n_links = sum(len(r.wiki_links) for r in rows)
    print(f"parsed {len(rows)} FRAMES rows ({n_links} wiki links)")

    side = sidecar_mod.open_sidecar()
    if side is None:
        print(
            f"sidecar not found at {sidecar_mod.DB_PATH} — download+parse OK, "
            "title->QID mapping deferred; rerun after ingest/build_sidecar.py."
        )
        return 0

    records: list[QuestionRecord] = []
    unmapped = 0
    for row in rows:
        rec = row_to_record(row, side)
        if rec is None:
            unmapped += 1
        else:
            records.append(rec)
    print(f"mapped {len(records)} rows to QIDs ({unmapped} dropped: no mappable title)")

    if not records:
        return 1
    if args.dry_run:
        for r in records[:5]:
            print(r.to_jsonl_line())
        print(f"(dry run) {len(records)} records NOT written")
    else:
        append_records(args.out, records)
        print(f"appended {len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
