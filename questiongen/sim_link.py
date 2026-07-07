"""SID-style similarity-link question generation (source="sim_link").

Idea: build multihop questions whose FINAL hop is text-only — it exists in
the abstracts/embedding space but is NOT a triple in the graph — forcing a
hybrid vector+graph trajectory at solve time.

Construction (per question):
  1. pick an anchor entity A (random point from the qdrant index);
  2. graph hop: pick one triple (A)-[r]->(B) from neo4j; (r, B) identify A
     in the question without naming it;
  3. text link: among A's semantically-nearest neighbors (qdrant), keep
     entities S such that
        (a) S's title is mentioned in A's abstract (verifiable text link,
            via the sqlite sidecar), and
        (b) there is NO triple between A and S in either direction (checked
            against neo4j) — so the hop cannot be resolved by graph tools;
  4. answers = the surviving S set (capped), bridges = {A, B}, difficulty=2.

The golden set is exact by construction (deterministic given the index +
graph + sidecar state).

Service guarding: requires live qdrant + neo4j + sidecar. `main()` prints a
clear SKIP message when any is missing. The candidate-selection logic
(`find_text_link_candidates`, `build_scaffold`) is pure and unit-tested
against in-memory fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys
from collections.abc import Callable, Sequence

from ecs import config

from . import sidecar as sidecar_mod
from . import verbalize
from .kg_patterns import REL_LABELS, Neo4jUnavailable, neo4j_available, run_cypher
from .schema import QuestionRecord, append_records

DEFAULT_OUT = os.path.join(config.DATA_DIR, "questions", "sim_link.jsonl")
NEIGHBOR_K = 25  # semantic neighbors to consider per anchor
MAX_ANSWERS = 10  # cap the text-linked answer set
MIN_TITLE_LEN = 4  # avoid spurious mentions of very short titles


# ---------------------------------------------------------------------------
# Pure logic (unit-tested with in-memory fixtures)
# ---------------------------------------------------------------------------


def title_mentioned(title: str, abstract: str, min_len: int = MIN_TITLE_LEN) -> bool:
    """Word-boundary, case-insensitive mention of `title` inside `abstract`."""
    if not title or not abstract or len(title) < min_len:
        return False
    pattern = r"(?<!\w)" + re.escape(title.lower()) + r"(?!\w)"
    return re.search(pattern, abstract.lower()) is not None


def find_text_link_candidates(
    anchor_qid: str,
    anchor_abstract: str | None,
    neighbors: Sequence[tuple[str, str]],  # (qid, title), similarity-ordered
    has_edge: Callable[[str, str], bool],  # graph triple in either direction?
    max_answers: int = MAX_ANSWERS,
) -> list[tuple[str, str]]:
    """Neighbors that are text-linked to the anchor but NOT graph-linked.

    Keeps similarity order; dedupes; excludes the anchor itself. This is the
    exact golden-answer set for a sim_link question.
    """
    if not anchor_abstract:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = {anchor_qid}
    for qid, title in neighbors:
        if not qid or qid in seen:
            continue
        seen.add(qid)
        if not title_mentioned(title, anchor_abstract):
            continue
        if has_edge(anchor_qid, qid):
            continue
        out.append((qid, title))
        if len(out) >= max_answers:
            break
    return out


def build_scaffold(
    rel_pid: str,
    rel_label: str,
    b_title: str,
    n_answers: int,
) -> str:
    """Deterministic semantics scaffold (also the LLM verbalizer input).

    Describes the anchor via its graph hop (r -> B) without naming it, then
    asks for the text-linked entities — the final hop is text-only.
    """
    noun = "entity" if n_answers == 1 else "entities"
    return (
        f"Consider the subject whose \"{rel_label}\" ({rel_pid}) is "
        f"\"{b_title}\". Find the {n_answers} {noun} that are discussed in "
        "that subject's Wikipedia abstract (and are semantically closest to "
        "it) but have NO direct relationship with it in the knowledge graph. "
        "Do not report the subject itself."
    )


def record_id(anchor_qid: str, answer_qids: Sequence[str]) -> str:
    key = anchor_qid + "|" + "|".join(sorted(answer_qids))
    return "sim-" + hashlib.sha1(key.encode()).hexdigest()[:12]


def to_record(
    anchor_qid: str,
    bridge_qid: str,
    answers: list[tuple[str, str]],
    question: str,
) -> QuestionRecord:
    answer_qids = [q for q, _ in answers]
    return QuestionRecord(
        id=record_id(anchor_qid, answer_qids),
        question=question,
        answer_qids=answer_qids,
        bridge_qids=[anchor_qid, bridge_qid],
        source="sim_link",
        cypher=None,
        difficulty=2,  # one graph hop + one text-only hop
    )


# ---------------------------------------------------------------------------
# Live-service adapters (guarded; not unit-tested — need qdrant/neo4j/sidecar)
# ---------------------------------------------------------------------------


def qdrant_available() -> bool:
    try:
        _qdrant().get_collection(config.QDRANT_COLLECTION)
        return True
    except Exception:
        return False


_client = None


def _qdrant():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient  # lazy import

        _client = QdrantClient(url=config.QDRANT_URL, timeout=10)
    return _client


def sample_anchor_points(n: int) -> list[dict]:
    """Randomly sample points ({qid, title}) from the qdrant collection."""
    from qdrant_client import models

    client = _qdrant()
    try:
        res = client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=models.SampleQuery(sample=models.Sample.RANDOM),
            limit=n,
            with_payload=True,
        )
        points = res.points
    except Exception:
        # older server without random sampling: fall back to first page
        points, _ = client.scroll(
            collection_name=config.QDRANT_COLLECTION, limit=n, with_payload=True
        )
    return [p.payload for p in points if p.payload and p.payload.get("qid")]


def semantic_neighbors(qid: str, k: int = NEIGHBOR_K) -> list[tuple[str, str]]:
    """Nearest neighbors of an indexed entity, as (qid, title) pairs."""
    from qdrant_client import models

    client = _qdrant()
    hits, _ = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="qid", match=models.MatchValue(value=qid))]
        ),
        limit=1,
        with_vectors=True,
    )
    if not hits or hits[0].vector is None:
        return []
    res = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=hits[0].vector,
        limit=k + 1,  # the anchor itself comes back first
        with_payload=True,
    )
    out: list[tuple[str, str]] = []
    for p in res.points:
        payload = p.payload or {}
        nq, nt = payload.get("qid"), payload.get("title")
        if nq and nq != qid:
            out.append((nq, nt or nq))
    return out


def graph_edge_exists(a: str, b: str) -> bool:
    rows = run_cypher(
        "MATCH (x:Entity {qid: $a})-[r]-(y:Entity {qid: $b}) RETURN count(r) AS n",
        {"a": a, "b": b},
    )
    return bool(rows and rows[0]["n"] > 0)


def graph_hop(anchor_qid: str, rng: random.Random) -> tuple[str, str, str] | None:
    """One outgoing triple (pid, b_qid, b_title) identifying the anchor."""
    rows = run_cypher(
        "MATCH (a:Entity {qid: $qid})-[r]->(b:Entity) "
        "RETURN type(r) AS pid, b.qid AS qid, b.title AS title LIMIT 25",
        {"qid": anchor_qid},
    )
    if not rows:
        return None
    row = rng.choice(rows)
    return row["pid"], row["qid"], row["title"] or row["qid"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def generate_one(
    rng: random.Random,
    side: sidecar_mod.SidecarLike,
    use_llm: bool,
) -> tuple[QuestionRecord | None, str]:
    anchors = sample_anchor_points(1)
    if not anchors:
        return None, "no anchor sampled from qdrant"
    anchor_qid = anchors[0]["qid"]

    hop = graph_hop(anchor_qid, rng)
    if hop is None:
        return None, f"{anchor_qid}: no outgoing graph edge"
    pid, b_qid, b_title = hop

    neighbors = semantic_neighbors(anchor_qid)
    candidates = find_text_link_candidates(
        anchor_qid, side.abstract(anchor_qid), neighbors, graph_edge_exists
    )
    if not candidates:
        return None, f"{anchor_qid}: no text-linked, graph-unlinked neighbors"

    rel_label = side.relation_label(pid) or REL_LABELS.get(pid, pid)
    scaffold = build_scaffold(pid, rel_label, b_title, len(candidates))
    if use_llm:
        relations = [f"{pid} ({rel_label})"]
        question = verbalize.verbalize(scaffold, relations, style="direct")
    else:
        question = scaffold

    record = to_record(anchor_qid, b_qid, candidates, question)
    return record, f"{anchor_qid}: ok ({len(candidates)} answers via {pid})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=None)
    args = ap.parse_args(argv)

    # --- availability gates (this pipeline needs all three live services) ---
    if not qdrant_available():
        print(
            f"SKIP: qdrant collection '{config.QDRANT_COLLECTION}' not reachable "
            f"at {config.QDRANT_URL} — start it with `docker compose up -d "
            "qdrant` and build the index (ingest/) first."
        )
        return 0
    if not neo4j_available():
        print(
            f"SKIP: neo4j not reachable at {config.NEO4J_URI} — start it with "
            "`docker compose up -d neo4j` and load Wikidata5M first."
        )
        return 0
    side = sidecar_mod.open_sidecar()
    if side is None:
        print(
            f"SKIP: sidecar not found at {sidecar_mod.DB_PATH} — build it with "
            "ingest/build_sidecar.py first."
        )
        return 0

    use_llm = verbalize.have_api_key()
    if not use_llm:
        print(
            "WARNING: ANTHROPIC_API_KEY not set — questions will use the "
            "deterministic scaffold text."
        )

    rng = random.Random(args.seed)
    max_attempts = args.max_attempts or max(10 * args.n, 20)
    records: list[QuestionRecord] = []
    attempts = 0
    while len(records) < args.n and attempts < max_attempts:
        attempts += 1
        try:
            record, status = generate_one(rng, side, use_llm)
        except Neo4jUnavailable as exc:
            print(f"SKIP: {exc}")
            return 0
        print(f"[{attempts}] {status}")
        if record is not None:
            records.append(record)

    if not records:
        print(f"no questions produced after {attempts} attempts")
        return 1
    if args.dry_run:
        for r in records:
            print(r.to_jsonl_line())
        print(f"(dry run) {len(records)} records NOT written")
    else:
        append_records(args.out, records)
        print(f"appended {len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
