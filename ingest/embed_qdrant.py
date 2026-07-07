"""Embed entity texts and upsert into the Qdrant `wiki_entities` collection.

- Collection: 1024-dim cosine, vectors stored on disk (DESIGN.md).
- Document text: `title + ". " + abstract`, abstract pre-truncated to
  ~400 tokens (whitespace-word approximation; the model tokenizer truncates
  hard at 512 tokens as a backstop).
- Point id: integer part of the QID (Q42 -> 42) — deterministic, so re-runs
  overwrite instead of duplicating.
- Payload: {qid, title} only; abstract lives in the sqlite sidecar.
- Checkpointed: after each upserted batch the last processed qid is written to
  data/qdrant_embed_progress.json; a killed run resumes where it left off.
  Entities are streamed from the sidecar in qid order (primary-key range scan).

Usage:
  .venv/bin/python ingest/embed_qdrant.py --limit 20000     # smoke test
  .venv/bin/python ingest/embed_qdrant.py                   # full corpus (~1 day on a 3060)
  .venv/bin/python ingest/embed_qdrant.py --restart         # ignore checkpoint, recreate collection
Env: ECS_EMBED_MODEL (fallback model), ECS_EMBED_BATCH (encode batch size).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecs import config  # noqa: E402

SIDECAR = os.path.join(config.DATA_DIR, "sidecar.db")
CHECKPOINT = os.path.join(config.DATA_DIR, "qdrant_embed_progress.json")

ABSTRACT_WORD_LIMIT = 400  # ~400 tokens per DESIGN.md
UPSERT_BATCH = 512


def doc_text(title: str, abstract: str | None) -> str:
    words = (abstract or "").split()
    return f"{title}. {' '.join(words[:ABSTRACT_WORD_LIMIT])}"


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"last_qid": "", "done": 0}


def save_checkpoint(state: dict) -> None:
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT)


def ensure_collection(client, recreate: bool) -> None:
    from qdrant_client import models

    exists = client.collection_exists(config.QDRANT_COLLECTION)
    if exists and recreate:
        client.delete_collection(config.QDRANT_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=config.EMBED_DIM,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
        )
        print(f"created collection {config.QDRANT_COLLECTION}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="stop after N entities (0 = all)")
    ap.add_argument("--restart", action="store_true", help="recreate collection, reset checkpoint")
    ap.add_argument("--batch", type=int, default=UPSERT_BATCH, help="entities per upsert batch")
    args = ap.parse_args()

    from qdrant_client import QdrantClient, models

    from ecs.embedder import Embedder

    client = QdrantClient(url=config.QDRANT_URL, timeout=120)
    ensure_collection(client, recreate=args.restart)
    if args.restart and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    state = load_checkpoint()
    print(f"resuming after qid={state['last_qid']!r} ({state['done']:,} done)")

    emb = Embedder()
    print(f"model={emb.model_name} device={emb.device} dim={emb.dim} batch={emb.batch_size}")
    if emb.dim != config.EMBED_DIM:
        sys.exit(
            f"model dim {emb.dim} != collection dim {config.EMBED_DIM}; "
            "the fallback model cannot write into this collection"
        )
    tokenizer = emb.model.tokenizer

    con = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    (total_entities,) = con.execute("SELECT COUNT(*) FROM entity").fetchone()
    target = args.limit if args.limit else total_entities

    done_this_run = 0
    tokens_this_run = 0
    t_start = time.time()
    last_qid = state["last_qid"]
    while True:
        rows = con.execute(
            "SELECT qid, title, abstract FROM entity WHERE qid > ? "
            "ORDER BY qid LIMIT ?",
            (last_qid, args.batch),
        ).fetchall()
        if not rows:
            break
        texts = [doc_text(t, a) for _, t, a in rows]
        # measured token count (real tokenizer, incl. truncation to max_seq_length)
        enc = tokenizer(texts, truncation=True, max_length=emb.model.max_seq_length)
        n_tokens = sum(len(ids) for ids in enc["input_ids"])
        vecs = emb.embed_docs(texts)
        client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=models.Batch(
                ids=[int(qid[1:]) for qid, _, _ in rows],
                vectors=vecs.tolist(),
                payloads=[{"qid": qid, "title": title} for qid, title, _ in rows],
            ),
            wait=True,
        )
        last_qid = rows[-1][0]
        state = {"last_qid": last_qid, "done": state["done"] + len(rows)}
        save_checkpoint(state)
        done_this_run += len(rows)
        tokens_this_run += n_tokens
        if done_this_run % (args.batch * 8) < args.batch:
            dt = time.time() - t_start
            print(
                f"{state['done']:,}/{total_entities:,} entities | this run: "
                f"{done_this_run:,} in {dt:.0f}s | {done_this_run/dt:.1f} ent/s | "
                f"{tokens_this_run/dt:,.0f} tokens/s",
                flush=True,
            )
        if args.limit and done_this_run >= args.limit:
            break

    dt = time.time() - t_start
    if done_this_run:
        eps = done_this_run / dt
        remaining = total_entities - state["done"]
        print(
            f"\nDONE this run: {done_this_run:,} entities, {tokens_this_run:,} tokens "
            f"in {dt:.0f}s\n  throughput: {eps:.1f} entities/s, "
            f"{tokens_this_run/dt:,.0f} tokens/s\n  full-corpus extrapolation: "
            f"{total_entities/eps/3600:.1f} h total, {remaining/eps/3600:.1f} h remaining"
        )
    info = client.get_collection(config.QDRANT_COLLECTION)
    print(f"collection points: {info.points_count:,}")


if __name__ == "__main__":
    main()
