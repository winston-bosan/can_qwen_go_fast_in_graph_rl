"""Parity + throughput benchmark: llama.cpp (q8_0 GGUF, /v1/embeddings) vs
sentence-transformers fp16 for microsoft/harrier-oss-v1-0.6b.

Subcommands:
  parity     cosine(ST, llama.cpp) over N real doc texts + instruction-prefixed
             queries. Acceptance: mean doc cosine >= 0.98.
  retrieval  embed the first N entities (must already be in the main qdrant
             collection with ST vectors) into two temp collections (ST vectors
             copied from the main collection; llama.cpp vectors freshly
             embedded) and compare top-10 results for realistic queries.
             Acceptance: mean overlap >= 8/10. Temp collections are dropped
             afterwards unless --keep.
  bench      throughput (ent/s, tok/s) via concurrent /v1/embeddings requests.

The llama.cpp server must be running with:
  --embeddings --pooling last --embd-normalize 2
(tokenizer parity was verified: llama.cpp add_special=true appends the same
trailing <|endoftext|> that the HF/ST tokenizer does; no BOS on either side).

Usage:
  .venv/bin/python ingest/bench_llamacpp.py parity   [--n-docs 256]
  .venv/bin/python ingest/bench_llamacpp.py retrieval [--n-entities 20480] [--keep]
  .venv/bin/python ingest/bench_llamacpp.py bench    [--n-docs 4096] [--workers 8] [--req-batch 64]
Env: ECS_LLAMACPP_URL (default http://localhost:7802)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecs import config  # noqa: E402

LLAMACPP_URL = os.environ.get("ECS_LLAMACPP_URL", "http://localhost:7802")
SIDECAR = os.path.join(config.DATA_DIR, "sidecar.db")

QUERIES = [
    "Which Ivy League university is located in Cambridge, Massachusetts?",
    "Who developed the theory of general relativity?",
    "What is the capital city of France?",
    "Which physicist formulated the equations of classical electromagnetism?",
    "What company designs the iPhone?",
    "Who wrote The Hitchhiker's Guide to the Galaxy?",
    "Which planet is third from the Sun?",
    "Who was the first female Nobel laureate?",
    "What is the largest search engine company?",
    "Which US president was born in Hawaii?",
    "What species are modern humans?",
    "Which river flows through Paris?",
    "Who painted the Mona Lisa?",
    "What is the longest river in South America?",
    "Which country hosted the 2008 Summer Olympics?",
    "Who founded Microsoft together with Paul Allen?",
    "What mountain is the highest on Earth?",
    "Which element has the chemical symbol Au?",
    "Who directed the film Jaws?",
    "What organization awards the Fields Medal?",
]


def doc_text(title: str, abstract: str | None) -> str:
    words = (abstract or "").split()
    return f"{title}. {' '.join(words[:400])}"


def load_docs(n: int, offset: int = 0) -> tuple[list[str], list[str]]:
    con = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT qid, title, abstract FROM entity ORDER BY qid LIMIT ? OFFSET ?",
        (n, offset),
    ).fetchall()
    con.close()
    return [q for q, _, _ in rows], [doc_text(t, a) for _, t, a in rows]


def query_prefix(q: str) -> str:
    return f"Instruct: {config.QUERY_INSTRUCTION}\nQuery: {q}"


def lcpp_embed(
    texts: list[str], workers: int = 8, req_batch: int = 64
) -> np.ndarray:
    """Embed via llama.cpp /v1/embeddings with concurrent batched requests."""
    chunks = [texts[i : i + req_batch] for i in range(0, len(texts), req_batch)]

    def one(chunk: list[str]) -> np.ndarray:
        with httpx.Client(timeout=600) as client:
            r = client.post(f"{LLAMACPP_URL}/v1/embeddings", json={"input": chunk})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            return np.array([d["embedding"] for d in data], dtype=np.float32)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(one, chunks))
    return np.vstack(parts)


def st_embedder():
    from ecs.embedder import Embedder

    return Embedder()


def count_tokens(texts: list[str]) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    enc = tok(texts, truncation=True, max_length=512)
    return sum(len(x) for x in enc["input_ids"])


def cmd_parity(args) -> None:
    _, docs = load_docs(args.n_docs)
    queries = [query_prefix(q) for q in QUERIES[:16]]
    emb = st_embedder()
    st_docs = emb.embed_docs(docs)
    st_q = emb.embed_docs(queries)  # prefix already applied; embed raw
    lc_docs = lcpp_embed(docs)
    lc_q = lcpp_embed(queries)
    d_cos = np.sum(st_docs * lc_docs, axis=1)
    q_cos = np.sum(st_q * lc_q, axis=1)
    print(f"docs   n={len(docs)}  mean={d_cos.mean():.5f}  min={d_cos.min():.5f}")
    print(f"queries n={len(queries)}  mean={q_cos.mean():.5f}  min={q_cos.min():.5f}")
    ok = d_cos.mean() >= 0.98
    print("PARITY", "PASS (mean doc cosine >= 0.98)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_retrieval(args) -> None:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=config.QDRANT_URL, timeout=120)
    qids, docs = load_docs(args.n_entities)
    ids = [int(q[1:]) for q in qids]

    # ST vectors: copy from the main collection (already embedded by the run)
    st_vecs: dict[int, list[float]] = {}
    for i in range(0, len(ids), 1024):
        recs = client.retrieve(
            config.QDRANT_COLLECTION, ids=ids[i : i + 1024], with_vectors=True
        )
        for rec in recs:
            st_vecs[rec.id] = rec.vector
    missing = [i for i in ids if i not in st_vecs]
    if missing:
        sys.exit(
            f"{len(missing)} of {len(ids)} entities not in {config.QDRANT_COLLECTION} "
            "— lower --n-entities to the embedded prefix"
        )

    print(f"embedding {len(docs)} docs via llama.cpp ...")
    lc_vecs = lcpp_embed(docs, workers=args.workers, req_batch=args.req_batch)

    tmp_st, tmp_lc = "bench_st_test", "bench_lcpp_test"
    for name, dim in ((tmp_st, len(next(iter(st_vecs.values())))), (tmp_lc, lc_vecs.shape[1])):
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )
    for i in range(0, len(ids), 1024):
        chunk = ids[i : i + 1024]
        client.upsert(tmp_st, points=models.Batch(
            ids=chunk, vectors=[st_vecs[j] for j in chunk]), wait=True)
        client.upsert(tmp_lc, points=models.Batch(
            ids=chunk, vectors=lc_vecs[i : i + 1024].tolist()), wait=True)

    emb = st_embedder()
    overlaps = []
    for q in QUERIES:
        st_qv = emb.embed_query(q).tolist()
        lc_qv = lcpp_embed([query_prefix(q)])[0].tolist()
        top_st = [p.id for p in client.query_points(tmp_st, query=st_qv, limit=10).points]
        top_lc = [p.id for p in client.query_points(tmp_lc, query=lc_qv, limit=10).points]
        ov = len(set(top_st) & set(top_lc))
        overlaps.append(ov)
        print(f"  overlap {ov:2d}/10  {q[:60]}")
    mean_ov = float(np.mean(overlaps))
    print(f"mean top-10 overlap: {mean_ov:.2f}/10  min: {min(overlaps)}/10")
    if not args.keep:
        client.delete_collection(tmp_st)
        client.delete_collection(tmp_lc)
    ok = mean_ov >= 8.0
    print("RETRIEVAL", "PASS (mean overlap >= 8/10)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_bench(args) -> None:
    _, docs = load_docs(args.n_docs, offset=args.offset)
    n_tokens = count_tokens(docs)
    lcpp_embed(docs[: args.req_batch * 2], workers=2, req_batch=args.req_batch)  # warmup
    t0 = time.time()
    lcpp_embed(docs, workers=args.workers, req_batch=args.req_batch)
    dt = time.time() - t0
    eps = len(docs) / dt
    print(
        f"llama.cpp: {len(docs)} docs, {n_tokens:,} tokens in {dt:.1f}s -> "
        f"{eps:.1f} ent/s, {n_tokens/dt:,.0f} tokens/s "
        f"(workers={args.workers}, req_batch={args.req_batch})"
    )
    print(f"full corpus (4,944,834): {4944834/eps/3600:.1f} h")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--n-docs", type=int, default=256)
    p.set_defaults(fn=cmd_parity)
    p = sub.add_parser("retrieval")
    p.add_argument("--n-entities", type=int, default=20480)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--req-batch", type=int, default=64)
    p.add_argument("--keep", action="store_true")
    p.set_defaults(fn=cmd_retrieval)
    p = sub.add_parser("bench")
    p.add_argument("--n-docs", type=int, default=4096)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--req-batch", type=int, default=64)
    p.set_defaults(fn=cmd_bench)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
