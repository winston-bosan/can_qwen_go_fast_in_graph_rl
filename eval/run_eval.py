"""Evaluation harness: questions JSONL + pluggable policy -> report JSON.

Policies implement `predict(record) -> list[str]` (QIDs ordered by relevance).
Shipped baselines:
  * RandomBaseline     — k random QIDs from a pool (default: the union of all
                         golden QIDs in the eval file — a "gold-pool" floor)
  * VectorOnlyBaseline — the tool server's /vector_search on the raw question,
                         top-k QIDs (guarded: skips when the server is down)

Metrics come from eval/metrics.py (the reward contract): mean two-tier NDCG,
recall@50, set-F1 — overall and per source.

Usage:
  .venv/bin/python -m eval.run_eval --questions data/questions/kg_pattern.jsonl \\
      --policy random --out data/reports/random.json
  .venv/bin/python -m eval.run_eval --questions ... --policy vector
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Protocol

from ecs import config
from questiongen.schema import QuestionRecord, load_records

from .metrics import f1_set, ndcg_two_tier, recall_at_k

DEFAULT_K = 50


class Policy(Protocol):
    name: str

    def predict(self, record: QuestionRecord) -> list[str]: ...


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class RandomBaseline:
    """Ranks k random QIDs drawn from a pool.

    Without an explicit pool, uses the union of golden QIDs across the eval
    file — a floor that is (a) reproducible and (b) harder than random noise,
    since every guess is at least a plausible entity.
    """

    name = "random"

    def __init__(self, pool: list[str] | None = None, k: int = DEFAULT_K, seed: int = 0):
        self.pool = list(pool) if pool else []
        self.k = k
        self.rng = random.Random(seed)

    @classmethod
    def from_records(cls, records: list[QuestionRecord], k: int = DEFAULT_K, seed: int = 0):
        pool = sorted({q for r in records for q in (*r.answer_qids, *r.bridge_qids)})
        return cls(pool=pool, k=k, seed=seed)

    def predict(self, record: QuestionRecord) -> list[str]:
        if not self.pool:
            return []
        n = min(self.k, len(self.pool))
        return self.rng.sample(self.pool, n)


class VectorOnlyBaseline:
    """Tool server's /vector_search on the raw question text, top-k QIDs."""

    name = "vector_only"

    def __init__(self, k: int = DEFAULT_K, base_url: str | None = None, timeout: float = 30.0):
        self.k = min(k, 50)  # endpoint contract: k <= 50
        self.base_url = (base_url or config.TOOLSERVER_URL).rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        import httpx

        try:
            r = httpx.get(self.base_url + "/openapi.json", timeout=3.0)
            return r.status_code < 500
        except Exception:
            return False

    def predict(self, record: QuestionRecord) -> list[str]:
        import httpx

        resp = httpx.post(
            self.base_url + "/vector_search",
            json={"query": record.question, "k": self.k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        hits = resp.json()
        if isinstance(hits, dict):  # tolerate {"results": [...]} wrapping
            hits = hits.get("results", [])
        return [h["qid"] for h in hits if h.get("qid")]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def evaluate(
    records: list[QuestionRecord], policy: Policy, k: int = DEFAULT_K
) -> dict:
    """Run a policy over the records; return the report dict."""
    per_question: list[dict] = []
    by_source: dict[str, list[dict]] = defaultdict(list)
    t0 = time.time()
    for r in records:
        predicted = policy.predict(r)
        answer, bridge = set(r.answer_qids), set(r.bridge_qids)
        row = {
            "id": r.id,
            "source": r.source,
            "difficulty": r.difficulty,
            "n_predicted": len(predicted),
            "ndcg": ndcg_two_tier(predicted, answer, bridge, k=k),
            f"recall@{k}": recall_at_k(predicted, answer, k=k),
            "f1": f1_set(predicted, answer),
        }
        per_question.append(row)
        by_source[r.source].append(row)

    def agg(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            "mean_ndcg": sum(x["ndcg"] for x in rows) / n if n else 0.0,
            f"mean_recall@{k}": sum(x[f"recall@{k}"] for x in rows) / n if n else 0.0,
            "mean_f1": sum(x["f1"] for x in rows) / n if n else 0.0,
        }

    return {
        "policy": policy.name,
        "k": k,
        "seconds": round(time.time() - t0, 3),
        **agg(per_question),
        "per_source": {src: agg(rows) for src, rows in sorted(by_source.items())},
        "per_question": per_question,
    }


def write_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def summarize(report: dict) -> str:
    k = report["k"]
    return (
        f"{report['policy']}: n={report['n']}  "
        f"NDCG={report['mean_ndcg']:.4f}  "
        f"recall@{k}={report[f'mean_recall@{k}']:.4f}  "
        f"F1={report['mean_f1']:.4f}"
    )


def build_policy(name: str, records: list[QuestionRecord], k: int, seed: int) -> Policy | None:
    """Returns None (after printing a skip message) when a policy's service is down."""
    if name == "random":
        return RandomBaseline.from_records(records, k=k, seed=seed)
    if name == "vector":
        policy = VectorOnlyBaseline(k=k)
        if not policy.available():
            print(
                f"SKIP: tool server not reachable at {policy.base_url} — start "
                "it (toolserver/app.py) before running the vector baseline."
            )
            return None
        return policy
    if name == "claude":
        from .claude_baseline import ClaudeBaseline  # lazy: needs anthropic + toolserver

        policy = ClaudeBaseline(k=k)
        reason = policy.unavailable_reason()
        if reason:
            print(f"SKIP: {reason}")
            return None
        return policy
    raise ValueError(f"unknown policy {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", required=True, help="questions JSONL path")
    ap.add_argument("--policy", default="random", choices=["random", "vector", "claude"])
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="report JSON path (default: data/reports/<policy>.json)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.questions):
        print(f"SKIP: questions file not found: {args.questions}")
        return 0
    records = load_records(args.questions)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"SKIP: no records in {args.questions}")
        return 0

    policy = build_policy(args.policy, records, args.k, args.seed)
    if policy is None:
        return 0

    report = evaluate(records, policy, k=args.k)
    out = args.out or os.path.join(config.DATA_DIR, "reports", f"{policy.name}.json")
    write_report(report, out)
    print(summarize(report))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
