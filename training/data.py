"""Dataset pipeline: data/questions/*.jsonl -> verl RLHFDataset parquet.

Input records (questiongen contract, DESIGN.md):
    {id, question, answer_qids: [...], bridge_qids: [...], source, cypher?, difficulty}

- Deterministic train/val split by sha256(id) -- stable across regeneration
  and across question files, no leakage when questiongen appends new data.
- Curriculum option: "off" (hash-shuffled), "sorted" (easy -> hard),
  "mixed" (difficulty-stratified round-robin so every batch sees all tiers).
- Emits the fields verl's RLHFDataset + sglang multi-turn rollout expect:
  prompt (chat messages), data_source, reward_model.ground_truth,
  extra_info.tools_kwargs / need_tools_kwargs.

Usage:
    python -m training.data --questions-dir data/questions --out-dir data/rl \
        --val-frac 0.05 --curriculum off
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import random
from typing import Any, Iterable

import training  # noqa: F401  (sys.path setup)
from training.env.tools import TOOL_NAMES

logger = logging.getLogger(__name__)

# Kept deliberately short: it is re-tokenized into every rollout turn.
# The tool JSON schemas themselves are injected by the chat template
# (sglang/Qwen3 hermes format), not duplicated here.
SYSTEM_PROMPT = """\
You are an entity-retrieval agent working over a Wikipedia/Wikidata knowledge base.
Given a question, find the Wikidata entities needed to answer it using the available
tools (vector_search, get_entity, get_neighbors, find_paths). Search iteratively:
start broad with vector_search, then use the graph tools to verify and expand.
Entity IDs are QIDs like Q42. You have a limited number of tool calls, so be efficient.

When you are done, output ONLY your final answer as a fenced block listing QIDs,
one per line, ordered by relevance (most relevant first), at most 50 lines:

```entities
Q123  # optional short comment
Q456
```

Include every entity needed to answer the question (tier-1) and the key evidence /
bridge entities you used to find them (tier-2), ranked with tier-1 first."""

_DIFFICULTY_ORDER = {"trivial": 0, "easy": 1, "medium": 2, "hard": 3, "very_hard": 4}


def _difficulty_rank(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(_DIFFICULTY_ORDER.get(value.strip().lower(), 2))
    return 2.0  # unknown -> middle of the pack


def _split_of(record_id: str, val_frac: float) -> str:
    """Stable split: sha256 of the id, uniform in [0, 1)."""
    h = int.from_bytes(hashlib.sha256(str(record_id).encode()).digest()[:8], "big")
    return "val" if (h / 2**64) < val_frac else "train"


def load_questions(questions_dir: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(questions_dir, "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no *.jsonl question files under {questions_dir}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    skipped = 0
    for path in paths:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    logger.warning("bad JSON at %s:%d", path, lineno)
                    continue
                if not rec.get("id") or not rec.get("question") or not rec.get("answer_qids"):
                    skipped += 1
                    continue
                if rec["id"] in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(rec["id"])
                records.append(rec)
    logger.info("loaded %d questions from %d files (%d skipped)", len(records), len(paths), skipped)
    return records


def to_rl_row(rec: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    """One question -> one verl RLHFDataset row."""
    ground_truth = json.dumps(
        {
            "answer_qids": list(rec.get("answer_qids") or []),
            "bridge_qids": list(rec.get("bridge_qids") or []),
        }
    )
    return {
        "data_source": f"ecs/{rec.get('source', 'unknown')}",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rec["question"]},
        ],
        "ability": "entity-retrieval",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "split": split,
            "id": rec["id"],
            "difficulty": str(rec.get("difficulty", "")),
            # verl sglang multi-turn: which tools this sample may use and the
            # per-instance create kwargs. Our tools are stateless; the question
            # id doubles as a non-empty struct child (Arrow cannot serialize
            # empty structs to parquet) and as a tool-side log correlation key.
            "need_tools_kwargs": True,
            "tools_kwargs": {
                name: {"create_kwargs": {"question_id": rec["id"]}} for name in TOOL_NAMES
            },
        },
    }


def apply_curriculum(
    records: list[dict[str, Any]], curriculum: str, seed: int = 0
) -> list[dict[str, Any]]:
    """Order the *training* records.

    off    -- deterministic shuffle (trainer may reshuffle anyway).
    sorted -- easy -> hard; pair with verl's sequential (non-shuffling) sampler,
              otherwise ordering is a no-op (see configs: data.shuffle=False).
    mixed  -- stratified round-robin over difficulty tiers: every contiguous
              batch window contains all difficulty levels.
    """
    rng = random.Random(seed)
    if curriculum == "off":
        out = records[:]
        rng.shuffle(out)
        return out
    if curriculum == "sorted":
        out = records[:]
        rng.shuffle(out)  # stable tie-breaking inside a tier
        out.sort(key=lambda r: _difficulty_rank(r.get("difficulty")))
        return out
    if curriculum == "mixed":
        tiers: dict[float, list[dict[str, Any]]] = {}
        for r in records:
            tiers.setdefault(_difficulty_rank(r.get("difficulty")), []).append(r)
        for tier in tiers.values():
            rng.shuffle(tier)
        queues = [tiers[k] for k in sorted(tiers)]
        out = []
        while any(queues):
            for q in queues:
                if q:
                    out.append(q.pop())
        return out
    raise ValueError(f"unknown curriculum {curriculum!r} (off|sorted|mixed)")


def build_splits(
    records: Iterable[dict[str, Any]],
    val_frac: float = 0.05,
    curriculum: str = "off",
    seed: int = 0,
    max_val: int | None = 512,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_recs, val_recs = [], []
    for rec in records:
        (val_recs if _split_of(rec["id"], val_frac) == "val" else train_recs).append(rec)
    train_recs = apply_curriculum(train_recs, curriculum, seed)
    val_recs.sort(key=lambda r: r["id"])  # stable eval order
    if max_val is not None and len(val_recs) > max_val:
        val_recs = val_recs[:max_val]
    train_rows = [to_rl_row(r, i, "train") for i, r in enumerate(train_recs)]
    val_rows = [to_rl_row(r, i, "val") for i, r in enumerate(val_recs)]
    return train_rows, val_rows


def write_parquet(rows: list[dict[str, Any]], path: str) -> None:
    import pandas as pd  # heavy import kept local

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    logger.info("wrote %d rows -> %s", len(rows), path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions-dir", default=os.path.join(repo_root, "data", "questions"))
    p.add_argument("--out-dir", default=os.path.join(repo_root, "data", "rl"))
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--curriculum", choices=["off", "sorted", "mixed"], default="off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-val", type=int, default=512)
    args = p.parse_args(argv)

    records = load_questions(args.questions_dir)
    train_rows, val_rows = build_splits(
        records, args.val_frac, args.curriculum, args.seed, args.max_val
    )
    write_parquet(train_rows, os.path.join(args.out_dir, "train.parquet"))
    write_parquet(val_rows, os.path.join(args.out_dir, "val.parquet"))
    print(f"train={len(train_rows)} val={len(val_rows)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
