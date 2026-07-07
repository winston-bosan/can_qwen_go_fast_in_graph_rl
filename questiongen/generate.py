"""CLI: generate KG-pattern questions -> data/questions/kg_pattern.jsonl.

Pipeline per question: sample_pattern -> execute (exact golden sets, filters)
-> LLM verbalize (Prompt A) -> round-trip faithfulness check (Prompt B)
-> append JSONL.

LLM: config.QGEN_MODEL via OpenRouter (OPENROUTER_API_KEY, auto-loaded from
the repo-root .env by ecs.config). Prints a token-usage/cost summary at the
end so cost per accepted question is measurable.

Degrades gracefully:
  * neo4j down          -> clear skip message, exit 0
  * OPENROUTER_API_KEY unset -> falls back to the deterministic semantics
    scaffold as the question text and skips the round-trip check (use
    --require-llm to forbid this in production runs)

Usage:
  .venv/bin/python -m questiongen.generate --n 20
  .venv/bin/python -m questiongen.generate --n 5 --dry-run
  .venv/bin/python -m questiongen.generate --n 1 --seed-qid Q49108 --template chain_film_director_school
"""

from __future__ import annotations

import argparse
import os
import random
import sys

from ecs import config

from . import kg_patterns, verbalize
from .schema import QuestionRecord, append_records

DEFAULT_OUT = os.path.join(config.DATA_DIR, "questions", "kg_pattern.jsonl")


def generate_one(
    rng: random.Random,
    seed_qid: str | None,
    template: str | None,
    style: str | None,
    use_llm: bool,
    client=None,
) -> tuple[QuestionRecord | None, str]:
    """Attempt one question; returns (record | None, status_message)."""
    pattern = kg_patterns.sample_pattern(
        seed_qid=seed_qid, template=template, rng=rng
    )
    if pattern is None:
        return None, "sampler found no anchor binding"
    result = kg_patterns.execute(pattern)
    if result is None:
        return None, f"{pattern.template.name}: answer set empty or >{kg_patterns.MAX_ANSWERS}"
    answers, bridges = result

    relations = kg_patterns.relation_vocabulary(pattern.template)
    chosen_style = style or rng.choice(sorted(verbalize.STYLE_VARIANTS))

    if use_llm:
        question = verbalize.verbalize(
            pattern.semantics, relations, style=chosen_style, client=client
        )
        verdict = verbalize.roundtrip_check(
            question, relations, list(pattern.template.relations), client=client
        )
        if not verdict.accept:
            return None, f"{pattern.template.name}: round-trip rejected ({verdict.reason})"
    else:
        question = pattern.semantics  # deterministic scaffold fallback

    record = kg_patterns.to_record(pattern, answers, bridges, question)
    return record, f"{pattern.template.name}: ok ({len(answers)} answers, {len(bridges)} bridges)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=10, help="questions to generate")
    ap.add_argument("--dry-run", action="store_true", help="print records; don't write")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSONL (appended)")
    ap.add_argument("--seed-qid", default=None, help="pin the primary anchor QID")
    ap.add_argument("--template", default=None, choices=sorted(kg_patterns.TEMPLATES_BY_NAME))
    ap.add_argument("--style", default=None, choices=sorted(verbalize.STYLE_VARIANTS))
    ap.add_argument("--seed", type=int, default=None, help="rng seed")
    ap.add_argument(
        "--max-attempts", type=int, default=None,
        help="sampling attempts before giving up (default 10*n)",
    )
    ap.add_argument(
        "--require-llm", action="store_true",
        help="fail instead of falling back to scaffold questions without an API key",
    )
    args = ap.parse_args(argv)

    if not kg_patterns.neo4j_available():
        print(
            f"SKIP: neo4j not reachable at {config.NEO4J_URI} — start it with "
            "`docker compose up -d neo4j` and load Wikidata5M "
            "(ingest/load_neo4j.py) before generating KG-pattern questions."
        )
        return 0

    use_llm = verbalize.have_api_key()
    client = None
    if use_llm:
        client = verbalize._client()  # one client for the whole run
    else:
        msg = (
            "OPENROUTER_API_KEY not set — questions will use the deterministic "
            "semantics scaffold and skip the round-trip check."
        )
        if args.require_llm:
            print(f"ERROR: {msg}")
            return 1
        print(f"WARNING: {msg}")

    rng = random.Random(args.seed)
    max_attempts = args.max_attempts or max(10 * args.n, 20)
    records: list[QuestionRecord] = []
    attempts = 0
    while len(records) < args.n and attempts < max_attempts:
        attempts += 1
        try:
            record, status = generate_one(
                rng, args.seed_qid, args.template, args.style, use_llm, client=client
            )
        except kg_patterns.Neo4jUnavailable as exc:
            print(f"SKIP: {exc}")
            return 0
        print(f"[{attempts}] {status}")
        if record is not None:
            records.append(record)

    if use_llm and verbalize.USAGE.calls:
        u = verbalize.USAGE
        print(f"LLM usage ({verbalize.QGEN_MODEL}): {u.summary()}")
        if records:
            print(
                f"accepted {len(records)}/{attempts} attempts "
                f"(${u.cost_usd / len(records):.4f} per accepted question)"
            )

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
