"""CLI: generate KG-pattern questions -> data/questions/*.jsonl (parallel).

Pipeline per question: sample_pattern -> execute (exact golden sets, filters)
-> LLM verbalize (Prompt A) -> round-trip judge(s) (Prompt B) -> append JSONL.

Engine properties (full-corpus hardening):
  * ~16 worker threads (--workers) — neo4j driver and the OpenAI client are
    both thread-safe; each attempt is an independent unit of work
  * dedupe by (template, sorted anchor QIDs) — that tuple IS the record id,
    so the id set doubles as the dedupe key; keys are reserved before any
    LLM spend
  * difficulty-mix targeting (--mix, default 40% 2-hop / 35% 3-hop /
    25% 4-hop): next attempt's difficulty is drawn weighted by remaining
    quota, so the mix converges without starving hard tiers
  * dual/N-judge accept (--judge-models, comma-separated): a question is
    kept only if EVERY judge accepts (cross-model judging; see
    verbalize.JUDGE_MODEL)
  * budget guard (--budget, default $80): aborts when OpenRouter-reported
    cumulative cost crosses the cap
  * checkpointed append: each accepted record is written+flushed
    immediately; on restart, existing records are preloaded (ids + mix), so
    the run is resumable toward the same --n total
  * stats line every 100 accepted (and every 500 attempts)

LLM: config.QGEN_MODEL via OpenRouter (OPENROUTER_API_KEY, auto-loaded from
the repo-root .env by ecs.config).

Degrades gracefully:
  * neo4j down          -> clear skip message, exit 0
  * OPENROUTER_API_KEY unset -> falls back to the deterministic semantics
    scaffold as the question text and skips the round-trip check (use
    --require-llm to forbid this in production runs)

Usage:
  .venv/bin/python -m questiongen.generate --n 20
  .venv/bin/python -m questiongen.generate --n 15000 --workers 16 \\
      --judge-models deepseek/deepseek-v4-pro,google/gemini-3.1-flash-lite \\
      --out data/questions/kg_full.jsonl --budget 80
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ecs import config

from . import kg_patterns, verbalize
from .schema import QuestionRecord

DEFAULT_OUT = os.path.join(config.DATA_DIR, "questions", "kg_pattern.jsonl")
DEFAULT_WORKERS = 16
DEFAULT_BUDGET_USD = 80.0
DIFFICULTY_TARGETS: dict[int, float] = {2: 0.40, 3: 0.35, 4: 0.25}

TEMPLATES_BY_HOPS: dict[int, list[kg_patterns.PatternTemplate]] = {}
for _t in kg_patterns.PATTERNS:
    TEMPLATES_BY_HOPS.setdefault(_t.hops, []).append(_t)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def parse_mix(spec: str) -> dict[int, float]:
    """'40,35,25' -> {2: .40, 3: .35, 4: .25} (normalized)."""
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 3 or any(p < 0 for p in parts) or sum(parts) <= 0:
        raise ValueError(f"bad --mix {spec!r}; expected three non-negative numbers")
    total = sum(parts)
    return {2: parts[0] / total, 3: parts[1] / total, 4: parts[2] / total}


def choose_difficulty(
    by_difficulty: dict[int, int],
    n: int,
    rng: random.Random,
    targets: dict[int, float] = DIFFICULTY_TARGETS,
) -> int:
    """Draw the next attempt's difficulty weighted by remaining quota.

    Quota-weighted (not deficit-greedy) so a hard-to-fill tier pulls the mix
    toward its target without starving the others.
    """
    remaining = {
        d: max(targets[d] * n - by_difficulty.get(d, 0), 0.0) for d in targets
    }
    total = sum(remaining.values())
    weights = remaining if total > 0 else dict(targets)  # quota met: target ratios
    ds = sorted(weights)
    pick = rng.random() * sum(weights[d] for d in ds)
    acc = 0.0
    for d in ds:
        acc += weights[d]
        if pick <= acc and weights[d] > 0:
            return d
    return max(ds, key=lambda d: weights[d])


def preload_existing(path: str) -> tuple[set[str], Counter]:
    """Read an existing output file -> (record ids, difficulty counts).

    The record id encodes (template, sorted anchor QIDs), so this both
    resumes the count toward --n and seeds the dedupe set.
    """
    ids: set[str] = set()
    by_difficulty: Counter = Counter()
    if not os.path.exists(path):
        return ids, by_difficulty
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn tail line from a killed run
            if obj.get("id") in ids:
                continue
            ids.add(obj["id"])
            by_difficulty[int(obj.get("difficulty", 0))] += 1
    return ids, by_difficulty


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class GenStats:
    accepted: int = 0
    attempts: int = 0
    sampler_miss: int = 0
    title_filtered: int = 0  # title-hygiene pre-filter ('/', '#', empty, ==QID)
    kg_filtered: int = 0
    dupes: int = 0
    anchor_rejects: int = 0  # verbalizer's anchor-type gate fired
    judge_rejects: Counter = field(default_factory=Counter)  # per judge model
    errors: int = 0
    by_difficulty: Counter = field(default_factory=Counter)

    @property
    def judge_rejected(self) -> int:
        return sum(self.judge_rejects.values())


SNIPPET_CHARS = 150


def anchor_descriptions(pattern, get_abstract) -> list[str]:
    """'"MIT" — must be an educational institution (...) — "The Massachusetts..."'

    One line per anchor for the verbalizer's anchor-type gate. `get_abstract`
    maps qid -> abstract text (or None); snippets are truncated to
    SNIPPET_CHARS and flattened to one line.
    """
    out = []
    tmpl = pattern.template
    for slot, role in zip(tmpl.anchors, tmpl.roles):
        a = pattern.anchors[slot]
        abstract = (get_abstract(a["qid"]) if get_abstract else None) or ""
        snippet = " ".join(abstract.split())[:SNIPPET_CHARS]
        out.append(
            f'"{a["title"]}" — must be {role} — abstract: '
            f'"{snippet or "(no abstract available)"}"'
        )
    return out


class _ThreadLocalSidecar:
    """qid -> abstract via one read-only sqlite connection per worker thread
    (sqlite3 connections must not be shared across threads)."""

    def __init__(self, path: str):
        self._path = path
        self._tls = threading.local()

    def abstract(self, qid: str) -> str | None:
        from . import sidecar as sidecar_mod

        side = getattr(self._tls, "side", None)
        if side is None:
            side = self._tls.side = sidecar_mod.SqliteSidecar(self._path)
        return side.abstract(qid)


class GenEngine:
    """Thread-parallel generation with dedupe, mix targeting, budget guard.

    The four pipeline stages are injectable so tests run the engine entirely
    against in-memory fakes.
    """

    def __init__(
        self,
        n: int,
        out_path: str | None,
        judge_models: list[str],
        *,
        workers: int = DEFAULT_WORKERS,
        budget_usd: float = DEFAULT_BUDGET_USD,
        seed: int | None = None,
        max_attempts: int | None = None,
        style: str | None = None,
        use_llm: bool = True,
        targets: dict[int, float] = DIFFICULTY_TARGETS,
        stats_every: int = 100,
        sample_fn=None,
        execute_fn=None,
        verbalize_fn=None,
        judge_fn=None,
        abstract_fn=None,  # qid -> abstract text; arms the anchor-type gate
        log=print,
    ):
        self.n = n
        self.out_path = out_path
        self.judge_models = judge_models
        self.workers = max(1, workers)
        self.budget_usd = budget_usd
        self.seed = seed if seed is not None else random.randrange(1 << 30)
        self.max_attempts = max_attempts or max(10 * n, 20)
        self.style = style
        self.use_llm = use_llm
        self.targets = targets
        self.stats_every = stats_every
        self.log = log

        self._sample = sample_fn or kg_patterns.sample_pattern
        self._execute = execute_fn or kg_patterns.execute
        self._verbalize = verbalize_fn or verbalize.verbalize
        self._judge = judge_fn or verbalize.roundtrip_check
        self._abstract = abstract_fn

        self.stats = GenStats()
        self.stop_reason: str | None = None
        self.records: list[QuestionRecord] = []  # only kept when out_path is None
        self._lock = threading.Lock()
        self._fh = None
        self._t0 = time.time()
        self._session_accepted = 0

        if out_path:
            ids, by_diff = preload_existing(out_path)
            self._seen = ids
            self.stats.accepted = sum(by_diff.values())
            self.stats.by_difficulty = by_diff
            if ids:
                self.log(
                    f"resuming: {self.stats.accepted} existing records in "
                    f"{out_path} (mix {dict(sorted(by_diff.items()))})"
                )
        else:
            self._seen = set()

    # -- stop conditions (call under lock) ----------------------------------

    def _should_stop(self) -> bool:
        if self.stop_reason:
            return True
        if self.stats.accepted >= self.n:
            self.stop_reason = "target reached"
            return True
        if self.stats.attempts >= self.max_attempts:
            self.stop_reason = f"max attempts ({self.max_attempts}) exhausted"
            return True
        if self.use_llm and verbalize.USAGE.cost_usd >= self.budget_usd:
            self.stop_reason = (
                f"BUDGET GUARD: cumulative cost ${verbalize.USAGE.cost_usd:.2f} "
                f">= ${self.budget_usd:.2f} — aborting"
            )
            return True
        return False

    # -- one attempt ---------------------------------------------------------

    def _attempt(self) -> bool:
        """Run one attempt; False = engine should stop."""
        with self._lock:
            if self._should_stop():
                return False
            self.stats.attempts += 1
            attempt_no = self.stats.attempts
            rng = random.Random((self.seed << 20) ^ attempt_no)
            difficulty = choose_difficulty(
                self.stats.by_difficulty, self.n, rng, self.targets
            )
        template = rng.choice(TEMPLATES_BY_HOPS[difficulty])

        try:
            pattern = self._sample(template=template, rng=rng)
        except kg_patterns.Neo4jUnavailable as exc:
            with self._lock:
                self.stop_reason = str(exc)
            return False
        if pattern is None:
            with self._lock:
                self.stats.sampler_miss += 1
            self._maybe_attempt_stats()
            return True

        # title hygiene (free) — junk anchors never reach the LLM
        if not kg_patterns.anchors_ok(pattern):
            with self._lock:
                self.stats.title_filtered += 1
            self._maybe_attempt_stats()
            return True

        # dedupe BEFORE any LLM spend; reserve the key (a KG-filtered pattern
        # would fail identically next time, so the reservation stays)
        with self._lock:
            if pattern.id in self._seen:
                self.stats.dupes += 1
                return True
            self._seen.add(pattern.id)

        result = self._execute(pattern)
        if result is None:
            with self._lock:
                self.stats.kg_filtered += 1
            self._maybe_attempt_stats()
            return True
        answers, bridges = result

        relations = kg_patterns.relation_vocabulary(pattern.template)
        chosen_style = self.style or rng.choice(sorted(verbalize.STYLE_VARIANTS))
        try:
            if self.use_llm:
                question = self._verbalize(
                    pattern.semantics,
                    relations,
                    style=chosen_style,
                    anchors_info=anchor_descriptions(pattern, self._abstract),
                )
                for jm in self.judge_models:
                    verdict = self._judge(
                        question, relations, list(pattern.template.relations), model=jm
                    )
                    if not verdict.accept:
                        with self._lock:
                            self.stats.judge_rejects[jm] += 1
                        self._maybe_attempt_stats()
                        return True
            else:
                question = pattern.semantics  # deterministic scaffold fallback
        except verbalize.AnchorRejected as exc:
            with self._lock:
                self.stats.anchor_rejects += 1
            self.log(f"anchor-type gate: {exc}")
            self._maybe_attempt_stats()
            return True
        except Exception as exc:
            with self._lock:
                self.stats.errors += 1
                if self.stats.errors >= 50:
                    self.stop_reason = f"too many LLM errors (last: {exc})"
                    return False
            self.log(f"WARN attempt {attempt_no}: {type(exc).__name__}: {exc}")
            return True

        record = kg_patterns.to_record(pattern, answers, bridges, question)
        with self._lock:
            if self.stats.accepted >= self.n:
                return False
            self.stats.accepted += 1
            self._session_accepted += 1
            self.stats.by_difficulty[difficulty] += 1
            self._write(record)
            if self.stats.accepted % self.stats_every == 0:
                self.log(self._stats_line())
        return True

    # -- output / stats ------------------------------------------------------

    def _write(self, record: QuestionRecord) -> None:  # caller holds lock
        if self.out_path is None:
            self.records.append(record)
            return
        if self._fh is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
            self._fh = open(self.out_path, "a", encoding="utf-8")
        self._fh.write(record.to_jsonl_line() + "\n")
        self._fh.flush()

    def _maybe_attempt_stats(self) -> None:
        with self._lock:
            if self.stats.attempts % 500 == 0:
                self.log(self._stats_line())

    def _stats_line(self) -> str:  # caller holds lock
        s = self.stats
        elapsed = max(time.time() - self._t0, 1e-6)
        rate = self._session_accepted / elapsed * 60
        cost = verbalize.USAGE.cost_usd
        per_q = cost / max(self._session_accepted, 1)
        remaining = max(self.n - s.accepted, 0)
        eta_h = (remaining / max(rate, 1e-6)) / 60
        mix = " ".join(f"{d}:{s.by_difficulty.get(d, 0)}" for d in sorted(self.targets))
        return (
            f"[stats] accepted={s.accepted}/{self.n} attempts={s.attempts} "
            f"kg_filtered={s.kg_filtered} title_filtered={s.title_filtered} "
            f"dupes={s.dupes} anchor_rej={s.anchor_rejects} "
            f"judge_rej={s.judge_rejected} miss={s.sampler_miss} errors={s.errors} | "
            f"cost=${cost:.2f} (${per_q:.4f}/q) | {rate:.1f} q/min | "
            f"ETA {eta_h:.1f}h | mix {mix}"
        )

    # -- run ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            try:
                if not self._attempt():
                    return
            except Exception as exc:  # never let a worker die silently
                with self._lock:
                    self.stats.errors += 1
                self.log(f"WARN worker error: {type(exc).__name__}: {exc}")

    def run(self) -> GenStats:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for _ in range(self.workers):
                pool.submit(self._worker)
        with self._lock:
            self.log(self._stats_line())
            if self.stop_reason:
                self.log(f"stopped: {self.stop_reason}")
            if self._fh:
                self._fh.close()
                self._fh = None
        return self.stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=10, help="target TOTAL records in --out")
    ap.add_argument("--dry-run", action="store_true", help="print records; don't write")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSONL (appended, resumable)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                    help="abort when cumulative OpenRouter cost (USD) crosses this")
    ap.add_argument("--judge-models", default=None,
                    help="comma-separated; question kept only if ALL accept "
                    f"(default: {verbalize.JUDGE_MODEL}; env ECS_JUDGE_MODEL)")
    ap.add_argument("--mix", default=None,
                    help="difficulty mix 2,3,4-hop as e.g. '40,35,25' (default)")
    ap.add_argument("--style", default=None, choices=sorted(verbalize.STYLE_VARIANTS))
    ap.add_argument("--seed", type=int, default=None, help="rng seed")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="attempt cap before giving up (default 10*n)")
    ap.add_argument("--stats-every", type=int, default=100)
    ap.add_argument("--require-llm", action="store_true",
                    help="fail instead of falling back to scaffold questions without an API key")
    args = ap.parse_args(argv)

    if not kg_patterns.neo4j_available():
        print(
            f"SKIP: neo4j not reachable at {config.NEO4J_URI} — start it with "
            "`docker compose up -d neo4j` and load Wikidata5M "
            "(ingest/load_neo4j.py) before generating KG-pattern questions."
        )
        return 0

    use_llm = verbalize.have_api_key()
    if not use_llm:
        msg = (
            "OPENROUTER_API_KEY not set — questions will use the deterministic "
            "semantics scaffold and skip the round-trip check."
        )
        if args.require_llm:
            print(f"ERROR: {msg}")
            return 1
        print(f"WARNING: {msg}")

    judge_models = (
        [m.strip() for m in args.judge_models.split(",") if m.strip()]
        if args.judge_models
        else [verbalize.JUDGE_MODEL]
    )
    targets = parse_mix(args.mix) if args.mix else DIFFICULTY_TARGETS

    # sidecar abstracts arm the verbalizer's anchor-type gate
    from . import sidecar as sidecar_mod

    abstract_fn = None
    if sidecar_mod.available():
        abstract_fn = _ThreadLocalSidecar(sidecar_mod.DB_PATH).abstract
    elif use_llm:
        print(
            f"WARNING: sidecar not found at {sidecar_mod.DB_PATH} — the "
            "anchor-type gate runs without abstract snippets (weaker)."
        )

    engine = GenEngine(
        n=args.n,
        out_path=None if args.dry_run else args.out,
        judge_models=judge_models,
        workers=args.workers,
        budget_usd=args.budget,
        seed=args.seed,
        max_attempts=args.max_attempts,
        style=args.style,
        use_llm=use_llm,
        targets=targets,
        stats_every=args.stats_every,
        abstract_fn=abstract_fn,
    )
    print(
        f"generating: n={args.n} out={'(dry run)' if args.dry_run else args.out} "
        f"workers={engine.workers} gen={verbalize.QGEN_MODEL} "
        f"judges={judge_models} budget=${args.budget:.0f} "
        f"mix={ {d: round(v, 2) for d, v in targets.items()} }"
    )
    stats = engine.run()

    if use_llm and verbalize.USAGE.calls:
        u = verbalize.USAGE
        print(f"LLM usage ({verbalize.QGEN_MODEL} + judges): {u.summary()}")
        if engine._session_accepted:
            print(
                f"accepted {engine._session_accepted} this session / "
                f"{stats.attempts} attempts "
                f"(${u.cost_usd / engine._session_accepted:.4f} per accepted question)"
            )
        if stats.judge_rejects:
            print(f"judge rejections by model: {dict(stats.judge_rejects)}")
        if stats.anchor_rejects or stats.title_filtered:
            print(
                f"anchor guards: {stats.title_filtered} title-hygiene filtered, "
                f"{stats.anchor_rejects} anchor-type rejected"
            )

    if args.dry_run:
        for r in engine.records:
            print(r.to_jsonl_line())
        print(f"(dry run) {len(engine.records)} records NOT written")
        return 0 if engine.records else 1
    if stats.accepted == 0:
        print(f"no questions produced after {stats.attempts} attempts")
        return 1
    print(f"total {stats.accepted} records in {args.out}")
    return 0 if engine.stop_reason == "target reached" else 2


if __name__ == "__main__":
    sys.exit(main())
