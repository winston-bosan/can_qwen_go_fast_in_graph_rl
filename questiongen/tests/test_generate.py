"""Tests for the parallel generation engine (questiongen/generate.py).

The engine's four pipeline stages are injectable, so everything here runs
against in-memory fakes: dedupe, difficulty-mix targeting, dual-judge
accept, budget guard, checkpointed append + resume, and stop conditions."""

import json
import random
import threading
from collections import Counter

import pytest

from questiongen import generate as gen
from questiongen import kg_patterns as kp
from questiongen import verbalize as vb
from questiongen.generate import (
    DIFFICULTY_TARGETS,
    GenEngine,
    choose_difficulty,
    parse_mix,
    preload_existing,
)
from questiongen.kg_patterns import InstantiatedPattern


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_templates_by_hops_covers_all_tiers():
    assert set(gen.TEMPLATES_BY_HOPS) == {2, 3, 4}
    assert sum(len(v) for v in gen.TEMPLATES_BY_HOPS.values()) == len(kp.PATTERNS)


def test_parse_mix():
    assert parse_mix("40,35,25") == {2: 0.40, 3: 0.35, 4: 0.25}
    assert parse_mix("1,1,2") == {2: 0.25, 3: 0.25, 4: 0.50}
    with pytest.raises(ValueError):
        parse_mix("40,60")
    with pytest.raises(ValueError):
        parse_mix("-1,50,51")


def test_default_targets_are_the_contract_mix():
    assert DIFFICULTY_TARGETS == {2: 0.40, 3: 0.35, 4: 0.25}


def test_choose_difficulty_prefers_unfilled_quota():
    rng = random.Random(0)
    # 2-hop and 3-hop quotas full; only 4-hop remains
    counts = {2: 40, 3: 35, 4: 0}
    picks = {choose_difficulty(counts, 100, rng) for _ in range(50)}
    assert picks == {4}


def test_choose_difficulty_converges_to_mix():
    rng = random.Random(1)
    counts: dict[int, int] = {}
    n = 1000
    for _ in range(n):
        d = choose_difficulty(counts, n, rng)
        counts[d] = counts.get(d, 0) + 1
    assert counts[2] / n == pytest.approx(0.40, abs=0.02)
    assert counts[3] / n == pytest.approx(0.35, abs=0.02)
    assert counts[4] / n == pytest.approx(0.25, abs=0.02)


def test_choose_difficulty_when_quota_met_uses_target_ratios():
    rng = random.Random(2)
    counts = {2: 400, 3: 350, 4: 250}  # everything full (n=1000)
    picks = Counter(choose_difficulty(counts, 1000, rng) for _ in range(300))
    assert set(picks) <= {2, 3, 4}
    assert picks[2] > picks[4]  # still roughly target-shaped


def test_preload_existing(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [
        {"id": "kg-a", "difficulty": 2},
        {"id": "kg-b", "difficulty": 4},
        {"id": "kg-a", "difficulty": 2},  # duplicate line -> counted once
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + '\n{"id": "kg-torn", "diff'
    )  # torn tail line from a killed run -> ignored
    ids, by_diff = preload_existing(str(path))
    assert ids == {"kg-a", "kg-b"}
    assert by_diff == Counter({2: 1, 4: 1})


def test_preload_existing_missing_file(tmp_path):
    ids, by_diff = preload_existing(str(tmp_path / "nope.jsonl"))
    assert ids == set() and by_diff == Counter()


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


def make_pattern(template, i: int) -> InstantiatedPattern:
    anchors = {slot: {"qid": f"Q{i}{j}", "title": f"T{i}{j}"} for j, slot in enumerate(template.anchors)}
    return InstantiatedPattern(template=template, anchors=anchors)


class Fakes:
    """Injectable pipeline stages with call counting."""

    def __init__(self, judge_verdicts=None, unique_patterns=True):
        self.counter = 0
        self.lock = threading.Lock()
        self.verbalized = 0
        self.judged = Counter()
        self.judge_verdicts = judge_verdicts or {}  # model -> accept bool
        self.unique_patterns = unique_patterns

    def sample(self, template=None, rng=None, **kw):
        with self.lock:
            self.counter += 1
            i = self.counter
        if not self.unique_patterns:
            # always the SAME pattern (fixed template + anchors), regardless
            # of what the engine asked for — exercises the dedupe path
            return make_pattern(gen.TEMPLATES_BY_HOPS[2][0], 1)
        return make_pattern(template, i)

    def execute(self, pattern, **kw):
        return [f"Q{int(pattern.id[-6:], 16)}"], ["Q1"]

    def verbalize(self, semantics, relations, style=None, **kw):
        with self.lock:
            self.verbalized += 1
        return f"Question about {semantics[:20]}?"

    def judge(self, question, relations, expected, model=None, **kw):
        with self.lock:
            self.judged[model] += 1
        accept = self.judge_verdicts.get(model, True)
        return vb.JudgeResult(accept=accept, reason="fake", reconstructed_pids=list(expected), judge_unique=accept)


def make_engine(n=10, fakes=None, out_path=None, **kw):
    fakes = fakes or Fakes()
    logs = []
    engine = GenEngine(
        n=n,
        out_path=out_path,
        judge_models=kw.pop("judge_models", ["gen/model"]),
        workers=kw.pop("workers", 4),
        seed=kw.pop("seed", 42),
        sample_fn=fakes.sample,
        execute_fn=fakes.execute,
        verbalize_fn=fakes.verbalize,
        judge_fn=fakes.judge,
        log=logs.append,
        **kw,
    )
    return engine, fakes, logs


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------


def test_engine_reaches_target_and_writes_checkpointed(tmp_path):
    vb.USAGE.reset()
    out = str(tmp_path / "q.jsonl")
    engine, fakes, _ = make_engine(n=20, out_path=out)
    stats = engine.run()
    assert stats.accepted == 20
    assert engine.stop_reason == "target reached"
    lines = [json.loads(x) for x in open(out) if x.strip()]
    assert len(lines) == 20
    assert len({r["id"] for r in lines}) == 20  # all unique
    assert all(r["source"] == "kg_pattern" for r in lines)


def test_engine_mix_targeting(tmp_path):
    vb.USAGE.reset()
    out = str(tmp_path / "q.jsonl")
    engine, _, _ = make_engine(n=100, out_path=out)
    stats = engine.run()
    total = sum(stats.by_difficulty.values())
    assert total == 100
    assert stats.by_difficulty[2] / total == pytest.approx(0.40, abs=0.10)
    assert stats.by_difficulty[3] / total == pytest.approx(0.35, abs=0.10)
    assert stats.by_difficulty[4] / total == pytest.approx(0.25, abs=0.10)


def test_engine_dedupes_identical_patterns_before_llm():
    vb.USAGE.reset()
    # every sample returns the SAME pattern -> only 1 accepted, no LLM re-spend
    fakes = Fakes(unique_patterns=False)
    engine, fakes, _ = make_engine(n=5, fakes=fakes, max_attempts=30)
    stats = engine.run()
    assert stats.accepted == 1
    assert stats.dupes == 29 - stats.sampler_miss  # everything else was a dupe
    assert fakes.verbalized == 1  # dedupe fired BEFORE the LLM stage
    assert "max attempts" in engine.stop_reason


def test_engine_dual_judge_requires_all_accepts():
    vb.USAGE.reset()
    fakes = Fakes(judge_verdicts={"judge/a": True, "judge/b": False})
    engine, fakes, _ = make_engine(
        n=5, fakes=fakes, judge_models=["judge/a", "judge/b"], max_attempts=12
    )
    stats = engine.run()
    assert stats.accepted == 0
    assert stats.judge_rejects["judge/b"] > 0
    assert stats.judge_rejects["judge/a"] == 0  # a accepted; b vetoed
    assert fakes.judged["judge/a"] == fakes.judged["judge/b"]  # b ran after a


def test_engine_dual_judge_short_circuits_on_first_reject():
    vb.USAGE.reset()
    fakes = Fakes(judge_verdicts={"judge/a": False, "judge/b": True})
    engine, fakes, _ = make_engine(
        n=3, fakes=fakes, judge_models=["judge/a", "judge/b"], max_attempts=6
    )
    engine.run()
    assert fakes.judged["judge/a"] > 0
    assert fakes.judged["judge/b"] == 0  # never consulted after a's veto


def test_engine_budget_guard_aborts(monkeypatch):
    vb.USAGE.reset()
    vb.USAGE.cost_usd = 99.0  # already over budget
    engine, fakes, _ = make_engine(n=50, budget_usd=80.0)
    stats = engine.run()
    assert stats.accepted == 0
    assert "BUDGET GUARD" in engine.stop_reason
    assert fakes.verbalized == 0
    vb.USAGE.reset()


def test_engine_budget_ignored_without_llm():
    vb.USAGE.reset()
    vb.USAGE.cost_usd = 99.0
    engine, _, _ = make_engine(n=3, budget_usd=80.0, use_llm=False)
    stats = engine.run()
    assert stats.accepted == 3  # scaffold mode spends nothing; guard not applied
    vb.USAGE.reset()


def test_engine_resume_preloads_ids_and_mix(tmp_path):
    vb.USAGE.reset()
    out = str(tmp_path / "q.jsonl")
    engine1, _, _ = make_engine(n=10, out_path=out)
    engine1.run()
    # resume toward n=25 total: 15 more, no id collisions with the first run
    fakes2 = Fakes()
    fakes2.counter = 1000  # fresh anchor ids
    engine2, _, logs2 = make_engine(n=25, fakes=fakes2, out_path=out)
    assert engine2.stats.accepted == 10  # preloaded
    stats = engine2.run()
    assert stats.accepted == 25
    lines = [json.loads(x) for x in open(out) if x.strip()]
    assert len(lines) == 25
    assert len({r["id"] for r in lines}) == 25
    assert any("resuming: 10 existing" in x for x in logs2)


def test_engine_resume_already_complete(tmp_path):
    vb.USAGE.reset()
    out = str(tmp_path / "q.jsonl")
    engine1, _, _ = make_engine(n=5, out_path=out)
    engine1.run()
    engine2, fakes2, _ = make_engine(n=5, out_path=out)
    stats = engine2.run()
    assert stats.accepted == 5
    assert stats.attempts == 0  # nothing to do
    assert engine2.stop_reason == "target reached"
    assert fakes2.verbalized == 0


def test_engine_scaffold_mode_skips_llm():
    vb.USAGE.reset()
    engine, fakes, _ = make_engine(n=4, use_llm=False)
    stats = engine.run()
    assert stats.accepted == 4
    assert fakes.verbalized == 0 and sum(fakes.judged.values()) == 0
    assert all(r.question for r in engine.records)


def test_engine_neo4j_down_stops_gracefully():
    vb.USAGE.reset()

    def boom(template=None, rng=None, **kw):
        raise kp.Neo4jUnavailable("neo4j not reachable — skipping")

    fakes = Fakes()
    engine, _, _ = make_engine(n=5, fakes=fakes)
    engine._sample = boom
    stats = engine.run()
    assert stats.accepted == 0
    assert "neo4j not reachable" in engine.stop_reason


def test_engine_stats_line_shape():
    vb.USAGE.reset()
    engine, _, _ = make_engine(n=4, stats_every=2)
    logs = []
    engine.log = logs.append
    engine.run()
    stats_lines = [x for x in logs if x.startswith("[stats]")]
    assert stats_lines, logs
    line = stats_lines[-1]
    for token in ("accepted=", "attempts=", "cost=$", "q/min", "ETA", "mix"):
        assert token in line


def test_judge_model_env_override(monkeypatch):
    import importlib

    monkeypatch.setenv("ECS_JUDGE_MODEL", "some/other-judge")
    importlib.reload(vb)
    try:
        assert vb.JUDGE_MODEL == "some/other-judge"
    finally:
        monkeypatch.delenv("ECS_JUDGE_MODEL")
        importlib.reload(vb)
    # default: self-judge (generator model) until calibration verdict lands
    assert vb.JUDGE_MODEL == vb.QGEN_MODEL
