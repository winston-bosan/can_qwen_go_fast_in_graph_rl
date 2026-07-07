"""Tests for training/env/reward.py.

Metric note: eval/metrics.py (workstream B) may not exist yet. These tests run
against whichever implementation training.env.reward resolves; a dedicated test
asserts that eval.metrics is PREFERRED whenever importable, and another pins the
fallback (training/tests/reference_metrics.py, TEST-ONLY) plus its warning.
"""

from __future__ import annotations

import json
import math
import sys
import types

import pytest

from training.env import reward as reward_mod
from training.env.reward import (
    LengthPenaltyConfig,
    RewardConfig,
    compute_reward,
    compute_score,
)
from training.tests.reference_metrics import ndcg_two_tier as ref_ndcg


@pytest.fixture(autouse=True)
def _fresh_resolution():
    """Reset the lazy parse/metric resolution cache around every test."""
    reward_mod._reset_resolution_cache()
    yield
    reward_mod._reset_resolution_cache()


def block(*qids: str) -> str:
    return "Some reasoning...\n```entities\n" + "\n".join(qids) + "\n```\n"


ANSWER = ["Q691283"]
BRIDGE = ["Q42"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_perfect_answer_gets_ndcg_plus_format_bonus():
    sol = block("Q691283", "Q42")
    res = compute_reward(sol, ANSWER, BRIDGE)
    assert res.parsed
    assert res.task_score == pytest.approx(1.0)  # ideal ordering -> NDCG 1
    assert res.format_bonus == pytest.approx(0.1)
    assert res.total == pytest.approx(1.1)


def test_partial_answer_scores_between_zero_and_one():
    sol = block("Q999", "Q42")  # junk first, bridge second, answer missing
    res = compute_reward(sol, ANSWER, BRIDGE)
    assert res.parsed
    expected = ref_ndcg(["Q999", "Q42"], set(ANSWER), set(BRIDGE), k=50)
    assert 0.0 < res.task_score < 1.0
    assert res.task_score == pytest.approx(expected)
    assert res.total == pytest.approx(expected + 0.1)


def test_comments_and_ordering_respected():
    sol = block("Q42  # bridge first (wrong order)", "Q691283")
    res = compute_reward(sol, ANSWER, BRIDGE)
    assert res.parsed and res.n_entities == 2
    assert res.task_score < 1.0  # answer entity ranked second -> discounted


# ---------------------------------------------------------------------------
# Format handling / hard zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solution",
    [
        "",  # empty
        "The answer is St John's College (Q691283).",  # no fenced block
        "```entities\n```",  # empty block
        "```entities\nnot-a-qid\nalso bad\n```",  # block with no valid QIDs
        "```python\nQ691283\n```",  # wrong fence language
        None,
    ],
)
def test_unparseable_output_is_hard_zero(solution):
    res = compute_reward(solution, ANSWER, BRIDGE)
    assert not res.parsed
    assert res.total == 0.0
    assert res.format_bonus == 0.0  # no format bonus without a parseable block


def test_over_50_entities_truncated_before_scoring():
    # answer QID buried at position 56 -> must NOT contribute after truncation.
    # Parser-agnostic: holds whether truncation happens in ecs.answer (which
    # caps at MAX_ANSWER_ENTITIES itself) or in compute_reward's own guard.
    filler = [f"Q{i}" for i in range(1000, 1055)]  # 55 junk QIDs
    sol = block(*(filler + ["Q691283"]))
    res = compute_reward(sol, ANSWER, BRIDGE)
    assert res.parsed
    assert res.n_entities == 50
    assert res.task_score == pytest.approx(0.0)
    assert res.total == pytest.approx(0.1)  # format bonus only


def test_truncated_flag_with_fallback_parser(monkeypatch):
    # The internal fallback parser does NOT cap at 50, so compute_reward's own
    # truncation guard (and its `truncated` flag) is exercised deterministically.
    monkeypatch.setitem(sys.modules, "ecs.answer", None)
    reward_mod._reset_resolution_cache()
    filler = [f"Q{i}" for i in range(1000, 1055)]
    with pytest.warns(RuntimeWarning, match="ecs.answer"):
        res = compute_reward(block(*(filler + ["Q691283"])), ANSWER, BRIDGE)
    assert res.truncated
    assert res.n_entities == 50
    assert res.task_score == pytest.approx(0.0)
    assert res.total == pytest.approx(0.1)


def test_exactly_50_not_flagged_truncated():
    qids = ["Q691283"] + [f"Q{i}" for i in range(2000, 2049)]
    res = compute_reward(block(*qids), ANSWER, BRIDGE)
    assert res.n_entities == 50 and not res.truncated


# ---------------------------------------------------------------------------
# Config knobs: recall metric, length penalty
# ---------------------------------------------------------------------------


def test_recall_metric():
    cfg = RewardConfig(metric="recall")
    sol = block("Q691283", "Q42", "Q999")
    res = compute_reward(sol, ["Q691283", "Q123456"], BRIDGE, config=cfg)
    assert res.task_score == pytest.approx(0.5)  # 1 of 2 answer QIDs found
    assert res.total == pytest.approx(0.6)


def test_length_penalty_off_by_default():
    res = compute_reward(block("Q691283"), ANSWER, BRIDGE, response_tokens=100_000)
    assert res.length_penalty == 0.0


def test_length_penalty_applies_when_enabled():
    cfg = RewardConfig(
        length_penalty=LengthPenaltyConfig(enabled=True, coef=0.0001, target_tokens=1000)
    )
    res = compute_reward(block("Q691283", "Q42"), ANSWER, BRIDGE, response_tokens=3000, config=cfg)
    assert res.length_penalty == pytest.approx(0.2)
    assert res.total == pytest.approx(1.0 + 0.1 - 0.2)


def test_length_penalty_never_negative_total():
    cfg = RewardConfig(
        length_penalty=LengthPenaltyConfig(enabled=True, coef=1.0, target_tokens=0)
    )
    res = compute_reward(block("Q691283"), ANSWER, BRIDGE, response_tokens=10_000, config=cfg)
    assert res.total == 0.0


def test_invalid_metric_rejected(tmp_path):
    import yaml as _yaml

    path = tmp_path / "bad_reward.yaml"
    path.write_text(_yaml.safe_dump({"metric": "f1"}))
    with pytest.raises(ValueError):
        RewardConfig.load(str(path))


# ---------------------------------------------------------------------------
# verl entry point (compute_score) + ground-truth coercion
# ---------------------------------------------------------------------------


def test_compute_score_with_json_string_ground_truth():
    gt = json.dumps({"answer_qids": ANSWER, "bridge_qids": BRIDGE})
    score = compute_score("ecs/kg-pattern", block("Q691283", "Q42"), gt, {"index": 0})
    assert score == pytest.approx(1.1)


def test_compute_score_with_dict_and_bare_list_ground_truth():
    assert compute_score("ecs/x", block("Q691283"), {"answer_qids": ANSWER}) > 0
    assert compute_score("ecs/x", block("Q691283"), ANSWER) > 0  # bare list = answer set


def test_compute_score_unparseable_is_zero():
    gt = {"answer_qids": ANSWER, "bridge_qids": BRIDGE}
    assert compute_score("ecs/x", "no block here", gt) == 0.0


# ---------------------------------------------------------------------------
# Import-path preference: eval.metrics must win whenever importable
# ---------------------------------------------------------------------------


def test_prefers_eval_metrics_when_importable(monkeypatch):
    sentinel = 0.4242

    fake_metrics = types.ModuleType("eval.metrics")

    def fake_ndcg(predicted, answer, bridge, k=50):
        return sentinel

    fake_metrics.ndcg_two_tier = fake_ndcg
    fake_eval = types.ModuleType("eval")
    fake_eval.metrics = fake_metrics
    monkeypatch.setitem(sys.modules, "eval", fake_eval)
    monkeypatch.setitem(sys.modules, "eval.metrics", fake_metrics)
    reward_mod._reset_resolution_cache()

    res = compute_reward(block("Q691283"), ANSWER, BRIDGE)
    assert res.task_score == pytest.approx(sentinel), (
        "reward must use eval.metrics.ndcg_two_tier when importable"
    )


def test_fallback_to_reference_metrics_warns(monkeypatch):
    # Force `import eval.metrics` to fail even if workstream B has landed it.
    monkeypatch.setitem(sys.modules, "eval.metrics", None)
    monkeypatch.delitem(sys.modules, "eval", raising=False)
    reward_mod._reset_resolution_cache()

    with pytest.warns(RuntimeWarning, match="eval.metrics"):
        res = compute_reward(block("Q691283", "Q42"), ANSWER, BRIDGE)
    assert res.task_score == pytest.approx(
        ref_ndcg(["Q691283", "Q42"], set(ANSWER), set(BRIDGE), k=50)
    )


def test_reference_metric_sanity():
    # spot-check the reference implementation itself
    assert ref_ndcg(["Q1"], {"Q1"}, set()) == pytest.approx(1.0)
    assert ref_ndcg([], {"Q1"}, set()) == 0.0
    assert ref_ndcg(["Q2"], {"Q1"}, set()) == 0.0
    # bridge-only prediction vs ideal [answer, bridge]
    got = ref_ndcg(["Q2"], {"Q1"}, {"Q2"})
    ideal = 2.0 / math.log2(2) + 1.0 / math.log2(3)
    assert got == pytest.approx(1.0 / ideal)
    # answer/bridge overlap: answer tier wins
    assert ref_ndcg(["Q1"], {"Q1"}, {"Q1"}) == pytest.approx(1.0)
