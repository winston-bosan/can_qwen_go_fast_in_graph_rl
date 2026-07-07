"""Tests for eval/metrics.py — the reward contract.

These tests double as executable documentation of the two-tier NDCG's
behaviour, in particular exactly how much (or little) it penalizes
"entity dumping" — see test_entity_dumping_*.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import f1_set, ndcg_two_tier, recall_at_k


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


# ---------------------------------------------------------------------------
# ndcg_two_tier — basic ordering behaviour
# ---------------------------------------------------------------------------


def test_perfect_order_is_one():
    answer = {"Q1", "Q2", "Q3"}
    bridge = {"Q10", "Q11"}
    predicted = ["Q1", "Q2", "Q3", "Q10", "Q11"]
    assert ndcg_two_tier(predicted, answer, bridge) == pytest.approx(1.0)


def test_perfect_order_answers_only():
    assert ndcg_two_tier(["Q5"], {"Q5"}, set()) == pytest.approx(1.0)


def test_reversed_order_penalized():
    """Bridges ranked above answers score < 1.0 (exact value asserted)."""
    answer = {"Q1", "Q2"}
    bridge = {"Q10"}
    predicted = ["Q10", "Q2", "Q1"]  # bridge first, answers after

    ideal = dcg([2, 2, 1])
    got = dcg([1, 2, 2])
    expected = got / ideal
    assert ndcg_two_tier(predicted, answer, bridge) == pytest.approx(expected)
    assert expected == pytest.approx(0.86708, abs=1e-4)
    # and strictly worse than the perfect order
    assert expected < 1.0


def test_order_within_tier_does_not_matter():
    answer = {"Q1", "Q2"}
    a = ndcg_two_tier(["Q1", "Q2"], answer, set())
    b = ndcg_two_tier(["Q2", "Q1"], answer, set())
    assert a == pytest.approx(b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Entity dumping — executable documentation of the penalty (or lack thereof)
# ---------------------------------------------------------------------------


def test_entity_dumping_junk_after_correct_is_free():
    """DOCUMENTED PROPERTY: appending junk AFTER all golden entities costs
    exactly NOTHING in two-tier NDCG.

    5 correct answers followed by 40 junk entities scores exactly the same
    as the 5 correct answers alone (NDCG = 1.0, penalty = 0.0):

      * junk has gain 0, so it adds nothing to the DCG;
      * junk sits at positions 6..45, so it displaces nothing;
      * the ideal DCG depends only on the golden sets, not the prediction.

    In other words NDCG alone provides NO incentive against dumping once all
    golden entities are ranked first. The anti-dumping pressure in this
    project comes from (a) the 50-line answer cap — when the golden set is
    large, junk occupies slots that could have held golden entities (see
    test_entity_dumping_displacement_beyond_k below) — and (b) the set-F1
    metric reported alongside, which drops from 1.0 to 0.2 in this exact
    scenario (precision 5/45): see test_entity_dumping_f1_contrast.
    """
    answer = {f"Q{i}" for i in range(1, 6)}  # 5 golden answers
    correct = [f"Q{i}" for i in range(1, 6)]
    junk = [f"Q{i}" for i in range(1000, 1040)]  # 40 junk entities

    clean = ndcg_two_tier(correct, answer, set())
    dumped = ndcg_two_tier(correct + junk, answer, set())

    assert clean == pytest.approx(1.0)
    assert dumped == pytest.approx(1.0)
    assert clean - dumped == pytest.approx(0.0)  # penalty is exactly zero


def test_entity_dumping_junk_before_correct_is_expensive():
    """Contrast case: the SAME 40 junk entities placed BEFORE the 5 correct
    ones push the answers to positions 41-45 and cost ~69% of the score
    (NDCG drops from 1.0 to ~0.311)."""
    answer = {f"Q{i}" for i in range(1, 6)}
    correct = [f"Q{i}" for i in range(1, 6)]
    junk = [f"Q{i}" for i in range(1000, 1040)]

    got = ndcg_two_tier(junk + correct, answer, set())
    ideal = dcg([2] * 5)
    expected = dcg([0] * 40 + [2] * 5) / ideal
    assert got == pytest.approx(expected)
    assert got == pytest.approx(0.31068, abs=1e-4)


def test_entity_dumping_displacement_beyond_k():
    """When the golden set is large, dumping DOES cost: junk occupying slots
    inside the top-k pushes further golden entities past the k cutoff."""
    answer = {f"Q{i}" for i in range(1, 11)}  # 10 golden answers
    first5 = [f"Q{i}" for i in range(1, 6)]
    last5 = [f"Q{i}" for i in range(6, 11)]
    junk45 = [f"Q{i}" for i in range(1000, 1045)]

    # 5 correct + 45 junk fills k=50; the other 5 golden can never appear.
    dumped = ndcg_two_tier(first5 + junk45 + last5, answer, set(), k=50)
    honest = ndcg_two_tier(first5 + last5, answer, set(), k=50)
    assert honest == pytest.approx(1.0)
    assert dumped < honest
    expected = dcg([2] * 5) / dcg([2] * 10)
    assert dumped == pytest.approx(expected)
    assert dumped == pytest.approx(0.64893, abs=1e-4)


def test_entity_dumping_f1_contrast():
    """F1 punishes the dump that NDCG ignores: 5 correct + 40 junk gives
    precision 5/45, recall 1.0 -> F1 = 0.2 (vs 1.0 without junk)."""
    answer = {f"Q{i}" for i in range(1, 6)}
    correct = [f"Q{i}" for i in range(1, 6)]
    junk = [f"Q{i}" for i in range(1000, 1040)]
    assert f1_set(correct, answer) == pytest.approx(1.0)
    assert f1_set(correct + junk, answer) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Bridge credit / tiers
# ---------------------------------------------------------------------------


def test_bridge_only_gets_partial_credit():
    answer = {"Q1"}
    bridge = {"Q10", "Q11"}
    predicted = ["Q10", "Q11"]  # no answers at all, only bridges
    ideal = dcg([2, 1, 1])
    expected = dcg([1, 1]) / ideal
    got = ndcg_two_tier(predicted, answer, bridge)
    assert got == pytest.approx(expected)
    assert 0.0 < got < 1.0


def test_answer_membership_wins_over_bridge():
    """An entity present in both sets counts with gain 2, and the ideal does
    not double-count it."""
    answer = {"Q1"}
    bridge = {"Q1", "Q2"}  # Q1 in both
    # perfect list: Q1 (gain 2), Q2 (gain 1)
    assert ndcg_two_tier(["Q1", "Q2"], answer, bridge) == pytest.approx(1.0)
    # Q1 alone earns 2/ideal, not 3/ideal
    ideal = dcg([2, 1])
    assert ndcg_two_tier(["Q1"], answer, bridge) == pytest.approx(dcg([2]) / ideal)


def test_junk_only_scores_zero():
    assert ndcg_two_tier(["Q99", "Q98"], {"Q1"}, {"Q2"}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Duplicates, empties, truncation
# ---------------------------------------------------------------------------


def test_duplicates_counted_once_first_occurrence():
    answer = {"Q1", "Q2"}
    # duplicated Q1 must not earn gain twice; first occurrence keeps its rank
    dup = ndcg_two_tier(["Q1", "Q1", "Q1", "Q2"], answer, set())
    ideal = dcg([2, 2])
    expected = dcg([2, 2]) / ideal  # Q1 at rank 0, Q2 at rank 1 after dedup
    assert dup == pytest.approx(expected) == pytest.approx(1.0)


def test_duplicates_do_not_displace_after_dedup():
    """Dedup happens BEFORE truncation: k=2 with ['Q1','Q1','Q2'] still
    sees both golden entities."""
    answer = {"Q1", "Q2"}
    assert ndcg_two_tier(["Q1", "Q1", "Q2"], answer, set(), k=2) == pytest.approx(1.0)


def test_empty_prediction_is_zero():
    assert ndcg_two_tier([], {"Q1"}, {"Q2"}) == 0.0


def test_empty_golden_is_zero():
    assert ndcg_two_tier(["Q1", "Q2"], set(), set()) == 0.0


def test_truncation_at_k_prediction_side():
    """A golden entity ranked beyond k earns nothing."""
    answer = {"Q1"}
    junk = [f"Q{i}" for i in range(100, 150)]  # 50 junk fills k=50
    assert ndcg_two_tier(junk + ["Q1"], answer, set(), k=50) == 0.0


def test_truncation_at_k_ideal_side():
    """Ideal DCG is truncated at k: with 3 answers and k=2 a perfect top-2
    scores 1.0."""
    answer = {"Q1", "Q2", "Q3"}
    assert ndcg_two_tier(["Q1", "Q2"], answer, set(), k=2) == pytest.approx(1.0)


def test_k_zero_or_negative():
    assert ndcg_two_tier(["Q1"], {"Q1"}, set(), k=0) == 0.0
    assert recall_at_k(["Q1"], {"Q1"}, k=0) == 0.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_basic():
    assert recall_at_k(["Q1", "Q9"], {"Q1", "Q2"}) == pytest.approx(0.5)
    assert recall_at_k(["Q1", "Q2"], {"Q1", "Q2"}) == pytest.approx(1.0)


def test_recall_truncates_at_k():
    junk = [f"Q{i}" for i in range(100, 150)]
    assert recall_at_k(junk + ["Q1"], {"Q1"}, k=50) == 0.0
    assert recall_at_k(["Q1"] + junk, {"Q1"}, k=50) == pytest.approx(1.0)


def test_recall_duplicates_dont_burn_slots():
    """Dedup before truncation: k=2 with a duplicate still reaches Q2."""
    assert recall_at_k(["Q1", "Q1", "Q2"], {"Q1", "Q2"}, k=2) == pytest.approx(1.0)


def test_recall_empty_answer_is_zero():
    assert recall_at_k(["Q1"], set()) == 0.0


def test_recall_empty_prediction_is_zero():
    assert recall_at_k([], {"Q1"}) == 0.0


# ---------------------------------------------------------------------------
# f1_set
# ---------------------------------------------------------------------------


def test_f1_perfect():
    assert f1_set(["Q1", "Q2"], {"Q1", "Q2"}) == pytest.approx(1.0)


def test_f1_partial():
    # precision 1/2, recall 1/3 -> F1 = 0.4
    assert f1_set(["Q1", "Q9"], {"Q1", "Q2", "Q3"}) == pytest.approx(0.4)


def test_f1_duplicates_are_set_semantics():
    assert f1_set(["Q1", "Q1"], {"Q1"}) == pytest.approx(1.0)


def test_f1_empty_cases():
    assert f1_set([], {"Q1"}) == 0.0
    assert f1_set(["Q1"], set()) == 0.0
    assert f1_set([], set()) == 0.0
    assert f1_set(["Q9"], {"Q1"}) == 0.0
