"""Tests for eval/run_eval.py with in-memory records and dummy policies."""

import json

import pytest

from eval.run_eval import RandomBaseline, evaluate, summarize, write_report
from questiongen.schema import QuestionRecord


def rec(i: int, answers, bridges=(), source="kg_pattern") -> QuestionRecord:
    return QuestionRecord(
        id=f"q{i}",
        question=f"question {i}?",
        answer_qids=list(answers),
        bridge_qids=list(bridges),
        source=source,
        difficulty=2,
    )


RECORDS = [
    rec(1, ["Q1", "Q2"], ["Q10"]),
    rec(2, ["Q3"], [], source="sim_link"),
]


class PerfectPolicy:
    name = "perfect"

    def predict(self, record):
        return list(record.answer_qids) + list(record.bridge_qids)


class EmptyPolicy:
    name = "empty"

    def predict(self, record):
        return []


def test_evaluate_perfect_policy():
    report = evaluate(RECORDS, PerfectPolicy())
    assert report["n"] == 2
    assert report["mean_ndcg"] == pytest.approx(1.0)
    assert report["mean_recall@50"] == pytest.approx(1.0)
    # F1 counts bridges as false positives on q1: p=2/3, r=1 -> 0.8; q2 -> 1.0
    assert report["mean_f1"] == pytest.approx((0.8 + 1.0) / 2)
    assert set(report["per_source"]) == {"kg_pattern", "sim_link"}
    assert report["per_source"]["sim_link"]["n"] == 1


def test_evaluate_empty_policy_floors_at_zero():
    report = evaluate(RECORDS, EmptyPolicy())
    assert report["mean_ndcg"] == 0.0
    assert report["mean_recall@50"] == 0.0
    assert report["mean_f1"] == 0.0


def test_random_baseline_reproducible_and_pooled():
    p1 = RandomBaseline.from_records(RECORDS, k=3, seed=42)
    p2 = RandomBaseline.from_records(RECORDS, k=3, seed=42)
    assert p1.pool == sorted({"Q1", "Q2", "Q10", "Q3"})
    a = [p1.predict(r) for r in RECORDS]
    b = [p2.predict(r) for r in RECORDS]
    assert a == b  # same seed -> same predictions
    assert all(len(x) == 3 for x in a)


def test_random_baseline_empty_pool():
    assert RandomBaseline(pool=[], k=5).predict(RECORDS[0]) == []


def test_write_report_and_summarize(tmp_path):
    report = evaluate(RECORDS, PerfectPolicy())
    out = tmp_path / "sub" / "report.json"
    write_report(report, str(out))
    loaded = json.loads(out.read_text())
    assert loaded["policy"] == "perfect"
    assert loaded["n"] == 2
    line = summarize(report)
    assert "perfect" in line and "NDCG=1.0000" in line


def test_per_question_rows_present():
    report = evaluate(RECORDS, PerfectPolicy(), k=50)
    assert [r["id"] for r in report["per_question"]] == ["q1", "q2"]
    row = report["per_question"][0]
    assert {"ndcg", "recall@50", "f1", "source", "difficulty"} <= set(row)
