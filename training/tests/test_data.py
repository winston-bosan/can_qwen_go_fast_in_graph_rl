"""Tests for training/data.py: split determinism, curriculum, verl row format."""

from __future__ import annotations

import json

import pytest

from training.data import (
    SYSTEM_PROMPT,
    _split_of,
    apply_curriculum,
    build_splits,
    load_questions,
    to_rl_row,
)
from training.env.tools import TOOL_NAMES


def rec(i: int, difficulty="medium") -> dict:
    return {
        "id": f"q-{i:05d}",
        "question": f"Sample question {i}?",
        "answer_qids": [f"Q{i}"],
        "bridge_qids": [f"Q{i + 1000}"],
        "source": "kg-pattern",
        "difficulty": difficulty,
    }


def test_split_is_deterministic_and_roughly_proportional():
    ids = [f"q-{i}" for i in range(5000)]
    splits1 = [_split_of(i, 0.05) for i in ids]
    splits2 = [_split_of(i, 0.05) for i in ids]
    assert splits1 == splits2
    val_frac = splits1.count("val") / len(splits1)
    assert 0.03 < val_frac < 0.07
    # membership independent of other records: same id always same side
    assert _split_of("q-42", 0.05) == splits1[42]


def test_build_splits_no_overlap_and_val_cap():
    records = [rec(i) for i in range(1000)]
    train, val = build_splits(records, val_frac=0.1, max_val=20)
    assert len(val) == 20
    train_ids = {r["extra_info"]["id"] for r in train}
    val_ids = {r["extra_info"]["id"] for r in val}
    assert not train_ids & val_ids
    assert len(train) + len(val) <= len(records)


def test_row_format_matches_verl_expectations():
    row = to_rl_row(rec(7), index=3, split="train")
    assert row["data_source"] == "ecs/kg-pattern"
    assert row["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert row["prompt"][1]["role"] == "user"
    gt = json.loads(row["reward_model"]["ground_truth"])
    assert gt == {"answer_qids": ["Q7"], "bridge_qids": ["Q1007"]}
    assert row["reward_model"]["style"] == "rule"
    xi = row["extra_info"]
    assert xi["need_tools_kwargs"] is True
    assert set(xi["tools_kwargs"]) == set(TOOL_NAMES)
    for kw in xi["tools_kwargs"].values():
        # non-empty struct: Arrow cannot write empty structs to parquet
        assert kw == {"create_kwargs": {"question_id": "q-00007"}}


def test_curriculum_sorted_orders_easy_to_hard():
    records = [rec(i, d) for i, d in enumerate(["hard", "easy", "medium", "easy", "hard"])]
    out = apply_curriculum(records, "sorted", seed=1)
    ranks = [r["difficulty"] for r in out]
    assert ranks == sorted(ranks, key=lambda d: {"easy": 1, "medium": 2, "hard": 3}[d])


def test_curriculum_mixed_interleaves_tiers():
    records = [rec(i, "easy") for i in range(10)] + [rec(100 + i, "hard") for i in range(10)]
    out = apply_curriculum(records, "mixed", seed=1)
    # every window of 2 must contain one easy and one hard while both remain
    for i in range(0, 20, 2):
        window = {out[i]["difficulty"], out[i + 1]["difficulty"]}
        assert window == {"easy", "hard"}


def test_curriculum_off_is_deterministic_shuffle():
    records = [rec(i) for i in range(50)]
    a = apply_curriculum(records, "off", seed=7)
    b = apply_curriculum(records, "off", seed=7)
    assert a == b
    assert [r["id"] for r in a] != [r["id"] for r in records]  # actually shuffled


def test_curriculum_unknown_raises():
    with pytest.raises(ValueError):
        apply_curriculum([], "linear-warmup", seed=0)


def test_load_questions_skips_bad_lines_and_dupes(tmp_path):
    good = rec(1)
    lines = [
        json.dumps(good),
        json.dumps(good),  # duplicate id -> skipped
        "not json at all",
        json.dumps({"id": "x", "question": "no answer set?"}),  # missing answer_qids
        json.dumps(rec(2)),
    ]
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    records = load_questions(str(tmp_path))
    assert [r["id"] for r in records] == ["q-00001", "q-00002"]


def test_load_questions_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_questions(str(tmp_path))
