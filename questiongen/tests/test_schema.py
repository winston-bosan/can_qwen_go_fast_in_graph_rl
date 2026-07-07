import pytest
from pydantic import ValidationError

from questiongen.schema import QuestionRecord, append_records, load_records


def make(**kw) -> QuestionRecord:
    base = dict(
        id="kg-abc",
        question="Which films were directed by someone who studied at MIT?",
        answer_qids=["Q1", "Q2"],
        bridge_qids=["Q10"],
        source="kg_pattern",
        cypher="MATCH ...",
        difficulty=2,
    )
    base.update(kw)
    return QuestionRecord(**base)


def test_roundtrip_jsonl():
    r = make()
    line = r.to_jsonl_line()
    assert QuestionRecord.from_jsonl_line(line) == r


def test_cypher_optional_and_excluded_when_none():
    r = make(cypher=None, source="frames")
    assert "cypher" not in r.to_jsonl_line()


def test_source_literal_enforced():
    with pytest.raises(ValidationError):
        make(source="wikihop")


def test_qid_format_enforced():
    with pytest.raises(ValidationError):
        make(answer_qids=["Q1", "banana"])
    with pytest.raises(ValidationError):
        make(bridge_qids=["P57"])


def test_qids_deduplicated_preserving_order():
    r = make(answer_qids=["Q2", "Q1", "Q2", "Q1"])
    assert r.answer_qids == ["Q2", "Q1"]


def test_difficulty_positive():
    with pytest.raises(ValidationError):
        make(difficulty=0)


def test_append_and_load(tmp_path):
    path = str(tmp_path / "sub" / "q.jsonl")
    append_records(path, [make(id="a"), make(id="b")])
    append_records(path, [make(id="c")])
    got = load_records(path)
    assert [r.id for r in got] == ["a", "b", "c"]
