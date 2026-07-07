"""Tests for verbalize.py.

Prompt construction, response parsing, and the round-trip verdict are pure
and tested offline. Tests that hit the Anthropic API run only when
ANTHROPIC_API_KEY is set (they are the integration smoke check)."""

import os

import pytest

from questiongen import verbalize as vb

HAVE_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

SEMANTICS = (
    'Find the films (answer entities) whose director (bridge) was educated '
    'at "Massachusetts Institute of Technology" (Q49108).'
)
RELATIONS = ["P57 (director)", "P69 (educated at)"]


# ---------------------------------------------------------------------------
# Offline: prompts and parsing
# ---------------------------------------------------------------------------


def test_module_imports_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert vb.have_api_key() is False
    with pytest.raises(vb.ApiKeyMissing, match="ANTHROPIC_API_KEY"):
        vb._client()


def test_three_style_variants_exist():
    assert len(vb.STYLE_VARIANTS) == 3


def test_verbalize_prompt_contents():
    for style in vb.STYLE_VARIANTS:
        p = vb.build_verbalize_prompt(SEMANTICS, RELATIONS, style)
        assert SEMANTICS in p
        assert "P57 (director)" in p and "P69 (educated at)" in p
        assert vb.STYLE_VARIANTS[style] in p
        assert "<question>" in p
        # paraphrase-diversity instructions present
        assert "Paraphrase-diversity" in p
        assert "Do not reveal" in p


def test_verbalize_prompt_rejects_unknown_style():
    with pytest.raises(ValueError, match="unknown style"):
        vb.build_verbalize_prompt(SEMANTICS, RELATIONS, "haiku")


def test_parse_question():
    assert (
        vb.parse_question("blah <question>Which films\n  qualify?</question>")
        == "Which films qualify?"
    )
    assert vb.parse_question("no tags here") is None
    assert vb.parse_question("<question>  </question>") is None


def test_judge_prompt_contains_only_question_and_relations():
    p = vb.build_judge_prompt("Which films qualify?", RELATIONS)
    assert "Which films qualify?" in p
    assert "- P57 (director)" in p
    # the judge must never see the pattern semantics
    assert SEMANTICS not in p
    assert '"pids"' in p and '"unique"' in p and '"reason"' in p


def test_parse_judge_response_extracts_json():
    text = 'Sure!\n{"pids": ["p57", "P69"], "unique": true, "reason": "clear"}'
    got = vb.parse_judge_response(text)
    assert got == {"pids": ["P57", "P69"], "unique": True, "reason": "clear"}


def test_parse_judge_response_no_json_raises():
    with pytest.raises(ValueError, match="no JSON"):
        vb.parse_judge_response("I cannot answer that.")


def test_judge_verdict_accepts_matching_multiset():
    parsed = {"pids": ["P69", "P57"], "unique": True, "reason": "ok"}
    v = vb.judge_verdict(parsed, ["P57", "P69"])
    assert v.accept and v.judge_unique


def test_judge_verdict_rejects_relation_mismatch():
    parsed = {"pids": ["P57"], "unique": True, "reason": "ok"}
    v = vb.judge_verdict(parsed, ["P57", "P69"])
    assert not v.accept
    assert "relation mismatch" in v.reason


def test_judge_verdict_respects_multiplicity():
    # P161 traversed twice (co-star pattern) != traversed once
    parsed = {"pids": ["P161"], "unique": True, "reason": "ok"}
    assert not vb.judge_verdict(parsed, ["P161", "P161"]).accept
    parsed2 = {"pids": ["P161", "P161"], "unique": True, "reason": "ok"}
    assert vb.judge_verdict(parsed2, ["P161", "P161"]).accept


def test_judge_verdict_rejects_ambiguous():
    parsed = {"pids": ["P57", "P69"], "unique": False, "reason": "ambiguous"}
    v = vb.judge_verdict(parsed, ["P57", "P69"])
    assert not v.accept and "ambiguous" in v.reason


# ---------------------------------------------------------------------------
# Live API (skipped without ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_KEY, reason="ANTHROPIC_API_KEY not set")
def test_live_verbalize_and_roundtrip():
    question = vb.verbalize(SEMANTICS, RELATIONS, style="direct")
    assert question.endswith("?") or len(question) > 10
    assert "Q49108" not in question  # QIDs must not leak
    verdict = vb.roundtrip_check(question, RELATIONS, ["P57", "P69"])
    assert isinstance(verdict.accept, bool)
    assert verdict.reason
