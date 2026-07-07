"""Tests for verbalize.py (OpenRouter / OpenAI SDK).

Prompt construction, response parsing, the round-trip verdict, retry/backoff
and usage tracking are pure and tested offline with stub clients. Tests that
hit OpenRouter run only when OPENROUTER_API_KEY is set (loaded from the
repo-root .env by ecs.config)."""

import os
from types import SimpleNamespace

import pytest

from questiongen import verbalize as vb

HAVE_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))

SEMANTICS = (
    'Find the films (answer entities) whose director (bridge) was educated '
    'at "Massachusetts Institute of Technology" (Q49108).'
)
RELATIONS = ["P57 (director)", "P69 (educated at)"]


# ---------------------------------------------------------------------------
# Stub client machinery (offline)
# ---------------------------------------------------------------------------


def make_response(text: str, prompt_tokens=100, completion_tokens=20, cost=0.0005):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
        ),
    )


class FakeError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"http {status_code}")


class StubClient:
    """OpenAI-shaped stub: yields queued errors first, then queued responses."""

    def __init__(self, responses, errors=()):
        self._responses = list(responses)
        self._errors = list(errors)
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1
                outer.last_kwargs = kwargs
                if outer._errors:
                    raise outer._errors.pop(0)
                return outer._responses.pop(0)

        self.chat = SimpleNamespace(completions=_Completions())


# ---------------------------------------------------------------------------
# Offline: key guard, prompts, parsing
# ---------------------------------------------------------------------------


def test_module_imports_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert vb.have_api_key() is False
    with pytest.raises(vb.ApiKeyMissing, match="OPENROUTER_API_KEY"):
        vb._client()


def test_default_model_comes_from_config():
    from ecs import config

    assert vb.QGEN_MODEL == config.QGEN_MODEL


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
# Offline: _chat plumbing — stub client, retry/backoff, usage tracking
# ---------------------------------------------------------------------------


def test_chat_uses_stub_and_requests_usage():
    stub = StubClient([make_response("hello")])
    got = vb._chat("prompt", client=stub, model="test/model")
    assert got == "hello"
    assert stub.last_kwargs["model"] == "test/model"
    assert stub.last_kwargs["extra_body"] == {"usage": {"include": True}}


def test_chat_tracks_usage_and_cost():
    vb.USAGE.reset()
    stub = StubClient(
        [make_response("a", 100, 10, 0.001), make_response("b", 200, 20, 0.002)]
    )
    vb._chat("p1", client=stub)
    vb._chat("p2", client=stub)
    assert vb.USAGE.calls == 2
    assert vb.USAGE.prompt_tokens == 300
    assert vb.USAGE.completion_tokens == 30
    assert vb.USAGE.cost_usd == pytest.approx(0.003)
    assert "300 prompt" in vb.USAGE.summary()
    vb.USAGE.reset()
    assert vb.USAGE.calls == 0 and vb.USAGE.cost_usd == 0.0


def test_chat_retries_on_429_then_succeeds(monkeypatch):
    delays = []
    monkeypatch.setattr(vb.time, "sleep", delays.append)
    stub = StubClient(
        [make_response("recovered")], errors=[FakeError(429), FakeError(503)]
    )
    got = vb._chat("p", client=stub, base_delay=0.5)
    assert got == "recovered"
    assert stub.calls == 3
    assert len(delays) == 2
    assert delays[1] > delays[0] >= 0.5  # exponential backoff


def test_chat_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(vb.time, "sleep", lambda s: None)
    stub = StubClient([], errors=[FakeError(429)] * 5)
    with pytest.raises(FakeError):
        vb._chat("p", client=stub, max_retries=2)
    assert stub.calls == 3  # initial + 2 retries


def test_chat_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr(vb.time, "sleep", lambda s: None)
    stub = StubClient([make_response("never")], errors=[FakeError(400)])
    with pytest.raises(FakeError):
        vb._chat("p", client=stub)
    assert stub.calls == 1  # 400 is not retryable


def test_verbalize_and_roundtrip_via_stub():
    vb.USAGE.reset()
    stub = StubClient(
        [
            make_response("<question>Which films were made by MIT alumni?</question>"),
            make_response('{"pids": ["P57", "P69"], "unique": true, "reason": "ok"}'),
        ]
    )
    q = vb.verbalize(SEMANTICS, RELATIONS, style="direct", client=stub)
    assert q == "Which films were made by MIT alumni?"
    verdict = vb.roundtrip_check(q, RELATIONS, ["P57", "P69"], client=stub)
    assert verdict.accept
    assert vb.USAGE.calls == 2


# ---------------------------------------------------------------------------
# Live OpenRouter (skipped without OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_KEY, reason="OPENROUTER_API_KEY not set")
def test_live_verbalize_and_roundtrip():
    vb.USAGE.reset()
    question = vb.verbalize(SEMANTICS, RELATIONS, style="direct")
    assert question.endswith("?") or len(question) > 10
    assert "Q49108" not in question  # QIDs must not leak
    verdict = vb.roundtrip_check(question, RELATIONS, ["P57", "P69"])
    assert isinstance(verdict.accept, bool)
    assert verdict.reason
    # usage was tracked for both calls
    assert vb.USAGE.calls == 2
    assert vb.USAGE.prompt_tokens > 0 and vb.USAGE.completion_tokens > 0


# ---------------------------------------------------------------------------
# Anchor-type gate + verbalizer constraints (full-run guards)
# ---------------------------------------------------------------------------

ANCHORS_INFO = [
    '"Yale Banner" — must be an educational institution (...) — abstract: "The Yale Banner is the yearbook of Yale University..."',
]


def test_diversity_rules_p69_and_qualifier_constraints():
    assert "graduate of" in vb.DIVERSITY_RULES
    assert "graduated from" in vb.DIVERSITY_RULES
    assert "attended" in vb.DIVERSITY_RULES
    assert "NEVER add type qualifiers" in vb.DIVERSITY_RULES


def test_prompt_without_anchors_info_has_no_gate():
    p = vb.build_verbalize_prompt(SEMANTICS, RELATIONS, "direct")
    assert "<reject>" not in p and "Anchor-type gate" not in p


def test_prompt_with_anchors_info_arms_the_gate():
    p = vb.build_verbalize_prompt(SEMANTICS, RELATIONS, "direct", ANCHORS_INFO)
    assert ANCHORS_INFO[0] in p
    assert "Anchor-type gate" in p
    assert "<reject>" in p
    assert "yearbook" in p  # failure classes spelled out for the model


def test_parse_reject():
    assert (
        vb.parse_reject('<reject>anchor "Yale Banner" is a yearbook, not a school</reject>')
        == 'anchor "Yale Banner" is a yearbook, not a school'
    )
    assert vb.parse_reject("<question>ok?</question>") is None
    assert vb.parse_reject("<reject>  </reject>") == "anchor rejected (no reason given)"


def test_verbalize_raises_anchor_rejected_on_sentinel():
    stub = StubClient([make_response("<reject>anchor is a yearbook</reject>")])
    with pytest.raises(vb.AnchorRejected, match="yearbook"):
        vb.verbalize(SEMANTICS, RELATIONS, client=stub, anchors_info=ANCHORS_INFO)


def test_verbalize_passes_anchors_info_through():
    stub = StubClient([make_response("<question>Which films qualify?</question>")])
    q = vb.verbalize(SEMANTICS, RELATIONS, client=stub, anchors_info=ANCHORS_INFO)
    assert q == "Which films qualify?"
    sent_prompt = stub.last_kwargs["messages"][0]["content"]
    assert ANCHORS_INFO[0] in sent_prompt


def test_chat_retries_empty_completions(monkeypatch):
    monkeypatch.setattr(vb.time, "sleep", lambda s: None)
    stub = StubClient([make_response(""), make_response("  "), make_response("ok")])
    assert vb._chat("p", client=stub) == "ok"
    assert stub.calls == 3


def test_chat_raises_after_persistent_empty(monkeypatch):
    monkeypatch.setattr(vb.time, "sleep", lambda s: None)
    stub = StubClient([make_response("")] * 3)
    with pytest.raises(vb.EmptyCompletion):
        vb._chat("p", client=stub, max_retries=2)
