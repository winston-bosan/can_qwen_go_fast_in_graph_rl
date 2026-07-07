"""Offline tests for the Claude baseline stub: schema plumbing and guards.

The agentic loop itself needs ANTHROPIC_API_KEY + a live tool server and is
exercised via `python -m eval.run_eval --policy claude`, not unit tests."""

from eval.claude_baseline import MAX_TOOL_CALLS, ClaudeBaseline, tool_schemas


def test_tool_schemas_are_anthropic_format():
    tools = tool_schemas()
    names = {t["name"] for t in tools}
    assert {"vector_search", "get_entity", "get_neighbors", "find_paths"} <= names
    for t in tools:
        assert set(t) >= {"name", "description", "input_schema"}
        assert t["input_schema"]["type"] == "object"


def test_max_tool_calls_contract():
    assert MAX_TOOL_CALLS == 15
    assert ClaudeBaseline().max_tool_calls == 15


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reason = ClaudeBaseline().unavailable_reason()
    assert reason and "ANTHROPIC_API_KEY" in reason


def test_unavailable_without_toolserver(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    baseline = ClaudeBaseline(base_url="http://127.0.0.1:1")  # nothing listens here
    reason = baseline.unavailable_reason()
    assert reason and "tool server" in reason


def test_uses_shared_answer_parser():
    # the baseline must score through the same parser as the trained policy
    from ecs.answer import parse_entities

    text = "done!\n```entities\nQ1 # answer\nq2\nQ1\n```"
    assert parse_entities(text) == ["Q1", "Q2"]
