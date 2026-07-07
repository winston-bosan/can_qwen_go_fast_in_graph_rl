"""Offline tests for the OpenAI-compatible agent baseline: schema plumbing,
guards, the tool-calling loop against a stubbed client, and the deprecated
claude_baseline shim. Live runs go through `python -m eval.run_eval
--policy agent --model ...`, not unit tests."""

import json
from types import SimpleNamespace

import pytest

from eval.agent_baseline import (
    DEFAULT_MODEL,
    MAX_TOOL_CALLS,
    AgentBaseline,
    tool_schemas,
)
from questiongen.schema import QuestionRecord

RECORD = QuestionRecord(
    id="q1", question="Which films?", answer_qids=["Q1"],
    source="kg_pattern", difficulty=2,
)


# ---------------------------------------------------------------------------
# Schemas / config
# ---------------------------------------------------------------------------


def test_tool_schemas_are_openai_format():
    tools = tool_schemas()
    names = {t["function"]["name"] for t in tools}
    assert {"vector_search", "get_entity", "get_neighbors", "find_paths"} <= names
    for t in tools:
        assert t["type"] == "function"
        fn = t["function"]
        assert set(fn) >= {"name", "description", "parameters"}
        assert fn["parameters"]["type"] == "object"


def test_defaults_follow_config():
    from ecs import config

    assert DEFAULT_MODEL == config.QGEN_MODEL
    assert MAX_TOOL_CALLS == 15
    b = AgentBaseline()
    assert b.max_tool_calls == 15
    assert b.model == config.QGEN_MODEL
    assert b.name == f"agent:{config.QGEN_MODEL}"


def test_model_is_a_parameter():
    b = AgentBaseline(model="anthropic/claude-sonnet-5")
    assert b.model == "anthropic/claude-sonnet-5"
    assert b.name == "agent:anthropic/claude-sonnet-5"


# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reason = AgentBaseline().unavailable_reason()
    assert reason and "OPENROUTER_API_KEY" in reason


def test_unavailable_without_toolserver(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-not-real")
    baseline = AgentBaseline(base_url="http://127.0.0.1:1")  # nothing listens here
    reason = baseline.unavailable_reason()
    assert reason and "tool server" in reason


# ---------------------------------------------------------------------------
# The loop, with a stubbed OpenAI client + stubbed tool transport
# ---------------------------------------------------------------------------


def tc(id_, name, args):
    return SimpleNamespace(
        id=id_,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def msg(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


class StubOpenAI:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                # snapshot: the loop mutates the messages list between calls
                outer.requests.append({**kwargs, "messages": list(kwargs["messages"])})
                return outer._responses.pop(0)

        self.chat = SimpleNamespace(completions=_Completions())


def make_baseline(responses, tool_log):
    b = AgentBaseline(model="test/model")
    b._client = StubOpenAI(responses)
    b._call_tool = lambda name, args: tool_log.append((name, args)) or '{"ok": true}'
    return b


def test_loop_executes_tools_then_parses_answer():
    tool_log = []
    baseline = make_baseline(
        [
            msg(tool_calls=[tc("c1", "vector_search", {"query": "films", "k": 5})]),
            msg(tool_calls=[tc("c2", "get_neighbors", {"qid": "Q5"})]),
            msg(content="found them\n```entities\nQ1 # film\nQ2\n```"),
        ],
        tool_log,
    )
    got = baseline.predict(RECORD)
    assert got == ["Q1", "Q2"]
    assert tool_log == [
        ("vector_search", {"query": "films", "k": 5}),
        ("get_neighbors", {"qid": "Q5"}),
    ]
    stub = baseline._client
    # transcript grows correctly: system+user, then +assistant+tool per round
    assert [m["role"] for m in stub.requests[0]["messages"]] == ["system", "user"]
    assert [m["role"] for m in stub.requests[2]["messages"]] == [
        "system", "user", "assistant", "tool", "assistant", "tool",
    ]
    assert stub.requests[0]["tool_choice"] == "auto"
    assert stub.requests[0]["model"] == "test/model"


def test_loop_enforces_tool_budget():
    tool_log = []
    # the model asks for one tool call per round, forever
    responses = [
        msg(tool_calls=[tc(f"c{i}", "get_entity", {"qid": f"Q{i}"})])
        for i in range(MAX_TOOL_CALLS)
    ]
    # after the budget is used up, tool_choice becomes "none" -> final answer
    responses.append(msg(content="```entities\nQ7\n```"))
    baseline = make_baseline(responses, tool_log)
    got = baseline.predict(RECORD)
    assert got == ["Q7"]
    assert len(tool_log) == MAX_TOOL_CALLS  # not one more
    assert baseline._client.requests[-1]["tool_choice"] == "none"


def test_loop_returns_empty_on_malformed_answer():
    baseline = make_baseline([msg(content="no entities block, sorry")], [])
    assert baseline.predict(RECORD) == []


def test_uses_shared_answer_parser():
    # the baseline must score through the same parser as the trained policy
    from ecs.answer import parse_entities

    text = "done!\n```entities\nQ1 # answer\nq2\nQ1\n```"
    assert parse_entities(text) == ["Q1", "Q2"]


# ---------------------------------------------------------------------------
# Deprecated shim
# ---------------------------------------------------------------------------


def test_claude_baseline_shim_warns_and_pins_frontier_model():
    import importlib
    import sys

    sys.modules.pop("eval.claude_baseline", None)
    with pytest.warns(DeprecationWarning, match="agent_baseline"):
        shim = importlib.import_module("eval.claude_baseline")
    b = shim.ClaudeBaseline()
    assert isinstance(b, AgentBaseline)
    assert b.model == shim.FRONTIER_MODEL == "anthropic/claude-sonnet-5"
