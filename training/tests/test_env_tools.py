"""Tests for training/env/tools.py: error-as-string semantics, retry, mocks, schemas."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from training.env.tools import (
    TOOL_NAMES,
    MockToolClient,
    ToolServerClient,
    get_tool_schemas,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mock_tools.json")


def test_schemas_cover_all_four_endpoints():
    schemas = get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == set(TOOL_NAMES)
    for s in schemas:
        assert s["type"] == "function"
        assert "parameters" in s["function"]
        assert s["function"]["parameters"]["type"] == "object"


def test_mock_client_returns_json_and_round_robins():
    client = MockToolClient(FIXTURE)
    first = asyncio.run(client.call("vector_search", {"query": "x", "k": 5}))
    second = asyncio.run(client.call("vector_search", {"query": "y", "k": 5}))
    third = asyncio.run(client.call("vector_search", {"query": "z", "k": 5}))
    r1, r2, r3 = json.loads(first), json.loads(second), json.loads(third)
    assert r1[0]["qid"] == "Q42"
    assert r2[0]["qid"] == "Q691283"
    assert r3 == r2  # last fixture repeats once the list is exhausted


def test_mock_client_unknown_tool_is_error_string():
    client = MockToolClient(FIXTURE)
    out = asyncio.run(client.call("nuke_database", {}))
    assert out.startswith("ERROR:")


def test_live_client_unreachable_server_returns_error_string_not_raise():
    # Nothing listens on this port; both the call and its single retry must
    # fail fast and come back as an ERROR string the model can read.
    client = ToolServerClient(base_url="http://127.0.0.1:1", timeout=0.5)

    async def go():
        try:
            return await client.call("get_entity", {"qid": "Q42"})
        finally:
            await client.close()

    out = asyncio.run(go())
    assert isinstance(out, str) and out.startswith("ERROR:")


def test_live_client_rejects_unknown_tool_without_network():
    client = ToolServerClient(base_url="http://127.0.0.1:1", timeout=0.5)
    out = asyncio.run(client.call("drop_tables", {}))
    assert out.startswith("ERROR: unknown tool")


def test_truncation_marker():
    client = ToolServerClient(base_url="http://127.0.0.1:1", max_result_chars=10)
    assert client._truncate("0123456789ABC").startswith("0123456789...")
    assert client._truncate("short") == "short"
