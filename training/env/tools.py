"""Adapter between the ECS FastAPI tool server (:7801) and the trainer's tool-calling interface.

Two layers:

1. ``ToolServerClient`` -- framework-agnostic async httpx client for the four
   endpoints (vector_search / get_entity / get_neighbors / find_paths).
   Timeouts, retry-once, and *errors returned as strings* (never raised into
   the rollout loop -- the model sees the error text as tool output and can
   recover, matching SID-1's "tools may fail, the policy must cope" setup).

2. verl ``BaseTool`` subclasses (one per endpoint) used by verl's sglang
   multi-turn rollout via ``training/configs/tools_ecs.yaml``. Import-guarded
   so this module works on boxes without verl (e.g. rollout_smoke.py locally).

Tool JSON schemas: prefer ``toolserver.schema`` (owned by workstream A, single
source of truth once it lands); a byte-for-byte fallback matching DESIGN.md is
kept here so training code and the smoke test run before/without it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

import training  # noqa: F401  (sys.path setup)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.environ.get("ECS_TOOL_TIMEOUT", "30"))
# Cap tool output injected into the context; hub entities / long abstracts can
# otherwise blow the token budget of a turn.
MAX_TOOL_RESULT_CHARS = int(os.environ.get("ECS_TOOL_RESULT_MAX_CHARS", "6000"))

TOOL_NAMES = ("vector_search", "get_entity", "get_neighbors", "find_paths")


# --------------------------------------------------------------------------
# Tool schemas (OpenAI function-call style)
# --------------------------------------------------------------------------

_FALLBACK_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": (
                "Semantic search over Wikipedia/Wikidata entity abstracts. "
                "Returns the k most similar entities to the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "k": {
                        "type": "integer",
                        "description": "Number of results (1-50).",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity",
            "description": (
                "Fetch one entity by Wikidata QID: title, abstract, aliases, "
                "and in/out degree in the knowledge graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "qid": {"type": "string", "description": "Wikidata QID, e.g. 'Q42'."},
                },
                "required": ["qid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": (
                "List graph edges of an entity (paginated). Optionally filter "
                "by relation P-id and direction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "qid": {"type": "string", "description": "Wikidata QID, e.g. 'Q42'."},
                    "relation": {
                        "type": "string",
                        "description": "Optional Wikidata P-id filter, e.g. 'P69'.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["out", "in", "both"],
                        "default": "both",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
                "required": ["qid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_paths",
            "description": "Find knowledge-graph paths between two entities (up to max_hops).",
            "parameters": {
                "type": "object",
                "properties": {
                    "src_qid": {"type": "string", "description": "Source QID."},
                    "dst_qid": {"type": "string", "description": "Destination QID."},
                    "max_hops": {"type": "integer", "minimum": 1, "maximum": 4, "default": 3},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": ["src_qid", "dst_qid"],
            },
        },
    },
]


def get_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style tool schema list; prefers toolserver.schema when available."""
    try:
        from toolserver import schema as ts_schema  # workstream A

        for attr in ("OPENAI_TOOLS", "TOOL_SCHEMAS", "OPENAI_TOOL_SCHEMAS", "tool_schemas"):
            schemas = getattr(ts_schema, attr, None)
            if schemas:
                return list(schemas)
        logger.warning(
            "toolserver.schema importable but exposes no TOOL_SCHEMAS; using fallback schemas"
        )
    except ImportError:
        logger.info("toolserver.schema not available yet; using fallback schemas from DESIGN.md")
    return _FALLBACK_TOOL_SCHEMAS


# --------------------------------------------------------------------------
# Async client
# --------------------------------------------------------------------------


def _toolserver_url() -> str:
    try:
        from ecs.config import TOOLSERVER_URL

        return TOOLSERVER_URL
    except ImportError:  # pragma: no cover - ecs.config exists per DESIGN.md
        return os.environ.get("ECS_TOOLSERVER_URL", "http://localhost:7801")


class ToolServerClient:
    """Async client for the four tool endpoints.

    Every call returns a *string* (JSON on success, ``"ERROR: ..."`` on any
    failure). Failures never raise: transient tool-server hiccups must not
    kill a rollout batch; the policy sees the error text and can retry or
    change strategy.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_result_chars: int = MAX_TOOL_RESULT_CHARS,
    ):
        self.base_url = (base_url or _toolserver_url()).rstrip("/")
        self.timeout = timeout
        self.max_result_chars = max_result_chars
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=min(10.0, self.timeout)),
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch one tool call; retry once on transport/5xx errors."""
        if name not in TOOL_NAMES:
            return f"ERROR: unknown tool '{name}'. Available tools: {', '.join(TOOL_NAMES)}."
        if not isinstance(arguments, dict):
            return "ERROR: tool arguments must be a JSON object."

        last_err = ""
        for attempt in (0, 1):  # retry exactly once
            try:
                client = await self._get_client()
                resp = await client.post(f"/{name}", json=arguments)
                if resp.status_code >= 500:
                    last_err = f"ERROR: tool server returned {resp.status_code}: {resp.text[:300]}"
                    if attempt == 0:
                        await asyncio.sleep(0.5)
                        continue
                    return last_err
                if resp.status_code >= 400:
                    # Client error (bad QID, validation, ...) -- model's fault,
                    # do not retry; surface the message so it can self-correct.
                    return f"ERROR: {resp.status_code}: {resp.text[:300]}"
                return self._truncate(resp.text)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = f"ERROR: tool call '{name}' failed: {type(e).__name__}: {e}"
                if attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
            except Exception as e:  # never raise into the rollout loop
                logger.exception("unexpected tool-call failure")
                return f"ERROR: tool call '{name}' failed unexpectedly: {type(e).__name__}: {e}"
        return last_err

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_result_chars:
            return text
        return text[: self.max_result_chars] + '... [truncated, use "limit"/"offset" to page]'

    # Convenience typed wrappers -------------------------------------------------

    async def vector_search(self, query: str, k: int = 10) -> str:
        return await self.call("vector_search", {"query": query, "k": k})

    async def get_entity(self, qid: str) -> str:
        return await self.call("get_entity", {"qid": qid})

    async def get_neighbors(self, qid: str, **kwargs: Any) -> str:
        return await self.call("get_neighbors", {"qid": qid, **kwargs})

    async def find_paths(self, src_qid: str, dst_qid: str, **kwargs: Any) -> str:
        return await self.call("find_paths", {"src_qid": src_qid, "dst_qid": dst_qid, **kwargs})


class MockToolClient:
    """Drop-in replacement for ToolServerClient backed by a fixture JSON file.

    Fixture format: ``{tool_name: <response object> | [<response objects>]}``.
    A list is consumed round-robin per tool so multi-turn trajectories see
    varied output. Used by ``rollout_smoke.py --mock-tools`` and tests.
    """

    def __init__(self, fixture_path: str):
        with open(fixture_path) as f:
            self._fixtures: dict[str, Any] = json.load(f)
        self._counters: dict[str, int] = {}

    async def close(self) -> None:  # interface parity
        return None

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in TOOL_NAMES:
            return f"ERROR: unknown tool '{name}'. Available tools: {', '.join(TOOL_NAMES)}."
        fixture = self._fixtures.get(name)
        if fixture is None:
            return f"ERROR: no mock fixture for tool '{name}'"
        if isinstance(fixture, list):
            i = self._counters.get(name, 0)
            self._counters[name] = i + 1
            fixture = fixture[min(i, len(fixture) - 1)]
        return json.dumps(fixture)


# --------------------------------------------------------------------------
# verl BaseTool adapters (used via configs/tools_ecs.yaml; require verl)
# --------------------------------------------------------------------------

try:
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

    _HAVE_VERL = True
except ImportError:
    _HAVE_VERL = False

if _HAVE_VERL:
    import uuid

    class _EcsBaseTool(BaseTool):
        """Shared plumbing: stateless per-instance, one shared async client."""

        endpoint: str = ""

        def __init__(self, config: dict, tool_schema: "OpenAIFunctionToolSchema" = None):
            if tool_schema is None:
                schema_dict = next(
                    s for s in get_tool_schemas() if s["function"]["name"] == self.endpoint
                )
                tool_schema = OpenAIFunctionToolSchema.model_validate(schema_dict)
            super().__init__(config, tool_schema)
            self._client = ToolServerClient(
                base_url=config.get("base_url"),
                timeout=float(config.get("timeout", DEFAULT_TIMEOUT)),
                max_result_chars=int(config.get("max_result_chars", MAX_TOOL_RESULT_CHARS)),
            )

        def get_openai_tool_schema(self) -> "OpenAIFunctionToolSchema":
            return self.tool_schema

        async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, "ToolResponse"]:
            return instance_id or str(uuid.uuid4()), ToolResponse(text="")

        async def execute(
            self, instance_id: str, parameters: dict[str, Any], **kwargs
        ) -> tuple["ToolResponse", float, dict]:
            result = await self._client.call(self.endpoint, parameters)
            # Step reward 0.0: all learning signal comes from the terminal
            # NDCG reward (env/reward.py), as in SID-1.
            return ToolResponse(text=result), 0.0, {"error": result.startswith("ERROR:")}

        async def calc_reward(self, instance_id: str, **kwargs) -> float:
            return 0.0

        async def release(self, instance_id: str, **kwargs) -> None:
            return None

    class VectorSearchTool(_EcsBaseTool):
        endpoint = "vector_search"

    class GetEntityTool(_EcsBaseTool):
        endpoint = "get_entity"

    class GetNeighborsTool(_EcsBaseTool):
        endpoint = "get_neighbors"

    class FindPathsTool(_EcsBaseTool):
        endpoint = "find_paths"
