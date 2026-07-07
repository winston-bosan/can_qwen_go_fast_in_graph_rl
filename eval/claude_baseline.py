"""Claude agentic baseline (stub): frontier model + the tool server's tools.

Runs Claude in a manual tool-use loop against the four tool-server endpoints
(vector_search / get_entity / get_neighbors / find_paths), max 15 tool calls,
then parses the final ```entities block with src/ecs/answer.py — the same
parser the trained policy is scored with.

Tool schemas come from toolserver/schema.py (ANTHROPIC_TOOLS); a local
fallback matching DESIGN.md is used if that import is unavailable. Tool name
== endpoint path (POST /<name>).

Needs ANTHROPIC_API_KEY and a running tool server; `unavailable_reason()`
reports which is missing so eval/run_eval.py can skip cleanly.
"""

from __future__ import annotations

import os

from ecs import config
from ecs.answer import parse_entities  # contract: src/ecs/answer.py
from questiongen.schema import QuestionRecord

BASELINE_MODEL = os.environ.get("ECS_BASELINE_MODEL", "claude-opus-4-8")
MAX_TOOL_CALLS = 15

SYSTEM_PROMPT = """\
You retrieve Wikidata entities relevant to a question, using the provided
search tools over a Wikipedia/Wikidata knowledge base (~4.8M entities).

Report EVERY entity needed to answer the question — the answer entities AND
the bridge/evidence entities you traversed to find them — ordered by
relevance (answers first). Finish your reply with exactly one fenced block:

```entities
Q123  # optional comment
Q456
```

Max 50 lines. QIDs only. Do not guess QIDs you have not seen in tool results.
"""

# Fallback schemas per DESIGN.md, used only when toolserver.schema is absent.
_FALLBACK_TOOLS: list[dict] = [
    {
        "name": "vector_search",
        "description": "Semantic search over entity abstracts; returns the k most similar entities (qid, title, score, snippet).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_entity",
        "description": "Look up one entity by QID: title, abstract, aliases, degree_in, degree_out.",
        "input_schema": {
            "type": "object",
            "properties": {"qid": {"type": "string", "pattern": "^Q\\d+$"}},
            "required": ["qid"],
        },
    },
    {
        "name": "get_neighbors",
        "description": "Paginated knowledge-graph edges of an entity (src, rel, rel_label, dst, dst_title).",
        "input_schema": {
            "type": "object",
            "properties": {
                "qid": {"type": "string", "pattern": "^Q\\d+$"},
                "relation": {"type": "string"},
                "direction": {"type": "string", "enum": ["out", "in", "both"], "default": "both"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": ["qid"],
        },
    },
    {
        "name": "find_paths",
        "description": "Shortest paths between two entities in the knowledge graph (max_hops <= 4).",
        "input_schema": {
            "type": "object",
            "properties": {
                "src_qid": {"type": "string", "pattern": "^Q\\d+$"},
                "dst_qid": {"type": "string", "pattern": "^Q\\d+$"},
                "max_hops": {"type": "integer", "minimum": 1, "maximum": 4, "default": 3},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["src_qid", "dst_qid"],
        },
    },
]


def tool_schemas() -> list[dict]:
    """Anthropic-format tool schemas, preferring the tool server's own."""
    try:
        from toolserver.schema import ANTHROPIC_TOOLS  # owned by workstream A

        return list(ANTHROPIC_TOOLS)
    except Exception:
        return _FALLBACK_TOOLS


class ClaudeBaseline:
    name = "claude_agentic"

    def __init__(
        self,
        k: int = 50,
        model: str = BASELINE_MODEL,
        base_url: str | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ):
        self.k = k
        self.model = model
        self.base_url = (base_url or config.TOOLSERVER_URL).rstrip("/")
        self.max_tool_calls = max_tool_calls
        self._client = None

    # -- availability ------------------------------------------------------

    def unavailable_reason(self) -> str | None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "ANTHROPIC_API_KEY not set — cannot run the Claude baseline."
        import httpx

        try:
            httpx.get(self.base_url + "/openapi.json", timeout=3.0)
        except Exception:
            return (
                f"tool server not reachable at {self.base_url} — start "
                "toolserver/app.py before running the Claude baseline."
            )
        return None

    # -- tool plumbing -----------------------------------------------------

    def _call_tool(self, name: str, args: dict) -> tuple[str, bool]:
        """POST /<name>; returns (result_text, is_error)."""
        import httpx

        try:
            resp = httpx.post(f"{self.base_url}/{name}", json=args, timeout=60.0)
            body = resp.text
            if len(body) > 20_000:  # keep hub-sized results out of context
                body = body[:20_000] + "\n... (truncated)"
            return body, resp.status_code >= 400
        except Exception as exc:
            return f"tool call failed: {type(exc).__name__}: {exc}", True

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    # -- the agentic loop --------------------------------------------------

    def predict(self, record: QuestionRecord) -> list[str]:
        client = self._get_client()
        tools = tool_schemas()
        messages: list[dict] = [{"role": "user", "content": record.question}]
        tool_calls = 0
        response = None

        while True:
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                break
            if tool_calls >= self.max_tool_calls:
                # budget exhausted: demand the final answer without more tools
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tu.id,
                                "content": "tool budget exhausted — reply with your final ```entities block now.",
                                "is_error": True,
                            }
                            for tu in tool_uses
                        ],
                    }
                )
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    tool_choice={"type": "none"},
                    messages=messages,
                )
                break

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                tool_calls += 1
                body, is_error = self._call_tool(tu.name, dict(tu.input))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": body,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

        final_text = "".join(b.text for b in response.content if b.type == "text")
        return parse_entities(final_text, max_entities=self.k)
