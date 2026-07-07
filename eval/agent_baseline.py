"""Agentic tool-calling baseline over any OpenAI-compatible model (OpenRouter).

Runs a model in a function-calling loop against the four tool-server
endpoints (vector_search / get_entity / get_neighbors / find_paths), max 15
tool calls, then parses the final ```entities block with src/ecs/answer.py —
the same parser the trained policy is scored with.

The model is a parameter, so the same harness gives both:
  * a frontier baseline, e.g.  --model anthropic/claude-sonnet-5
  * a same-class baseline,     --model deepseek/deepseek-v4-pro (default:
    config.QGEN_MODEL)
both routed through OpenRouter (base_url = config.OPENROUTER_BASE_URL, key =
$OPENROUTER_API_KEY, auto-loaded from the repo-root .env by ecs.config).

Tool schemas come from toolserver/schema.py (OPENAI_TOOLS); a local fallback
matching DESIGN.md is used if that import is unavailable. Tool name ==
endpoint path (POST /<name>).

Needs OPENROUTER_API_KEY and a running tool server; `unavailable_reason()`
reports which is missing so eval/run_eval.py can skip cleanly. Tool-server
calls retry once on transient failure (the server may restart mid-run).

Note: eval/claude_baseline.py is a deprecated shim kept for import
compatibility; it re-exports this class pinned to a Claude model.
"""

from __future__ import annotations

import json
import os

from ecs import config
from ecs.answer import parse_entities  # contract: src/ecs/answer.py
from questiongen.schema import QuestionRecord

MAX_TOOL_CALLS = 15
DEFAULT_MODEL = config.QGEN_MODEL
FRONTIER_MODEL = "anthropic/claude-sonnet-5"  # frontier baseline via OpenRouter

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

# Fallback schemas per DESIGN.md (OpenAI function-calling format), used only
# when toolserver.schema is absent.
_FALLBACK_PARAMS: dict[str, dict] = {
    "vector_search": {
        "description": "Semantic search over entity abstracts; returns the k most similar entities (qid, title, score, snippet).",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["query"],
        },
    },
    "get_entity": {
        "description": "Look up one entity by QID: title, abstract, aliases, degree_in, degree_out.",
        "schema": {
            "type": "object",
            "properties": {"qid": {"type": "string", "pattern": "^Q\\d+$"}},
            "required": ["qid"],
        },
    },
    "get_neighbors": {
        "description": "Paginated knowledge-graph edges of an entity (src, rel, rel_label, dst, dst_title).",
        "schema": {
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
    "find_paths": {
        "description": "Shortest paths between two entities in the knowledge graph (max_hops <= 4).",
        "schema": {
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
}

_FALLBACK_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": spec["schema"],
        },
    }
    for name, spec in _FALLBACK_PARAMS.items()
]


def tool_schemas() -> list[dict]:
    """OpenAI-format tool schemas, preferring the tool server's own."""
    try:
        from toolserver.schema import OPENAI_TOOLS  # owned by workstream A

        return list(OPENAI_TOOLS)
    except Exception:
        return _FALLBACK_TOOLS


class AgentBaseline:
    """OpenAI-compatible function-calling loop against the tool server."""

    def __init__(
        self,
        k: int = 50,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ):
        self.k = k
        self.model = model
        self.base_url = (base_url or config.TOOLSERVER_URL).rstrip("/")
        self.max_tool_calls = max_tool_calls
        self._client = None

    @property
    def name(self) -> str:
        return f"agent:{self.model}"

    # -- availability ------------------------------------------------------

    def unavailable_reason(self) -> str | None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            return (
                "OPENROUTER_API_KEY not set (repo-root .env or environment) — "
                "cannot run the agent baseline."
            )
        import httpx

        try:
            httpx.get(self.base_url + "/openapi.json", timeout=3.0)
        except Exception:
            return (
                f"tool server not reachable at {self.base_url} — start "
                "toolserver/app.py before running the agent baseline."
            )
        return None

    # -- tool plumbing -----------------------------------------------------

    def _call_tool(self, name: str, args: dict) -> str:
        """POST /<name>, one retry on transient failure (server may restart)."""
        import httpx

        last = ""
        for attempt in range(2):
            try:
                resp = httpx.post(f"{self.base_url}/{name}", json=args, timeout=60.0)
                body = resp.text
                if len(body) > 20_000:  # keep hub-sized results out of context
                    body = body[:20_000] + "\n... (truncated)"
                if resp.status_code >= 500 and attempt == 0:
                    last = body
                    continue
                return body
            except Exception as exc:
                last = f"tool call failed: {type(exc).__name__}: {exc}"
                if attempt == 0:
                    import time

                    time.sleep(1.0)
        return last

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=config.OPENROUTER_BASE_URL,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        return self._client

    # -- the agentic loop --------------------------------------------------

    def predict(self, record: QuestionRecord) -> list[str]:
        client = self._get_client()
        tools = tool_schemas()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record.question},
        ]
        tool_calls_used = 0

        while True:
            over_budget = tool_calls_used >= self.max_tool_calls
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=messages,
                tools=tools,
                tool_choice="none" if over_budget else "auto",
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return parse_entities(msg.content or "", max_entities=self.k)

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                tool_calls_used += 1
                if tool_calls_used > self.max_tool_calls:
                    body = (
                        "tool budget exhausted — reply with your final "
                        "```entities block now."
                    )
                else:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    body = self._call_tool(tc.function.name, args)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": body}
                )


def main(argv: list[str] | None = None) -> int:
    """Convenience: answer one ad-hoc question from the command line."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args(argv)

    baseline = AgentBaseline(k=args.k, model=args.model)
    reason = baseline.unavailable_reason()
    if reason:
        print(f"SKIP: {reason}")
        return 0
    record = QuestionRecord(
        id="adhoc", question=args.question, answer_qids=["Q1"],
        source="kg_pattern", difficulty=1,
    )
    for qid in baseline.predict(record):
        print(qid)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
