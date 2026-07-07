"""DEPRECATED shim — use eval/agent_baseline.py.

The baseline harness was generalized to any OpenAI-compatible model routed
through OpenRouter (DESIGN.md amendment retiring the anthropic SDK path).
This module is kept so existing imports keep working; it re-exports the
generic loop pinned to a Claude model. New code should do:

    from eval.agent_baseline import AgentBaseline
    AgentBaseline(model="anthropic/claude-sonnet-5")   # frontier baseline
"""

from __future__ import annotations

import warnings

from .agent_baseline import (  # noqa: F401  (re-exports)
    FRONTIER_MODEL,
    MAX_TOOL_CALLS,
    SYSTEM_PROMPT,
    AgentBaseline,
    tool_schemas,
)

warnings.warn(
    "eval.claude_baseline is deprecated; use eval.agent_baseline.AgentBaseline "
    "(model routed through OpenRouter).",
    DeprecationWarning,
    stacklevel=2,
)


class ClaudeBaseline(AgentBaseline):
    """AgentBaseline pinned to a Claude model via OpenRouter (deprecated)."""

    def __init__(self, k: int = 50, model: str = FRONTIER_MODEL, **kw):
        super().__init__(k=k, model=model, **kw)
