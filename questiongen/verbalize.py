"""LLM verbalization of KG patterns + round-trip faithfulness check.

Prompt A (verbalize): given the pattern semantics with entity titles
substituted, produce a natural-language question in one of three style
variants, with paraphrase-diversity instructions.

Prompt B (round-trip judge): receives ONLY the candidate question and the
list of relations available in the graph, reconstructs the relational
pattern it implies, and judges whether the question uniquely maps back to
the intended pattern semantics (accept/reject + reason). We accept a
question only when (a) the judge says the mapping is unique and (b) the
reconstructed relation multiset equals the template's.

LLM access (DESIGN.md, amended): OpenAI SDK against OpenRouter —
base_url = config.OPENROUTER_BASE_URL, api_key = $OPENROUTER_API_KEY
(auto-loaded from the repo-root .env by ecs.config), model =
config.QGEN_MODEL (default "deepseek/deepseek-v4-pro"). The anthropic SDK
path is retired for generation.

Robustness/observability:
  * `_chat` retries with exponential backoff + jitter on 429/5xx/connection
    errors (OpenRouter rate limits);
  * every response's token usage (and OpenRouter's reported cost, requested
    via `usage: {include: true}`) is accumulated in the module-level `USAGE`
    tracker and logged, so cost per accepted question is measurable.

The module imports cleanly without the key; API-touching helpers raise
`ApiKeyMissing` with a clear message, and tests skip.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field

from ecs import config  # side effect: loads repo-root .env (OPENROUTER_API_KEY)

log = logging.getLogger(__name__)

QGEN_MODEL = config.QGEN_MODEL

STYLE_VARIANTS: dict[str, str] = {
    "direct": "a plain, direct factual question ('Which films ... ?')",
    "narrative": (
        "a question wrapped in a one-sentence scenario or research framing "
        "('A film archivist is cataloguing ... which films qualify?')"
    ),
    "trivia": (
        "a compact quiz/trivia phrasing ('Name the films that ...')"
    ),
}

DIVERSITY_RULES = """\
Paraphrase-diversity rules:
- Do NOT copy the relation names verbatim; use natural synonyms
  (e.g. "educated at" -> "studied at", "attended"; "cast member" -> "starred in").
- Vary sentence structure; do not start with "Which entity".
- Refer to anchor entities by their titles exactly as given; never invent
  extra qualifiers (dates, nationalities) that are not in the pattern.
- Never mention Wikidata, QIDs, Cypher, graphs, or the word "entity".
- The question must ask for ALL qualifying items, not one example.
- Do not reveal or enumerate any answers."""


class ApiKeyMissing(RuntimeError):
    """Raised when OPENROUTER_API_KEY is not set; message explains the skip."""


def have_api_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _client():
    if not have_api_key():
        raise ApiKeyMissing(
            "OPENROUTER_API_KEY is not set (repo-root .env or environment) — "
            "skipping LLM verbalization / round-trip check."
        )
    from openai import OpenAI  # lazy: module must import without the SDK configured

    return OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Usage tracking + retrying chat call
# ---------------------------------------------------------------------------


@dataclass
class UsageTracker:
    """Accumulates token usage / cost across calls (module-level: `USAGE`)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    _t0: float = field(default_factory=time.time)

    def add(self, usage) -> None:
        self.calls += 1
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        # OpenRouter reports credit cost on usage.cost when requested
        self.cost_usd += getattr(usage, "cost", 0.0) or 0.0

    def reset(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self._t0 = time.time()

    def summary(self) -> str:
        return (
            f"{self.calls} LLM calls, {self.prompt_tokens} prompt + "
            f"{self.completion_tokens} completion tokens, "
            f"${self.cost_usd:.4f}, {time.time() - self._t0:.0f}s elapsed"
        )


USAGE = UsageTracker()

RETRYABLE_STATUS = {429, 500, 502, 503, 504, 520, 522, 524}


def _status_of(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


def _is_retryable(exc: Exception) -> bool:
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS or status >= 500
    # connection/timeout errors carry no status; retry them too
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def _chat(
    prompt: str,
    client=None,
    model: str = QGEN_MODEL,
    max_tokens: int = 1024,
    max_retries: int = 4,
    base_delay: float = 2.0,
) -> str:
    """One chat completion with backoff on 429/5xx; tracks usage; returns text."""
    client = client or _client()
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"usage": {"include": True}},  # OpenRouter: report cost
            )
            usage = getattr(resp, "usage", None)
            USAGE.add(usage)
            log.info(
                "qgen call model=%s prompt_tokens=%s completion_tokens=%s cost=%s",
                model,
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
                getattr(usage, "cost", None),
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # openai typed errors; SDK import stays lazy
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            log.warning(
                "retryable LLM error (%s, status=%s); retry %d/%d in %.1fs",
                type(exc).__name__, _status_of(exc), attempt + 1, max_retries, delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop either returns or raises


# ---------------------------------------------------------------------------
# Prompt A — verbalization
# ---------------------------------------------------------------------------


def build_verbalize_prompt(semantics: str, relations: list[str], style: str) -> str:
    if style not in STYLE_VARIANTS:
        raise ValueError(f"unknown style {style!r}; one of {sorted(STYLE_VARIANTS)}")
    return f"""You write benchmark questions over a knowledge graph of Wikipedia entities.

Underlying pattern (the exact semantics your question must preserve):
{semantics}

Relations traversed by the pattern: {", ".join(relations)}

Write ONE natural-language question with exactly these semantics, phrased as
{STYLE_VARIANTS[style]}.

{DIVERSITY_RULES}

Reply with the question alone inside <question></question> tags."""


def parse_question(text: str) -> str | None:
    m = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
    if not m:
        return None
    q = " ".join(m.group(1).split())
    return q or None


def verbalize(
    semantics: str,
    relations: list[str],
    style: str = "direct",
    client=None,
    model: str = QGEN_MODEL,
) -> str:
    """Prompt A: verbalize a pattern; returns the question text."""
    prompt = build_verbalize_prompt(semantics, relations, style)
    text = _chat(prompt, client=client, model=model)
    q = parse_question(text)
    if not q:
        raise ValueError(f"verbalizer returned no <question> block: {text!r}")
    return q


# ---------------------------------------------------------------------------
# Prompt B — round-trip faithfulness judge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeResult:
    accept: bool
    reason: str
    reconstructed_pids: list[str]
    judge_unique: bool


def build_judge_prompt(question: str, relations_available: list[str]) -> str:
    rels = "\n".join(f"- {r}" for r in relations_available)
    return f"""You audit benchmark questions for a knowledge graph.

The graph's ONLY available relations are:
{rels}

Candidate question:
{question}

Task: reconstruct the graph pattern this question describes, using only the
relations above. Then judge whether the question maps back to ONE unique
pattern — reject it if it is ambiguous (could be read as different relation
chains), underspecified, or requires relations/attributes not in the list.

Reply with ONLY a JSON object, no prose:
{{"pids": ["P.."], "unique": true|false, "reason": "<one sentence>"}}
where "pids" lists every relation traversal in your reconstruction (repeat a
P-id if it is traversed twice)."""


def parse_judge_response(text: str) -> dict:
    """Extract the first JSON object from the judge's reply (robust)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"judge reply contains no JSON object: {text!r}")
    obj = json.loads(m.group(0))
    pids = [str(p).strip().upper() for p in obj.get("pids", [])]
    return {
        "pids": pids,
        "unique": bool(obj.get("unique", False)),
        "reason": str(obj.get("reason", "")),
    }


def judge_verdict(parsed: dict, expected_pids: list[str]) -> JudgeResult:
    """Pure decision: accept iff judge says unique AND relation multisets match."""
    match = Counter(parsed["pids"]) == Counter(p.upper() for p in expected_pids)
    accept = parsed["unique"] and match
    reason = parsed["reason"]
    if not match:
        reason = (
            f"relation mismatch: reconstructed {sorted(parsed['pids'])} vs "
            f"expected {sorted(p.upper() for p in expected_pids)}; " + reason
        )
    return JudgeResult(
        accept=accept,
        reason=reason,
        reconstructed_pids=parsed["pids"],
        judge_unique=parsed["unique"],
    )


def roundtrip_check(
    question: str,
    relations_available: list[str],
    expected_pids: list[str],
    client=None,
    model: str = QGEN_MODEL,
) -> JudgeResult:
    """Prompt B: judge whether `question` uniquely maps back to the pattern.

    `relations_available` are human-readable entries like 'P69 (educated at)'
    — the judge sees only the question and this vocabulary, never the pattern.
    """
    prompt = build_judge_prompt(question, relations_available)
    text = _chat(prompt, client=client, model=model)
    parsed = parse_judge_response(text)
    return judge_verdict(parsed, expected_pids)
