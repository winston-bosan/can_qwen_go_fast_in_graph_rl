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

Uses the anthropic SDK (model: claude-sonnet-5, env ANTHROPIC_API_KEY).
The module imports cleanly without the key; API-touching helpers raise
`ApiKeyMissing` with a clear message, and tests skip.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass

VERBALIZER_MODEL = os.environ.get("ECS_VERBALIZER_MODEL", "claude-sonnet-5")

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
    """Raised when ANTHROPIC_API_KEY is not set; message explains the skip."""


def have_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client():
    if not have_api_key():
        raise ApiKeyMissing(
            "ANTHROPIC_API_KEY is not set — skipping LLM verbalization / "
            "round-trip check (export the key to enable them)."
        )
    import anthropic  # lazy: module must import without the SDK configured

    return anthropic.Anthropic()


def _text_of(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


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
    model: str = VERBALIZER_MODEL,
) -> str:
    """Prompt A: verbalize a pattern; returns the question text."""
    client = client or _client()
    prompt = build_verbalize_prompt(semantics, relations, style)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    q = parse_question(_text_of(resp))
    if not q:
        raise ValueError(f"verbalizer returned no <question> block: {_text_of(resp)!r}")
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
    model: str = VERBALIZER_MODEL,
) -> JudgeResult:
    """Prompt B: judge whether `question` uniquely maps back to the pattern.

    `relations_available` are human-readable entries like 'P69 (educated at)'
    — the judge sees only the question and this vocabulary, never the pattern.
    """
    client = client or _client()
    prompt = build_judge_prompt(question, relations_available)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    parsed = parse_judge_response(_text_of(resp))
    return judge_verdict(parsed, expected_pids)
