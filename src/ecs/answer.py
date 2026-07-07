"""Parser for the model's final answer format (DESIGN.md):

    ```entities
    Q123  # optional comment
    Q456
    ```

`parse_entities` returns an ordered, deduplicated list of QIDs (max 50).
Tolerant of: trailing comments (# or //), commas, list bullets, surrounding
whitespace, lowercase q, and extra prose around the fenced block. If several
``entities`` blocks are present, the last one wins (models sometimes draft
then revise).
"""

from __future__ import annotations

import re

from .config import MAX_ANSWER_ENTITIES

_BLOCK_RE = re.compile(
    r"```[ \t]*entities[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_QID_RE = re.compile(r"\b[Qq](\d+)\b")


def extract_block(text: str) -> str | None:
    """Return the body of the last ```entities fenced block, or None."""
    matches = _BLOCK_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1]


def parse_entities(text: str, max_entities: int = MAX_ANSWER_ENTITIES) -> list[str]:
    """Parse the answer text; return ordered unique QIDs (<= max_entities).

    Returns [] when no valid ```entities block is found (format penalty is
    the caller's concern — see eval/metrics.py).
    """
    body = extract_block(text)
    if body is None:
        return []
    qids: list[str] = []
    seen: set[str] = set()
    for raw_line in body.splitlines():
        # strip trailing comments before matching
        line = raw_line.split("#", 1)[0].split("//", 1)[0].strip()
        if not line:
            continue
        m = _QID_RE.search(line)
        if not m:
            continue
        qid = "Q" + m.group(1)
        if qid in seen:
            continue
        seen.add(qid)
        qids.append(qid)
        if len(qids) >= max_entities:
            break
    return qids
