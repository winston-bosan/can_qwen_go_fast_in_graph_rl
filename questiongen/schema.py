"""Pydantic model for the question JSONL record (DESIGN.md contract).

One JSON object per line:

    {id, question, answer_qids: [...], bridge_qids: [...],
     source: "kg_pattern"|"sim_link"|"frames", cypher?, difficulty}

`difficulty` is the hop count of the underlying pattern (for FRAMES, a
documented proxy — see eval/frames.py).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Source = Literal["kg_pattern", "sim_link", "frames"]

_QID_RE = re.compile(r"^Q\d+$")


class QuestionRecord(BaseModel):
    """A single generated/evaluated question with its exact golden sets."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer_qids: list[str]
    bridge_qids: list[str] = Field(default_factory=list)
    source: Source
    cypher: str | None = None
    difficulty: int = Field(ge=1, description="hop count of the pattern")

    @field_validator("answer_qids", "bridge_qids")
    @classmethod
    def _valid_qids(cls, v: list[str]) -> list[str]:
        for qid in v:
            if not _QID_RE.match(qid):
                raise ValueError(f"invalid QID: {qid!r} (expected 'Q<digits>')")
        # deduplicate preserving order — golden sets are sets
        seen: set[str] = set()
        out: list[str] = []
        for qid in v:
            if qid not in seen:
                seen.add(qid)
                out.append(qid)
        return out

    def to_jsonl_line(self) -> str:
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "QuestionRecord":
        return cls.model_validate_json(line)


def append_records(path: str, records: list[QuestionRecord]) -> None:
    """Append records to a JSONL file, creating parent dirs as needed."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_jsonl_line() + "\n")


def load_records(path: str) -> list[QuestionRecord]:
    """Read a JSONL question file, skipping blank lines."""
    out: list[QuestionRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(QuestionRecord.from_jsonl_line(line))
    return out
