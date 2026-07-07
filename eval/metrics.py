"""Reward / evaluation metrics — THE single source of truth.

Imported by the training workstream (reward) and by eval/run_eval.py.
Do not change the public signatures below; they are part of the workstream
contract (DESIGN.md):

    ndcg_two_tier(predicted, answer, bridge, k=50) -> float
    recall_at_k(predicted, answer, k=50) -> float
    f1_set(predicted, answer) -> float

Two-tier NDCG (SID-1 recipe):
  * gain 2  — entity in the answer set
  * gain 1  — entity in the bridge/evidence set (answer membership wins when
              an entity appears in both)
  * gain 0  — anything else
  * log2 positional discount: gain_i / log2(i + 2) for 0-indexed position i
  * predicted list is deduplicated preserving first occurrence, then
    truncated at k
  * ideal DCG is computed from the golden sets (all answers first, then
    bridges), truncated at k
  * empty golden (answer and bridge both empty) -> 0.0

Known property (covered by tests): appending junk entities *after* all golden
entities does not lower NDCG at all — junk has gain 0 and displaces nothing.
The anti-dumping pressure comes from the 50-entity answer cap (junk displaces
golden entities the model could have listed) and from set-F1 / precision
metrics reported alongside.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

__all__ = ["ndcg_two_tier", "recall_at_k", "f1_set"]

GAIN_ANSWER = 2.0
GAIN_BRIDGE = 1.0


def _dedup_truncate(predicted: Sequence[str], k: int) -> list[str]:
    """Deduplicate preserving first occurrence; truncate to k items."""
    seen: set[str] = set()
    out: list[str] = []
    for qid in predicted:
        if qid in seen:
            continue
        seen.add(qid)
        out.append(qid)
        if len(out) >= k:
            break
    return out


def _dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_two_tier(
    predicted: list[str],
    answer: set[str],
    bridge: set[str],
    k: int = 50,
) -> float:
    """Two-tier NDCG@k with gains {answer: 2, bridge: 1, other: 0}.

    ``predicted`` is deduplicated (first occurrence wins) and truncated at k.
    Ideal DCG comes from the golden sets truncated at k. Returns 0.0 when the
    golden sets are both empty, and 0.0 for an empty prediction.
    """
    if k <= 0:
        return 0.0
    answer = set(answer)
    bridge = set(bridge)
    n_answer = len(answer)
    n_bridge_only = len(bridge - answer)
    if n_answer == 0 and n_bridge_only == 0:
        return 0.0

    ideal_gains = [GAIN_ANSWER] * n_answer + [GAIN_BRIDGE] * n_bridge_only
    idcg = _dcg(ideal_gains[:k])
    if idcg <= 0.0:
        return 0.0

    ranked = _dedup_truncate(predicted, k)
    gains = (
        GAIN_ANSWER if qid in answer else GAIN_BRIDGE if qid in bridge else 0.0
        for qid in ranked
    )
    return _dcg(gains) / idcg


def recall_at_k(predicted: list[str], answer: set[str], k: int = 50) -> float:
    """Fraction of answer entities present in the (deduped) top-k prediction.

    Returns 0.0 when the answer set is empty.
    """
    answer = set(answer)
    if not answer or k <= 0:
        return 0.0
    top = _dedup_truncate(predicted, k)
    return len(answer.intersection(top)) / len(answer)


def f1_set(predicted: list[str], answer: set[str]) -> float:
    """Set-based F1 between the full predicted list (as a set) and the answer set.

    Duplicates in ``predicted`` are ignored (set semantics). Returns 0.0 when
    either side is empty or there is no overlap.
    """
    pred = set(predicted)
    answer = set(answer)
    if not pred or not answer:
        return 0.0
    inter = len(pred & answer)
    if inter == 0:
        return 0.0
    precision = inter / len(pred)
    recall = inter / len(answer)
    return 2 * precision * recall / (precision + recall)
