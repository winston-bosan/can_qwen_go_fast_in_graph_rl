"""TEST-ONLY reference copy of the two-tier NDCG metric.

The single source of truth is ``eval/metrics.py`` (workstream B) with the
agreed signature::

    ndcg_two_tier(predicted: list[str], answer: set[str], bridge: set[str], k: int = 50) -> float

This minimal mirror exists ONLY so training/tests (and the pre-integration
smoke test) can run before eval/metrics.py lands. training/env/reward.py
always prefers ``eval.metrics`` when importable and emits a RuntimeWarning
when it has to fall back to this file. DO NOT extend this file; fix
eval/metrics.py instead.
"""

from __future__ import annotations

import math


def ndcg_two_tier(
    predicted: list[str], answer: set[str], bridge: set[str], k: int = 50
) -> float:
    """Two-tier NDCG@k: gain 2 for answer-set entities, 1 for bridge, 0 else."""
    answer = set(answer)
    bridge = set(bridge) - answer  # answer tier wins on overlap
    ranked = list(predicted)[:k]

    def _gain(qid: str) -> float:
        if qid in answer:
            return 2.0
        if qid in bridge:
            return 1.0
        return 0.0

    dcg = sum(_gain(q) / math.log2(i + 2) for i, q in enumerate(ranked))
    ideal_gains = ([2.0] * len(answer) + [1.0] * len(bridge))[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0
