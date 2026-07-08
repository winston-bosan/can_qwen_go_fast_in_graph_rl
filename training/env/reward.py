"""Terminal reward for GRPO: two-tier entity-NDCG + format reward + shaping penalties.

Follows SID-1's reward shape:
  - task reward   : ndcg_two_tier(predicted, answer, bridge, k=50)  (or plain recall,
                    selectable in the yaml config -- useful for ablations)
  - format reward : +0.1 if the final message contains a well-formed ```entities block
  - hard zero     : unparseable final message -> total reward 0.0 (no partial credit;
                    this is the strongest lever to make the format stick without SFT)
  - dump penalty  : w_dump * junk_count / k. MEASURED (eval/tests/test_metrics.py,
                    DESIGN.md reward bullet as amended): junk appended AFTER correct
                    entities costs exactly 0.0 two-tier NDCG, so the metric alone has
                    NO anti-dumping pressure. This is a TRAINING-ONLY shaping term
                    (eval/metrics.py stays pure); junk = predicted entities (after
                    dedup/truncation at k) in neither the answer nor the bridge set.
                    Default w_dump=0.2 -> a fully-dumped 50-line list with 5 correct
                    costs 0.2*45/50 = 0.18: meaningful, not dominant. w_dump: 0 disables.
  - length penalty: mild, per-token over a budget, DEFAULT OFF (SID-1 controls length
                    with max-token scheduling, not the reward; knob kept for ablations)

Total = max(0, task + format_bonus - dump_penalty - length_penalty); hard 0 on
unparseable output is unaffected by any knob.

Single sources of truth (this module implements NO metric of its own):
  - entity parsing : ``ecs.answer.parse_entities``  (src/ecs/, workstream A)
  - NDCG           : ``eval.metrics.ndcg_two_tier`` (eval/, workstream B)

Both are resolved lazily with clearly-marked fallbacks so the smoke test runs
before the sibling workstreams land; ``eval.metrics`` / ``ecs.answer`` always
win when importable (tests assert this).

verl entry point: ``compute_score`` matches verl's ``custom_reward_function``
signature (see configs/*.yaml -> custom_reward_function.path/name).
"""

import json
import logging
import os
import re
import sys
import types
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml

import training  # noqa: F401  (sys.path setup)

# verl's custom_reward_function loader execs this file as a module WITHOUT
# registering it in sys.modules. Python 3.12's @dataclass then crashes in
# dataclasses._is_type (sys.modules.get(cls.__module__).__dict__ -> None) when
# annotations are strings. We therefore avoid `from __future__ import
# annotations` here AND self-register a shim so later introspection always
# finds a live module object.
if __name__ not in sys.modules:  # pragma: no cover - only under verl's loader
    _shim = types.ModuleType(__name__)
    _shim.__dict__.update(globals())
    sys.modules[__name__] = _shim

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
DEFAULT_REWARD_CONFIG_PATH = os.path.abspath(os.path.join(_CONFIG_DIR, "reward.yaml"))

MAX_ENTITIES = 50
try:
    from ecs.config import MAX_ANSWER_ENTITIES as MAX_ENTITIES  # type: ignore # noqa: F811
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class LengthPenaltyConfig:
    enabled: bool = False
    # penalty = coef * max(0, response_tokens - target_tokens); clamped so the
    # total reward never goes below 0 for a parseable answer.
    coef: float = 0.0001
    target_tokens: int = 3072


@dataclass
class RewardConfig:
    metric: str = "two_tier"  # "two_tier" (NDCG) | "recall" (answer-set recall@k)
    k: int = 50
    format_bonus: float = 0.1
    # Over-reporting (dump) penalty weight: reward -= w_dump * junk_count / k,
    # junk = predicted entities (post dedup/truncation at k) in neither the
    # answer nor the bridge set. 0 disables. See module docstring / DESIGN.md.
    w_dump: float = 0.2
    length_penalty: LengthPenaltyConfig = field(default_factory=LengthPenaltyConfig)

    @classmethod
    def load(cls, path: str | None = None) -> "RewardConfig":
        path = path or os.environ.get("ECS_REWARD_CONFIG", DEFAULT_REWARD_CONFIG_PATH)
        if not os.path.exists(path):
            logger.warning("reward config %s not found; using defaults", path)
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        lp = LengthPenaltyConfig(**(raw.pop("length_penalty", None) or {}))
        known = {k: v for k, v in raw.items() if k in {"metric", "k", "format_bonus", "w_dump"}}
        cfg = cls(length_penalty=lp, **known)
        if cfg.metric not in ("two_tier", "recall"):
            raise ValueError(f"reward metric must be 'two_tier' or 'recall', got {cfg.metric!r}")
        return cfg


_default_config: RewardConfig | None = None


def get_config() -> RewardConfig:
    global _default_config
    if _default_config is None:
        _default_config = RewardConfig.load()
    return _default_config


# --------------------------------------------------------------------------
# Lazy resolution of parse_entities / ndcg_two_tier (prefer canonical modules)
# --------------------------------------------------------------------------

_ENTITIES_BLOCK_RE = re.compile(r"```entities\s*\n(.*?)```", re.DOTALL)
_QID_RE = re.compile(r"^Q\d+$")


def _fallback_parse_entities(text: str) -> list[str]:
    """FALLBACK ONLY -- minimal mirror of the DESIGN.md answer format, used
    until ``src/ecs/answer.py`` (workstream A) lands. Last fenced ```entities
    block wins; one QID per line; trailing ``# comment`` stripped; order kept;
    duplicates dropped; invalid lines ignored. Returns [] when unparseable."""
    blocks = _ENTITIES_BLOCK_RE.findall(text or "")
    if not blocks:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in blocks[-1].splitlines():
        qid = line.split("#", 1)[0].strip()
        if _QID_RE.match(qid) and qid not in seen:
            seen.add(qid)
            out.append(qid)
    return out


_parse_entities: Callable[[str], list[str]] | None = None
_ndcg_two_tier: Callable[..., float] | None = None


def _resolve_parse_entities() -> Callable[[str], list[str]]:
    global _parse_entities
    if _parse_entities is None:
        try:
            from ecs.answer import parse_entities  # canonical (workstream A)

            _parse_entities = parse_entities
        except ImportError:
            warnings.warn(
                "ecs.answer not importable yet -- using training.env.reward's fallback "
                "parser. src/ecs/answer.py is the single source of truth once it lands.",
                RuntimeWarning,
                stacklevel=2,
            )
            _parse_entities = _fallback_parse_entities
    return _parse_entities


def _resolve_ndcg() -> Callable[..., float]:
    global _ndcg_two_tier
    if _ndcg_two_tier is None:
        try:
            from eval.metrics import ndcg_two_tier  # canonical (workstream B)

            _ndcg_two_tier = ndcg_two_tier
        except ImportError:
            warnings.warn(
                "eval.metrics not importable yet -- falling back to the TEST-ONLY "
                "reference copy in training/tests/reference_metrics.py. Do NOT train "
                "against this fallback; eval/metrics.py is the single source of truth.",
                RuntimeWarning,
                stacklevel=2,
            )
            from training.tests.reference_metrics import ndcg_two_tier

            _ndcg_two_tier = ndcg_two_tier
    return _ndcg_two_tier


def _recall_fallback(predicted: list[str], answer: set[str], k: int = 50) -> float:
    if not answer:
        return 0.0
    top = list(dict.fromkeys(predicted))[:k]
    return len(answer.intersection(top)) / len(answer)


def _resolve_recall() -> Callable[..., float]:
    global _recall_at_k
    if _recall_at_k is None:
        try:
            from eval.metrics import recall_at_k  # canonical (workstream B)

            _recall_at_k = recall_at_k
        except ImportError:
            _recall_at_k = _recall_fallback
    return _recall_at_k


_recall_at_k: Callable[..., float] | None = None


def _reset_resolution_cache() -> None:
    """Test hook: force re-resolution of parse_entities / metric functions."""
    global _parse_entities, _ndcg_two_tier, _recall_at_k
    _parse_entities = None
    _ndcg_two_tier = None
    _recall_at_k = None


# --------------------------------------------------------------------------
# Reward computation
# --------------------------------------------------------------------------


@dataclass
class RewardResult:
    total: float
    task_score: float = 0.0
    format_bonus: float = 0.0
    dump_penalty: float = 0.0
    junk_count: int = 0
    length_penalty: float = 0.0
    parsed: bool = False
    n_entities: int = 0
    truncated: bool = False


def compute_reward(
    solution_str: str,
    answer_qids: list[str] | set[str],
    bridge_qids: list[str] | set[str] | None = None,
    response_tokens: int | None = None,
    config: RewardConfig | None = None,
) -> RewardResult:
    cfg = config or get_config()
    try:
        predicted = _resolve_parse_entities()(solution_str or "")
    except Exception:
        # A reward function must never crash a training step; a parser
        # exception on adversarial model output counts as unparseable.
        logger.warning("parse_entities raised; treating output as unparseable", exc_info=True)
        predicted = []
    if not predicted:
        # Unparseable / empty answer block -> hard zero, including format bonus.
        return RewardResult(total=0.0, parsed=False)

    truncated = len(predicted) > MAX_ENTITIES
    predicted = predicted[:MAX_ENTITIES]

    answer = set(answer_qids or ())
    bridge = set(bridge_qids or ()) - answer

    if cfg.metric == "recall":
        task = _resolve_recall()(predicted, answer, k=cfg.k)
    else:
        task = _resolve_ndcg()(predicted, answer, bridge, k=cfg.k)

    fmt = cfg.format_bonus

    # Over-reporting penalty: two-tier NDCG is provably indifferent to junk
    # appended after the golden entities (see module docstring); charge it here.
    # `predicted` is already deduped (parser contract) and truncated at k<=50.
    golden = answer | bridge
    junk_count = sum(1 for q in predicted if q not in golden)
    dump_penalty = cfg.w_dump * junk_count / cfg.k if cfg.w_dump else 0.0

    length_penalty = 0.0
    if cfg.length_penalty.enabled and response_tokens is not None:
        over = max(0, response_tokens - cfg.length_penalty.target_tokens)
        length_penalty = cfg.length_penalty.coef * over

    total = max(0.0, task + fmt - dump_penalty - length_penalty)
    return RewardResult(
        total=total,
        task_score=task,
        format_bonus=fmt,
        dump_penalty=dump_penalty,
        junk_count=junk_count,
        length_penalty=length_penalty,
        parsed=True,
        n_entities=len(predicted),
        truncated=truncated,
    )


def _coerce_ground_truth(ground_truth: Any) -> tuple[list[str], list[str]]:
    """Accept dict / JSON string / bare list forms of the golden sets."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return [], []
    if isinstance(ground_truth, dict):
        return list(ground_truth.get("answer_qids") or []), list(
            ground_truth.get("bridge_qids") or []
        )
    if isinstance(ground_truth, (list, tuple)):
        return list(ground_truth), []
    return [], []


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> float:
    """verl ``custom_reward_function`` entry point.

    ``ground_truth`` carries ``{"answer_qids": [...], "bridge_qids": [...]}``
    (dict or JSON string -- parquet round-trips it as a string), as written by
    ``training/data.py``. ``extra_info["response_tokens"]`` enables the length
    penalty when the knob is on.
    """
    answer_qids, bridge_qids = _coerce_ground_truth(ground_truth)
    if isinstance(extra_info, dict):
        response_tokens = extra_info.get("response_tokens")
        if not bridge_qids:
            bridge_qids = list(extra_info.get("bridge_qids") or [])
    else:
        response_tokens = None
    return compute_reward(
        solution_str,
        answer_qids,
        bridge_qids,
        response_tokens=response_tokens,
    ).total
