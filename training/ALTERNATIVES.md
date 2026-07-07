# Alternative validation path: Unsloth + ART (LoRA GRPO on the local 3060)

*Assessment only (2026-07-07) — no implementation. Requested as a possible
bonus validation path that runs locally once the embedding run finishes.*

**What it is.** [OpenPipe ART](https://github.com/OpenPipe/ART) (Agent
Reinforcement Trainer) wraps Unsloth's GRPOTrainer for multi-turn agent RL:
you write a python `rollout(model, scenario) -> art.Trajectory` function that
drives the policy through an OpenAI-compatible endpoint (ART serves the
current LoRA via vLLM/Unsloth), attach a scalar reward, and ART handles GRPO
grouping, LoRA updates, and checkpointing. Actively maintained (supports
Qwen3.x-era models as of mid-2026); outputs LoRA adapters, not full-FT
checkpoints.

## What integration would look like

Small — our environment pieces already exist and are framework-agnostic:

1. **Rollout fn** (~100 LoC): messages = `training/data.py:SYSTEM_PROMPT` +
   question; tools = `training/env/tools.py:get_tool_schemas()` passed as
   OpenAI `tools=`; loop ≤8 turns dispatching tool calls through
   `ToolServerClient` (same error-string semantics); on final message call
   `training/env/reward.py:compute_reward(...)` → `trajectory.reward`. This is
   essentially `rollout_smoke.py:run_episode` rewritten against an OpenAI
   client instead of raw `transformers.generate` — the tool-call parsing we
   hand-rolled there disappears (ART's endpoint does hermes parsing for Qwen3).
2. **Scenario feed** (~30 LoC): iterate `training/data.py:load_questions()` +
   `build_splits()` train records (skip the parquet step).
3. **Config/runner** (~50-100 LoC + yaml): model `Qwen/Qwen3-0.6B`, LoRA r=16-32,
   4-bit base, group size 4-6, lr ~1e-5, W&B logging.

**Effort: ~1.5-2 days.** Day 1: rollout fn + runner working end-to-end against
the live toolserver (mock-tools first, same fixtures). Day 2: 3060 memory
tuning (4-bit base + LoRA + vLLM sleep/wake on 12GB forces small groups and
~4k context; if colocated serving+training thrashes, fall back to ART's
separate-server mode with CPU offload — slower but stable) plus an overnight
run (~50-150 GRPO steps at 3060 throughput). Add ~0.5 day if ART's Unsloth
backend fights the 12GB card on context length.

## What it WOULD validate (cheaply, before/alongside pod rental)

- **Reward shaping in anger**: does the format gate + dump penalty + NDCG
  actually move under optimization pressure (reward climbing, no dump
  exploit, no format collapse)? This is the highest-value signal — it
  de-risks the reward design, which transfers 1:1 to the verl run since both
  import the same `compute_reward` / `eval.metrics`.
- **Question quality / difficulty mix**: reward variance per difficulty tier,
  fraction of zero-reward groups (GRPO gets no gradient from all-zero groups —
  if most groups are flat-zero, questiongen needs easier tiers first).
- **Toolserver under sustained training load** (hours of continuous mixed
  traffic on the local stack; complements `loadtest_toolserver.py`).
- **Prompt wording + tool schema ergonomics** at near-zero cost per iteration.

## What it would NOT validate

- **The verl stack itself** — sglang multi-turn rollout, `format:` parser
  pass-through, TI/TO delta tokenization, FSDP sharding, length-stage
  resume: none of it is exercised. The pod smoke of `validate_0p6b.yaml`
  remains mandatory.
- **Full-FT dynamics**: LoRA-GRPO ≠ full-FT GRPO (SID-1 is full-FT); learning
  rates, KL behavior, and collapse modes don't transfer quantitatively.
- **Scale realism**: 3060 forces ~4k context, group 4-6, batch ~8 — advantage
  estimates are noisier and length scheduling can't be rehearsed; throughput
  ~20-60 trajectories/hour vs ~256/step on the pod.

## Verdict

Worth doing **in parallel** as a reward-design and data-quality testbed once
the embedding run frees the 3060 — it de-risks exactly the parts the verl smoke
test can't (optimization-pressure behavior of the reward), for ~2 days and $0.
It is **not** a substitute for the verl validation run, and its LoRA artifacts
are throwaway (the main run is full-FT under verl).
