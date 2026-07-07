# training/ — GRPO RL environment (workstream C)

SID-1-style GRPO (no SFT) for a multi-turn tool-calling entity-retrieval policy.
Reference: https://www.sid.ai/research/sid-1-technical-report

## Framework decision: **verl** (over SkyRL)

Decided 2026-07-07, against verl 0.8.0 (2026-06-01) and SkyRL main.

verl's multi-turn tool story is first-class and documented **today**:

- Native multi-turn rollout on the sglang backend
  (`actor_rollout_ref.rollout.multi_turn.enable=true`) with hermes-format tool
  calling — exactly what Qwen3 emits — and a config-driven tool registry
  (`tool_config_path` → YAML of `BaseTool` subclasses + OpenAI-style schemas).
  Docs: https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html
- Delta-based tokenization of each new message with a
  `tokenization_sanity_check_mode` guard — i.e. the rollout engine's token ids
  are what the learner trains on (**tokens-in/tokens-out**), which is one of the
  SID-1 gotchas handled at the framework level rather than by us.
- GRPO is a config switch (`algorithm.adv_estimator=grpo`) with explicit
  control over the length-bias knobs (`loss_agg_mode`,
  `norm_adv_by_std_in_grpo`) — see "SID-1 gotchas" below.
- Custom terminal reward is a two-line config (`custom_reward_function.path/name`)
  pointing at a plain Python function → `training/env/reward.py`.
- DESIGN.md already fixes verl; nothing found justifies deviating.

SkyRL (NovaSky) is credible — skyrl-train + skyrl-gym have a clean env
abstraction and its SkyRL-Agent paper targets exactly long-horizon tool agents —
but it is younger, has a smaller install base, its tool integration is
gym-env-shaped (we'd write more glue for hermes tool-calling + TI/TO), and it
buys us nothing verl lacks for this task. Revisit only if we hit a hard verl
blocker (e.g. sglang multi-turn instability at 16k context).

## Layout

```
env/tools.py        adapter: 4 toolserver endpoints -> trainer tool interface
                    (async httpx, timeout, retry-once, errors returned as strings)
env/reward.py       terminal reward: two-tier NDCG (eval/metrics.py) + format
                    bonus + optional length penalty; verl compute_score entry
data.py             data/questions/*.jsonl -> verl parquet; hash split; curriculum
configs/
  validate_0p6b.yaml  Qwen3-0.6B, group 8, batch 32, 8k ctx, 1-2 GPUs
  main_4b.yaml        Qwen3-4B, group 16, batch 64, 16k ctx, 8xH100 (B300 notes inside)
  tools_ecs.yaml      verl tool registry (schemas mirror toolserver/schema.py)
  reward.yaml         two_tier vs recall, format bonus, length-penalty knob
rollout_smoke.py    trainer-free end-to-end check (HF model + real tool loop + reward)
launch/             setup_remote.sh, run_validate.sh, run_main.sh, common.sh
tests/              pytest suite; reference_metrics.py is a TEST-ONLY NDCG mirror
```

## Quickstart (local, no GPU rental)

```bash
pytest training/tests -q                          # 38 tests
python -m training.rollout_smoke --mock-tools     # no tool server needed
python -m training.rollout_smoke                  # against live :7801
python -m training.data                           # data/questions/*.jsonl -> data/rl/*.parquet
```

`rollout_smoke.py` runs Qwen3-0.6B (fits the 3060 in fp16, ~17s/episode) through
the same schemas / hermes tool-call format / answer parser / reward the verl run
uses. Verified passing 2026-07-07: 2 assistant turns, 5 tool calls, reward
0.860 (NDCG 0.760 + 0.1 format) on the built-in sample question.

## Reward (env/reward.py + configs/reward.yaml)

- parse final assistant message with `ecs.answer.parse_entities` (single source
  of truth; ≤50 QIDs)
- **unparseable → hard 0.0 total** (no format bonus, no partial credit)
- task score: `eval.metrics.ndcg_two_tier(predicted, answer, bridge, k=50)`
  (single source of truth; `metric: recall` in reward.yaml switches to
  `recall_at_k` for ablations)
- `+0.1` format bonus for a well-formed block
- length penalty knob (`length_penalty.enabled`, default **off**) —
  `coef * max(0, response_tokens - target)`; SID-1 controls length via token
  scheduling instead, keep this off unless ablating

## SID-1 gotchas → where they are handled

| SID-1 item | Where |
|---|---|
| GRPO without SFT | configs: `adv_estimator: grpo`, `use_kl_loss: false` (no SFT anchor) |
| NDCG reward | env/reward.py → eval/metrics.py `ndcg_two_tier` |
| Format reward | reward.yaml `format_bonus: 0.1` + hard-0 gate on unparseable |
| Length scheduling | staged runs in launch/run_*.sh (`LENGTH_SCHEDULE`), resume via `trainer.resume_mode: auto`; verl can't ramp `max_response_length` in-run |
| Tokens-in/tokens-out | verl sglang delta tokenization + `multi_turn.tokenization_sanity_check_mode: strict` |
| GRPO length-norm caveat | `loss_agg_mode: token-mean` (batch-token mean, not per-seq mean); `norm_adv_by_std_in_grpo` documented in configs — flip to `false` (Dr. GRPO) if length collapse appears |
| Tool errors mid-rollout | env/tools.py returns `"ERROR: ..."` strings (retry-once first), never raises into the rollout |

## Length scheduling

`run_validate.sh` / `run_main.sh` chain verl runs with increasing
`data.max_response_length`, each stage resuming the previous checkpoint:

```bash
LENGTH_SCHEDULE="4096:60 8192:140 12288" ./training/launch/run_main.sh
# "<max_response_tokens>:<cumulative global step>"; last stage may omit steps
```

## Remote topology

**Preferred: everything on the training box.**

```
[training box: 8xH100 or B300]
  verl (sglang rollout + FSDP actor)
  toolserver :7801  <- uvicorn toolserver.app:app
  neo4j + qdrant    <- docker compose up -d   (rsync data/neo4j, data/qdrant)
  embedder (harrier-0.6b) shares a GPU with rollout or runs on CPU
```

Sync: `rsync -av --exclude .venv data/neo4j data/qdrant data/questions box:.../data/`
(~30–40GB: qdrant 4.8M×1024-dim on-disk vectors ≈ 20GB + HNSW, neo4j ≈ 5–10GB).
Then `training/launch/setup_remote.sh` (idempotent), edit `training/launch/env.sh`.

**Fallback: tunnel to this dev box** (`ssh -R 7801:localhost:7801 box`,
`ECS_TOOLSERVER_URL=http://localhost:7801`). Do the math before choosing this:

- Tool-call volume: validate = 32 q × 8 rollouts × ~6 calls ≈ **1.5k calls/step**;
  main = 64 × 16 × ~6 ≈ **6k calls/step** (the oft-quoted 512-rollout step is ~3k).
- Latency: calls within one trajectory are *serial* → ~6 × (100–150ms tunnel RTT
  + server time) ≈ +1s wall per trajectory; trajectories run concurrently, so
  RTT alone costs only ~1–2s/step. **The real risk is dev-box throughput**: the
  3060-backed `vector_search` (harrier embed) sustains ~20–30 QPS, so 6k
  calls/step ≈ **3–5 min/step added**, serialized behind one tunnel — likely
  dominating step time. Bandwidth is irrelevant (6k × ~4KB ≈ 25MB/step).
- Tunnel flaps surface as `ERROR:` tool outputs → reward noise. Colocate for the
  main run; the tunnel is acceptable for short validation runs only.

## Sizing & token budgets

**validate_0p6b (1–2 GPUs):** 256 trajectories/step (32×8), context ceiling
8k → ceiling ≈ **2.1M tokens/step**; typical trajectory ≈ 4–5k tokens (~1.2k
prompt+schemas, ~6 tool exchanges, short final) → ≈ **1.0–1.3M tokens/step
processed, ~0.2–0.3M policy-generated**. Trivial for one H100; fits a single
24GB card with `gpu_memory_utilization≈0.4` and micro-batch 2.

**main_4b (8xH100):** 1024 trajectories/step (64×16), ceiling 16k →
**≈16.8M tokens/step ceiling**; typical ≈ 6–8k/traj → **6–8M tokens/step
processed, ~1.5–3M generated**. At ~20–30k decode tok/s aggregate for a 4B on
8xH100 (sglang, tp=2, colocated at 0.55 mem util) expect rollout ≈ 1.5–3 min +
update ≈ 1 min → **~3–5 min/step**; tool calls (6k/step against a local
toolserver at ≥200 QPS) add <1 min.

**Single B300 (288GB) re-size** (documented in main_4b.yaml header, runnable via
`ECS_EXTRA_OVERRIDES`, colocated sglang+FSDP on the one device):
`trainer.n_gpus_per_node=1`, `tensor_model_parallel_size=1`,
`gpu_memory_utilization=0.45` (~130GB for rollout weights+KV; 4B actor +
grads + Adam ≈ 64GB + activations leave ample headroom),
`ppo_micro_batch_size_per_gpu=16`. Same batch/group hyperparameters (gradient
accumulation absorbs the difference) → identical learning dynamics, ~3–4×
wall-clock per step vs the 8xH100 node.

## Data

`python -m training.data --questions-dir data/questions --out-dir data/rl \
   --val-frac 0.05 --curriculum off|sorted|mixed`

- split: sha256(id) → stable train/val membership across regenerations
- `sorted` = easy→hard (pair with `data.shuffle=false`), `mixed` =
  difficulty-stratified round-robin, `off` = seeded shuffle
- output rows carry `reward_model.ground_truth` =
  `{"answer_qids": [...], "bridge_qids": [...]}` (JSON) consumed by
  `env/reward.py:compute_score`, and `extra_info.tools_kwargs` for verl's
  multi-turn tool rollout.
