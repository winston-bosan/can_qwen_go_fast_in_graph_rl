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
                    bonus - dump penalty - optional length penalty
data.py             data/questions/*.jsonl -> verl parquet; hash split; curriculum
configs/
  validate_0p6b.yaml  Qwen3-0.6B, group 8, batch 32, 8k ctx, 1-2 GPUs (PRIMARY)
  validate_0p8b.yaml  Qwen3.5-0.8B variant -- PREPARED, NOT CLEARED (see below)
  main_4b.yaml        Qwen3-4B, group 16, batch 64, 16k ctx, 8xH100 (B300 notes inside)
  tools_ecs.yaml      verl tool registry (schemas mirror toolserver/schema.py)
  reward.yaml         two_tier vs recall, format bonus, w_dump, length penalty
ALTERNATIVES.md     assessment: Unsloth+ART LoRA validation path on the local 3060
rollout_smoke.py    trainer-free end-to-end check (HF model + real tool loop + reward)
launch/
  setup_remote.sh          generic GPU box: venv + verl[sglang]==0.8.0 + dataset
  setup_runpod.sh          RunPod pod: native qdrant/neo4j/toolserver (no docker!)
  make_data_tarball.sh     run LOCALLY: data_bundle.tar.zst + manifest + checksums
  loadtest_toolserver.py   verify toolserver QPS on the pod before training
  run_validate.sh / run_main.sh / common.sh   staged-length GRPO launches
tests/              pytest suite; reference_metrics.py is a TEST-ONLY NDCG mirror
```

## Quickstart (local, no GPU rental)

```bash
pytest training/tests -q                          # 45 tests
python -m training.rollout_smoke --mock-tools     # no tool server needed
python -m training.rollout_smoke                  # against live :7801
python -m training.data                           # data/questions/*.jsonl -> data/rl/*.parquet
```

`rollout_smoke.py` runs Qwen3-0.6B (fits the 3060 in fp16, ~17s/episode) through
the same schemas / hermes tool-call format / answer parser / reward the verl run
uses. Verified passing 2026-07-07: 2 assistant turns, 5 tool calls, reward
0.8562 (NDCG 0.7602 + 0.1 format − 0.004 dump penalty for 1 junk entity) on the
built-in sample question.

## Reward (env/reward.py + configs/reward.yaml)

`total = max(0, task + format_bonus − dump_penalty − length_penalty)`

- parse final assistant message with `ecs.answer.parse_entities` (single source
  of truth; ≤50 QIDs)
- **unparseable → hard 0.0 total** (no format bonus, no partial credit; no knob
  changes this)
- task score: `eval.metrics.ndcg_two_tier(predicted, answer, bridge, k=50)`
  (single source of truth; `metric: recall` in reward.yaml switches to
  `recall_at_k` for ablations)
- `+0.1` format bonus for a well-formed block
- **dump penalty** `w_dump * junk_count / k` (default `w_dump: 0.2`, `0`
  disables): junk = predicted entities (after dedup/truncation at k=50) in
  neither the answer nor the bridge set. Rationale — MEASURED by workstream B
  (`eval/tests/test_metrics.py`, DESIGN.md amendment): junk appended *after*
  correct entities costs exactly 0.0 two-tier NDCG (40 junk after 5 correct:
  1.0 → 1.0; the same junk *before*: 0.3107), so the metric alone exerts no
  anti-dumping pressure. Calibration: a fully-dumped 50-line list with 5
  correct costs 0.2·45/50 = **0.18** — meaningful, not dominant. This is a
  *training-only shaping term*; `eval/metrics.py` stays pure and eval reports
  F1 alongside NDCG to expose dumping.
- length penalty knob (`length_penalty.enabled`, default **off**) —
  `coef * max(0, response_tokens - target)`; SID-1 controls length via token
  scheduling instead, keep this off unless ablating

## Qwen3.5 option (prepared, NOT cleared — 0.6B stays primary)

`configs/validate_0p8b.yaml` preps a switch of the validation policy to
**Qwen/Qwen3.5-0.8B** (March 2026: 24 layers, 256K context, agentic-tuned,
thinking + non-thinking). **Compat verdict (checked 2026-07-07): not clean
under verl 0.8 + sglang today** — run a 1-step pod smoke before committing to
it. Findings:

1. **Tool-call format (the blocker)**: Qwen3.5 does *not* emit hermes-style
   `<tool_call>{json}</tool_call>`; official vLLM/SGLang serving guidance is
   `--tool-call-parser qwen3_coder` (the XML-ish `<function=...>` format).
   verl 0.8's `multi_turn.format` defaults to `hermes` and its docs list
   "hermes, llama3_json, ..." — whether `format: qwen3_coder` passes through
   to sglang's FunctionCallParser inside the *rollout* is **unverified**.
   Worse, upstream reports the family's tool-call emission is unstable without
   `tool_choice` forcing (verl#6223), which an RL rollout cannot apply. The
   0p8b config sets `format: qwen3_coder`; if verl rejects it, the model is
   blocked (do NOT fall back to hermes — it would silently strip tool calls
   and train on broken trajectories).
2. **Thinking default**: Qwen3.5 thinks *by default* (Qwen3-0.6B in our smoke
   was templated with `enable_thinking=False` explicitly). Decision: train
   **non-thinking** for rollout-length control — thinking tokens would consume
   the SID-1 length-scheduled response budget. Enforced via
   `data.apply_chat_template_kwargs.enable_thinking: false` in the config
   (same kwarg the smoke test uses).
3. **Stack versions**: checkpoints ship the multimodal-flavored
   `Qwen3_5ForConditionalGeneration` class → needs a transformers version with
   the Qwen3.5 arch (was git-main at release), and sglang with Qwen3.5 support
   (verl 0.8.0 pins `sglang==0.5.12`; sglang documents Qwen3.5 but the pinned
   version is unverified). Known open issues for Qwen3.5 *RL training*:
   verl#6549 (CUDA illegal memory access in `torch_chunk_gated_delta_rule`,
   vLLM+FSDP2, 9B/27B) and vllm#36275 (no text-only class → weight-name
   mismatch in colocated GRPO). verl 0.8 release notes claim Qwen3.5 support
   for **Megatron and VeOmni only** — our path (FSDP + sglang) is not on that
   list. The 0.8B dense tier may dodge the GDN-kernel issues (reports conflict
   on whether small tiers carry linear-attention layers), but that is exactly
   what the pod smoke must prove.

**Clearance procedure** (cheap, ~30 min on the validation pod): bare
`sglang.launch_server --model-path <Qwen3.5 model>` + one tool-call request;
then a 1-step run (`trainer.total_training_steps=1`) and check (a) no config
rejection on `format`, (b) rollout log shows parsed tool calls, (c)
`tokenization_sanity_check_mode: strict` stays quiet. Until then:
**Qwen3-0.6B primary**.

**CLEARANCE RUN 2026-07-08 (Qwen3.5-2B, vast A100): FAILED at arch load** —
the blocker is the pinned stack, not the tool-call path:

- `sglang==0.5.8` (verl 0.8.0's exact `sglang`-extra pin) has no `qwen3_5`
  model module (`srt/models/` tops out at qwen3 / qwen3_next / qwen3_vl);
  bare `launch_server` dies with `KeyError: 'qwen3_5'`.
- `transformers==4.57.1` (resolved by verl 0.8.0) rejects the checkpoint —
  `ValueError: ... model type 'qwen3_5' but Transformers does not recognize
  this architecture` — so the FSDP actor side is equally blocked, and
  sglang's generic `TransformersForCausalLM` fallback dead-ends on the same
  error.
- **Positives confirmed while checking** (reusable for any future Qwen3.5/3.6
  attempt): verl 0.8's agent-loop ToolParser registry DOES include
  `qwen3_coder` (`Qwen3XMLToolParser` in
  `verl/experimental/agent_loop/tool_parser.py`); the Qwen3.5 chat template
  DOES emit that format (`<function=` markers) and honors `enable_thinking`;
  and the agent loop applies `data.apply_chat_template_kwargs` on **every**
  turn, so non-thinking mode holds across whole trajectories.
- Unblocking = sglang ≥ the release that added `qwen3_5` + transformers ≥5.x
  + re-verifying torch/flash-attn — i.e. leaving verl 0.8.0's tested matrix
  and re-running the training smoke. Zero-compat-risk fallback:
  **Qwen3-1.7B** (hermes-native sibling) — user's call.

### Qwen3.5-4B for the MAIN run? (finding only — decision is the user's)

Qwen3.5 ships a 4B tier (`Qwen/Qwen3.5-4B`, plus 2B/9B siblings; the
0.8B–27B tiers are dense, MoE starts at 35B-A3B).

**Pro**: newer post-training with explicit agentic/tool-use tuning (that is our
exact task shape); 256K native context (Qwen3-4B needs YaRN past 32K — moot at
our 16k, but headroom is free); family consistency with a 0.8B validation run
(same template, parser, and gotchas validated once); likely better zero-shot
tool-calling from step 0, which matters for GRPO-without-SFT where early
reward signal comes from format-lucky rollouts.

**Con**: every blocker above, amplified by a longer run (tool-parser
pass-through unverified in verl's FSDP+sglang path; open illegal-memory and
weight-name issues filed specifically against Qwen3.5 RL training; thinking
default needs template-kwarg hygiene everywhere); recommended serving sampling
(`presence_penalty=2.0`) diverges from GRPO-neutral sampling — unclear how the
model behaves at plain temp=1.0 over 9-turn trajectories; far less community
RL precedent (mid-2026 RL papers still overwhelmingly train Qwen3/Qwen2.5);
switching mid-project re-baselines everything (0.6B validation results stop
predicting 4B main-run behavior across a family boundary).

**Suggested path**: keep `main_4b.yaml` on Qwen3-4B; if the 0.8B clearance
smoke passes cleanly AND the Qwen3.5-0.8B validation run beats the Qwen3-0.6B
run on val NDCG at equal steps, clone `main_4b.yaml` for Qwen3.5-4B then.

## SID-1 gotchas → where they are handled

| SID-1 item | Where |
|---|---|
| GRPO without SFT | configs: `adv_estimator: grpo`, `use_kl_loss: false` (no SFT anchor) |
| NDCG reward | env/reward.py → eval/metrics.py `ndcg_two_tier` |
| Format reward | reward.yaml `format_bonus: 0.1` + hard-0 gate on unparseable |
| Entity dumping (NDCG blind spot) | reward.yaml `w_dump: 0.2` → `w_dump·junk/k` training-only penalty (metric itself stays pure in eval/metrics.py) |
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
  toolserver :7801  <- uvicorn toolserver.app:app --workers 4
  neo4j + qdrant    <- NATIVE binaries via setup_runpod.sh (RunPod pods are
                       containers; docker-in-docker is NOT available, so
                       docker-compose cannot run inside a pod)
  embedder (harrier-270m, 640-dim) shares a GPU with rollout or runs on CPU
```

Sync: ship `data_bundle.tar.zst` from `training/launch/make_data_tarball.sh`
(~20–30GB: qdrant 4.8M×640-dim on-disk vectors ≈ 12GB + HNSW, neo4j ≈ 4–10GB,
sidecar.db ≈ 5GB). Full RunPod walkthrough in the next section; a generic
(non-RunPod) box with working docker can instead use docker-compose +
`training/launch/setup_remote.sh`.

**Fallback: tunnel to this dev box** (`ssh -R 7801:localhost:7801 box`,
`ECS_TOOLSERVER_URL=http://localhost:7801`). Do the math before choosing this:

- Tool-call volume: validate = 32 q × 8 rollouts × ~6 calls ≈ **1.5k calls/step**;
  main = 64 × 16 × ~6 ≈ **6k calls/step** (the oft-quoted 512-rollout step is ~3k).
- Latency: calls within one trajectory are *serial* → ~6 × (100–150ms tunnel RTT
  + server time) ≈ +1s wall per trajectory; trajectories run concurrently, so
  RTT alone costs only ~1–2s/step. **The real risk is dev-box throughput**: the
  3060-backed `vector_search` (harrier-270m embed) sustains ~30–60 QPS, so 6k
  calls/step ≈ **2–4 min/step added**, serialized behind one tunnel — likely
  dominating step time. Bandwidth is irrelevant (6k × ~4KB ≈ 25MB/step).
- Tunnel flaps surface as `ERROR:` tool outputs → reward noise. Colocate for the
  main run; the tunnel is acceptable for short validation runs only.

## RunPod deployment

### Recommended pod spec

| item | validation (0.6B) | main (4B) |
|---|---|---|
| GPU | 1× H100 80GB / A100 80GB (or 1× B300) | 8× H100 SXM **or** 1× B300 288GB |
| Image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` | same |
| Volume (`/workspace`) | ≥ 150GB | ≥ 250GB |
| Container disk | ≥ 50GB | ≥ 50GB |
| Ports | TCP 22 (SSH) only — all ECS services bind 127.0.0.1 | same |

- **Image rationale**: a `-devel` image is required (nvcc for the best-effort
  flash-attn build); CUDA toolkit 12.8.1 matches the cu128 wheels of
  torch 2.9.1 that `verl[sglang]==0.8.0` pins. The image's preinstalled
  torch 2.8 is *never used* — `setup_remote.sh` builds its own venv. Do not
  pip-install into the system python (the RunPod 2.8 template has a known
  torch-fallback packaging bug, runpod/containers#114; our venv sidesteps it).
- **Disk math**: data bundle ~15–20GB compressed + ~25GB expanded (640-dim
  index), HF weights (0.6B ≈ 1.5GB, 4B ≈ 8GB), FSDP checkpoints (4B: ~30GB per save incl.
  optimizer state — keep `save_freq` modest), wandb/logs. Put repo, venv and
  data on the `/workspace` volume: RunPod wipes the container layer on
  restart but keeps the volume, and `setup_runpod.sh` is idempotent on top.
- **Env vars**: none needed at provision time; all `ECS_*` default to
  localhost. `WANDB_API_KEY` goes into `training/launch/env.sh` after setup.

### Exact sequence

```bash
# 0. LOCAL box, AFTER the embedding run completes (script refuses otherwise):
docker compose stop qdrant neo4j            # cold copy required
training/launch/make_data_tarball.sh        # -> data/data_bundle.tar.zst
docker compose start qdrant neo4j
rsync -avP data/data_bundle.tar.zst root@<pod>:/workspace/

# 1. POD: get the repo (volume-persistent path)
cd /workspace && git clone <repo-url> entity_component_search
cd entity_component_search

# 2. services (native qdrant+neo4j+toolserver) + data restore + training venv
training/launch/setup_runpod.sh --restore /workspace/data_bundle.tar.zst
#   add --import-csv to rebuild neo4j from CSVs instead of the binary store
#   (use when neo4j versions diverge from 5.26.x; qdrant must stay 1.18.x)

# 3. health checks (also printed at the end of setup_runpod.sh)
training/launch/setup_runpod.sh --status     # ports + scripts/status.py counts

# 4. verify toolserver throughput BEFORE burning GPU-hours
.venv-train/bin/python training/launch/loadtest_toolserver.py --duration 30 --concurrency 64

# 5. end-to-end env check against the LIVE stack, then train
.venv-train/bin/python -m training.rollout_smoke
training/launch/run_validate.sh              # then: run_main.sh
```

### CUDA / version landmines (checked 2026-07)

1. **torch pin**: `verl[sglang]==0.8.0` pins `torch==2.9.1`; the default PyPI
   wheel is **cu128** → NVIDIA driver ≥ 550 on the host. Check `nvidia-smi`
   right after provisioning; RunPod lets you filter hosts by CUDA version —
   pick 12.8+.
2. **B300 = Blackwell Ultra (sm_103)**: needs cu128+ kernels end-to-end.
   torch 2.9.1 cu128 ships Blackwell (sm_100/sm_120) kernels and runs sm_103
   via PTX forward-compat; if you see "no kernel image" errors, reinstall with
   `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130`
   (still satisfies verl's `==2.9.1` pin). sglang's `sgl-kernel` (pinned by
   verl's sglang extra) has Blackwell builds; verify with a 1-prompt
   `sglang.launch_server` before training. flash-attn may fail to compile for
   sm_103 — it is best-effort in `setup_remote.sh` and training runs without it.
3. **No docker-in-docker on GPU pods** — hence `setup_runpod.sh` runs qdrant
   (standalone binary), neo4j (tarball + headless JDK 21, Java 17 fallback)
   and the toolserver natively under `setsid`/pidfiles
   (`.services/run/*.{pid,log}`).
4. **Binary-restore version pins**: qdrant storage dir needs qdrant **1.18.x**
   on the pod (dev box runs 1.18.2; `QDRANT_VERSION` env) — there is no
   re-import route for qdrant short of re-embedding, so always match. neo4j
   store needs **5.26.x** (`NEO4J_VERSION` env); on any skew use
   `--import-csv` (entities.csv/triples.csv travel in the bundle; ~10–20 min
   for 21M edges + index build, same command as `ingest/load_neo4j.py`).
5. **Toolserver VRAM**: each uvicorn worker (default `TOOLSERVER_WORKERS=4`)
   lazily loads its own harrier-270m embedder ≈ 0.7GB VRAM → ~3GB total. On
   the 8×H100 pod set `TOOLSERVER_CUDA_VISIBLE_DEVICES=0` so the embedders sit
   in GPU0's sglang headroom (`gpu_memory_utilization: 0.55` leaves ~35GB); on
   a single B300 the default is fine.
6. **Embedder/index coupling**: the qdrant index and the query path are both
   harrier-**270m / 640-dim** (canonical since DESIGN.md 2026-07-07, commit
   b82cce7; `src/ecs/config.py` is the source of truth). Never mix embedder
   models between documents and queries. harrier-0.6b (1024-dim) is the
   quality-**upgrade path** and requires a full re-embed of the collection
   (new index dimension) — plan a fresh data bundle if taken.

### Recommendations for workstream A (toolserver — not edited here)

- An `ECS_EMBED_DEVICE` env knob (cuda:N / cpu) for the embedder would replace
  the `CUDA_VISIBLE_DEVICES` workaround above.
- A single shared embedder process (or micro-batching queue across uvicorn
  workers) would cut embedder VRAM from workers× to 1× and raise
  vector_search QPS; today's per-worker lazy load is why `setup_runpod.sh`
  fires warmup requests.

## Sizing & token budgets

**validate_0p6b (1–2 GPUs):** 256 trajectories/step (32×8), context ceiling
8k → ceiling ≈ **2.1M tokens/step**; typical trajectory ≈ 4–5k tokens (~1.2k
prompt+schemas, ~6 tool exchanges, short final) → ≈ **1.0–1.3M tokens/step
processed, ~0.2–0.3M policy-generated**. Trivial for one H100; fits a single
24GB card with `gpu_memory_utilization≈0.4` and micro-batch 2.

### Validation pod tiers (0.6B/0.8B full-FT, colocated sglang+FSDP, ~250 steps)

Trainer-state math for **0.8B full-FT with Adam** (FSDP mixed precision):
bf16 params 1.6GB + fp32 master copy 3.2GB + fp32 Adam m/v 6.4GB + bf16 grads
1.6GB = **12.8GB ≈ 13GB** — the quoted ~13GB is correct (fp32 grad-reduce
variant: ~14.4GB; Qwen3-0.6B: ~9.6GB, same shape ×0.75). On top of that:
activations with gradient checkpointing at 8k ctx ≈ 1–2GB per micro-batch
sequence, sglang engine = `gpu_memory_utilization × VRAM`, plus ~2–3GB
CUDA/NCCL overhead.

| tier | fit | settings (ECS_EXTRA_OVERRIDES) |
|---|---|---|
| **minimum: 1× 48GB** (L40S / RTX 6000 Ada; 16 vCPU / 64GB RAM / 150GB volume) | 13GB trainer + ~19GB sglang @ 0.40 + ~4GB activations + overhead ≈ **40–43GB — fits, tight** | `trainer.n_gpus_per_node=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.40 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2` |
| **sweet spot: 1× 80GB** (H100/A100) | config defaults as-is; headroom for KV at 9-turn trajectories and no accum-step inflation | none (optionally `gpu_memory_utilization=0.55`, micro-batch 8) |

Notes for the 48GB tier: rollout decode is the bottleneck (~2× slower steps vs
80GB from the smaller KV pool); keep `TOOLSERVER_CUDA_VISIBLE_DEVICES=0` (the
270m embedders' ~3GB share the same card — budgeted in the overhead above);
if OOM appears at the 6144-response stage, `actor.fsdp_config.optimizer_offload=true`
buys ~6.4GB for ~15% step-time cost. 16 vCPU / 64GB RAM is comfortably enough
(neo4j 8G heap + 8G pagecache + qdrant + toolserver workers ≈ 24GB RSS).

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
