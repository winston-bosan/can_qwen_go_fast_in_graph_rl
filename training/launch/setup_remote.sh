#!/usr/bin/env bash
# Idempotent setup for a rented training box (RunPod B300 / 8xH100 node).
#
# Assumptions (see training/README.md "Remote topology"):
#   * This repo is present on the box, either by
#       git clone <repo-url> ~/entity_component_search
#     or, from the dev box (excludes gitignored bulk data by default):
#       rsync -av --exclude data/ --exclude .venv/ --exclude checkpoints/ \
#           ~/here_we_go_again/entity_component_search/ box:~/entity_component_search/
#   * Question JSONL is synced separately (small):
#       rsync -av ~/.../data/questions/ box:~/entity_component_search/data/questions/
#   * If the tool server + neo4j + qdrant run ON this box (preferred), also
#     rsync data/neo4j data/qdrant and `docker compose up -d` from the repo
#     root, then `uvicorn toolserver.app:app --port 7801`. If they stay on the
#     dev box, tunnel: `ssh -R 7801:localhost:7801 box` and set
#     ECS_TOOLSERVER_URL accordingly (latency notes in README).
#
# Safe to re-run; every step is a no-op when already done.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${ECS_TRAIN_VENV:-$ROOT/.venv-train}"
PY="${ECS_PYTHON:-python3}"

echo "== ecs training setup: repo=$ROOT venv=$VENV =="

# --- sanity ------------------------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi not found (GPU box?)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
    || { echo "FATAL: python >= 3.10 required"; exit 1; }

# --- venv + deps ---------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    "$PY" -m venv "$VENV"
fi
"$VENV/bin/pip" install --upgrade -q pip
"$VENV/bin/pip" install -q -r "$ROOT/training/requirements.txt"

# flash-attn: needs torch importable at build time -> separate, tolerant step
# (sglang ships its own attention kernels; flash-attn only speeds up FSDP side).
if ! "$VENV/bin/python" -c 'import flash_attn' 2>/dev/null; then
    "$VENV/bin/pip" install -q flash-attn --no-build-isolation \
        || echo "WARN: flash-attn build failed -- continuing without it (slower FSDP attention)"
fi

"$VENV/bin/python" - <<'EOF'
import torch, verl, sglang, transformers
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)")
print(f"verl {verl.__version__} | sglang {sglang.__version__} | transformers {transformers.__version__}")
EOF

# --- env file (edit me) --------------------------------------------------------
ENV_FILE="$ROOT/training/launch/env.sh"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<'EOF'
# Sourced by run_validate.sh / run_main.sh -- edit for this box. NOT committed
# with secrets: keep WANDB_API_KEY in your shell profile or paste it here and
# never commit (file is only a template when hand-edited).
export ECS_TOOLSERVER_URL="${ECS_TOOLSERVER_URL:-http://localhost:7801}"
export ECS_REWARD_CONFIG="${ECS_REWARD_CONFIG:-}"        # empty = training/configs/reward.yaml
# export WANDB_API_KEY=...
# export WANDB_MODE=offline          # uncomment for air-gapped runs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
EOF
    echo "wrote template $ENV_FILE -- edit it (tool server URL, wandb)"
fi

# --- dataset -------------------------------------------------------------------
if [ ! -f "$ROOT/data/rl/train.parquet" ]; then
    if compgen -G "$ROOT/data/questions/*.jsonl" >/dev/null; then
        echo "building parquet dataset from data/questions/*.jsonl"
        (cd "$ROOT" && "$VENV/bin/python" -m training.data)
    else
        echo "WARN: no data/questions/*.jsonl yet -- rsync them, then run: python -m training.data"
    fi
else
    echo "dataset ok: data/rl/train.parquet"
fi

# --- tool server reachability (non-fatal) ---------------------------------------
source "$ENV_FILE"
if curl -sf -m 3 -X POST "$ECS_TOOLSERVER_URL/get_entity" \
        -H 'content-type: application/json' -d '{"qid":"Q42"}' >/dev/null 2>&1; then
    echo "tool server reachable at $ECS_TOOLSERVER_URL"
else
    echo "WARN: tool server NOT reachable at $ECS_TOOLSERVER_URL -- start it (or fix the tunnel) before training"
fi

echo "== setup done. Next: training/launch/run_validate.sh (smoke: python -m training.rollout_smoke --mock-tools) =="
