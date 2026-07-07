#!/usr/bin/env bash
# Main GRPO run: Qwen3-4B on one 8xH100 node. See configs/main_4b.yaml
# (header documents the single-B300 re-size; pass it via ECS_EXTRA_OVERRIDES).
#
#   ./training/launch/run_main.sh
#   LENGTH_SCHEDULE="4096:60 8192:140 12288" ./run_main.sh
#
#   # single RunPod B300 (288GB) example:
#   ECS_EXTRA_OVERRIDES="trainer.n_gpus_per_node=1 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16" \
#     ./training/launch/run_main.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# SID-1-style ramp: cheap short rollouts while the format + tool loop is being
# learned, full 12288-token budget only for the final leg.
SCHEDULE=(${LENGTH_SCHEDULE:-"4096:60" "8192:140" "12288"})
run_stages main_4b "${SCHEDULE[@]}"
