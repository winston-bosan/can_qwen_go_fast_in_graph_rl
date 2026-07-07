#!/usr/bin/env bash
# Validation GRPO run: Qwen3-0.6B, 1-2 GPUs. See configs/validate_0p6b.yaml.
#
#   ./training/launch/run_validate.sh                      # default schedule
#   LENGTH_SCHEDULE="2048:40 6144" ./run_validate.sh       # custom stages
#   ECS_EXTRA_OVERRIDES="trainer.n_gpus_per_node=1" ...    # extra hydra overrides
#
# Prereqs: setup_remote.sh done; tool server reachable at $ECS_TOOLSERVER_URL;
# data/rl/train.parquet built (python -m training.data).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Two short stages are enough to *validate* length scheduling mechanics on the
# small model: 2048-cap for the first 30 steps, then the full 6144 budget.
# CONFIG_NAME=validate_0p8b switches to the (experimental, not yet cleared)
# Qwen3.5-0.8B config -- see its header + README before using.
SCHEDULE=(${LENGTH_SCHEDULE:-"2048:30" "6144"})
run_stages "${CONFIG_NAME:-validate_0p6b}" "${SCHEDULE[@]}"
