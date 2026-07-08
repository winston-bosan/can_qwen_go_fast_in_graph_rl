# Shared launch plumbing for run_validate.sh / run_main.sh (sourced, not run).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${ECS_TRAIN_VENV:-$ROOT/.venv-train}"
[ -x "$VENV/bin/python" ] || { echo "FATAL: $VENV missing -- run training/launch/setup_remote.sh"; exit 1; }
[ -f "$ROOT/training/launch/env.sh" ] && source "$ROOT/training/launch/env.sh"

# SID-1 length scheduling: verl cannot ramp data.max_response_length inside a
# run, so we chain runs. Each stage is "<max_response_tokens>:<cumulative_step
# target>"; the LAST stage may omit ":steps" to run to the config's natural
# end. trainer.resume_mode=auto (set in the yaml) makes stage N+1 resume from
# stage N's latest checkpoint in trainer.default_local_dir.
run_stages() {
    local config_name="$1"; shift
    local -a stages=("$@")
    cd "$ROOT"
    export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    # sglang scheduler subprocesses get LD_PRELOADed with torch_memory_saver,
    # which links libcudart.so.12. That lib lives in the venv's pip-provided
    # nvidia dirs, invisible to child processes by default -> the scheduler
    # dies with exit 127 ("libcudart.so.12: cannot open shared object file").
    # Export the venv CUDA lib dirs so every descendant process can resolve it.
    local nvlibs
    nvlibs="$("$VENV/bin/python" - <<'PY'
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
dirs = sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib"))) + [os.path.join(sp, "torch", "lib")]
print(":".join(d for d in dirs if os.path.isdir(d)))
PY
)"
    export LD_LIBRARY_PATH="$nvlibs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    for stage in "${stages[@]}"; do
        local resp="${stage%%:*}" steps="${stage#*:}" until_msg=""
        local -a extra=("data.max_response_length=$resp")
        if [ "$steps" != "$stage" ]; then
            extra+=("trainer.total_training_steps=$steps")
            until_msg=" (until global step $steps)"
        fi
        echo "== stage: max_response_length=$resp$until_msg =="
        "$VENV/bin/python" -m verl.trainer.main_ppo \
            --config-path="$ROOT/training/configs" \
            --config-name="$config_name" \
            "${extra[@]}" \
            ${ECS_EXTRA_OVERRIDES:-}
    done
}
