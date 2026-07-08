#!/usr/bin/env bash
# Build the Qwen3.5-capable training venv (parallel to the default .venv-train,
# which stays the qualified Qwen3 fallback). Drive verl with it via
# ECS_TRAIN_VENV=<venv path> (read by training/launch/common.sh).
#
# MATRIX (chosen 2026-07-08, see training/README.md "Qwen3.5 option"):
#   verl 0.8.0        -- unchanged (latest release; no newer verl exists)
#   sglang 0.5.10     -- newest release that BOTH ships srt/models/qwen3_5.py
#                        (+ vendored srt/configs/qwen3_5.py) AND still pins
#                        torch==2.9.1 (0.5.11+ jumps to torch 2.11)
#   transformers 5.3.0-- sglang 0.5.10's own pin; first-party qwen3_5 support
#   torch 2.9.1+cu128 -- UNCHANGED from the qualified stack (driver >= 550)
#   flash-attn 2.8.3  -- same prebuilt cu13torch2.9 cp312 wheel (NO source
#                        builds: box nvcc is CUDA 13 vs torch cu128)
#   scipy 1.17.1      -- MUST pin: verl drags numpy to 1.26.4 while a floating
#                        scipy resolves to 1.18 (needs numpy>=2) -> crashes at
#                        sglang import ("numpy has no attribute 'long'", then a
#                        masked transformers AutoProcessor import error)
#
# verl is installed WITHOUT its [sglang] extra on purpose: that extra pins
# sglang==0.5.8 exactly (no qwen3_5). tensordict is pinned to the extra's range.
set -euxo pipefail

VENV="${1:-$HOME/.venv-train-q35}"

python3 -m venv "$VENV"
P="$VENV/bin/pip"
$P install -q -U pip
$P install -q "sglang[openai,srt]==0.5.10"     # pulls torch==2.9.1, transformers==5.3.0
# cachetools: verl imports it but doesn't declare it; sglang 0.5.8's dep tree
# used to provide it, 0.5.10's no longer does.
$P install -q "verl==0.8.0" "tensordict>=0.8.0,!=0.9.0,<=0.10.0" "scipy==1.17.1" cachetools
$P install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
# Gated-DeltaNet fused kernels: WITHOUT both causal-conv1d AND fla importable,
# transformers' modeling_qwen3_5 falls back to the naive
# torch_chunk_gated_delta_rule, which crashes with a CUDA illegal memory
# access under FSDP compute_log_prob (verl#6549 -- reproduced here at 2B).
# is_fast_path_available in modeling_qwen3_5.py requires all four symbols.
# causal-conv1d: prebuilt wheel (same no-nvcc rule as flash-attn);
# flash-linear-attention: pure triton, no build step.
$P install -q "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
$P install -q flash-linear-attention
$P install -q httpx pyyaml orjson pandas pyarrow

"$VENV/bin/python" - <<'PY'
import torch, sglang, transformers, verl, tensordict, scipy, flash_attn
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("sglang", sglang.__version__, "| tfm", transformers.__version__,
      "| verl", verl.__version__, "| tensordict", tensordict.__version__,
      "| scipy", scipy.__version__, "| flash_attn", flash_attn.__version__)
import transformers.models.qwen3_5.modeling_qwen3_5 as m
assert m.is_fast_path_available, "GDN fused kernels missing (causal-conv1d/fla) -> naive path crashes, see verl#6549"
print("qwen3_5 GDN fast path: ACTIVE")
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
assert (x @ x).float().abs().mean().item() > 0, "bf16 matmul failed"
from transformers import AutoConfig, AutoProcessor  # AutoProcessor: scipy-pin canary
c = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
assert c.model_type == "qwen3_5", c.model_type
print("qwen3_5 arch OK; BUILD_OK")
PY
