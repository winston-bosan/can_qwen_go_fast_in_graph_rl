"""Workstream C: RL training environment (GRPO à la SID-1) for entity retrieval.

Layout:
  env/tools.py    -- tool-server adapter (async httpx client + verl BaseTool subclasses)
  env/reward.py   -- reward function (two-tier NDCG + format reward + length penalty)
  data.py         -- JSONL -> verl parquet dataset conversion, split, curriculum
  configs/        -- verl (hydra) run configs + tool schema config + reward config
  rollout_smoke.py-- trainer-free end-to-end smoke test of the multi-turn loop
  launch/         -- remote setup + run scripts
"""

import os
import sys

# The repo is not necessarily pip-installed on the training box; make
# `ecs.*` (src layout), `eval.*`, and `toolserver.*` importable when running
# from anywhere inside the repo.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
