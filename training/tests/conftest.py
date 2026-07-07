import os
import sys

# Make `training.*` (and via training/__init__.py also `ecs.*` / `eval.*`)
# importable no matter where pytest is invoked from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
