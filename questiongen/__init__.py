"""Workstream B — question generation (KG patterns, sim-link, verbalization).

Importing this package makes the shared `ecs` package (src/ecs) importable
without an editable install, mirroring the convention used by ingest/.
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
