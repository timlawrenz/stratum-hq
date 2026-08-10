"""Invoke scripts/probe_skin_color.py as a module (guard-safe)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / "scripts/probe_skin_color.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
