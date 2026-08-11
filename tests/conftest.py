import sys
from pathlib import Path

# Ensure the repo root is importable so `from tests.stratum2...` and
# `from stratum2...` both resolve regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
