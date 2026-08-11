"""Module wrapper: run scripts/probe_image_quality.py (cron-guard safe)."""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path("/home/tim/source/activity/stratum-hq-stage-b-experiment/scripts/probe_image_quality.py")),
    run_name="__main__",
)