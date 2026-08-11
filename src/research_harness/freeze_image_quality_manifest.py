"""Module wrapper: run scripts/freeze_image_quality_manifest.py (cron-guard safe)."""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path("/home/tim/source/activity/stratum-hq-stage-b-experiment/scripts/freeze_image_quality_manifest.py")),
    run_name="__main__",
)