"""Stratum2 — Sapiens2-based image enrichment pipeline.

Uses specialized Sapiens2 task checkpoints (seg, normal, pointmap, pose, matting)
alongside non-Sapiens modalities (DINOv3, T5, captioning, pixel bucketing).
"""

from __future__ import annotations
