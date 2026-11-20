"""Candidate trackers behind spana.tracking.protocol.Tracker (bench/ only).

`make_tracker` builds an online candidate; `refine` applies an offline one (GTA-Link) to a finished run. Modules for
code without a commercial licence are marked EVAL-ONLY (benchmark use only) and are imported lazily from here.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Candidate


def make_tracker(cand: Candidate, cache, external_dir: Path):
    """A fresh tracker for one clip (cache: bench.cache.DetectionCache)."""
    fps = cache.fps
    if cand.name == "spana":
        from .spana_ref import SpanaReference
        return SpanaReference(cand, fps, cache.size, cache.doc["detection_config"])
    if cand.implementation.get("package") == "trackers":
        from .roboflow import RoboflowTracker
        return RoboflowTracker(cand, fps)
    if cand.name == "deep_eiou":
        from .deep_eiou import DeepEIoUTracker
        return DeepEIoUTracker(cand, fps, external_dir)
    raise ValueError(f"no adapter for online candidate {cand.name}")


def refine(cand: Candidate, run: list, cache, external_dir: Path) -> list:
    if cand.name == "gta_link":
        from .gta_link import gta_link
        return gta_link(cand, run, cache, external_dir)
    raise ValueError(f"no adapter for offline candidate {cand.name}")
