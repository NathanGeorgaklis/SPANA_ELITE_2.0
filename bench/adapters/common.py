"""Shared adapter helpers."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from spana.records import Detection


def det_index_of_box(box_xyxy: Sequence[float], detections: Sequence[Detection], atol: float = 1e-3) -> int:
    """The detection a tracker reported back (trackers that return boxes, not indices). Exact match or error."""
    if not detections:
        raise ValueError("a track was reported on a frame without detections")
    arr = np.array([d.box for d in detections], float)
    err = np.abs(arr - np.asarray(box_xyxy, float)[None, :]).max(axis=1)
    k = int(np.argmin(err))
    if err[k] > atol:
        raise ValueError(f"reported box {list(box_xyxy)} is not one of the frame's detections (off by {err[k]:.4f})")
    return k
