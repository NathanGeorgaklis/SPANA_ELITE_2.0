"""EVAL-ONLY (benchmark use only; never imported from spana/). Candidate C: Deep-EIoU, official code read from the
pinned checkout in bench/external/. The repository has no licence, so nothing of it is copied into this repo."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np

from spana.records import Detection
from spana.tracking.protocol import FrameInput, TrackOutput

from ..config import Candidate
from ..external import verified_path
from .common import det_index_of_box


class DeepEIoUTracker:
    def __init__(self, cand: Candidate, fps: float, external_dir):
        path = str(verified_path(external_dir, cand))
        if path not in sys.path:
            sys.path.insert(0, path)
        if not hasattr(np, "float"):
            # Their code (2023, numpy < 1.24) uses the alias np.float, which was exactly the builtin float; restoring
            # the alias changes no behaviour and leaves their checkout untouched.
            np.float = float                                    # type: ignore[attr-defined]
        from tracker.Deep_EIoU import Deep_EIoU                    # their package name
        self.name = cand.name
        self.min_box_area = cand.params["min_box_area"]
        self.t = Deep_EIoU(SimpleNamespace(**cand.params), frame_rate=fps)

    def update(self, detections: Sequence[Detection], frame: FrameInput) -> List[TrackOutput]:
        if frame.embeddings is None:
            raise ValueError("Deep-EIoU needs per-detection embeddings (build the cache with --embed)")
        arr = np.array([[*d.box, d.score] for d in detections], float).reshape(-1, 5)
        out = []
        for s in self.t.update(arr, np.asarray(frame.embeddings, float)):
            x, y, w, h = s.last_tlwh
            if w * h <= self.min_box_area:                      # their sport_track.py output filter
                continue
            k = det_index_of_box((x, y, x + w, y + h), detections)
            out.append(TrackOutput(f"T{int(s.track_id):06d}", k, tuple(detections[k].box)))
        return out
