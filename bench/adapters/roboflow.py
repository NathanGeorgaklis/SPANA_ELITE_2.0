"""Candidates A / B: Roboflow `trackers` (Apache-2.0) at the pinned version, published defaults from config."""
from __future__ import annotations

from importlib.metadata import version
from typing import List, Sequence

import numpy as np

from spana.records import Detection
from spana.tracking.protocol import FrameInput, TrackOutput

from ..config import Candidate


class RoboflowTracker:
    def __init__(self, cand: Candidate, fps: float):
        impl = cand.implementation
        if version("trackers") != impl["version"]:
            raise ValueError(f"trackers {version('trackers')} installed, config pins {impl['version']}")
        import trackers
        self.name, self.needs_image = cand.name, cand.needs_image
        self.t = getattr(trackers, impl["class"])(frame_rate=fps, **cand.params)

    def update(self, detections: Sequence[Detection], frame: FrameInput) -> List[TrackOutput]:
        import supervision as sv
        n = len(detections)
        det = sv.Detections(xyxy=np.array([d.box for d in detections], float).reshape(n, 4),
                            confidence=np.array([d.score for d in detections], float),
                            class_id=np.zeros(n, int),            # class-agnostic, as our tracker uses them
                            data={"idx": np.arange(n)})
        if self.needs_image:
            if frame.image is None:
                raise ValueError(f"{self.name} needs the frame image (camera motion compensation)")
            res = self.t.update(det, frame.image)
        else:
            res = self.t.update(det)
        out = []
        for tid, i in zip(res.tracker_id, res.data["idx"]):
            if tid >= 0:                                        # -1 = not yet confirmed
                out.append(TrackOutput(f"T{int(tid):06d}", int(i), tuple(detections[int(i)].box)))
        return out
