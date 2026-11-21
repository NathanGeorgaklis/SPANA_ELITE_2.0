"""Reference row: the production tracker driven exactly as `spana.cli track` drives it (config/default.yaml)."""
from __future__ import annotations

import dataclasses
from typing import List, Sequence

from spana.appearance import describe
from spana.config import load_config
from spana.records import Detection, IdentityState, PositionSource
from spana.tracking.camera import CameraMotionEstimator
from spana.tracking.protocol import FrameInput, TrackOutput
from spana.tracking.tracker import MultiPlayerTracker

from ..config import REPO, Candidate


class SpanaReference:
    def __init__(self, cand: Candidate, fps: float, size: tuple, detection_config: dict | None = None):
        self.name = cand.name
        self.cfg = load_config(REPO / cand.implementation["config"])
        if detection_config is not None and detection_config != dataclasses.asdict(self.cfg.detection):
            raise ValueError("cached detections were made under a detection config other than default.yaml's")
        if self.cfg.reid_resolve.enabled:
            raise ValueError("reference run expects reid_resolve off (the product default)")
        w, h = size
        self.cam = CameraMotionEstimator(self.cfg.camera, w, h, fps)
        self.trk = MultiPlayerTracker(self.cfg, w, h)

    def update(self, detections: Sequence[Detection], frame: FrameInput) -> List[TrackOutput]:
        if frame.image is None:
            raise ValueError("the SPANA tracker needs the frame image (camera motion, appearance)")
        boxes = [d.box for d in detections]
        ft = self.trk.update(frame.index, frame.t, detections, self.cam.observe(frame.image, boxes),
                             describe(frame.image, boxes, self.cfg.appearance))
        return [TrackOutput(o.track_id, o.detection_index, tuple(detections[o.detection_index].box),
                            o.identity == IdentityState.CERTAIN)
                for o in ft.tracks if o.source == PositionSource.OBSERVED and o.confirmed]
