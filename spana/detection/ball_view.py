"""Ball view of a multi-class detector: the ball candidates the ball tracker consumes.

The counterpart of PersonView: keeps boxes whose class is in `ball_classes` (config/roboflow.yaml `detector`) with
score >= ball.score_min, then merges overlapping ball boxes with NMS at ball.nms_iou. No size floor: a distant ball
is a few pixels. Nothing here knows any class name; deciding which candidate is the ball is the tracker's job.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..config import BallConfig
from ..records import Detection
from .postprocess import nms


class BallViewError(ValueError):
    pass


class BallView:
    def __init__(self, detector, ball_classes: Sequence[str], cfg: BallConfig):
        names = tuple(detector.class_names)
        missing = [c for c in ball_classes if c not in names]
        if missing or not ball_classes:
            raise BallViewError(f"ball classes {list(ball_classes)} not all in the detector's classes {names}")
        floor = detector.model_id.get("confidence")
        if floor is None or float(floor) > cfg.score_min:
            raise BallViewError(f"the detector's request confidence {floor!r} must be known and <= ball.score_min "
                                f"{cfg.score_min}")
        self._det, self._cfg = detector, cfg
        self._names = names
        self._ball_ids = frozenset(names.index(c) for c in ball_classes)
        self._ball_classes = tuple(ball_classes)

    @property
    def class_names(self) -> tuple:
        return self._names

    @property
    def model_id(self) -> dict:
        return {**self._det.model_id, "class_names": list(self._names),
                "ball_view": {"ball_classes": list(self._ball_classes)}}

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        dets = [d for d in self._det.detect(frame_bgr)
                if d.class_id in self._ball_ids and d.score >= self._cfg.score_min
                and d.box[2] > d.box[0] and d.box[3] > d.box[1]]
        if not dets:
            return []
        keep = nms(np.array([d.box for d in dets], float), np.array([d.score for d in dets], float),
                   self._cfg.nms_iou)
        return [Detection(dets[k].box, dets[k].score, dets[k].class_id, False) for k in keep]
