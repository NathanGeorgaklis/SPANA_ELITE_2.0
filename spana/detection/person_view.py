"""Person view of a multi-class detector: the `PlayerDetector` the tracker consumes.

A multi-class `Detector` (e.g. the football detector: player / goalkeeper / referee / ball) is turned into person
detections with the same post-processing rules the person-only backend applies (config `detection`):
  1. keep boxes whose class is in `person_classes`; every other class (e.g. the ball) is dropped;
  2. score >= detection.score_low, box height >= detection.min_box_height_frac x frame height;
  3. class-agnostic NMS at detection.nms_iou: one person boxed as two classes keeps the higher-scoring box, whose
     class is kept as a hint (class_id indexes the detector's class_names);
  4. detection.multi_body flag on the kept boxes.
The wrapped detector must be asked for boxes down to detection.score_low (fails closed otherwise: the tracker's
low-score association round would get nothing). Nothing here knows any class name: the list comes from config.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..config import DetectionConfig
from ..profiling import StageTimer
from ..records import Detection
from .postprocess import flag_multi_body, nms


class PersonViewError(ValueError):
    pass


class PersonView:
    def __init__(self, detector, person_classes: Sequence[str], cfg: DetectionConfig):
        names = tuple(detector.class_names)
        missing = [c for c in person_classes if c not in names]
        if missing or not person_classes:
            raise PersonViewError(f"person classes {list(person_classes)} not all in the detector's classes {names}")
        floor = detector.model_id.get("confidence")
        if floor is None or float(floor) > cfg.score_low:
            raise PersonViewError(f"the detector's request confidence {floor!r} must be known and <= detection.score_low "
                                  f"{cfg.score_low} (the low-score association round needs those boxes)")
        self._det, self._cfg = detector, cfg
        self._names = names
        self._person_ids = frozenset(names.index(c) for c in person_classes)
        self._person_classes = tuple(person_classes)
        self.timer = StageTimer(False)   # spana.profiling; a no-op unless the caller sets an enabled timer

    def set_timer(self, timer: StageTimer) -> None:
        self.timer = timer
        if hasattr(self._det, "timer"):
            self._det.timer = timer

    @property
    def class_names(self) -> tuple:
        return self._names

    @property
    def model_id(self) -> dict:
        return {**self._det.model_id, "class_names": list(self._names),
                "person_view": {"person_classes": list(self._person_classes)}}

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        raw = self._det.detect(frame_bgr)
        with self.timer.stage("detect.person_view"):
            return self._persons(raw, frame_bgr.shape[0])

    def _persons(self, raw: List[Detection], h: int) -> List[Detection]:
        c = self._cfg
        dets = [d for d in raw
                if d.class_id in self._person_ids and d.score >= c.score_low
                and d.box[3] - d.box[1] >= c.min_box_height_frac * h and d.box[2] > d.box[0]]
        if not dets:
            return []
        boxes = np.array([d.box for d in dets], float)
        scores = np.array([d.score for d in dets], float)
        keep = nms(boxes, scores, c.nms_iou)                     # score desc, ties by index: deterministic
        flags = flag_multi_body(boxes[keep], scores[keep], c)
        return [Detection(dets[k].box, dets[k].score, dets[k].class_id, f) for k, f in zip(keep, flags)]
