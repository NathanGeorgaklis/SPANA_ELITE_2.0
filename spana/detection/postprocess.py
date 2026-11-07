"""Model-agnostic pre/post-processing for single-stage box detectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from ..config import DetectionConfig
from ..geometry import iou_matrix
from ..records import Detection


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        b = np.asarray(boxes, dtype=float).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_x) / self.scale
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_y) / self.scale
        return b


def letterbox(frame_bgr: np.ndarray, size: int, pad_value: int, anchor: str = "center"
              ) -> Tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio into a size x size canvas, placed at the centre or the top-left."""
    if anchor not in ("center", "top_left"):
        raise ValueError(f"unknown letterbox anchor {anchor!r}")
    h, w = frame_bgr.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    px, py = ((size - nw) // 2, (size - nh) // 2) if anchor == "center" else (0, 0)
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, Letterbox(s, float(px), float(py))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Greedy NMS; deterministic tie-break by index."""
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    keep: List[int] = []
    if not order:
        return keep
    ious = iou_matrix(boxes, boxes)
    suppressed = np.zeros(len(scores), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= ious[i] > iou_thr
    return keep


def filter_detections(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, *, frame_w: int, frame_h: int,
                      cfg: DetectionConfig, keep_class: int) -> List[Detection]:
    """Class filter -> clip to image -> score/size filter -> NMS. Order of output: score desc, then index."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    class_ids = np.asarray(class_ids).reshape(-1)
    if len(boxes) == 0:
        return []
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_h)
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    ok = (class_ids == keep_class) & (scores >= cfg.score_low) & (h >= cfg.min_box_height_frac * frame_h) & (w > 0)
    idx = np.flatnonzero(ok)
    keep = nms(boxes[idx], scores[idx], cfg.nms_iou)
    return [Detection(tuple(float(v) for v in boxes[idx[k]]), float(scores[idx[k]]), int(keep_class)) for k in keep]
