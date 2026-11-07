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
    keep = suppress_unions(boxes[idx][keep], scores[idx][keep], cfg, keep)
    flags = flag_multi_body(boxes[idx][keep], scores[idx][keep], cfg)
    return [Detection(tuple(float(v) for v in boxes[idx[k]]), float(scores[idx[k]]), int(keep_class), bool(f))
            for k, f in zip(keep, flags)]


def flag_multi_body(boxes: np.ndarray, scores: np.ndarray, cfg: DetectionConfig) -> List[bool]:
    """Flag boxes that may hold more than one person: the box encloses another, confident detection (inside it by
    >= multibody_member_ioa of its own area, score >= multibody_member_score) and is >= multibody_min_extent times
    wider or taller than it, so a large part of the box is not explained by that person.

    Real DEV finding: two players in contact produce one merged box next to one player's own box. Deleting such
    boxes would lose real people; flagged boxes are kept and the tracker treats them as ambiguous evidence.
    Uses only post-suppression boxes; independent of input order."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(boxes)
    if n < 2:
        return [False] * n
    iw = np.clip(np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1]), 0, None)
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-9)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-9)
    ioa_in = iw * ih / (w * h)[:, None]                        # [a, b] = share of box a inside box b
    extent = np.maximum(w[None, :] / w[:, None], h[None, :] / h[:, None])   # [a, b] = how much b exceeds a
    member = (ioa_in >= cfg.multibody_member_ioa) & (np.asarray(scores)[:, None] >= cfg.multibody_member_score) & \
        (extent >= cfg.multibody_min_extent)
    np.fill_diagonal(member, False)
    return [bool(v) for v in member.any(axis=0)]


def suppress_unions(boxes: np.ndarray, scores: np.ndarray, cfg: DetectionConfig, ids: List[int]) -> List[int]:
    """Drop union boxes: a box containing two DISTINCT detections (each inside it by >= union_member_ioa of its own
    area, mutual IoU <= union_member_max_iou) that are each MORE confident than the box itself.

    Real DEV finding: a low-resolution pass merges two depth-aligned players into one box that survives NMS next to
    their separate boxes (IoU with each is low). A single player's full box normally outscores its partial boxes, so
    it is not affected. Decisions use only the boxes that survived NMS; the result is independent of input order."""
    n = len(boxes)
    if n < 3:
        return list(ids)
    iw = np.clip(np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1]), 0, None)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    ioa_in = iw * ih / area[:, None]          # [i, j] = share of box i inside box j
    ious = iou_matrix(boxes, boxes)
    drop = set()
    for j in range(n):
        members = [i for i in range(n) if i != j and ioa_in[i, j] >= cfg.union_member_ioa and scores[i] > scores[j]]
        if any(ious[a, b] <= cfg.union_member_max_iou for x, a in enumerate(members) for b in members[x + 1:]):
            drop.add(j)
    return [k for pos, k in enumerate(ids) if pos not in drop]
