"""Box geometry. Boxes are float arrays [x1, y1, x2, y2] in source-video pixels."""
from __future__ import annotations

import numpy as np

Box = np.ndarray  # shape (4,)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def centers(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def heights(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return boxes[:, 3] - boxes[:, 1]


def box_to_cxcywh(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def cxcywh_to_box(v: np.ndarray) -> np.ndarray:
    cx, cy, w, h = (float(x) for x in v[:4])
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def affine_scale(affine: np.ndarray) -> float:
    """Isotropic scale of a 2x3 similarity transform."""
    return float(np.sqrt(abs(np.linalg.det(np.asarray(affine, dtype=float)[:, :2]))))


def apply_affine_points(affine: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    A = np.asarray(affine, dtype=float)
    return pts @ A[:, :2].T + A[:, 2]
