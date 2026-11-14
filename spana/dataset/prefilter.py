"""Pre-label quality filters: blur gate, grass (on-pitch) mask, ball tiling and box merging.

Used ONLY to prepare pre-labels for human review (tools/roboflow_prelabel.py). The runtime detector never uses these:
a pre-label that is dropped here costs a human a few clicks, a wrong runtime filter would hide real players.
All sizes are fractions of the frame or come from config (prelabel.blur / grass / ball_tiling); nothing is
specific to a match, venue or camera.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


def blur_score(frame_bgr: np.ndarray, analysis_width_px: int) -> float:
    """Variance of the Laplacian of a grey copy resized to `analysis_width_px` (higher = sharper)."""
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    if w != analysis_width_px:
        grey = cv2.resize(grey, (analysis_width_px, max(1, int(round(h * analysis_width_px / w)))),
                          interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def _odd_kernel(frac: float, height: int) -> int:
    k = int(round(frac * height))
    return k + 1 if k % 2 == 0 else k


def grass_mask(frame_bgr: np.ndarray, g) -> np.ndarray:
    """Boolean mask of the grass area. `g` is a prelabel.grass config section."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(g.hsv_lower, np.uint8), np.array(g.hsv_upper, np.uint8))
    k = _odd_kernel(g.close_frac, h)
    if k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= g.min_component_frac * h * w
    if g.convex_hull:
        m = np.zeros((h, w), np.uint8)
        for k in np.flatnonzero(keep):
            cv2.fillConvexPoly(m, cv2.convexHull(cv2.findNonZero((labels == k).astype(np.uint8))), 255)
    else:
        m = keep[labels].astype(np.uint8) * 255
    k = _odd_kernel(g.dilate_frac, h)
    if k > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m > 0


def bottom_centre(box: Sequence[float]) -> Tuple[float, float]:
    x1, _, x2, y2 = box
    return (x1 + x2) / 2.0, float(y2)


def on_grass(mask: np.ndarray, box: Sequence[float]) -> bool:
    """True when the box's bottom-centre point (feet / ball contact point) lies inside the grass mask."""
    h, w = mask.shape
    x, y = bottom_centre(box)
    xi = min(w - 1, max(0, int(x)))
    yi = min(h - 1, max(0, int(np.ceil(y)) - 1))   # last pixel row inside the box
    return bool(mask[yi, xi])


def foot_unobserved(box: Sequence[float], frame_height: int, margin_frac: float) -> bool:
    """True when the box's bottom edge lies within margin_frac x frame height of the frame's bottom border: the body
    is cut by the frame, so its bottom-centre is not a foot / ground-contact point and says nothing about the grass."""
    return float(box[3]) >= frame_height * (1.0 - margin_frac)


def tiles(width: int, height: int, tile_px: int, overlap_frac: float) -> List[Tuple[int, int, int, int]]:
    """Overlapping (x1, y1, x2, y2) windows covering the whole frame, deterministic, edge tiles flush with the
    border. A dimension no larger than tile_px gets one window."""
    def starts(n: int) -> List[int]:
        if n <= tile_px:
            return [0]
        stride = max(1, int(round(tile_px * (1.0 - overlap_frac))))
        s = list(range(0, n - tile_px, stride))
        return s + [n - tile_px]
    return [(x, y, min(width, x + tile_px), min(height, y + tile_px)) for y in starts(height) for x in starts(width)]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def contains_centre(a: Sequence[float], b: Sequence[float]) -> bool:
    """True when either box contains the other's centre (a ball cut by a tile edge vs. the whole ball)."""
    def inside(p, q):
        cx, cy = (q[0] + q[2]) / 2.0, (q[1] + q[3]) / 2.0
        return p[0] <= cx <= p[2] and p[1] <= cy <= p[3]
    return inside(a, b) or inside(b, a)


def merge_boxes(boxes: List[dict], merge_iou: float) -> Tuple[List[dict], int]:
    """Greedy NMS over {box, score} dicts: highest score first; a box is merged into a kept one when their IoU
    exceeds merge_iou or one contains the other's centre. -> (kept, number merged away). Deterministic."""
    order = sorted(boxes, key=lambda b: (-b["score"], b["box"]))
    kept: List[dict] = []
    for b in order:
        if any(iou(b["box"], k["box"]) > merge_iou or contains_centre(b["box"], k["box"]) for k in kept):
            continue
        kept.append(b)
    return kept, len(boxes) - len(kept)
