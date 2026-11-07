"""Generic appearance descriptor: an HSV hue-saturation histogram of the torso band of a box.

No kit colours or team knowledge. It is only used to VETO or DISTINGUISH re-associations after an occlusion
(see tracking/tracker.py), never to force a match. Too-small crops return None: the descriptor is then UNKNOWN,
not a guess.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from .config import AppearanceConfig


def torso_descriptor(frame_bgr: np.ndarray, box: Sequence[float], cfg: AppearanceConfig) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = (float(v) for v in box)
    h, w = y2 - y1, x2 - x1
    if h < cfg.min_crop_height_px:
        return None
    cx = (x1 + x2) / 2
    half_w = w * cfg.torso_width_frac / 2
    ty1, ty2 = y1 + cfg.torso_top_frac * h, y1 + cfg.torso_bottom_frac * h
    H, W = frame_bgr.shape[:2]
    xa, xb = int(max(0, round(cx - half_w))), int(min(W, round(cx + half_w)))
    ya, yb = int(max(0, round(ty1))), int(min(H, round(ty2)))
    if xb - xa < 2 or yb - ya < 2:
        return None
    hsv = cv2.cvtColor(frame_bgr[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [cfg.hue_bins, cfg.sat_bins], [0, 180, 0, 256]).reshape(-1)
    total = float(hist.sum())
    return None if total <= 0 else (hist / total).astype(np.float64)


def describe(frame_bgr: np.ndarray, boxes: Sequence[Sequence[float]], cfg: AppearanceConfig
             ) -> List[Optional[np.ndarray]]:
    return [torso_descriptor(frame_bgr, b, cfg) if cfg.enabled else None for b in boxes]


def similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Bhattacharyya coefficient of two normalised histograms, in [0, 1]; None if either is unknown."""
    if a is None or b is None:
        return None
    return float(np.clip(np.sum(np.sqrt(a * b)), 0.0, 1.0))
