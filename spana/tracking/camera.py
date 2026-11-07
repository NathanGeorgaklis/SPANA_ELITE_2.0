"""Inter-frame camera motion (similarity transform) and shot-cut detection.

Players are masked out so the estimate follows the pitch/background, not the players.
Failure modes are explicit: CUT (shot change) and UNKNOWN (no reliable estimate). Never a guessed transform.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ..config import CameraConfig
from ..records import CameraObservation, CameraStatus


class CameraMotionEstimator:
    def __init__(self, cfg: CameraConfig, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.k = min(1.0, cfg.analysis_width_px / float(frame_w))  # source -> analysis scale
        self.size = (max(1, int(round(frame_w * self.k))), max(1, int(round(frame_h * self.k))))
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._prev_mask: Optional[np.ndarray] = None

    def _mask(self, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        m = np.full((self.size[1], self.size[0]), 255, dtype=np.uint8)
        for b in boxes:
            x1, y1, x2, y2 = (int(round(v * self.k)) for v in b)
            m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
        return m

    def observe(self, frame_bgr: np.ndarray, player_boxes: Sequence[Sequence[float]]) -> CameraObservation:
        small = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        mask = self._mask(player_boxes)

        prev_gray, prev_hist, prev_mask = self._prev_gray, self._prev_hist, self._prev_mask
        self._prev_gray, self._prev_hist, self._prev_mask = gray, hist, mask

        if prev_gray is None:
            return CameraObservation.identity(CameraStatus.FIRST)
        dist = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        if dist > self.cfg.cut_hist_distance:
            return CameraObservation.identity(CameraStatus.CUT, hist_distance=dist)

        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=self.cfg.max_features, qualityLevel=self.cfg.feature_quality,
                                      minDistance=self.cfg.feature_min_distance_frac * self.size[0],
                                      mask=prev_mask)
        if pts is None or len(pts) < self.cfg.min_inliers:
            return CameraObservation.identity(CameraStatus.UNKNOWN, hist_distance=dist)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
        good = st.reshape(-1) == 1
        if good.sum() < self.cfg.min_inliers:
            return CameraObservation.identity(CameraStatus.UNKNOWN, hist_distance=dist)
        diag = float(np.hypot(*self.size))
        A, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                             ransacReprojThreshold=self.cfg.ransac_reproj_frac * diag)
        n_inl = int(inl.sum()) if inl is not None else 0
        if A is None or n_inl < self.cfg.min_inliers:
            return CameraObservation.identity(CameraStatus.UNKNOWN, hist_distance=dist, inliers=n_inl)
        scale = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
        if not scale > 0 or abs(np.log(scale)) > self.cfg.max_scale_change:
            return CameraObservation.identity(CameraStatus.UNKNOWN, hist_distance=dist, inliers=n_inl)
        # Analysis-frame transform -> source-pixel transform: linear part unchanged, translation / k.
        A_src = A.astype(float).copy()
        A_src[:, 2] /= self.k
        affine = tuple(tuple(float(v) for v in row) for row in A_src)
        return CameraObservation(CameraStatus.OK, affine, n_inl, dist)  # type: ignore[arg-type]
