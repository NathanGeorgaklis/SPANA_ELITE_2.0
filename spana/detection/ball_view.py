"""Ball view of a multi-class detector: the ball candidates the ball tracker consumes.

The counterpart of PersonView: keeps boxes whose class is in `ball_classes` (config/roboflow.yaml `detector`) with
score >= ball.score_min, then merges overlapping ball boxes with NMS at ball.nms_iou. No size floor: a distant ball
is a few pixels. Nothing here knows any class name; deciding which candidate is the ball is the tracker's job.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import BallConfig
from ..records import Detection
from .postprocess import nms


# the ball config keys the view uses: a ball run must share them
VIEW_KEYS = ("score_min", "nms_iou", "tile_size_frac", "tile_overlap_frac", "tile_edge_margin_frac")


def tile_grid(frame_w: int, frame_h: int, cfg: BallConfig) -> List[Tuple[int, int, int]]:
    """(x0, y0, side) of the ball tiles: square tiles of side tile_size_frac x frame height, the fewest evenly spaced
    ones that cover the frame with at least tile_overlap_frac of the side between neighbours. [] = no tiling
    (tile_size_frac 0, or one tile would be the whole frame)."""
    if cfg.tile_size_frac <= 0:
        return []
    side = max(1, min(int(round(cfg.tile_size_frac * frame_h)), frame_w, frame_h))
    overlap = int(math.ceil(cfg.tile_overlap_frac * side))

    def origins(length: int) -> List[int]:
        if side >= length:
            return [0]
        n = max(2, int(math.ceil((length - overlap) / max(1, side - overlap))))
        return sorted({int(round(i * (length - side) / (n - 1))) for i in range(n)})

    tiles = [(x, y, side) for y in origins(frame_h) for x in origins(frame_w)]
    return tiles if len(tiles) > 1 else []


class BallViewError(ValueError):
    pass


class BallView:
    def __init__(self, detector, ball_classes: Sequence[str], cfg: BallConfig, tile_detector=None):
        """tile_detector: the same detector WITHOUT a per-frame cache, run on the tile crops (None = no tiling
        possible: refused when cfg.tile_size_frac > 0)."""
        if cfg.tile_size_frac > 0 and tile_detector is None:
            raise BallViewError("ball.tile_size_frac > 0 needs a detector that can be run on tiles "
                                "(the local backend); set ball.tile_size_frac to 0 otherwise")
        self._tile_det = tile_detector
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

    def _balls(self, dets) -> List[Detection]:
        return [d for d in dets if d.class_id in self._ball_ids and d.score >= self._cfg.score_min
                and d.box[2] > d.box[0] and d.box[3] > d.box[1]]

    def _tile_balls(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Ball boxes of the tile passes, in frame coordinates. A box within the edge margin of an INTERNAL tile edge
        is cut off there and dropped: the neighbouring tile (or the full frame) sees that ball whole."""
        h, w = frame_bgr.shape[:2]
        out: List[Detection] = []
        for x0, y0, side in tile_grid(w, h, self._cfg):
            m = self._cfg.tile_edge_margin_frac * side
            for d in self._balls(self._tile_det.detect(np.ascontiguousarray(frame_bgr[y0:y0 + side, x0:x0 + side]))):
                x1, y1, x2, y2 = d.box
                if (x0 > 0 and x1 <= m) or (y0 > 0 and y1 <= m) or \
                        (x0 + side < w and x2 >= side - m) or (y0 + side < h and y2 >= side - m):
                    continue
                out.append(Detection((x1 + x0, y1 + y0, x2 + x0, y2 + y0), d.score, d.class_id, False))
        return out

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        dets = self._balls(self._det.detect(frame_bgr))
        if self._cfg.tile_size_frac > 0:
            dets = dets + self._tile_balls(frame_bgr)
        if not dets:
            return []
        keep = nms(np.array([d.box for d in dets], float), np.array([d.score for d in dets], float),
                   self._cfg.nms_iou)
        return [Detection(dets[k].box, dets[k].score, dets[k].class_id, False) for k in keep]
