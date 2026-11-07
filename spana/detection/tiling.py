"""Full-frame + overlapping-tile inference, for small and distant players.

A detector with a fixed input size sees a downscaled full frame, where distant players can be only a few
pixels tall. Tiles are run at a higher effective resolution and merged with the full-frame pass:

  * tile side = tile_size_frac x frame height (resolution independent), overlap = tile_overlap_frac x side;
  * a tile box within tile_edge_margin_frac x side of an INTERNAL tile edge (not a frame edge) is a
    truncated body and is dropped. A neighbouring tile or the full-frame pass sees that body whole, as long
    as the body is smaller than the overlap; larger bodies are covered by the full-frame pass;
  * everything is merged by the normal score/size filter and NMS in `filter_detections`.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from ..config import DetectionConfig
from ..records import Detection
from .postprocess import filter_detections

Infer = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


def tile_origins(length: int, side: int, overlap: int) -> List[int]:
    """Tile start positions covering [0, length): fixed step, last tile flush with the end."""
    if side >= length:
        return [0]
    step = max(1, side - overlap)
    starts = list(range(0, length - side, step))
    starts.append(length - side)
    return sorted(set(starts))


def tile_grid(frame_w: int, frame_h: int, cfg: DetectionConfig) -> List[Tuple[int, int, int]]:
    side = max(1, min(int(round(cfg.tile_size_frac * frame_h)), frame_w, frame_h))
    overlap = int(round(cfg.tile_overlap_frac * side))
    return [(x, y, side) for y in tile_origins(frame_h, side, overlap) for x in tile_origins(frame_w, side, overlap)]


def _internal_edge_ok(boxes: np.ndarray, x0: int, y0: int, side: int, frame_w: int, frame_h: int,
                      margin: float) -> np.ndarray:
    ok = np.ones(len(boxes), dtype=bool)
    if x0 > 0:
        ok &= boxes[:, 0] > x0 + margin
    if y0 > 0:
        ok &= boxes[:, 1] > y0 + margin
    if x0 + side < frame_w:
        ok &= boxes[:, 2] < x0 + side - margin
    if y0 + side < frame_h:
        ok &= boxes[:, 3] < y0 + side - margin
    return ok


def detect_tiled(infer: Infer, frame_bgr: np.ndarray, cfg: DetectionConfig, class_id: int) -> List[Detection]:
    h, w = frame_bgr.shape[:2]
    all_boxes, all_scores = [], []
    b, s = infer(frame_bgr)
    all_boxes.append(np.asarray(b, float).reshape(-1, 4))
    all_scores.append(np.asarray(s, float).reshape(-1))
    if cfg.tiling_enabled:
        tiles = tile_grid(w, h, cfg)
        if len(tiles) > 1:
            for x0, y0, side in tiles:
                tb, ts = infer(frame_bgr[y0:y0 + side, x0:x0 + side])
                tb = np.asarray(tb, float).reshape(-1, 4) + np.array([x0, y0, x0, y0], float)
                keep = _internal_edge_ok(tb, x0, y0, side, w, h, cfg.tile_edge_margin_frac * side)
                all_boxes.append(tb[keep])
                all_scores.append(np.asarray(ts, float).reshape(-1)[keep])
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    return filter_detections(boxes, scores, np.full(len(scores), class_id), frame_w=w, frame_h=h, cfg=cfg,
                             keep_class=class_id)
