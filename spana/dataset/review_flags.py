"""Review-priority flags for pre-labelled frames: pure rules over a frame's saved boxes and pixels.

Used ONLY to order human review (tools/roboflow_review_flags.py adds tags; annotations are never changed). A flag
means "look here first", never "this box is wrong". Every size is a fraction of the frame height and every threshold
comes from config (review_flags); nothing is specific to a match, venue or camera.

  ball_tiny      a ball box whose longer side is below review_flags.ball_tiny.min_size_frac x frame height
  ball_offpitch  a ball box whose centre lies outside the grass mask (prelabel.grass settings)
  player_edge    a person box whose foot point is within review_flags.player_edge.band_frac x frame height of the
                 grass mask's outer boundary, or outside the mask (possible sideline staff). Boxes cut by the
                 frame's bottom border have no observed foot point and are not judged
  dense          at least min_pairs pairs of person boxes with IoU >= pair_iou_min (occlusion: check for misses)
  ball_static    between two consecutive sampled frames of one match (at most max_gap_s apart) the camera moved,
                 yet a ball box stayed at the same place IN THE SCENE: its centre, carried into the next frame by
                 the camera motion estimated from background features, lands on a ball box there (likely a spare
                 ball, or a dead ball). No reliable camera estimate -> no flag
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .prefilter import bottom_centre, foot_unobserved, iou

FLAGS = ("ball_static", "ball_offpitch", "ball_tiny", "player_edge", "dense")


def boxes_from_annotation(annotation) -> List[dict]:
    """Saved annotation of one image (GET images/{id}: {"boxes": [{label, x, y, width, height}]}, centre format in
    pixels) -> [{class, box [x1, y1, x2, y2]}]. None / no boxes -> []. Any other shape raises ValueError."""
    if not annotation:
        return []
    boxes = annotation.get("boxes") if isinstance(annotation, dict) else None
    if not isinstance(boxes, list):
        raise ValueError("unexpected annotation shape")
    out = []
    for b in boxes:
        try:
            x, y, w, h = (float(b[k]) for k in ("x", "y", "width", "height"))
            label = str(b["label"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("unexpected annotation box shape") from None
        out.append({"class": label, "box": [x - w / 2, y - h / 2, x + w / 2, y + h / 2]})
    return out


def centre(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _at(mask: np.ndarray, x: float, y: float) -> Tuple[int, int]:
    h, w = mask.shape
    return min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))


def ball_tiny(balls: Sequence[dict], frame_height: int, min_size_frac: float) -> List[str]:
    limit = min_size_frac * frame_height
    return [f"ball box {[round(v, 1) for v in b['box']]}: longer side {side:.1f} px < {limit:.1f} px"
            for b in balls
            for side in [max(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1])] if side < limit]


def ball_offpitch(balls: Sequence[dict], mask: np.ndarray) -> List[str]:
    out = []
    for b in balls:
        x, y = centre(b["box"])
        if not mask[_at(mask, x, y)]:
            out.append(f"ball box {[round(v, 1) for v in b['box']]}: centre ({x:.1f}, {y:.1f}) outside grass mask")
    return out


def boundary_distance(mask: np.ndarray, reach_px: int) -> np.ndarray:
    """Per pixel: distance (px) to the nearest non-grass pixel, exact up to `reach_px`. The frame border does not
    count as a boundary (the grass is assumed to continue beyond the frame: the mask is extended by replication), so
    only the visible outer edge of the grass is measured."""
    h, w = mask.shape
    pad = reach_px + 2
    padded = cv2.copyMakeBorder(mask.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 3)[pad:pad + h, pad:pad + w]


def player_edge(people: Sequence[dict], mask: np.ndarray, band_frac: float, frame_edge_margin_frac: float) -> List[str]:
    h = mask.shape[0]
    band = band_frac * h
    dist = boundary_distance(mask, int(np.ceil(band)))
    out = []
    for b in people:
        if foot_unobserved(b["box"], h, frame_edge_margin_frac):
            continue
        x, y = bottom_centre(b["box"])
        d = float(dist[_at(mask, x, np.ceil(y) - 1)])
        if d <= band:
            where = "outside the grass mask" if d == 0 else f"{d:.1f} px inside the grass boundary (band {band:.1f} px)"
            out.append(f"{b['class']} box {[round(v, 1) for v in b['box']]}: foot point ({x:.1f}, {y:.1f}) {where}")
    return out


def dense(people: Sequence[dict], pair_iou_min: float, min_pairs: int) -> List[str]:
    pairs = sum(iou(a["box"], b["box"]) >= pair_iou_min for i, a in enumerate(people) for b in people[i + 1:])
    return [f"{pairs} overlapping person-box pairs (IoU >= {pair_iou_min}; flag from {min_pairs})"] if pairs >= min_pairs else []


# ------------------------------------------------------------------------------------------------ ball_static ----
def background_features(frame_bgr: np.ndarray, boxes: Sequence[dict], s) -> tuple:
    """ORB keypoints / descriptors of the frame outside every annotated box (players and balls move on their own).
    `s` is a review_flags.ball_static section."""
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    keep = np.full((h, w), 255, np.uint8)
    pad = int(round(s.box_pad_frac * h))
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b["box"])
        keep[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
    return cv2.ORB_create(s.max_features).detectAndCompute(grey, keep)


def camera_transform(feat_a: tuple, feat_b: tuple, frame_height: int, s) -> Optional[np.ndarray]:
    """2x3 similarity transform carrying frame A's background onto frame B's (RANSAC over ORB matches), or None when
    it is not reliable (fewer than min_inliers consistent matches)."""
    (ka, da), (kb, db) = feat_a, feat_b
    if da is None or db is None or len(ka) < s.min_inliers or len(kb) < s.min_inliers:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < s.min_inliers:
        return None
    a = np.float32([ka[m.queryIdx].pt for m in matches])
    b = np.float32([kb[m.trainIdx].pt for m in matches])
    t, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                             ransacReprojThreshold=s.ransac_reproj_frac * frame_height)
    if t is None or inliers is None or int(inliers.sum()) < s.min_inliers:
        return None
    return t


def ball_static(balls_a: Sequence[dict], balls_b: Sequence[dict], transform: Optional[np.ndarray], frame_height: int,
                s) -> List[Tuple[dict, dict, float, float]]:
    """-> [(ball in A, ball in B, residual px, camera shift px)]: A's ball centre carried into B by the camera
    transform lands within max_shift_frac x frame height of a ball in B, while the camera moved that point by at
    least min_camera_shift_frac x frame height."""
    if transform is None:
        return []
    out = []
    for a in balls_a:
        ax, ay = centre(a["box"])
        px, py = transform @ np.array([ax, ay, 1.0])
        shift = float(np.hypot(px - ax, py - ay))
        if shift < s.min_camera_shift_frac * frame_height:
            continue
        for b in balls_b:
            bx, by = centre(b["box"])
            resid = float(np.hypot(px - bx, py - by))
            if resid <= s.max_shift_frac * frame_height:
                out.append((a, b, resid, shift))
    return out


def frame_flags(cfg, frame_bgr: np.ndarray, boxes: Sequence[dict], mask: np.ndarray) -> Dict[str, List[str]]:
    """Single-frame flags -> {flag name: [reason, ...]} (only flags that fired). `mask`: grass mask of the frame."""
    r, ball = cfg.review_flags, cfg.prelabel.ball_class
    h = frame_bgr.shape[0]
    balls = [b for b in boxes if b["class"] == ball]
    people = [b for b in boxes if b["class"] != ball]
    found = {"ball_tiny": ball_tiny(balls, h, r.ball_tiny.min_size_frac),
             "ball_offpitch": ball_offpitch(balls, mask),
             "player_edge": player_edge(people, mask, r.player_edge.band_frac,
                                        cfg.prelabel.grass.frame_edge_margin_frac),
             "dense": dense(people, r.dense.pair_iou_min, r.dense.min_pairs)}
    return {k: v for k, v in found.items() if v}
