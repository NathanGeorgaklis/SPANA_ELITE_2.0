"""Detection metrics against frozen labels. Pure functions: boxes, classes and scores only; no domain logic.

A frame is {"gt": [{"class", "box"}], "pred": [{"class", "box", "score"}]} with boxes [x1, y1, x2, y2] in the same
pixel frame. Conventions:
  * AP: predictions of a class sorted by score over all frames; each is matched to the unmatched same-class label
    of its frame with the highest IoU >= the threshold; precision envelope, 101-point interpolation (COCO style).
    A class without labels has AP None and is left out of the mean.
  * Precision / recall / confusion are taken at one operating confidence.
  * Confusion ignores the class when matching (highest IoU first), so a player predicted as a referee counts in
    row player, column referee; unmatched labels go to column MISSED, unmatched predictions to row BACKGROUND.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MISSED, BACKGROUND = "MISSED", "BACKGROUND"
MAP_IOUS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_class(gt: List[dict], pred: List[dict], cls: str, iou_min: float) -> Tuple[List[Optional[int]], List[bool]]:
    """Same-class greedy match of one frame, predictions in descending score.
    -> (for each label of the class: index into `pred` of its match or None, for each prediction of the class: hit)."""
    gi = [i for i, g in enumerate(gt) if g["class"] == cls]
    pi = sorted((i for i, p in enumerate(pred) if p["class"] == cls), key=lambda i: -pred[i]["score"])
    taken: Dict[int, int] = {}
    hits = []
    for p in pi:
        best, best_iou = None, iou_min
        for g in gi:
            if g in taken:
                continue
            v = iou(gt[g]["box"], pred[p]["box"])
            if v >= best_iou:
                best, best_iou = g, v
        if best is not None:
            taken[best] = p
        hits.append(best is not None)
    return [taken.get(g) for g in gi], hits


def average_precision(frames: List[dict], cls: str, iou_min: float) -> Optional[float]:
    n_gt = sum(g["class"] == cls for f in frames for g in f["gt"])
    if not n_gt:
        return None
    scored = []
    for f in frames:
        _, hits = match_class(f["gt"], f["pred"], cls, iou_min)
        scores = sorted((p["score"] for p in f["pred"] if p["class"] == cls), reverse=True)
        scored += list(zip(scores, hits))
    if not scored:
        return 0.0
    scored.sort(key=lambda t: -t[0])
    tp = np.cumsum([h for _, h in scored])
    fp = np.cumsum([not h for _, h in scored])
    recall = tp / n_gt
    precision = np.maximum.accumulate((tp / (tp + fp))[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def at_confidence(frames: List[dict], confidence: float) -> List[dict]:
    return [{**f, "pred": [p for p in f["pred"] if p["score"] >= confidence]} for f in frames]


def precision_recall(frames: List[dict], cls: str, iou_min: float) -> dict:
    """Counts at the predictions given (filter with at_confidence first)."""
    tp = fp = fn = 0
    for f in frames:
        matched, hits = match_class(f["gt"], f["pred"], cls, iou_min)
        tp += sum(hits)
        fp += len(hits) - sum(hits)
        fn += sum(m is None for m in matched)
    return {"labels": tp + fn, "predictions": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None}


def confusion(frames: List[dict], classes: Sequence[str], iou_min: float) -> Dict[str, Dict[str, int]]:
    """{label class or BACKGROUND: {predicted class or MISSED: count}} at the predictions given."""
    out = {r: {c: 0 for c in (*classes, MISSED)} for r in (*classes, BACKGROUND)}
    for f in frames:
        pairs = sorted(((iou(g["box"], p["box"]), gi, pi) for gi, g in enumerate(f["gt"])
                        for pi, p in enumerate(f["pred"])), reverse=True)
        g_used, p_used = set(), set()
        for v, gi, pi in pairs:
            if v < iou_min:
                break
            if gi in g_used or pi in p_used:
                continue
            g_used.add(gi)
            p_used.add(pi)
            out[f["gt"][gi]["class"]][f["pred"][pi]["class"]] += 1
        for gi, g in enumerate(f["gt"]):
            if gi not in g_used:
                out[g["class"]][MISSED] += 1
        for pi, p in enumerate(f["pred"]):
            if pi not in p_used:
                out[BACKGROUND][p["class"]] += 1
    return out


def recall_by_size(frames: List[dict], cls: str, iou_min: float, bucket_of) -> Dict[str, dict]:
    """Recall of `cls` labels grouped by bucket_of(short side in pixels), at the predictions given."""
    out: Dict[str, dict] = {}
    for f in frames:
        matched, _ = match_class(f["gt"], f["pred"], cls, iou_min)
        boxes = [g["box"] for g in f["gt"] if g["class"] == cls]
        for box, m in zip(boxes, matched):
            b = out.setdefault(bucket_of(min(box[2] - box[0], box[3] - box[1])), {"labels": 0, "found": 0})
            b["labels"] += 1
            b["found"] += m is not None
    for b in out.values():
        b["recall"] = round(b["found"] / b["labels"], 4)
    return out


def frame_errors(frame: dict, cls: str, iou_min: float) -> int:
    """Missed labels + false predictions of `cls` in one frame, at the predictions given."""
    matched, hits = match_class(frame["gt"], frame["pred"], cls, iou_min)
    return sum(m is None for m in matched) + len(hits) - sum(hits)


def summarize(frames: List[dict], classes: Sequence[str], confidence: float, iou_min: float) -> dict:
    per_class = {}
    op = at_confidence(frames, confidence)
    for c in classes:
        aps = [average_precision(frames, c, t) for t in MAP_IOUS]
        per_class[c] = {"ap50": None if aps[0] is None else round(aps[0], 4),
                        "ap50_95": None if aps[0] is None else round(float(np.mean(aps)), 4),
                        **precision_recall(op, c, iou_min)}
    have = [v for v in per_class.values() if v["ap50"] is not None]
    return {"map50": round(float(np.mean([v["ap50"] for v in have])), 4) if have else None,
            "map50_95": round(float(np.mean([v["ap50_95"] for v in have])), 4) if have else None,
            "per_class": per_class, "confusion": confusion(op, classes, iou_min)}
