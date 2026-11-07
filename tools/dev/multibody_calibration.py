# DEV diagnostic script (not product code). Calibrates the multi-body detection flag on DEV labels only.
# A detection is MULTI when it covers >= 2 labelled people (each person box inside it by >= COVER_IOA and its centre
# inside the detection); SINGLE when it covers exactly one. For each detection we print candidate generic features so
# the flag rule can be chosen from data. Usage: python -m tools.dev.multibody_calibration <label_dir> <detections.json>
import json
import sys
from pathlib import Path

import numpy as np

from tools.benchmark import load_frozen

COVER_IOA = 0.5


def _ioa(a, b):
    """Share of a inside b."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def _covers(d, g):
    cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    return d[0] <= cx <= d[2] and d[1] <= cy <= d[3] and _ioa(g, d) >= COVER_IOA


def flagged(boxes, scores, k, member_ioa, member_score, extent):
    """Box k encloses a confident detection (inside k by >= member_ioa, score >= member_score) and extends beyond
    it by >= extent in width or height."""
    b = boxes[k]
    for j, a in enumerate(boxes):
        if j == k or scores[j] < member_score or _ioa(a, b) < member_ioa:
            continue
        if max((b[2] - b[0]) / max(a[2] - a[0], 1e-9), (b[3] - b[1]) / max(a[3] - a[1], 1e-9)) >= extent:
            return True
    return False


GRID = [(mi, ms, ex) for mi in (0.6, 0.7) for ms in (0.5, 0.6) for ex in (1.15, 1.2, 1.3, 1.4)]


def main(label_dir, det_path, score_high="0.5"):
    frozen, gt = load_frozen(Path(label_dir))
    doc = json.loads(Path(det_path).read_text())
    sh = float(score_high)
    counts = {g: [0, 0, 0, 0] for g in GRID}          # multi flagged, multi spawnable flagged, single flagged, single spawnable flagged
    tot = [0, 0, 0, 0]
    for fr in doc["frames"]:
        f = fr["frame"]
        if f not in gt.players:
            continue
        boxes = [r[:4] for r in fr["detections"]]
        scores = [r[4] for r in fr["detections"]]
        people = list(gt.players[f].values())
        for k, d in enumerate(boxes):
            n = sum(_covers(d, g) for g in people)
            if n == 0:
                continue
            base = 0 if n >= 2 else 2
            hi = scores[k] >= sh
            tot[base] += 1
            tot[base + 1] += hi
            for g in GRID:
                if flagged(boxes, scores, k, *g):
                    counts[g][base] += 1
                    counts[g][base + 1] += hi
    print(f"{frozen['clip']['name']}: multi {tot[0]} (score>=high {tot[1]}), single {tot[2]} (score>=high {tot[3]})")
    for g, c in counts.items():
        print(f"  member_ioa {g[0]} member_score>={g[1]} extent>={g[2]}: multi {c[0]} ({c[1]} high)  single {c[2]} ({c[3]} high)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
