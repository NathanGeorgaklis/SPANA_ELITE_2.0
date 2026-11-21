"""Builds tests/data/fresh07_crossing_f40_90.json: the exact tracker inputs of the FRESH_07 crossing of players 14
and 15 (DEV; frames 40-90). The camera observation and appearance descriptors are computed on the whole frame with
every detection, as `spana.cli track` does. Detections, with their descriptors, are kept when they intersect the
ROI. The frozen DEV labels of the two players are kept on the labelled frames.

    PYTHONPATH=. .venv/bin/python records/silent_transfer_fix_2026-11-20/make_fixture.py \
        --video ~/Downloads/SPANA_fresh_clips/FRESH_07_G3_1205_midfield_passing_switch.mp4 \
        --detections runs/person_view_compare/F07_local.det.json \
        --labels data/labels/FRESH_07_G3_1205_midfield_passing_switch/v1 --out tests/data/fresh07_crossing_f40_90.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spana.artifacts import read_json
from spana.cli import describe, detections_from_frame
from spana.config import load_config
from spana.labels import parse_mot
from spana.tracking.camera import CameraMotionEstimator
from spana.video import VideoSource

FIRST, LAST = 40, 90
PLAYERS = ("14", "15")
ROI = (1300.0, 280.0, 1750.0, 520.0)   # x1, y1, x2, y2: both players' paths over the window, with margin


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    video = VideoSource(a.video)
    ddoc = read_json(a.detections)
    det_frames = {fr["frame"]: fr for fr in ddoc["frames"]}
    cam = CameraMotionEstimator(cfg.camera, video.meta.width, video.meta.height, video.meta.fps)
    gt_path = Path(a.labels) / "gt.txt"
    gt = parse_mot(gt_path.read_text(), video.meta.width, video.meta.height, sorted(det_frames))
    frames, labels, occluded = [], {}, {}
    first_seen = False
    for i, t, img in video.frames(min(det_frames), LAST + 1):
        dets = detections_from_frame(det_frames[i])
        boxes = [d.box for d in dets]
        obs = cam.observe(img, boxes)                 # every frame from the clip start: camera state as in the run
        if i < FIRST:
            continue
        desc = describe(img, boxes, cfg.appearance)
        keep = [k for k, b in enumerate(boxes) if b[2] > ROI[0] and b[0] < ROI[2] and b[3] > ROI[1] and b[1] < ROI[3]]
        frames.append({
            "frame": i, "t": t,
            "camera": {"status": obs.status.value if first_seen else "FIRST", "affine": [list(r) for r in obs.affine],
                       "inliers": obs.inliers, "hist_distance": obs.hist_distance,
                       "motion_std_px": obs.motion_std_px},
            "detections": [[*dets[k].box, dets[k].score, dets[k].class_id, int(dets[k].multi_body)] for k in keep],
            "descriptors": [None if desc[k] is None else [float(x) for x in desc[k]] for k in keep]})
        first_seen = True
        if any(p in PLAYERS for p in gt.players.get(i, {})):
            labels[str(i)] = {p: list(b) for p, b in gt.players[i].items() if p in PLAYERS}
            occluded[str(i)] = sorted(p for p in gt.occluded.get(i, set()) if p in PLAYERS)
    doc = {"source": {"clip": Path(a.video).stem, "split": "DEV", "labels_gt_sha256": sha(gt_path),
                      "detections_sha256": sha(Path(a.detections)), "video": video.meta.sha256,
                      "frames": [FIRST, LAST], "players": list(PLAYERS), "roi": list(ROI)},
           "frames": frames, "labels": labels, "occluded": occluded}
    Path(a.out).write_text(json.dumps(doc, separators=(",", ":")))
    print(a.out, len(frames), "frames", len(labels), "labelled")


if __name__ == "__main__":
    main()
