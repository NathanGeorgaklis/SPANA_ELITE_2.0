# DEV tool (not product code). Records the exact tracker inputs (detections, camera observation, appearance
# descriptors) of a frame range of a DEV clip, restricted to a region of interest, plus the frozen DEV labels of
# chosen players in that range, as a small JSON regression fixture.
#   python -m tools.dev.make_tracking_fixture LABEL_DIR VIDEO CONFIG DETECTIONS F0 F1 PLAYERS(comma) OUT
import json
import sys
from pathlib import Path

import numpy as np

from spana.appearance import describe
from spana.artifacts import detections_from_frame, read_json
from spana.config import load_config
from spana.tracking.camera import CameraMotionEstimator
from spana.video import VideoSource
from tools.benchmark import load_frozen

MARGIN_H = 1.5          # ROI = players' labelled boxes over the range, grown by this many body heights


def main(label_dir, video_path, cfg_path, det_path, f0, f1, players, out):
    f0, f1, players = int(f0), int(f1), players.split(",")
    fz, gt = load_frozen(Path(label_dir))
    boxes = [gt.players[f][g] for f in gt.frames if f0 <= f <= f1 for g in players if g in gt.players[f]]
    b = np.array(boxes); h = float(np.median(b[:, 3] - b[:, 1]))
    roi = [b[:, 0].min() - MARGIN_H * h, b[:, 1].min() - MARGIN_H * h, b[:, 2].max() + MARGIN_H * h,
           b[:, 3].max() + MARGIN_H * h]
    cfg = load_config(cfg_path)
    video, ddoc = VideoSource(video_path), read_json(det_path)
    det_frames = {fr["frame"]: fr for fr in ddoc["frames"]}
    cam = CameraMotionEstimator(cfg.camera, video.meta.width, video.meta.height, video.meta.fps)
    first = min(det_frames)
    frames = []
    for i, t, img in video.frames(first, f1 + 1):          # camera estimator needs the full history
        if (i - first) % ddoc["frame_stride"]:
            continue
        dets = detections_from_frame(det_frames[i])
        all_boxes = [d.box for d in dets]
        co = cam.observe(img, all_boxes)
        if i < f0:
            continue
        desc = describe(img, all_boxes, cfg.appearance)
        keep = [k for k, d in enumerate(dets) if roi[0] <= (d.box[0] + d.box[2]) / 2 <= roi[2]
                and roi[1] <= (d.box[1] + d.box[3]) / 2 <= roi[3]]
        frames.append({"frame": i, "t": t, "camera": {"status": co.status.value, "affine": co.affine,
                                                       "inliers": co.inliers, "hist_distance": co.hist_distance,
                                                       "motion_std_px": co.motion_std_px},
                       "detections": [[*dets[k].box, dets[k].score, dets[k].class_id, int(dets[k].multi_body)]
                                      for k in keep],
                       "descriptors": [None if desc[k] is None else [round(float(v), 6) for v in desc[k]]
                                       for k in keep]})
    labels = {str(f): {g: gt.players[f][g] for g in players if g in gt.players[f]}
              for f in gt.frames if f0 <= f <= f1}
    occluded = {str(f): sorted(set(players) & gt.occluded.get(f, set())) for f in gt.frames if f0 <= f <= f1}
    doc = {"source": {"clip": fz["clip"]["name"], "split": fz["split"], "labels_gt_sha256": fz["gt_sha256"],
                      "detections_sha256_inputs": ddoc["provenance"].get("inputs"), "frames": [f0, f1],
                      "players": players, "roi": [round(v, 1) for v in roi]},
           "frames": frames, "labels": labels, "occluded": occluded}
    Path(out).write_text(json.dumps(doc, separators=(",", ":")))
    print(out, len(frames), "frames")


if __name__ == "__main__":
    main(*sys.argv[1:])
