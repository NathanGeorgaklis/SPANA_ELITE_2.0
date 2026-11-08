# DEV diagnostic script (not product code). How often is an appearance crop UNKNOWN only because the whole box
# overlaps another detection (IoU > clean_max_iou) while the torso band itself is not covered by any other box?
# Reported for (a) all confident detections and (b) the IDENTITY_UNCERTAIN tracks of an identity_resolution_probe
# run: observations before the trigger (own appearance) and NO_CLEAN_CROP frames while uncertain.
#   python tools/dev/torso_clean_check.py CONFIG DETECTIONS TRACKS IDPROBE_JSON
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from spana.config import load_config  # noqa: E402
from spana.geometry import iou_matrix  # noqa: E402

TORSO_COVER_MAX = 0.10      # torso band "clean" when other boxes cover at most this share of it (diagnostic only)


def torso(b, ac):
    h, w = b[3] - b[1], b[2] - b[0]; cx = (b[0] + b[2]) / 2
    return [cx - w * ac.torso_width_frac / 2, b[1] + ac.torso_top_frac * h, cx + w * ac.torso_width_frac / 2,
            b[1] + ac.torso_bottom_frac * h]


def covered(t, others):
    """Share of rectangle t covered by the union of others (rasterised on a 20x20 grid)."""
    xs = np.linspace(t[0], t[2], 20); ys = np.linspace(t[1], t[3], 20)
    X, Y = np.meshgrid(xs, ys); m = np.zeros_like(X, bool)
    for o in others:
        m |= (X >= o[0]) & (X <= o[2]) & (Y >= o[1]) & (Y <= o[3])
    return float(m.mean())


def status(k, boxes, ac):
    b = boxes[k]
    if b[3] - b[1] < ac.min_crop_height_px:
        return "too_small"
    ious = iou_matrix(np.array([b]), np.array(boxes))[0]; ious[k] = 0
    if ious.max() <= ac.clean_max_iou:
        return "clean_now"
    others = [o for j, o in enumerate(boxes) if j != k]
    return "torso_clean" if covered(torso(b, ac), others) <= TORSO_COVER_MAX else "torso_covered"


def main(cfg_path, det_path, trk_path, probe_path):
    ac = load_config(cfg_path).appearance
    dets = {f["frame"]: [r[:4] for r in f["detections"]] for f in json.load(open(det_path))["frames"]}
    scores = {f["frame"]: [r[4] for r in f["detections"]] for f in json.load(open(det_path))["frames"]}
    c = {"clean_now": 0, "torso_clean": 0, "torso_covered": 0, "too_small": 0}
    for f, bs in dets.items():
        for k in range(len(bs)):
            if scores[f][k] >= 0.5:
                c[status(k, bs, ac)] += 1
    n = sum(c.values())
    print("confident detections:", {k: f"{v} ({v / n:.0%})" for k, v in c.items()})
    trk = {f["frame"]: {t["track_id"]: t for t in f["tracks"]} for f in json.load(open(trk_path))["frames"]}
    probe = json.load(open(probe_path))
    pre = {"own_unknown_tracks": 0, "would_gain_appearance": 0}
    during = {"torso_clean": 0, "torso_covered": 0, "too_small": 0, "clean_now": 0}
    for t in probe["tracks"]:
        tid, rows = t["track_id"], probe["rows"][t["track_id"]]
        involved = (not t["own_appearance_known"]) or t["rival_without_appearance"]
        if not t["own_appearance_known"]:
            pre["own_unknown_tracks"] += 1
            obs = [(f, x) for f, ts in trk.items() if f < t["first_frame"] and (x := ts.get(tid)) and
                   x["source"] == "OBSERVED"]
            if any(status(x["detection_index"], dets[f], ac) in ("torso_clean", "clean_now") for f, x in obs):
                pre["would_gain_appearance"] += 1
        for r in rows:
            if r["reason"] == "NO_CLEAN_CROP":
                x = trk[r["frame"]][tid]
                during[status(x["detection_index"], dets[r["frame"]], ac)] += 1
    print("uncertain tracks with unknown own appearance:", pre)
    print("NO_CLEAN_CROP frames while uncertain:", during)


if __name__ == "__main__":
    main(*sys.argv[1:])
