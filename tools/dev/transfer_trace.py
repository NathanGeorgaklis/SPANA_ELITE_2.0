# DEV diagnostic script (not product code). Replays tracking exactly as `spana track` (hash-checked) with a
# read-only probe and, for each identity transfer of the benchmark, traces the frames between the last labelled
# frame on the old player and the first on the new one:
#   * the transferring track's match decision (stage, cost, IoU, distance, contested, rivals);
#   * every detection near the track with its owner, plus the gate report (centre / Mahalanobis / height /
#     appearance / motion) of the transferring track AND of every other track for that detection;
#   * the labelled boxes of both players, linearly interpolated between labelled frames (for attribution only).
#   python -m tools.dev.transfer_trace LABEL_DIR VIDEO CONFIG DETECTIONS REF_TRACKS BENCHMARK OUT_JSON [--all]
import collections
import json
import sys
from pathlib import Path

import numpy as np

from spana.artifacts import sha256_doc
from tools.benchmark import load_frozen
from tools.dev import motion_gate_probe as M

WATCH = {"frames": set()}
REPORTS = collections.defaultdict(list)          # frame -> rows
NEAR_H = 1.5                                      # detections within this many track heights are reported


def _apply(self, matched, dets, desc, frame_index, time_s):
    if frame_index in WATCH["frames"]:
        owner = {i: self._tracks[k].track_id for k, (i, _) in matched.items()}
        for t in self._tracks:
            if t.state.value == "LOST" and time_s - t.last_observed_t > 1.0:
                continue
            for i, d in enumerate(dets):
                c = np.array([(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2])
                if np.hypot(*(c - t.kf.x[:2])) > NEAR_H * t.kf.height:
                    continue
                r = M._orig_gate(self, t, d, desc[i], time_s)
                REPORTS[frame_index].append({"track": t.track_id, "confirmed": t.confirmed, "state": t.state.value,
                                             "observed_last": t.observed_last_frame, "detection": i,
                                             "owner": owner.get(i), "gap_s": round(time_s - t.last_observed_t, 3),
                                             "vel_h_per_s": [round(float(v) / t.kf.height, 2) for v in t.kf.x[4:6]],
                                             **{k: r[k] for k in ("score", "dist_h", "gate_radius_h", "mahalanobis2",
                                                                  "height_ratio", "iou", "appearance_similarity",
                                                                  "motion_consistent", "failed")}})
    return M._orig_apply(self, matched, dets, desc, frame_index, time_s)


def _interp(gt, g, f):
    fs = sorted(x for x in gt.players if g in gt.players[x])
    lo = max((x for x in fs if x <= f), default=None); hi = min((x for x in fs if x >= f), default=None)
    if lo is None or hi is None:
        return None
    if lo == hi:
        return list(gt.players[lo][g])
    a, b, w = np.array(gt.players[lo][g]), np.array(gt.players[hi][g]), (f - lo) / (hi - lo)
    return [round(float(v), 1) for v in a + w * (b - a)]


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / max(u, 1e-9)


def main(label_dir, video, cfg_path, det_path, ref_tracks, bench, out_json, *flags):
    fz, gt = load_frozen(Path(label_dir))
    transfers = [t for t in json.load(open(bench))["events"]["transfers"] if "--all" in flags or not t["flagged"]]
    step = fz["frame_step"]
    for fl in flags:                      # --span=TRACK:F0:F1:FROM:TO traces an explicit frame range instead
        if fl.startswith("--span="):
            tid, f0, f1, a, b = fl[7:].split(":")
            transfers = [{"track_id": tid, "frame": int(f1), "from": a, "to": b, "flagged": None, "span": int(f1) - int(f0)}]
    for t in transfers:
        WATCH["frames"] |= set(range(t["frame"] - t.get("span", step) - 2, t["frame"] + 2))
    M._apply = _apply
    cfg, ddoc, frames, log = M.replay(video, cfg_path, det_path)
    if sha256_doc(frames) != sha256_doc(json.load(open(ref_tracks))["frames"]):
        raise SystemExit("probe changed behaviour")
    trk = {fr["frame"]: {o["track_id"]: o for o in fr["tracks"]} for fr in frames}
    dets = {fr["frame"]: fr["detections"] for fr in ddoc["frames"]}
    dec = collections.defaultdict(list)
    for d in log:
        dec[d["frame"]].append(d)
    out = []
    for t in transfers:
        tid, g_from, g_to, f1 = t["track_id"], t["from"], t["to"], t["frame"]
        rows = []
        for f in sorted(x for x in WATCH["frames"] if f1 - t.get("span", step) <= x <= f1 and x in trk):
            o = trk[f].get(tid)
            gf, gt_ = _interp(gt, g_from, f), _interp(gt, g_to, f)
            box = o["box"] if o else None
            m = next((d for d in dec[f] if d.get("track_id") == tid and d["kind"] in ("match", "recovery")), None)
            near = [r for r in REPORTS[f] if r["track"] == tid or (o and r["detection"] == o["detection_index"])]
            near_dets = sorted({r["detection"] for r in REPORTS[f] if r["track"] == tid})
            rows.append({
                "frame": f, "track_state": o and o["state"], "source": o and o["source"],
                "detection": o and o["detection_index"], "contested": o and o["contested"],
                "box": box, "iou_with_from_player": box and gf and round(_iou(box, gf), 3),
                "iou_with_to_player": box and gt_ and round(_iou(box, gt_), 3),
                "from_player_box_interp": gf, "to_player_box_interp": gt_,
                "match_decision": m and {k: m.get(k) for k in ("kind", "stage", "cost", "iou", "dist_h", "contested",
                                                               "rival_tracks", "rival_detections")},
                "other_decisions_for_track": [d for d in dec[f] if d.get("track_id") == tid
                                              and d["kind"] not in ("match", "unmatched")],
                "detections_near_track": [{"detection": i, "box": dets[f][i][:4], "score": dets[f][i][4],
                                           "multi_body": dets[f][i][6],
                                           "iou_from": gf and round(_iou(dets[f][i][:4], gf), 3),
                                           "iou_to": gt_ and round(_iou(dets[f][i][:4], gt_), 3)} for i in near_dets],
                "gate_reports": near})
        out.append({**t, "frames": rows})
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    for t in out:
        print(f"\n=== {t['track_id']} {t['from']}->{t['to']} at labelled frame {t['frame']}")
        for r in t["frames"]:
            md = r["match_decision"] or {}
            print(f"  f{r['frame']} {r['track_state']} {r['source']} det={r['detection']} contested={r['contested']} "
                  f"iou(from)={r['iou_with_from_player']} iou(to)={r['iou_with_to_player']} "
                  f"stage={md.get('stage')} cost={md.get('cost')} rivals={md.get('rival_tracks')}/{md.get('rival_detections')}")
            for d in r["detections_near_track"]:
                print(f"      d{d['detection']} s{d['score']:.2f} mb{d['multi_body']} iou_from={d['iou_from']} iou_to={d['iou_to']}")
            for g in r["gate_reports"]:
                print(f"        {g['track']}{'' if g['confirmed'] else '(u)'} {g['state'][:3]} d{g['detection']}"
                      f" owner={g['owner']} dist_h={g['dist_h']} m2={g['mahalanobis2']} hr={g['height_ratio']} "
                      f"iou={g['iou']} app={g['appearance_similarity']} motion={g['motion_consistent']} "
                      f"gap={g['gap_s']} failed={g['failed']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
