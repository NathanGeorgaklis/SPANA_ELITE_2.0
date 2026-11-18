"""DEV diagnostic (not product code): trace the gate / low-score breaks found by continuity_attribution.

    python -m tools.dev.gate_break_trace --attribution ATTRIBUTION.json --out OUT.json \
        --clip NAME DETECTIONS.json TRACKS.json DECISIONS.jsonl [--clip ...]

For every in-scope fragment (ByteTrack kept one track) whose cause is OTHER:GATE_FAILED:* or LOW_SCORE_UNMATCHED,
report at the divergence frame:
  track A        state, confirmed, frames since its last observation, its last observed box and its predicted box
  detection d    (the one ByteTrack used) box, score, multi_body flag, touches the image border
  geometry       height of d / predicted height and / last observed height; centre distance in body heights;
                 IoU with the prediction and with A's last observed box; whether d is nested in / encloses another
                 detection on the frame (partial / full body pair)
  tracker log    A's unmatched reason and gate report for d (when logged), and what happened to d (taken by,
                 spawned, refused)
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from spana.artifacts import read_json
from tools.benchmark import _ioa

BORDER_PX = 4.0     # reporting only: a box edge this close to the image edge is "at the border"


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def h(b) -> float:
    return b[3] - b[1]


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def trace(name, det_p, trk_p, dec_p, rows):
    dets_doc, trk_doc = read_json(det_p), read_json(trk_p)
    W, H = trk_doc["video"]["width"], trk_doc["video"]["height"]
    dets = {fr["frame"]: fr["detections"] for fr in dets_doc["frames"]}
    trk = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in trk_doc["frames"]}
    dec = collections.defaultdict(list)
    for line in Path(dec_p).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            dec[d["frame"]].append(d)
    out = []
    for r in rows:
        e = r["evidence"]
        f, di, a_id = e["frame"], e["detection"], r["from"]
        d = dets[f][di]
        box = d[:4]
        a = trk[f].get(a_id)
        last = None
        for g in range(f - 1, r["f0"] - 1, -1):
            t = trk.get(g, {}).get(a_id)
            if t is not None and t["source"] == "OBSERVED":
                last = (g, t["box"])
                break
        pred = None if a is None else a["box"]
        others = [o[:4] for k, o in enumerate(dets[f]) if k != di]
        nested_in = max((_ioa(box, o) for o in others), default=0.0)
        encloses = max((_ioa(o, box) for o in others), default=0.0)
        unm = next((x for x in dec[f] if x["kind"] == "unmatched" and x["track_id"] == a_id), None)
        cand = None if unm is None else next((c for c in unm["candidates"] if c["detection"] == di), None)
        fate = [x for x in dec[f] if x.get("detection") == di and x["kind"] in
                ("match", "spawn", "spawn_refused", "recovery", "doubt_link")]
        out.append({
            "clip": name, "player": r["player"], "cause": r["cause"], "frame": f, "from": a_id, "to": r["to"],
            "a_state": None if a is None else a["state"], "a_confirmed": None if a is None else a["confirmed"],
            "frames_since_observed": None if last is None else f - last[0],
            "det": {"box": box, "score": d[4], "multi_body": bool(d[6]) if len(d) > 6 else None,
                    "at_border": box[0] <= BORDER_PX or box[1] <= BORDER_PX or box[2] >= W - BORDER_PX
                    or box[3] >= H - BORDER_PX},
            "h_ratio_pred": None if pred is None else round(h(box) / h(pred), 3),
            "h_ratio_last": None if last is None else round(h(box) / h(last[1]), 3),
            "dist_h_last": None if last is None else round(
                ((centre(box)[0] - centre(last[1])[0]) ** 2 + (centre(box)[1] - centre(last[1])[1]) ** 2) ** 0.5
                / h(last[1]), 3),
            "iou_pred": None if pred is None else round(iou(box, pred), 3),
            "iou_last": None if last is None else round(iou(box, last[1]), 3),
            "nested_in_other_det": round(nested_in, 3), "encloses_other_det": round(encloses, 3),
            "unmatched_reason": None if unm is None else unm["reason"],
            "gate": None if cand is None else {k: cand[k] for k in (
                "dist_h", "gate_radius_h", "mahalanobis2", "height_ratio", "iou", "failed", "taken_by")},
            "fate_of_det": fate,
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--attribution", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clip", nargs=4, action="append", required=True,
                   metavar=("NAME", "DETECTIONS", "TRACKS", "DECISIONS"))
    a = p.parse_args()
    rows = [r for r in json.loads(Path(a.attribution).read_text())["rows"]
            if r["bytetrack_continuous"] and r["fragment"]
            and (r["cause"].startswith("OTHER:GATE_FAILED") or r["cause"] == "LOW_SCORE_UNMATCHED")]
    out = []
    for name, det_p, trk_p, dec_p in a.clip:
        out += trace(name, det_p, trk_p, dec_p, [r for r in rows if r["clip"] == name])
    Path(a.out).write_text(json.dumps(out, indent=1))
    for o in out:
        g = o["gate"] or {}
        print(o["clip"], o["player"], o["frame"], o["cause"], o["a_state"], "conf" if o["a_confirmed"] else "unconf",
              "gap", o["frames_since_observed"], "score %.2f" % o["det"]["score"],
              "border" if o["det"]["at_border"] else "", "mb" if o["det"]["multi_body"] else "",
              "h/pred", o["h_ratio_pred"], "h/last", o["h_ratio_last"], "d_last", o["dist_h_last"],
              "iou_pred", o["iou_pred"], "nested", o["nested_in_other_det"], "encl", o["encloses_other_det"],
              "m2", g.get("mahalanobis2"), "reason", o["unmatched_reason"],
              "fate", [(x["kind"], x.get("track_id"), x.get("reason")) for x in o["fate_of_det"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
