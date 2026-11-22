"""Appearance-veto track changes in detail (DEV analysis only): motion gating, exclusivity, overlap of the vetoed box,
spawn timing; and, on the doubt-link run, which remaining vetoes a narrower exclusivity rule could take.
    python veto_details.py DEFAULT_CHANGES.json DEFAULT_RUN_DIR DOUBT_CHANGES.json DOUBT_RUN_DIR"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from spana.geometry import iou_matrix

MOTION = {"centre_ok", "mahalanobis_ok", "height_ok"}


def vetoes(path):
    return [e for e in json.load(open(path))["changes"]
            if e.get("reason") == "GATE_FAILED" and e.get("first_failed_gate") == "appearance_veto"]


def decisions(run, k):
    return [json.loads(l) for l in open(Path(run) / f"{k}_ours.trk.decisions.jsonl")]


def main(dch, drun, bch, brun):
    v = vetoes(dch)
    dec = {k: decisions(drun, k) for k in {e["clip"] for e in v}}
    docs = {k: {fr["frame"]: fr for fr in json.load(open(Path(drun) / f"{k}_ours.trk.json"))["frames"]} for k in dec}
    c, ious, sims, ovl = Counter(), [], [], []
    for e in v:
        n = e["nearest"]
        ious.append(n["iou"]); sims.append(n["appearance_similarity"])
        u = next(x for x in dec[e["clip"]] if x["kind"] == "unmatched" and x["track_id"] == e["from"] and x["frame"] == e["miss_frame"])
        mg = [x for x in u["candidates"] if not set(x["failed"]) & MOTION]
        c[f"motion_gated_detections={len(mg)}"] += 1
        c["only_appearance_failed" if n["failed"] == ["appearance_veto"] else "other_gates_failed_too"] += 1
        c["vetoed_box_taken_by_other" if n["taken_by"] else "vetoed_box_free"] += 1
        c["A_" + str(e["a_identity"])] += 1
        sp = next((x for x in dec[e["clip"]] if x["kind"] == "spawn" and x["track_id"] == e["to"]), None)
        c["B_spawned_on_the_vetoed_box_same_frame" if sp and sp["frame"] == e["miss_frame"] and sp["detection"] == n["detection"]
          else "B_elsewhere_or_later"] += 1
        fr = docs[e["clip"]][e["miss_frame"]]
        obs = [t for t in fr["tracks"] if t["source"] == "OBSERVED"]
        own = [t for t in obs if t["detection_index"] == n["detection"]]
        if own:
            others = np.array([t["box"] for t in obs if t is not own[0]], float).reshape(-1, 4)
            ovl.append(float(iou_matrix(np.array([own[0]["box"]], float), others).max()) if len(others) else 0.0)
    out = {"vetoes": len(v), "counts": dict(c), "iou_with_prediction_median": float(np.median(ious)),
           "iou_ge_0.40": sum(i >= 0.40 for i in ious), "similarity_median": float(np.median(sims)),
           "similarity_range": [min(sims), max(sims)],
           "vetoed_box_max_iou_with_other_tracked_box": {"n": len(ovl), "le_0.05": sum(x <= 0.05 for x in ovl),
                                                         "max": max(ovl) if ovl else None}}
    # doubt-link run: what would a narrower exclusivity rule (no other UNMATCHED track gates the box) add?
    w = vetoes(bch)
    bdec = {k: decisions(brun, k) for k in {e["clip"] for e in w}}
    n2 = Counter()
    for e in w:
        u = next(x for x in bdec[e["clip"]] if x["kind"] == "unmatched" and x["track_id"] == e["from"] and x["frame"] == e["miss_frame"])
        mg = [x for x in u["candidates"] if not set(x["failed"]) & MOTION]
        if len(mg) != 1:
            n2["not_single_motion_gated"] += 1; continue
        x = mg[0]
        if x["taken_by"]:
            n2["box_taken_by_other_track"] += 1; continue
        if x["iou"] < 0.40:
            n2["iou_below_0.40"] += 1; continue
        others = [y for y in bdec[e["clip"]] if y["kind"] == "unmatched" and y["frame"] == e["miss_frame"]
                  and y["track_id"] != e["from"] and any(z["detection"] == x["detection"] and not set(z["failed"]) & MOTION
                                                         for z in y["candidates"])]
        n2["also_gated_by_another_unmatched_track" if others else "eligible_under_narrower_rule"] += 1
    out["doubt_link_run"] = {"remaining_vetoes": len(w), "narrower_rule": dict(n2)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
