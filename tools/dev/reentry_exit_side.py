"""DEV diagnostic (not product code): when the followed player LEFT THE PICTURE, does the returning track appear at
the edge they left by, and do the wrong candidates? Decides whether an exit-side rule can make re-entry offers safer
while the player is off screen (records/reentry_exit_side_2026-11-19).

    python -m tools.dev.reentry_exit_side --config CFG --out OUT.json --window-s 10 \
        --clip LABELS_DIR TRACKS.json EMBEDDINGS.json [--clip ...] [--target-report TARGET_SCORE.json ...]

For a lost track A (last observed on frame index i) and a candidate B (born on frame index b > i):
  off screen   A ended LEFT_FRAME by frame b, or A's last box centre carried through the per-frame camera transforms
               to frame b lies outside the picture
  exit side    the side of the picture the carried centre is beyond (left / right / top / bottom, the largest
               excess); if still inside (ended LEFT_FRAME at the edge), the nearest side of the carried last box
  edge_dist    B's birth box distance from that side, as a share of the frame width (left / right) or height
               (top / bottom); 0 = touching the edge
  unknown      a CUT / UNKNOWN camera frame between i and b (position cannot say anything)
Labelled re-entries as tools/dev/reid_reentry_eval.py (NESTED_V2 owners). With --target-report, every re-ID offer in
the report gets the same features and its verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from spana.artifacts import read_json
from spana.config import load_config
from spana.geometry import affine_scale, apply_affine_points
from spana.identity.reid import ReentryContext
from tools.dev.reid_reentry_eval import clip_events

_BREAK = ("CUT", "UNKNOWN")


def exit_features(ctx: ReentryContext, lost: str, lost_idx: int, cand: str) -> dict:
    frames = ctx.frames
    b = ctx.birth[cand]
    W, H = ctx.width, ctx.height
    x1, y1, x2, y2 = ctx.box_at[(lost, lost_idx)]
    pts = np.array([[(x1 + x2) / 2, (y1 + y2) / 2], [x1, y1], [x2, y2]], dtype=float)
    left_frame = False
    for k in range(lost_idx + 1, b + 1):
        cam = frames[k]["camera"]
        if cam["status"] in _BREAK:
            return {"status": "unknown"}
        pts = apply_affine_points(np.asarray(cam["affine"], dtype=float), pts)
        left_frame = left_frame or any(e[0] == lost and e[1] == "LEFT_FRAME" for e in frames[k]["ended"])
    cx, cy = pts[0]
    excess = {"left": -cx, "right": cx - W, "top": -cy, "bottom": cy - H}
    outside = max(excess.values()) > 0
    if not (outside or left_frame):
        return {"status": "in_picture"}
    if outside:
        side = max(excess, key=excess.get)
    else:                                   # ended at the edge, centre still inside: nearest side of the carried box
        bx1, by1 = pts[1]
        bx2, by2 = pts[2]
        gaps = {"left": bx1, "right": W - bx2, "top": by1, "bottom": H - by2}
        side = min(gaps, key=gaps.get)
    cb = ctx.box_at[(cand, b)]
    dist = {"left": cb[0] / W, "right": (W - cb[2]) / W, "top": cb[1] / H, "bottom": (H - cb[3]) / H}[side]
    return {"status": "off_screen", "side": side, "left_frame_ended": left_frame, "outside": bool(outside),
            "edge_dist": round(float(max(dist, 0.0)), 4),
            "gap_s": round(frames[b]["t"] - frames[lost_idx]["t"], 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-s", type=float, required=True)
    ap.add_argument("--clip", nargs=3, action="append", required=True, metavar=("LABELS", "TRACKS", "EMBEDDINGS"))
    ap.add_argument("--target-report", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config)
    rows, ctxs = [], {}
    for lab, trk, emb in a.clip:
        name = Path(lab).parent.name
        tdoc = read_json(trk)
        ctx = ctxs[name] = ReentryContext.of(tdoc)
        ev = clip_events(lab, trk, emb, a.window_s, cfg)["events"]
        for e in ev:
            if not e["in_window"]:
                continue
            li = ctx.last_obs[e["from"]]
            true_f = exit_features(ctx, e["from"], li, e["to"])
            if true_f["status"] != "off_screen":
                continue
            others = []
            for c, v in e["candidates"].items():
                if c == e["to"] or v["owner"] == e["player"]:
                    continue
                f = exit_features(ctx, e["from"], li, c)
                if f["status"] == "off_screen":
                    others.append({"track": c, "owner": v["owner"], "similarity": v["similarity"], **f})
            rows.append({"clip": name, "player": e["player"], "from": e["from"], "to": e["to"],
                         "true": true_f, "true_similarity": e["candidates"].get(e["to"], {}).get("similarity"),
                         "others": others})
    offers = []
    for rp in a.target_report:
        rep = json.loads(Path(rp).read_text())
        for c in rep["clips"]:
            ctx = ctxs.get(c["clip"])
            if ctx is None:
                continue
            for p in c["per_player"]:
                for r in p.get("relinks", []):
                    if r["kind"] != "reid":
                        continue
                    li = ctx.last_obs_upto(r["from_track"], next(i for i, fr in enumerate(ctx.frames)
                                                                 if fr["frame"] == r["proposed_frame"]))
                    offers.append({"report": Path(rp).name, "clip": c["clip"], "player": p["player"],
                                   "from": r["from_track"], "to": r["to_track"], "frame": r["proposed_frame"],
                                   "verdict": r["verdict"], "verdict_player": r["verdict_player"],
                                   "similarity": r["similarity"],
                                   **exit_features(ctx, r["from_track"], li, r["to_track"])})
    true_d = sorted(r["true"]["edge_dist"] for r in rows)
    other_d = sorted(o["edge_dist"] for r in rows for o in r["others"])
    q = lambda v: {k: round(float(np.quantile(v, k)), 3) for k in (0.5, 0.75, 0.9, 1.0)} if v else {}  # noqa: E731
    summary = {"off_screen_reentries": len(rows), "true_edge_dist_quantiles": q(true_d),
               "wrong_candidate_edge_dist_quantiles": q(other_d), "wrong_candidates": len(other_d),
               "by_threshold": {str(t): {"true_kept": sum(d <= t for d in true_d),
                                         "wrong_candidates_kept": sum(d <= t for d in other_d)}
                                for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)}}
    Path(a.out).write_text(json.dumps({"summary": summary, "reentries": rows, "offers": offers}, indent=1))
    print(json.dumps(summary, indent=1))
    for o in offers:
        print({k: o.get(k) for k in ("report", "clip", "player", "from", "to", "frame", "verdict", "verdict_player",
                                     "similarity", "status", "side", "edge_dist", "gap_s")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
