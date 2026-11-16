"""Cost of an overlap guard on the target follower, measured on DEV before deciding (analysis only, nothing tuned).

    python -m tools.guard_cost --config config/default.yaml --out REPORT.json \
        --clip data/labels/<clip>/v1 TRK.json BALL.json [--clip ...] \
        [--iou 0.2 0.3 0.4] [--min-s 0 0.1 0.2] [--hold-s 0 0.2] [--near-ball-h 1 3]

Guard (one variant per (iou, min_s, hold_s)): a TARGET answer becomes ABSTAIN once the followed track's box has
overlapped another live track's box (the tracker's per-frame `max_overlap_iou`) at >= iou continuously for at least
min_s seconds (0 = from the first overlapping frame), and stays ABSTAIN for hold_s seconds after that overlap ends.
The overlap counts as identity uncertainty; the tracker's own IDENTITY_UNCERTAIN already abstains.

Players are selected and followed exactly as in tools/target_score.py (first unambiguous NESTED_V2 match; v1
follower, config as given). Over each player's labelled frames from the selection, every v1 TARGET answer is
correct / wrong / unverified (as in target_score). Per variant this reports what the guard removes:
  correct_lost        correct answers turned into ABSTAIN (the cost)
  wrong_avoided       wrong answers turned into ABSTAIN (the benefit)
  unverified_removed
each in slices:
  all
  contested           the followed track's association was `contested` on that frame
  near_ball_<R>h      a ball box centre lies within R target-box heights of the target's foot point on that frame
                      (from a tools/ball_pass.py file; "none" = no ball file for the clip, slice not reported).
                      Ball recall on DEV is about 0.70, so near-ball moments are under-counted.
plus guarded_target_share: the share of ALL v1 TARGET frames (labelled or not) the guard turns into ABSTAIN.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from spana.artifacts import read_json, sha256_doc, validate_tracks, write_json_atomic
from spana.config import load_config
from spana.identity.follower import Selection, follow
from tools.benchmark import load_frozen
from tools.target_score import RULE, attributions, first_selection

KINDS = ("correct", "wrong", "unverified")


def guarded_frames(answers: List[dict], tracks_by_frame: Dict[int, dict], tid: str, iou: float, min_s: float,
                   hold_s: float) -> set:
    """Frames whose TARGET answer the guard turns into ABSTAIN."""
    out, run_start, last_guard = set(), None, None
    for a in answers:
        tr = next((t for t in tracks_by_frame[a["frame"]]["tracks"] if t["track_id"] == tid), None)
        ov = tr["max_overlap_iou"] if tr is not None else 0.0
        t = a["t"]
        if ov >= iou:
            run_start = t if run_start is None else run_start
            active = t - run_start >= min_s - 1e-9
        else:
            run_start, active = None, False
        if active:
            last_guard = t
        held = last_guard is not None and t - last_guard <= hold_s + 1e-9
        if (active or held) and a["state"] == "TARGET":
            out.add(a["frame"])
    return out


def near_ball(box, balls, radius_h: float) -> bool:
    foot = np.array([(box[0] + box[2]) / 2, box[3]])
    h = box[3] - box[1]
    return any(np.hypot((b[0] + b[2]) / 2 - foot[0], (b[1] + b[3]) / 2 - foot[1]) <= radius_h * h for b in balls)


def empty_slice() -> dict:
    return {**{f"v1_{k}": 0 for k in KINDS}, "correct_lost": 0, "wrong_avoided": 0, "unverified_removed": 0}


def analyse_clip(label_dir: Path, tracks_path: Path, ball_path: Optional[Path], cfg, grid, radii) -> dict:
    frozen, gt = load_frozen(label_dir)
    doc = read_json(tracks_path)
    validate_tracks(doc)
    if doc["video"]["sha256"] != frozen["clip"]["sha256"]:
        raise SystemExit("refusing: the tracks were made from a different video than the labels")
    balls = None
    if ball_path is not None:
        bd = read_json(ball_path)
        if bd["video"]["sha256"] != frozen["clip"]["sha256"]:
            raise SystemExit("refusing: the ball pass was made from a different video than the labels")
        balls = {fr["frame"]: fr["boxes"] for fr in bd["frames"]}
    attr = attributions(gt, doc)
    by_frame = {fr["frame"]: fr for fr in doc["frames"]}
    slices = ["all", "contested"] + ([f"near_ball_{r:g}h" for r in radii] if balls is not None else [])
    res = {v: {s: empty_slice() for s in slices} | {"target_frames": 0, "guarded_target_frames": 0} for v in grid}
    players = sorted({p for f in gt.frames for p in gt.players.get(f, {})}, key=int)
    for pid in players:
        sel = first_selection(pid, gt, attr)
        if sel is None:
            continue
        _, f0, tid = sel
        answers, _ = follow(doc, Selection.track_at_frame(tid, f0), cfg)
        ans = {a["frame"]: a for a in answers}
        n_target = sum(a["state"] == "TARGET" for a in answers)
        labelled = []
        for f in gt.frames:
            if f < f0 or pid not in gt.players.get(f, {}) or ans[f]["state"] != "TARGET":
                continue
            p, amb = attr[f].get(tid, (None, False))
            kind = "correct" if (p == pid and not amb) else "wrong" if (p is not None and not amb) else "unverified"
            tr = next(t for t in by_frame[f]["tracks"] if t["track_id"] == tid)
            tags = ["all"] + (["contested"] if tr["contested"] else [])
            if balls is not None:
                tags += [f"near_ball_{r:g}h" for r in radii if near_ball(ans[f]["box"], balls.get(f, []), r)]
            labelled.append((f, kind, tags))
        for v in grid:
            g = guarded_frames(answers, by_frame, tid, *v)
            res[v]["target_frames"] += n_target
            res[v]["guarded_target_frames"] += len(g)
            for f, kind, tags in labelled:
                for s in tags:
                    row = res[v][s]
                    row[f"v1_{kind}"] += 1
                    if f in g:
                        row[{"correct": "correct_lost", "wrong": "wrong_avoided",
                             "unverified": "unverified_removed"}[kind]] += 1
    return {"clip": frozen["clip"]["name"], "tracks_sha256": sha256_doc(doc),
            "ball_pass": None if ball_path is None else {"path": ball_path.name, "sha256": sha256_doc(read_json(ball_path))},
            "variants": [{"iou": v[0], "min_s": v[1], "hold_s": v[2], **res[v]} for v in grid]}


def pooled(clips: List[dict]) -> List[dict]:
    out = []
    for n, v in enumerate(clips[0]["variants"]):
        row = {k: v[k] for k in ("iou", "min_s", "hold_s")}
        for k in ("target_frames", "guarded_target_frames"):
            row[k] = sum(c["variants"][n][k] for c in clips)
        row["guarded_target_share"] = round(row["guarded_target_frames"] / max(1, row["target_frames"]), 4)
        names = {s for c in clips for s in c["variants"][n] if isinstance(c["variants"][n][s], dict)}
        for s in sorted(names):
            parts = [c["variants"][n][s] for c in clips if s in c["variants"][n]]
            agg = {k: sum(p[k] for p in parts) for k in empty_slice()}
            agg["correct_lost_share"] = round(agg["correct_lost"] / max(1, agg["v1_correct"]), 4)
            agg["clips"] = len(parts)
            row[s] = agg
        out.append(row)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--clip", nargs=3, action="append", required=True, metavar=("LABELS_DIR", "TRACKS_JSON", "BALL_JSON"),
                   help='BALL_JSON from tools/ball_pass.py, or "none"')
    p.add_argument("--iou", nargs="+", type=float, default=[0.2, 0.3, 0.4])
    p.add_argument("--min-s", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    p.add_argument("--hold-s", nargs="+", type=float, default=[0.0, 0.2])
    p.add_argument("--near-ball-h", nargs="+", type=float, default=[1.0, 3.0])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    grid = list(itertools.product(a.iou, a.min_s, a.hold_s))
    clips = [analyse_clip(Path(l), Path(t), None if b == "none" else Path(b), cfg, grid, a.near_ball_h)
             for l, t, b in a.clip]
    report = {"tool": "tools.guard_cost", "match_rule": RULE.name, "config_sha256": cfg.sha256(),
              "near_ball_h": a.near_ball_h, "clips": clips, "pooled": pooled(clips)}
    write_json_atomic(a.out, report)
    for r in report["pooled"]:
        sl = {s: (r[s]["correct_lost"], r[s]["v1_correct"], r[s]["wrong_avoided"], r[s]["v1_wrong"])
              for s in r if isinstance(r[s], dict)}
        print(f"iou {r['iou']:.2f} min {r['min_s']:.2f}s hold {r['hold_s']:.2f}s  guarded {r['guarded_target_share']:.3f}"
              f"  (lost/correct, avoided/wrong) {sl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
