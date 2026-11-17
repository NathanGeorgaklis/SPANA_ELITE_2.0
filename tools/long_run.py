"""Helpers for a long DEV run (a window of a full DEV match): analysis only, no product logic.

    python -m tools.long_run locate --clip CLIP.mp4 --video MATCH.mp4 --around S [S ...] [--search-s 90]
        Find where a short clip starts inside a longer video: one sequential decode (frame indices exact, as
        spana.video counts them), every frame within --search-s of each candidate time compared with the clip's
        first frame on 64x36 grey thumbnails (mean absolute difference); the best is confirmed with the clip frame
        5 s later. Prints JSON {frame, t, score, confirm_score, ...}.

    python -m tools.long_run subsample --detections DET.json --stride N --out OUT.json
        A detections artifact with every N-th frame of DET (frame_stride multiplied by N), for checking how the
        tracker does when fewer frames are analysed, without running the detector again. Per-frame detections are
        unchanged (the detector sees each frame on its own). Provenance names the source artifact.

    python -m tools.long_run select --labels data/labels/<clip>/v1 --clip-fps F --clip-offset-frame O \\
        --tracks WINDOW_TRK.json --player P [--rule NESTED_V2]
        Pick a labelled clip player's track in a window that contains the clip: label frames mapped to window
        frames (window frame = O + clip frame * window fps / clip fps), boxes scaled by the frame sizes, attributed
        to observed window tracks with the scorer's matching. Prints the first unambiguous {frame, track_id}.

    python -m tools.long_run overlap --labels data/labels/<clip>/v1 --clip-fps F --clip-offset-frame O \
        --tracks WINDOW_TRK.json --target TARGET.json --player P
        Score a long target run on the labelled clip it contains (same mapping as select): correct / wrong /
        unverified answers, candidates, abstain / lost, per labelled frame of the player.

    python -m tools.long_run summary --target TARGET.json
        Over a long target document: answered (TARGET) time, offers, abstain / lost time, gaps per minute, gap
        lengths, offers, track ends. Without labels nothing here says whether answers were right.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from spana.artifacts import read_json, refuse_existing, validate_detections, validate_target, write_json_atomic
from spana.evaluation import RULES, _match
from spana.provenance import build_provenance
from spana.video import sha256_file
from tools.benchmark import IGNORE_IOA, _ioa, load_frozen


# --------------------------------------------------------------------------------------------- locate
def _thumb(img: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def _clip_refs(clip: str, confirm_s: float):
    c = cv2.VideoCapture(clip)
    fps = c.get(cv2.CAP_PROP_FPS)
    ok, f0 = c.read()
    if not ok:
        raise SystemExit(f"cannot read {clip}")
    k, ref2 = 0, None
    while ok and k < int(round(confirm_s * fps)):
        ok, f = c.read()
        k += 1
        ref2 = f if ok else ref2
    c.release()
    return fps, _thumb(f0), (_thumb(ref2) if ref2 is not None else None)


def locate(clip: str, video: str, around: List[float], search_s: float, confirm_s: float = 5.0) -> dict:
    clip_fps, ref, ref2 = _clip_refs(clip, confirm_s)
    v = cv2.VideoCapture(video)
    fps = v.get(cv2.CAP_PROP_FPS)
    windows = [(max(0, int((a - search_s) * fps)), int((a + search_s) * fps)) for a in around]
    last = max(b for _, b in windows) + int(confirm_s * fps) + 2
    scores: Dict[int, float] = {}
    keep: Dict[int, np.ndarray] = {}
    for i in range(last + 1):
        if not v.grab():
            break
        inside = any(a <= i <= b for a, b in windows)
        later = any(a + int(confirm_s * fps) - 1 <= i <= b + int(confirm_s * fps) + 1 for a, b in windows)
        if inside or later:
            ok, img = v.retrieve()
            if not ok:
                continue
            t = _thumb(img)
            if inside:
                scores[i] = float(np.mean(np.abs(t - ref)))
            if later and ref2 is not None:
                keep[i] = t
    v.release()
    if not scores:
        raise SystemExit("no frames decoded in the search windows")
    best = min(scores, key=scores.get)
    j = best + int(round(confirm_s * fps))
    confirm = float(np.mean(np.abs(keep[j] - ref2))) if (ref2 is not None and j in keep) else None
    ordered = sorted(scores.values())
    return {"frame": best, "t": round(best / fps, 3), "score": round(scores[best], 3),
            "confirm_frame": j, "confirm_score": None if confirm is None else round(confirm, 3),
            "median_score_in_windows": round(float(np.median(ordered)), 3), "video_fps": fps, "clip_fps": clip_fps,
            "candidates_searched": around, "search_s": search_s}


# --------------------------------------------------------------------------------------------- subsample
def subsample(doc: dict, stride: int, argv: List[str], source_sha: str) -> dict:
    if stride < 1:
        raise SystemExit("--stride must be >= 1")
    validate_detections(doc)
    first = doc["frames"][0]["frame"]
    frames = [fr for fr in doc["frames"] if (fr["frame"] - first) % (stride * doc["frame_stride"]) == 0]
    out = dict(doc)
    out.update(frames=frames, frame_stride=doc["frame_stride"] * stride,
               provenance={**build_provenance(config_sha256=doc["provenance"].get("config_sha256", ""), argv=argv,
                                              inputs={"detections_source": source_sha}),
                           "derived_from": {"tool": "tools.long_run subsample", "stride": stride,
                                            "source_provenance": doc["provenance"]}})
    validate_detections(out)
    return out


# --------------------------------------------------------------------------------------------- select
def select(label_dir: Path, clip_fps: float, offset: int, tracks: dict, player: str, rule_name: str) -> dict:
    frozen, gt = load_frozen(label_dir)
    vw, vh = tracks["video"]["width"], tracks["video"]["height"]
    sx, sy = vw / frozen["clip"]["width"], vh / frozen["clip"]["height"]
    wfps = tracks["video"]["fps"]
    by = {fr["frame"]: fr for fr in tracks["frames"]}
    rule = RULES[rule_name]
    for f in gt.frames:
        if player not in gt.players.get(f, {}):
            continue
        wf = offset + int(round(f * wfps / clip_fps))
        fr = by.get(wf)
        if fr is None:
            continue
        sc = lambda b: (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)                  # noqa: E731
        players = {p: sc(b) for p, b in gt.players[f].items()}
        ign = [sc(b) for b in gt.ignore.get(f, [])]
        obs = [t for t in fr["tracks"] if t["source"] == "OBSERVED" and not any(_ioa(t["box"], g) >= IGNORE_IOA for g in ign)]
        m, _, amb = _match(sorted(players), players, [t["box"] for t in obs], rule, frozenset(gt.occluded.get(f, ())))
        hit = [obs[c]["track_id"] for c, p in m.items() if p == player and c not in amb]
        if hit:
            return {"frame": wf, "track_id": hit[0], "clip_frame": f, "player": player, "rule": rule_name}
    raise SystemExit(f"player {player} has no unambiguous track in the window")


def overlap(label_dir: Path, clip_fps: float, offset: int, tracks: dict, target: dict, player: str,
            rule_name: str) -> dict:
    """Score a long target document on the labelled clip it contains (same mapping as select): per labelled frame
    of the player, the answer is correct / wrong / unverified (TARGET), candidate_* (CANDIDATE), abstain or lost."""
    frozen, gt = load_frozen(label_dir)
    vw, vh = tracks["video"]["width"], tracks["video"]["height"]
    sx, sy = vw / frozen["clip"]["width"], vh / frozen["clip"]["height"]
    wfps = tracks["video"]["fps"]
    by = {fr["frame"]: fr for fr in tracks["frames"]}
    ans = {fr["frame"]: fr for fr in target["frames"]}
    rule = RULES[rule_name]
    counts = {k: 0 for k in ("correct", "wrong", "unverified", "candidate_correct", "candidate_wrong",
                             "candidate_unverified", "abstain", "lost", "before_selection")}
    events = []
    for f in gt.frames:
        if player not in gt.players.get(f, {}):
            continue
        wf = offset + int(round(f * wfps / clip_fps))
        if wf not in ans:
            counts["before_selection"] += 1
            continue
        a = ans[wf]
        if a["state"] in ("ABSTAIN", "LOST"):
            counts[a["state"].lower()] += 1
            continue
        sc = lambda b: (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)                  # noqa: E731
        players = {p: sc(b) for p, b in gt.players[f].items()}
        ign = [sc(b) for b in gt.ignore.get(f, [])]
        obs = [t for t in by[wf]["tracks"] if t["source"] == "OBSERVED" and
               not any(_ioa(t["box"], g) >= IGNORE_IOA for g in ign)]
        m, _, amb = _match(sorted(players), players, [t["box"] for t in obs], rule, frozenset(gt.occluded.get(f, ())))
        c = next((c for c, t in enumerate(obs) if t["track_id"] == a["track_id"]), None)
        p = m.get(c) if c is not None else None
        kind = "correct" if (p == player and c not in amb) else "wrong" if (p is not None and c not in amb) \
            else "unverified"
        counts[kind if a["state"] == "TARGET" else f"candidate_{kind}"] += 1
        if kind == "wrong":
            events.append({"window_frame": wf, "clip_frame": f, "state": a["state"], "track_id": a["track_id"],
                           "attributed_to": p})
    return {"player": player, "counts": counts, "events": events}


# --------------------------------------------------------------------------------------------- summary
def summary(doc: dict) -> dict:
    frames = doc["frames"]
    n = len(frames)
    dur = frames[-1]["t"] - frames[0]["t"]
    count = lambda s: sum(f["state"] == s for f in frames)                                  # noqa: E731
    gaps = doc.get("gaps", [])
    lengths = sorted(round(g["end_t"] - g["start_t"], 3) for g in gaps)
    q = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))] if lengths else None  # noqa: E731
    return {"frames": n, "duration_s": round(dur, 3),
            "shares": {s: round(count(s) / n, 4) for s in ("TARGET", "CANDIDATE", "ABSTAIN", "LOST")},
            "gaps": len(gaps), "gaps_per_min": round(len(gaps) / (dur / 60), 2) if dur else None,
            "gap_s": {"median": q(0.5), "p90": q(0.9), "max": lengths[-1] if lengths else None,
                      "over_1s": sum(x > 1 for x in lengths), "over_5s": sum(x > 5 for x in lengths),
                      "over_30s": sum(x > 30 for x in lengths)},
            "offers": len(doc.get("relinks", [])), "track_ends": len(doc.get("track_ends", [])),
            "user_actions": len(doc.get("user_actions", []))}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("locate")
    a.add_argument("--clip", required=True)
    a.add_argument("--video", required=True)
    a.add_argument("--around", nargs="+", type=float, required=True, help="candidate start times, seconds")
    a.add_argument("--search-s", type=float, default=90.0)
    b = sub.add_parser("subsample")
    b.add_argument("--detections", required=True)
    b.add_argument("--stride", type=int, required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("select")
    c.add_argument("--labels", required=True)
    c.add_argument("--clip-fps", type=float, required=True)
    c.add_argument("--clip-offset-frame", type=int, required=True)
    c.add_argument("--tracks", required=True)
    c.add_argument("--player", required=True)
    c.add_argument("--rule", default="NESTED_V2", choices=sorted(RULES))
    d = sub.add_parser("summary")
    d.add_argument("--target", required=True)
    e = sub.add_parser("overlap")
    for k in ("labels", "tracks", "target", "player"):
        e.add_argument(f"--{k}", required=True)
    e.add_argument("--clip-fps", type=float, required=True)
    e.add_argument("--clip-offset-frame", type=int, required=True)
    e.add_argument("--rule", default="NESTED_V2", choices=sorted(RULES))
    x = p.parse_args(argv)
    if x.cmd == "locate":
        print(json.dumps(locate(x.clip, x.video, x.around, x.search_s)))
    elif x.cmd == "subsample":
        refuse_existing(x.out)
        out = subsample(read_json(x.detections), x.stride, sys.argv, sha256_file(x.detections))
        print(write_json_atomic(x.out, out))
    elif x.cmd == "select":
        print(json.dumps(select(Path(x.labels), x.clip_fps, x.clip_offset_frame, read_json(x.tracks), x.player, x.rule)))
    elif x.cmd == "overlap":
        tdoc, gdoc = read_json(x.tracks), read_json(x.target)
        validate_target(gdoc, tdoc)
        print(json.dumps(overlap(Path(x.labels), x.clip_fps, x.clip_offset_frame, tdoc, gdoc, x.player, x.rule)))
    else:
        doc = read_json(x.target)
        validate_target(doc)
        print(json.dumps(summary(doc)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
