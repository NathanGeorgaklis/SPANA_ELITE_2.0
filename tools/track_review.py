"""DEV diagnostic review of a tracking run WITHOUT human labels (not a benchmark).

Flags candidate failures from the run's own evidence, then renders frame crops around each flag so a reviewer
can confirm or reject it by eye:

  FRAG   a confirmed track that was spawned naming possible predecessors, or that started within the gate of a
         track which ended LOST_EXPIRED / was still unobserved (candidate fragmentation)
  DUP    two OBSERVED confirmed tracks whose boxes nest (IoA >= DUP_IOA) for >= DUP_MIN_S (candidate duplicate)
  SWAP   a contested match, or a refused match / recovery (candidate ID switch or wrong recovery)
  SHORT  a confirmed track observed for < SHORT_S in total (candidate fragment or false positive)

Usage: python -m tools.track_review --video V --detections D --tracks T --decisions L --out DIR
Writes DIR/flags.json, DIR/overlay.mp4 (all tracks, ID labels) and DIR/flag_XXX.jpg crops.
The review thresholds below only select frames to look at; they never change tracker output.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from spana.artifacts import read_json

DUP_IOA = 0.6
DUP_MIN_S = 0.3
SHORT_S = 1.0
CROP_PAD = 2.0          # crop half-size in box heights around the flagged box


def _ioa(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def _colour(tid: str):
    h = (int(tid[1:]) * 47) % 180
    c = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def find_flags(tracks: dict, decisions: List[dict]) -> List[dict]:
    frames = tracks["frames"]
    t_of = {fr["frame"]: fr["t"] for fr in frames}
    flags: List[dict] = []
    obs_time: Dict[str, float] = defaultdict(float)
    confirmed: set = set()
    dt = np.median(np.diff([fr["t"] for fr in frames])) if len(frames) > 1 else 0.0
    nested: Dict[tuple, float] = defaultdict(float)
    for fr in frames:
        obs = [t for t in fr["tracks"] if t["source"] == "OBSERVED"]
        for t in obs:
            obs_time[t["track_id"]] += dt
            if t["confirmed"]:
                confirmed.add(t["track_id"])
        live_pairs = set()
        for a in obs:
            for b in obs:
                if a is b or not (a["confirmed"] and b["confirmed"]):
                    continue
                if _ioa(a["box"], b["box"]) >= DUP_IOA:
                    key = (a["track_id"], b["track_id"])
                    nested[key] += dt
                    live_pairs.add(key)
                    if abs(nested[key] - DUP_MIN_S) < dt / 2 + 1e-9:
                        flags.append({"kind": "DUP", "frame": fr["frame"], "tracks": list(key), "box": a["box"]})
        for key in list(nested):
            if key not in live_pairs:
                nested[key] = 0.0
    first_seen: Dict[str, dict] = {}
    for fr in frames:
        for t in fr["tracks"]:
            if t["track_id"] not in first_seen:
                first_seen[t["track_id"]] = {"frame": fr["frame"], "box": t["box"], "preds": t["possible_predecessors"]}
    for tid, f in first_seen.items():
        if tid in confirmed and f["preds"]:
            flags.append({"kind": "FRAG", "frame": f["frame"], "tracks": [tid, *f["preds"]], "box": f["box"]})
        if tid in confirmed and obs_time[tid] < SHORT_S:
            flags.append({"kind": "SHORT", "frame": f["frame"], "tracks": [tid], "box": f["box"]})
    by_frame = {fr["frame"]: fr for fr in frames}
    for d in decisions:
        if d["kind"] in ("match_refused", "recovery_refused") or (d["kind"] == "match" and d.get("contested")):
            fr = by_frame.get(d["frame"])
            tid = d.get("track_id")
            tr = next((t for t in fr["tracks"] if t["track_id"] == tid), None) if fr else None
            if tr is not None and tr["confirmed"]:
                flags.append({"kind": "SWAP", "frame": d["frame"], "tracks": [tid], "box": tr["box"],
                              "decision": {k: d[k] for k in ("kind", "reason", "rival_tracks", "hidden_track")
                                           if k in d}})
    flags.sort(key=lambda f: (f["frame"], f["kind"]))
    for f in flags:
        f["t"] = t_of[f["frame"]]
    return flags


def render(video: str, tracks: dict, flags: List[dict], out: Path, fps_out: float) -> None:
    by_frame = {fr["frame"]: fr for fr in tracks["frames"]}
    flag_frames = defaultdict(list)
    for k, f in enumerate(flags):
        flag_frames[f["frame"]].append((k, f))
    cap = cv2.VideoCapture(video)
    proc = None
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        fr = by_frame.get(idx)
        if fr is not None:
            vis = img.copy()
            for t in fr["tracks"]:
                x1, y1, x2, y2 = map(int, t["box"])
                c = _colour(t["track_id"])
                thick = 2 if t["source"] == "OBSERVED" else 1
                if t["source"] != "OBSERVED":
                    c = tuple(v // 2 for v in c)
                cv2.rectangle(vis, (x1, y1), (x2, y2), c, thick)
                label = f"{t['track_id'][1:].lstrip('0')}{'' if t['confirmed'] else '?'}{'*' if t['contested'] else ''}"
                cv2.putText(vis, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            cv2.putText(vis, f"f{idx} t={fr['t']:.2f}s cam={fr['camera']['status']}", (20, 1060),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            for k, f in flag_frames.get(idx, []):
                x1, y1, x2, y2 = f["box"]
                h = y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                r = CROP_PAD * h
                xa, xb = int(max(0, cx - r)), int(min(vis.shape[1], cx + r))
                ya, yb = int(max(0, cy - r)), int(min(vis.shape[0], cy + r))
                crop = cv2.resize(vis[ya:yb, xa:xb], None, fx=max(1.0, 300 / max(1, yb - ya)),
                                  fy=max(1.0, 300 / max(1, yb - ya)))
                cv2.imwrite(str(out / f"flag_{k:03d}_{f['kind']}_f{idx}.jpg"), crop)
            if proc is None:
                h, w = vis.shape[:2]
                proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                         "-s", f"{w}x{h}", "-r", str(fps_out), "-i", "-", "-c:v", "libx264",
                                         "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                                         str(out / "overlay.mp4")], stdin=subprocess.PIPE)
            proc.stdin.write(vis.tobytes())
        idx += 1
    cap.release()
    if proc is not None:
        proc.stdin.close()
        proc.wait()


def main() -> int:
    p = argparse.ArgumentParser()
    for k in ("video", "tracks", "decisions", "out"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tracks = read_json(a.tracks)
    decisions = [json.loads(l) for l in Path(a.decisions).read_text().splitlines() if l.strip()]
    flags = find_flags(tracks, decisions)
    (out / "flags.json").write_text(json.dumps(flags, indent=1))
    ts = [fr["t"] for fr in tracks["frames"]]
    render(a.video, tracks, flags, out, round(1.0 / float(np.median(np.diff(ts))), 3))
    counts: Dict[str, int] = defaultdict(int)
    for f in flags:
        counts[f["kind"]] += 1
    print(json.dumps({"flags": dict(counts), "frames": len(ts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
