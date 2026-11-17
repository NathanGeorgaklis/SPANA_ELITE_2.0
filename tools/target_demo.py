"""A synthetic demo video of the target follower (no match footage): drawn players on a drawn pitch, a scripted
tracks artifact (validated like a real one), the real follower and the real review renderer.

    python -m tools.target_demo --out target_demo.mp4

Script (25 fps, 9 s): the user taps #7 -> followed; #7 passes behind #4 (hidden: no answer); the detector misses
#7 and his track ends (lost); a new track the tracker links to the lost one is OFFERED, the user accepts; two
players in the same kit cross (not sure: no answer, and still unsure after); the user taps #7 again; camera cut
(every track ends, nothing carried across); the user taps #7 in the new shot. Identities are scripted, so the demo
shows the behaviour, not accuracy (accuracy: records/target_*).
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from spana.artifacts import validate_tracks
from spana.config import load_config
from spana.identity.follower import Selection, confirm_candidate, reselect, target_document
from tools.target_render import render

FPS, W, H, N_FRAMES, CUT = 25.0, 1280, 720, 225, 175
BLUE, RED = (200, 110, 30), (40, 40, 210)            # kits (BGR)
# player: (number, kit, keyframes [(frame, x_centre, y_feet, height)]); separate keyframes before / after the cut
PLAYERS = {
    "A": (7, BLUE, [(0, 260, 470, 150), (60, 600, 470, 150), (110, 760, 470, 150), (174, 980, 480, 150)],
          [(175, 520, 560, 230), (224, 760, 560, 230)]),
    "B": (9, BLUE, [(0, 1000, 420, 130), (110, 760, 470, 148), (174, 560, 520, 158)],
          [(175, 980, 520, 210), (224, 1100, 520, 210)]),
    "C": (4, RED, [(0, 470, 500, 168), (174, 500, 500, 168)],
          [(175, 260, 600, 240), (224, 300, 600, 240)]),
    "D": (11, RED, [(0, 1120, 560, 175), (174, 980, 600, 180)],
          [(175, 760, 400, 160), (224, 700, 400, 160)]),
}
CAPTIONS = [(0, "The tracker finds every player. The user taps theirs."),
            (12, "Tap: following #7"),
            (30, "#7 passes behind #4: hidden, so no answer (never a guess)"),
            (42, "Back in view: the tracker continues the same track"),
            (60, "The detector misses #7: lost"),
            (65, "A new track the tracker links to the lost one: offered, not claimed"),
            (80, "The user accepts the offer: following again"),
            (110, "Two players in the same kit cross: not sure, no answer"),
            (125, "Still unsure after the crossing: waits for the user"),
            (140, "The user taps #7 again: following"),
            (175, "Camera cut: every track ends, nothing is carried across"),
            (195, "The user taps #7 in the new shot: following")]


def position(pid: str, f: int) -> Tuple[float, float, float]:
    num, kit, before, after = PLAYERS[pid]
    keys = before if f < CUT else after
    for (f0, *a), (f1, *b) in zip(keys, keys[1:]):
        if f0 <= f <= f1:
            w = (f - f0) / max(1, f1 - f0)
            return tuple(float(x + (y - x) * w) for x, y in zip(a, b))      # type: ignore[return-value]
    return tuple(float(v) for v in keys[-1][1:])                           # type: ignore[return-value]


def box(pid: str, f: int) -> List[float]:
    x, y, h = position(pid, f)
    w = 0.42 * h
    return [round(x - w / 2, 1), round(y - h, 1), round(x + w / 2, 1), round(y, 1)]


# --------------------------------------------------------------------------------------------- scripted tracks
# per track: player, birth frame, end frame (exclusive) + end reason, per-frame state function, predecessors
def _t1(f):     # #7 before the relink: occluded 30-41, lost 60-69, ends 70
    return "OCCLUDED" if 30 <= f <= 41 else "LOST" if 60 <= f <= 69 else "ACTIVE"


TRACKS = {
    "T000001": ("A", 0, 70, "LOST_EXPIRED", _t1, []),
    "T000002": ("B", 0, CUT, "CAMERA_CUT", lambda f: "ACTIVE", []),
    "T000003": ("C", 0, CUT, "CAMERA_CUT", lambda f: "ACTIVE", []),
    "T000004": ("D", 0, CUT, "CAMERA_CUT", lambda f: "ACTIVE", []),
    "T000005": ("A", 64, CUT, "CAMERA_CUT", lambda f: "ACTIVE", ["T000001"]),
    "T000006": ("A", CUT, N_FRAMES, None, lambda f: "ACTIVE", []),
    "T000007": ("B", CUT, N_FRAMES, None, lambda f: "ACTIVE", []),
    "T000008": ("C", CUT, N_FRAMES, None, lambda f: "ACTIVE", []),
    "T000009": ("D", CUT, N_FRAMES, None, lambda f: "ACTIVE", []),
}
UNCERTAIN = {"T000002": ("T000005", 110, 124), "T000005": ("T000002", 110, 124)}   # same-kit crossing


def tracks_artifact() -> dict:
    frames, st, last_live = [], {}, set()
    for f in range(N_FRAMES):
        live = [tid for tid, (_, b, e, *_r) in TRACKS.items() if b <= f < e]
        ended = [[tid, TRACKS[tid][3]] for tid in sorted(last_live - set(live))]
        rows, k = [], 0
        for tid in sorted(live):
            pid, born, _, _, state_of, preds = TRACKS[tid]
            state = state_of(f)
            s = st.setdefault(tid, {"age": -1, "obs": 0, "miss": 0, "eps": 0, "unc": False})
            s["age"] += 1
            observed = state == "ACTIVE"
            s["obs"] += int(observed)
            s["miss"] = 0 if observed else s["miss"] + 1
            rival = UNCERTAIN.get(tid)
            unc = rival is not None and rival[1] <= f <= rival[2]
            s["eps"] += int(unc and not s["unc"])
            s["unc"] = unc
            rows.append({
                "track_id": tid, "state": state, "confirmed": f > born,
                "source": "OBSERVED" if observed else "PREDICTED", "box": box(pid, f),
                "detection_index": k if observed else None, "detection_score": 0.9 if observed else None,
                "confidence": 0.9, "contested": False, "age_frames": s["age"], "missed_frames": s["miss"],
                "time_since_observed_s": 0 if observed else round(s["miss"] / FPS, 6),
                "observed_frames": max(1, s["obs"]), "epoch": int(f >= CUT),
                "occluder": "T000003" if state == "OCCLUDED" else None, "max_overlap_iou": 0.0,
                "overlap_with": None, "recoveries": 0, "last_gap_s": 0, "possible_predecessors": list(preds),
                "identity": "IDENTITY_UNCERTAIN" if unc else "CERTAIN",
                "identity_rivals": [rival[0]] if unc else [], "uncertain_episodes": s["eps"]})
            k += int(observed)
        cam = "FIRST" if f == 0 else "CUT" if f == CUT else "OK"
        frames.append({"frame": f, "t": round(f / FPS, 6),
                       "camera": {"status": cam, "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "inliers": 0,
                                  "hist_distance": None}, "tracks": rows, "ended": ended})
        last_live = set(live)
    tracklets = []
    for tid, (pid, b, e, reason, state_of, preds) in TRACKS.items():
        last = min(e, N_FRAMES) - 1
        obs = [f for f in range(b, last + 1) if state_of(f) == "ACTIVE"]
        tracklets.append({"track_id": tid, "epoch": int(b >= CUT), "first_frame": b, "last_frame": last,
                          "last_observed_frame": obs[-1], "observed_frames": len(obs), "confirmed": True,
                          "end_reason": reason if e < N_FRAMES else None, "recoveries": 0,
                          "possible_predecessors": list(preds)})
    doc = {"schema": "spana.tracks", "schema_version": "2.1.0",
           "provenance": {"note": "synthetic demo (tools/target_demo.py); no footage"},
           "video": {"path": "synthetic", "sha256": "0" * 64, "fps": FPS, "width": W, "height": H,
                     "frame_count_reported": N_FRAMES, "timestamp_source": "synthetic"},
           "detections_sha256": "0" * 64, "frames": frames, "tracklets": tracklets}
    validate_tracks(doc)
    return doc


SELECTION = Selection.track_at_frame("T000001", 12)
ACTIONS = [confirm_candidate("T000005", 80), reselect("T000005", 140), reselect("T000006", 195)]


# --------------------------------------------------------------------------------------------- drawing
def pitch(f: int) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    base = (60, 140, 60) if f < CUT else (55, 125, 70)                    # the other shot looks different
    for i in range(0, W, 160):
        img[:, i:i + 80] = base
        img[:, i + 80:i + 160] = tuple(int(c * 0.9) for c in base)
    if f < CUT:
        cv2.line(img, (W // 2, 0), (W // 2, H), (230, 230, 230), 3)
        cv2.circle(img, (W // 2, H // 2), 110, (230, 230, 230), 3)
    else:
        cv2.rectangle(img, (-50, 140), (900, 700), (230, 230, 230), 3)   # penalty box, closer view
    return img


def draw_player(img, pid: str, f: int) -> None:
    num, kit, *_ = PLAYERS[pid]
    x1, y1, x2, y2 = (int(v) for v in box(pid, f))
    h, cx = y2 - y1, (x1 + x2) // 2
    head = int(0.09 * h)
    cv2.circle(img, (cx, y1 + head), head, (150, 180, 215), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1 + int(0.08 * (x2 - x1)), y1 + 2 * head), (x2 - int(0.08 * (x2 - x1)), y1 + int(0.55 * h)),
                  kit, -1)
    cv2.rectangle(img, (x1 + int(0.2 * (x2 - x1)), y1 + int(0.55 * h)), (x2 - int(0.2 * (x2 - x1)), y1 + int(0.7 * h)),
                  (240, 240, 240), -1)
    step = int(4 * np.sin(f / 2.0) * h / 150)
    cv2.line(img, (cx - h // 12, y1 + int(0.7 * h)), (cx - h // 12 - step, y2), (40, 40, 40), max(2, h // 25))
    cv2.line(img, (cx + h // 12, y1 + int(0.7 * h)), (cx + h // 12 + step, y2), (40, 40, 40), max(2, h // 25))
    cv2.putText(img, str(num), (cx - int(0.12 * h), y1 + int(0.42 * h)), cv2.FONT_HERSHEY_SIMPLEX, h / 220,
                (255, 255, 255), max(1, h // 60), cv2.LINE_AA)


def scene(f: int) -> np.ndarray:
    img = pitch(f)
    for pid in sorted(PLAYERS, key=lambda p: position(p, f)[1]):          # far players first
        draw_player(img, pid, f)
    return img


def caption(f: int) -> Optional[str]:
    return [c for s, c in CAPTIONS if s <= f][-1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    tracks = tracks_artifact()
    target = target_document(tracks, SELECTION, cfg, ["tools.target_demo"], ACTIONS)
    render(((f, scene(f)) for f in range(N_FRAMES)), tracks, target, a.out, FPS, W, H, caption)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
