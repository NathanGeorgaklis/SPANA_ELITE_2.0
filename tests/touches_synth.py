"""Scripted scenes for possession / touches v1: players, their tracks and the ball, written as validated spana.tracks
and spana.ball artifacts (no footage, no detector). Each scene also carries what a person would say per frame
(`truth`), written by hand from the script, never computed with the possession rule itself.

truth per frame: ("CONTROL", pid) | ("FREE",) | ("CONTESTED",) | ("AIR",) (ball in the air / not at a player's
depth: FREE away from players, never CONTROL) | ("UNSURE",) (identity or the ball is not known: must never be CONTROL).
`edge` frames (another player coming close) also accept CONTESTED / FREE / UNKNOWN; CONTROL by the wrong player
is never accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from spana.artifacts import validate_ball, validate_tracks

FPS, W, H = 25.0, 1280, 720
PH = 150.0                                    # player height in px


def box(x: float, feet: float, h: float = PH) -> List[float]:
    w = 0.42 * h
    return [round(x - w / 2, 1), round(feet - h, 1), round(x + w / 2, 1), round(feet, 1)]


@dataclass
class Track:
    pid: str
    born: int
    end: int                                  # exclusive
    end_reason: Optional[str] = None          # set when the track ends inside the scene
    state: Callable[[int], str] = lambda f: "ACTIVE"     # ACTIVE / LOST
    epoch: int = 0


@dataclass
class Scene:
    name: str
    n: int
    players: Dict[str, Callable[[int], Tuple[float, float]]]           # pid -> f -> (x centre, feet y)
    tracks: Dict[str, Track]
    ball: Callable[[int], Tuple[str, Optional[Tuple[float, float]], Optional[float], bool]]  # state, centre, diam, contested
    truth: Callable[[int], tuple]
    possessions: Optional[List[Tuple[str, int, int]]]                  # (pid, first, last); None = not scripted
    uncertain: Dict[str, Tuple[int, int, List[str]]] = field(default_factory=dict)   # tid -> (f0, f1 incl., rivals)
    cut: Optional[int] = None
    edge: Sequence[int] = ()

    def pid_of(self, tid: str) -> str:
        return self.tracks[tid].pid


def tracks_doc(s: Scene) -> dict:
    frames, st, last_live = [], {}, set()
    for f in range(s.n):
        live = sorted(tid for tid, t in s.tracks.items() if t.born <= f < t.end)
        ended = [[tid, s.tracks[tid].end_reason] for tid in sorted(last_live - set(live))]
        rows, k = [], 0
        for tid in live:
            t = s.tracks[tid]
            x, feet = s.players[t.pid](f)
            state = t.state(f)
            c = st.setdefault(tid, {"age": -1, "obs": 0, "miss": 0, "eps": 0, "unc": False})
            c["age"] += 1
            observed = state == "ACTIVE"
            c["obs"] += int(observed)
            c["miss"] = 0 if observed else c["miss"] + 1
            u = s.uncertain.get(tid)
            unc = u is not None and u[0] <= f <= u[1]
            c["eps"] += int(unc and not c["unc"])
            c["unc"] = unc
            rows.append({
                "track_id": tid, "state": state, "confirmed": f > t.born,
                "source": "OBSERVED" if observed else "PREDICTED", "box": box(x, feet),
                "detection_index": k if observed else None, "detection_score": 0.9 if observed else None,
                "confidence": 0.9, "contested": False, "age_frames": c["age"], "missed_frames": c["miss"],
                "time_since_observed_s": 0 if observed else round(c["miss"] / FPS, 6),
                "observed_frames": max(1, c["obs"]), "epoch": t.epoch, "occluder": None, "max_overlap_iou": 0.0,
                "overlap_with": None, "recoveries": 0, "last_gap_s": 0, "possible_predecessors": [],
                "identity": "IDENTITY_UNCERTAIN" if unc else "CERTAIN",
                "identity_rivals": list(u[2]) if unc else [], "uncertain_episodes": c["eps"]})
            k += int(observed)
        cam = "FIRST" if f == 0 else "CUT" if f == s.cut else "OK"
        frames.append({"frame": f, "t": round(f / FPS, 6),
                       "camera": {"status": cam, "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "inliers": 0,
                                  "hist_distance": None}, "tracks": rows, "ended": ended})
        last_live = set(live)
    tracklets = []
    for tid, t in s.tracks.items():
        last = min(t.end, s.n) - 1
        obs = [f for f in range(t.born, last + 1) if t.state(f) == "ACTIVE"]
        tracklets.append({"track_id": tid, "epoch": t.epoch, "first_frame": t.born, "last_frame": last,
                          "last_observed_frame": obs[-1], "observed_frames": len(obs), "confirmed": True,
                          "end_reason": t.end_reason if t.end < s.n else None, "recoveries": 0,
                          "possible_predecessors": []})
    doc = {"schema": "spana.tracks", "schema_version": "2.1.0",
           "provenance": {"note": f"synthetic touches scene {s.name}; no footage"},
           "video": {"path": "synthetic", "sha256": "1" * 64, "fps": FPS, "width": W, "height": H,
                     "frame_count_reported": s.n, "timestamp_source": "synthetic"},
           "detections_sha256": "0" * 64, "frames": frames, "tracklets": tracklets}
    validate_tracks(doc)
    return doc


_BALL_REASON = {"OBSERVED": "CONTINUING", "PREDICTED": "SHORT_GAP", "LOST": "NO_CONFIRMED_BALL",
                "AMBIGUOUS": "RIVAL_CANDIDATES"}


def ball_doc(s: Scene) -> dict:
    frames, hyps = [], {}
    for f in range(s.n):
        epoch = int(s.cut is not None and f >= s.cut)
        state, centre, diam, contested = s.ball(f)
        reason = _BALL_REASON[state]
        if f == s.cut:
            state, reason, centre, diam, contested = "LOST", "CAMERA_DISCONTINUITY", None, None, False
        answered = state in ("OBSERVED", "PREDICTED")
        hid = epoch + 1 if answered else None
        if answered:
            h = hyps.setdefault(hid, {"hypothesis": hid, "first_frame": f, "last_observed_frame": f,
                                      "observed_frames": 0, "confirmed_frame": f, "end_frame": None,
                                      "end_reason": None})
            if state == "OBSERVED":
                h["last_observed_frame"] = f
                h["observed_frames"] += 1
        frames.append({"frame": f, "t": round(f / FPS, 6),
                       "camera_status": "FIRST" if f == 0 else "CUT" if f == s.cut else "OK",
                       "state": state, "reason": reason, "hypothesis": hid,
                       "center": None if not answered else [round(v, 1) for v in centre],
                       "diameter": None if not answered else round(diam, 1),
                       "confidence": 0.8 if answered else None, "contested": contested and state == "OBSERVED",
                       "candidates": int(state == "OBSERVED"), "confirmed_live": len(hyps)})
    if s.cut is not None and 1 in hyps:
        hyps[1].update(end_frame=s.cut, end_reason="CAMERA_DISCONTINUITY")
    for h in hyps.values():
        h["observed_frames"] = max(1, h["observed_frames"])
    doc = {"schema": "spana.ball", "schema_version": "1.0.0",
           "provenance": {"note": f"synthetic touches scene {s.name}; no footage"},
           "video": {"path": "synthetic", "sha256": "1" * 64, "fps": FPS, "width": W, "height": H,
                     "frame_count_reported": s.n, "timestamp_source": "synthetic"},
           "detections_sha256": "0" * 64, "ball_config": {}, "frames": frames, "hypotheses": list(hyps.values())}
    validate_ball(doc)
    return doc


# --------------------------------------------------------------------------------------------- scenes
FEET = 500.0
D = 18.0                                     # ball diameter at a player's depth (0.12 of the height)


def at_feet(x: float, side: float = 1.0) -> Tuple[float, float]:
    """Ball centre at a dribbling distance in front of the feet point (0.15 h aside, 0.05 h above)."""
    return x + side * 0.15 * PH, FEET - 0.05 * PH


def seen(c, d=D, contested=False):
    return "OBSERVED", c, d, contested


def lerp(a: float, b: float, f: int, f0: int, f1: int) -> float:
    return a + (b - a) * min(1.0, max(0.0, (f - f0) / (f1 - f0)))


def dribble() -> Scene:
    ax = lambda f: 300 + 8 * f                                    # noqa: E731
    return Scene("dribble", 50, {"A": lambda f: (ax(f), FEET), "B": lambda f: (1150, FEET)},
                 {"T1": Track("A", 0, 50), "T2": Track("B", 0, 50)},
                 ball=lambda f: seen(at_feet(ax(f))),
                 truth=lambda f: ("UNSURE",) if f == 0 else ("CONTROL", "A"),   # frame 0: tracks not confirmed yet
                 possessions=[("A", 1, 49)])


def pass_between_teammates() -> Scene:
    def ball(f):
        if f < 20:
            return seen(at_feet(300))
        if f < 40:
            return seen((lerp(345, 855, f, 20, 39), FEET - 0.05 * PH))
        return seen(at_feet(900, -1))

    def truth(f):      # the pass leaves A's reach after frame 20 and enters B's at frame 39 (0.30 h from the feet)
        return ("UNSURE",) if f == 0 else ("CONTROL", "A") if f <= 20 else ("FREE",) if f <= 38 else ("CONTROL", "B")
    return Scene("pass", 60, {"A": lambda f: (300, FEET), "B": lambda f: (900, FEET)},
                 {"T1": Track("A", 0, 60), "T2": Track("B", 0, 60)}, ball=ball, truth=truth,
                 possessions=[("A", 1, 20), ("B", 39, 59)])


def fifty_fifty() -> Scene:
    return Scene("fifty_fifty", 40,
                 {"A": lambda f: (450 if f < 10 else 600, FEET), "B": lambda f: (830 if f < 10 else 676, FEET)},
                 {"T1": Track("A", 0, 40), "T2": Track("B", 0, 40)},
                 ball=lambda f: seen((640, FEET - 0.05 * PH)),
                 truth=lambda f: ("FREE",) if f < 10 else ("CONTESTED",), possessions=[])


def lofted_over_player() -> Scene:
    # the ball crosses the player's knees in the picture, in the air (small: further from the camera)
    return Scene("lofted", 40, {"C": lambda f: (640, FEET)}, {"T1": Track("C", 0, 40)},
                 ball=lambda f: seen((200 + 22 * f, FEET - 0.30 * PH), d=8.0),
                 truth=lambda f: ("AIR",), possessions=[])


def ball_behind_player() -> Scene:
    # at the feet line in the picture, but far behind the player (tiny ball)
    return Scene("behind", 30, {"C": lambda f: (640, FEET)}, {"T1": Track("C", 0, 30)},
                 ball=lambda f: seen((560 + 6 * f, FEET - 0.03 * PH), d=6.0),
                 truth=lambda f: ("AIR",), possessions=[])


def occluded_dribble() -> Scene:
    ax = lambda f: 250 + 8 * f                                    # noqa: E731

    def ball(f):
        if 15 <= f <= 19 or 30 <= f <= 39:
            return "PREDICTED", at_feet(ax(f)), D, False             # hidden: extrapolated, not seen
        if 40 <= f <= 44:
            return "LOST", None, None, False
        return seen(at_feet(ax(f)))

    def truth(f):
        return ("UNSURE",) if f == 0 or 15 <= f <= 19 or 30 <= f <= 44 else ("CONTROL", "A")
    return Scene("occluded_dribble", 60, {"A": lambda f: (ax(f), FEET), "B": lambda f: (1200, FEET)},
                 {"T1": Track("A", 0, 60), "T2": Track("B", 0, 60)}, ball=ball, truth=truth,
                 possessions=[("A", 1, 29), ("A", 45, 59)])              # 0.2 s bridged, 0.6 s not


def crossing_uncertain() -> Scene:
    ax = lambda f: 300 + 10 * f                                   # noqa: E731
    bx = lambda f: 900 - 10 * f                                   # noqa: E731

    def truth(f):
        if f == 0 or 24 <= f <= 36:
            return ("UNSURE",)
        return ("CONTROL", "A")
    return Scene("crossing_uncertain", 60, {"A": lambda f: (ax(f), FEET), "B": lambda f: (bx(f), FEET + 4)},
                 {"T1": Track("A", 0, 60), "T2": Track("B", 0, 60)},
                 ball=lambda f: seen(at_feet(ax(f))), truth=truth, possessions=None,
                 uncertain={"T1": (24, 36, ["T2"]), "T2": (24, 36, ["T1"])},
                 edge=tuple(range(15, 24)) + tuple(range(37, 46)))      # B near: CONTESTED is also right


def hidden_player_nearest() -> Scene:
    # the player on the ball is behind someone (track PREDICTED): nobody may be given the ball
    return Scene("hidden_nearest", 30, {"A": lambda f: (640, FEET), "B": lambda f: (760, FEET)},
                 {"T1": Track("A", 0, 30, state=lambda f: "LOST" if 10 <= f <= 19 else "ACTIVE"),
                  "T2": Track("B", 0, 30)},
                 ball=lambda f: seen(at_feet(640, -1)),
                 truth=lambda f: ("UNSURE",) if f == 0 or 10 <= f <= 19 else ("CONTROL", "A"),
                 possessions=[("A", 1, 9), ("A", 20, 29)])                # 11 frames apart: > bridge_s


def camera_cut() -> Scene:
    ax = lambda f: 300 + 8 * f if f < 25 else 500 + 8 * (f - 25)   # noqa: E731

    def ball(f):
        if 25 < f < 28:
            return "LOST", None, None, False                    # new shot: ball not confirmed yet
        return seen(at_feet(ax(f)))

    def truth(f):
        return ("UNSURE",) if f in (0, 25, 26, 27) else ("CONTROL", "A")
    return Scene("camera_cut", 50, {"A": lambda f: (ax(f), FEET), "B": lambda f: (1150, FEET)},
                 {"T1": Track("A", 0, 25, "CAMERA_CUT"), "T2": Track("B", 0, 25, "CAMERA_CUT"),
                  "T3": Track("A", 25, 50, epoch=1), "T4": Track("B", 25, 50, epoch=1)},
                 ball=ball, truth=truth, possessions=[("A", 1, 24), ("A", 28, 49)], cut=25)


SCENES = (dribble, pass_between_teammates, fifty_fifty, lofted_over_player, ball_behind_player, occluded_dribble,
          crossing_uncertain, hidden_player_nearest, camera_cut)
