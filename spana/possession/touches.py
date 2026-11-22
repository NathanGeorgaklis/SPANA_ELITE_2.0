"""Who is on the ball, per frame, fail closed (layer 8, possession / touches v1).

In: a validated `spana.tracks` and a validated `spana.ball` of the same video and frames. Out: `spana.touches` 1.0.0
(`spana.artifacts.validate_touches`).

Image space only (no pitch layer yet): the distance from the ball centre to a track's feet point (bottom-centre of
its box), in that track's body heights (box height). Per frame:
  UNKNOWN  CAMERA_DISCONTINUITY   the tracks' or the ball's camera is CUT / UNKNOWN (every track ends there anyway)
           BALL_NOT_OBSERVED      the ball is not OBSERVED (PREDICTED, LOST, AMBIGUOUS): only a seen ball counts
           BALL_CONTESTED         the ball answer itself has rival candidates
           TRACK_NOT_CERTAIN      the nearest track in reach, or a track within the rival margin, is not usable
                                  (usable = ACTIVE, OBSERVED, confirmed, CERTAIN): never hand the ball to a guess
           BALL_AIRBORNE_SUSPECT  the ball is higher in the picture than possession.max_ball_above_feet_h above the
                                  nearest feet (in the air, or behind the player)
           BALL_SCALE_MISMATCH    ball diameter / nearest track height outside [diameter_to_height_min, max]: the
                                  ball is not at that player's depth
  FREE     NO_TRACK_IN_REACH      the ball is seen and no track's feet are within possession.reach_h
  CONTESTED RIVAL_WITHIN_MARGIN   a second usable track within possession.rival_margin_h of the nearest one
  CONTROL  NEAREST_FEET           exactly one usable track is on the ball
Possessions: runs of CONTROL by one track whose CONTROL frames are at most possession.bridge_s apart. A run is never
bridged across another track's CONTROL, a CONTESTED frame or a camera discontinuity.
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

from ..config import PossessionConfig

TOUCHES_SCHEMA = ("spana.touches", "1.0.0")
STATES_REASONS = {
    "CONTROL": ("NEAREST_FEET",),
    "CONTESTED": ("RIVAL_WITHIN_MARGIN",),
    "FREE": ("NO_TRACK_IN_REACH",),
    "UNKNOWN": ("CAMERA_DISCONTINUITY", "BALL_NOT_OBSERVED", "BALL_CONTESTED", "TRACK_NOT_CERTAIN",
                "BALL_AIRBORNE_SUSPECT", "BALL_SCALE_MISMATCH"),
}
NO_MEASUREMENT = ("CAMERA_DISCONTINUITY", "BALL_NOT_OBSERVED", "BALL_CONTESTED")   # candidates are not computed
DECIMALS = 4
_BREAKS = ("CUT", "UNKNOWN")
_EPS = 1e-9


def usable(track: dict) -> bool:
    """A track whose position and identity can carry the ball: observed now, confirmed, identity CERTAIN."""
    return (track["state"] == "ACTIVE" and track["source"] == "OBSERVED" and track["confirmed"]
            and track["identity"] == "CERTAIN")


def geometry(box: Sequence[float], center: Sequence[float], diameter: float) -> Tuple[float, float, float]:
    """(distance to the feet point, height of the ball above the feet line, ball diameter / height), all relative
    to the box height."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    fx, fy = 0.5 * (x1 + x2), y2
    return math.hypot(center[0] - fx, center[1] - fy) / h, (fy - center[1]) / h, diameter / h


def frame_answer(tracks_frame: dict, ball_frame: dict, pc: PossessionConfig) -> Tuple[dict, Optional[dict]]:
    """The touches frame record and its decision-log row (None when nothing was measured)."""
    out = {"frame": ball_frame["frame"], "t": ball_frame["t"], "state": "UNKNOWN", "reason": None, "track_id": None,
           "ball_frame_state": ball_frame["state"], "candidates": []}
    if tracks_frame["camera"]["status"] in _BREAKS or ball_frame["camera_status"] in _BREAKS:
        out["reason"] = "CAMERA_DISCONTINUITY"
        return out, None
    if ball_frame["state"] != "OBSERVED":
        out["reason"] = "BALL_NOT_OBSERVED"
        return out, None
    if ball_frame["contested"]:
        out["reason"] = "BALL_CONTESTED"
        return out, None
    c, d = ball_frame["center"], ball_frame["diameter"]
    geo = {t["track_id"]: (geometry(t["box"], c, d), usable(t)) for t in tracks_frame["tracks"]}
    band = pc.reach_h + pc.rival_margin_h
    near = sorted((g[0], tid) for tid, (g, _) in geo.items() if g[0] <= band + _EPS)
    out["candidates"] = [{"track_id": tid, "distance_h": round(dist, DECIMALS), "usable": geo[tid][1]}
                         for dist, tid in near]
    log = {"frame": out["frame"], "candidates": out["candidates"]}
    if not near or near[0][0] > pc.reach_h + _EPS:
        out["state"], out["reason"] = "FREE", "NO_TRACK_IN_REACH"
        return out, {**log, "state": "FREE", "reason": out["reason"], "track_id": None}
    d0, tid0 = near[0]
    (_, above, ratio), ok0 = geo[tid0]
    log.update(nearest=tid0, above_feet_h=round(above, DECIMALS), diameter_to_height=round(ratio, DECIMALS))
    rivals = [tid for dist, tid in near[1:] if dist <= d0 + pc.rival_margin_h + _EPS]
    if not ok0 or any(not geo[r][1] for r in rivals):
        out["reason"] = "TRACK_NOT_CERTAIN"
    elif above > pc.max_ball_above_feet_h + _EPS:
        out["reason"] = "BALL_AIRBORNE_SUSPECT"
    elif not pc.diameter_to_height_min - _EPS <= ratio <= pc.diameter_to_height_max + _EPS:
        out["reason"] = "BALL_SCALE_MISMATCH"
    elif rivals:
        out["state"], out["reason"] = "CONTESTED", "RIVAL_WITHIN_MARGIN"
    else:
        out["state"], out["reason"], out["track_id"] = "CONTROL", "NEAREST_FEET", tid0
    return out, {**log, "state": out["state"], "reason": out["reason"], "track_id": out["track_id"]}


def possessions(frames: Sequence[dict], pc: PossessionConfig) -> List[dict]:
    """Runs of CONTROL by one track (module docstring)."""
    eps: List[dict] = []
    cur: Optional[dict] = None
    last_t = -math.inf
    pending = 0                              # non-CONTROL frames since the run's last CONTROL frame

    def close() -> None:
        nonlocal cur
        if cur is not None:
            eps.append(cur)
        cur = None

    for fr in frames:
        if fr["state"] == "CONTROL":
            if cur is not None and cur["track_id"] == fr["track_id"] and fr["t"] - last_t <= pc.bridge_s + _EPS:
                cur["last_frame"] = fr["frame"]
                cur["frames_control"] += 1
                cur["frames_gap"] += pending
            else:
                close()
                cur = {"track_id": fr["track_id"], "first_frame": fr["frame"], "last_frame": fr["frame"],
                       "frames_control": 1, "frames_gap": 0}
            last_t, pending = fr["t"], 0
        elif fr["state"] == "CONTESTED" or fr["reason"] == "CAMERA_DISCONTINUITY":
            close()
        else:
            pending += 1
    close()
    return eps


def touches(tracks_doc: dict, ball_doc: dict, pc: PossessionConfig) -> Tuple[List[dict], List[dict], List[dict]]:
    """(frames, possessions, decision log) of two validated artifacts of the same video and frames."""
    if tracks_doc["video"]["sha256"] != ball_doc["video"]["sha256"]:
        raise ValueError("tracks and ball artifacts come from different videos")
    tf = tracks_doc["frames"]
    bf = ball_doc["frames"]
    if [f["frame"] for f in tf] != [f["frame"] for f in bf] or \
            any(abs(a["t"] - b["t"]) > 1e-6 for a, b in zip(tf, bf)):
        raise ValueError("tracks and ball artifacts cover different frames")
    frames, log = [], []
    for a, b in zip(tf, bf):
        fr, row = frame_answer(a, b, pc)
        frames.append(fr)
        if row is not None:
            log.append(row)
    return frames, possessions(frames, pc), log


def touches_document(tracks_doc: dict, ball_doc: dict, pc: PossessionConfig, provenance: dict,
                     tracks_sha256: str, ball_sha256: str) -> Tuple[dict, List[dict]]:
    frames, eps, log = touches(tracks_doc, ball_doc, pc)
    doc = {"schema": TOUCHES_SCHEMA[0], "schema_version": TOUCHES_SCHEMA[1], "provenance": provenance,
           "video": tracks_doc["video"], "tracks_sha256": tracks_sha256, "ball_sha256": ball_sha256,
           "possession_config": dataclasses.asdict(pc), "frames": frames, "possessions": eps}
    return doc, log
