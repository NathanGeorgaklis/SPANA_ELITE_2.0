"""Ball path v1: which sightings are the ball, decided afterwards over the whole clip. Never a guessed position.

In: one validated `spana.ball` 1.3.0 document (per-frame `sightings` of every hypothesis + `camera_affine`).
Out: `spana.ball_path` 1.0.0 (`spana.artifacts.validate_ball_path`). Pure and deterministic: the document is a
function of the ball document and the `ball_path` config section. Design: records/ball_path_v1_2026-11-23/DESIGN.md.

1. Stretch: the sightings of one hypothesis, split wherever two consecutive sightings are more than max_hole_s
   apart, a CUT / UNKNOWN camera frame lies between them, or the hypothesis was found again far from where it was
   heading (the sighting's rejoin_d > the ball document's ball_config.reacquire_max_d): what follows a far rejoin
   may be another object and must earn its confidence on its own.
2. Confident stretch: duration (last - first sighting time) >= min_stretch_s AND density (sightings / processed
   frames from first to last) >= min_density. Any other stretch is never a ball answer.
3. One ball: frames covered by two or more confident stretches are UNKNOWN / CONFLICT for all of them.
4. Per frame: SEEN (exactly one confident stretch covers the frame and sighted the ball on it; carries the position)
   or UNKNOWN with a reason and no position: HOLE (inside the one covering stretch, not sighted), CONFLICT, GAP
   (between two path segments), NO_BALL (before the first / after the last path segment).
5. Path segment: a maximal run of frames covered by exactly one confident stretch (the same one) holding at least
   one SEEN frame. Gap: between consecutive segments, from the last SEEN frame of one to the first SEEN frame of the
   next, with a kind (first match): CAMERA_BREAK (a CUT / UNKNOWN camera frame in (from, to]), CONFLICT (a conflict
   frame inside), IMPLAUSIBLE (the from position, carried through the camera transforms, is farther from the to
   position than the ball tracker's own speed bound allows), OPEN (the only kind "left here, arrived there" may be
   read from). A gap's `frames` counts the processed frames strictly between its two ends (with a frame stride > 1
   these are not video frames; `duration_s` is always in seconds).
Note: a run covered by one confident stretch that holds no SEEN frame (possible only between conflict frames) is not
a segment; its frames stay HOLE (inside that stretch, not sighted), wherever the run lies.
"""
from __future__ import annotations

import dataclasses
import math
from bisect import bisect_right
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import BallPathConfig

PATH_SCHEMA = ("spana.ball_path", "1.0.0")
BALL_VERSION_NEEDED = "1.3.0"                     # sightings + camera_affine
BALL_CONFIG_KEYS = ("reacquire_max_d", "gate_base_d", "max_speed_d_per_s", "min_diameter_px")   # read from the ball doc
STATES_REASONS = {"SEEN": ("CONFIDENT_STRETCH",), "UNKNOWN": ("HOLE", "CONFLICT", "GAP", "NO_BALL")}
GAP_KINDS = ("CAMERA_BREAK", "CONFLICT", "IMPLAUSIBLE", "OPEN")
STRETCH_KEYS = ("stretch", "hypothesis", "first_frame", "last_frame", "sightings", "frames", "duration_s", "density",
                "confident")
FRAME_KEYS = ("frame", "t", "state", "reason", "stretch", "center", "diameter", "score")
GAP_KEYS = ("from", "to", "frames", "duration_s", "kind", "distance_d", "allowed_d")
GAP_END_KEYS = ("frame", "t", "center", "diameter", "stretch")
_BREAKS = ("CUT", "UNKNOWN")
_EPS = 1e-9


def stretches(frames: Sequence[dict], pc: BallPathConfig, far_rejoin_d: float) -> List[dict]:
    """Every stretch of the clip (confident or not), ordered by first frame then hypothesis, ids S000001..."""
    seen_at: Dict[str, List[int]] = {}             # hypothesis -> indices into `frames` where it was sighted
    far: set = set()                               # (hypothesis, index): found again far from its prediction
    for k, fr in enumerate(frames):
        for s in fr["sightings"]:
            seen_at.setdefault(s["hypothesis"], []).append(k)
            if s["rejoin_d"] is not None and s["rejoin_d"] > far_rejoin_d + _EPS:
                far.add((s["hypothesis"], k))
    breaks = [k for k, fr in enumerate(frames) if fr["camera_status"] in _BREAKS]          # sorted

    def broken(a: int, b: int) -> bool:            # a CUT / UNKNOWN camera frame in (a, b]
        return bisect_right(breaks, b) > bisect_right(breaks, a)

    rows = []
    for hid in sorted(seen_at):
        idx, start = seen_at[hid], 0
        for j in range(1, len(idx) + 1):
            split = j == len(idx) or frames[idx[j]]["t"] - frames[idx[j - 1]]["t"] > pc.max_hole_s + _EPS \
                or broken(idx[j - 1], idx[j]) or (hid, idx[j]) in far
            if not split:
                continue
            run = idx[start:j]
            first, last = run[0], run[-1]
            duration = frames[last]["t"] - frames[first]["t"]
            density = len(run) / (last - first + 1)
            rows.append({"hypothesis": hid, "first_frame": frames[first]["frame"], "last_frame": frames[last]["frame"],
                         "sightings": len(run), "frames": last - first + 1, "duration_s": round(duration, 6),
                         "density": round(density, 4),
                         "confident": duration >= pc.min_stretch_s - _EPS and density >= pc.min_density - _EPS,
                         "_first": first, "_last": last, "_seen": set(run)})
            start = j
    rows.sort(key=lambda r: (r["first_frame"], r["hypothesis"]))
    for n, r in enumerate(rows, 1):
        r["stretch"] = f"S{n:06d}"
    return rows


def _carry(p: Tuple[float, float], affine: Sequence[Sequence[float]]) -> Tuple[float, float]:
    (a, b, c), (d, e, f) = affine
    return a * p[0] + b * p[1] + c, d * p[0] + e * p[1] + f


def _gap_end(row: dict) -> dict:
    return {k: row[k] for k in GAP_END_KEYS}


def ball_path(ball_doc: dict, pc: BallPathConfig) -> Tuple[List[dict], List[dict], List[dict]]:
    """(frames, stretches, gaps) of a validated spana.ball 1.3.0 document."""
    if ball_doc.get("schema_version") != BALL_VERSION_NEEDED:
        raise ValueError(f"the ball path needs a spana.ball {BALL_VERSION_NEEDED} document (sightings, camera_affine)")
    frames = ball_doc["frames"]
    bc = ball_doc["ball_config"]
    missing = [k for k in BALL_CONFIG_KEYS if not isinstance(bc, dict) or not isinstance(bc.get(k), (int, float))]
    if missing:
        raise ValueError(f"the ball document's ball_config lacks {missing}")
    rows = stretches(frames, pc, bc["reacquire_max_d"])
    confident = [r for r in rows if r["confident"]]
    cover: List[List[dict]] = [[] for _ in frames]
    for r in confident:
        for k in range(r["_first"], r["_last"] + 1):
            cover[k].append(r)

    out: List[dict] = []
    for k, fr in enumerate(frames):
        row = {"frame": fr["frame"], "t": fr["t"], "state": "UNKNOWN", "reason": "NO_BALL", "stretch": None,
               "center": None, "diameter": None, "score": None}
        if len(cover[k]) > 1:
            row["reason"] = "CONFLICT"
        elif len(cover[k]) == 1:
            r = cover[k][0]
            row["stretch"], row["reason"] = r["stretch"], "HOLE"
            if k in r["_seen"]:
                s = next(x for x in fr["sightings"] if x["hypothesis"] == r["hypothesis"])
                row.update(state="SEEN", reason="CONFIDENT_STRETCH", center=list(s["center"]),
                           diameter=s["diameter"], score=s["score"])
        out.append(row)

    # path segments: maximal runs covered by exactly one (the same) confident stretch, holding a SEEN frame
    segments: List[Tuple[int, int]] = []           # (first SEEN index, last SEEN index) per segment
    k = 0
    while k < len(out):
        if len(cover[k]) != 1:
            k += 1
            continue
        j = k
        while j + 1 < len(out) and len(cover[j + 1]) == 1 and cover[j + 1][0] is cover[k][0]:
            j += 1
        seen = [i for i in range(k, j + 1) if out[i]["state"] == "SEEN"]
        if seen:
            segments.append((seen[0], seen[-1]))
        k = j + 1

    gaps: List[dict] = []
    for (_, a), (b, _) in zip(segments, segments[1:]):
        for i in range(a + 1, b):
            if out[i]["reason"] == "NO_BALL":
                out[i]["reason"] = "GAP"
        duration = frames[b]["t"] - frames[a]["t"]
        gap = {"from": _gap_end(out[a]), "to": _gap_end(out[b]), "frames": b - a - 1,
               "duration_s": round(duration, 6), "kind": "OPEN", "distance_d": None, "allowed_d": None}
        if any(frames[i]["camera_status"] in _BREAKS for i in range(a + 1, b + 1)):
            gap["kind"] = "CAMERA_BREAK"
        else:
            p = tuple(out[a]["center"])
            for i in range(a + 1, b + 1):
                p = _carry(p, frames[i]["camera_affine"])
            d = max(out[a]["diameter"], out[b]["diameter"], bc["min_diameter_px"])
            dist_d = math.hypot(out[b]["center"][0] - p[0], out[b]["center"][1] - p[1]) / d
            allowed = bc["gate_base_d"] + bc["max_speed_d_per_s"] * duration
            gap["distance_d"], gap["allowed_d"] = round(dist_d, 3), round(allowed, 3)
            if any(out[i]["reason"] == "CONFLICT" for i in range(a + 1, b)):
                gap["kind"] = "CONFLICT"
            elif dist_d > allowed + _EPS:
                gap["kind"] = "IMPLAUSIBLE"
        gaps.append(gap)

    return out, [{k: r[k] for k in STRETCH_KEYS} for r in rows], gaps


def ball_path_document(ball_doc: dict, pc: BallPathConfig, provenance: dict, ball_sha256: str) -> dict:
    frames, rows, gaps = ball_path(ball_doc, pc)
    return {"schema": PATH_SCHEMA[0], "schema_version": PATH_SCHEMA[1], "provenance": provenance,
            "video": ball_doc["video"], "ball_sha256": ball_sha256, "ball_path_config": dataclasses.asdict(pc),
            "frames": frames, "stretches": rows, "gaps": gaps}


def as_ball_answers(path_doc: dict) -> dict:
    """The path in the shape `tools.ball_score` scores: SEEN = an OBSERVED answer, everything else = no answer."""
    return {"video": path_doc["video"],
            "frames": [{"frame": f["frame"], "state": "OBSERVED" if f["state"] == "SEEN" else "LOST",
                        "center": f["center"]} for f in path_doc["frames"]]}


def summary(path_doc: dict) -> dict:
    fr = path_doc["frames"]
    count = lambda key, v: sum(f[key] == v for f in fr)      # noqa: E731
    return {"frames": len(fr), "seen": count("state", "SEEN"),
            **{r.lower(): count("reason", r) for r in STATES_REASONS["UNKNOWN"]},
            "stretches": len(path_doc["stretches"]),
            "confident_stretches": sum(s["confident"] for s in path_doc["stretches"]),
            "gaps": {k: sum(g["kind"] == k for g in path_doc["gaps"]) for k in GAP_KINDS}}
