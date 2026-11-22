"""Frozen DEV ball labels (spana.ball_labels 1.0.0): one forced answer per labelled frame for the match ball.

Each labelled frame has exactly one entry:
  VISIBLE  the match ball is seen: `box` [x1, y1, x2, y2] in source pixels (tight around the ball)
  HIDDEN   the match ball is in the picture but not seen (behind a player, under a foot); `point` [x, y] is the
           annotator's best guess or null. Never scored as a position: it only says "a tracker must not claim it seen"
  OUT      the match ball is not in the picture
  UNSURE   the annotator cannot tell: the frame is not scored
`other_balls` (any state) boxes balls that are not the match ball (spare balls, warm-up balls): a tracker locked on
one is wrong, but differently from locking on noise.
Frozen labels are never edited: a correction is a new version with a reason.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCHEMA = ("spana.ball_labels", "1.0.0")
STATES = ("VISIBLE", "HIDDEN", "OUT", "UNSURE")
RECORD_FIELDS = ("schema", "schema_version", "split", "clip", "frame_step", "label_version", "annotator",
                 "frozen_utc", "frames")
Box = Tuple[float, float, float, float]


class BallLabelError(ValueError):
    pass


@dataclass(frozen=True)
class BallLabel:
    frame: int
    state: str
    box: Optional[Box]
    point: Optional[Tuple[float, float]]
    other_balls: Tuple[Box, ...]


def _box(v, w: int, h: int, where: str) -> Box:
    if not (isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)):
        raise BallLabelError(f"{where}: box must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(x) for x in v)
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise BallLabelError(f"{where}: box outside the image or empty")
    return x1, y1, x2, y2


def validate_record(rec: dict, allowed_frames: Optional[Sequence[int]] = None) -> Dict[int, BallLabel]:
    """The exported record is complete and consistent, or it is refused. Returns frame -> label."""
    missing = [k for k in RECORD_FIELDS if k not in rec]
    if missing:
        raise BallLabelError(f"ball labels missing fields: {missing}")
    if (rec["schema"], rec["schema_version"]) != SCHEMA:
        raise BallLabelError(f"unsupported schema {rec['schema']} {rec['schema_version']}")
    if rec["split"] != "DEV":
        raise BallLabelError("only DEV ball labels are accepted now")
    if not str(rec["annotator"]).strip():
        raise BallLabelError("annotator missing")
    w, h = int(rec["clip"]["width"]), int(rec["clip"]["height"])
    allowed = None if allowed_frames is None else set(allowed_frames)
    out: Dict[int, BallLabel] = {}
    for e in rec["frames"]:
        f = e.get("frame")
        if not isinstance(f, int) or f < 0:
            raise BallLabelError(f"bad frame index {f!r}")
        if f in out:
            raise BallLabelError(f"frame {f} labelled twice")
        if allowed is not None and f not in allowed:
            raise BallLabelError(f"frame {f} is not a frame of the pack")
        state = e.get("state")
        if state not in STATES:
            raise BallLabelError(f"frame {f}: state must be one of {STATES}")
        box = point = None
        if state == "VISIBLE":
            box = _box(e.get("box"), w, h, f"frame {f}")
        elif e.get("box") is not None:
            raise BallLabelError(f"frame {f}: only VISIBLE has a box")
        if state == "HIDDEN" and e.get("point") is not None:
            p = e["point"]
            if not (isinstance(p, list) and len(p) == 2 and 0 <= p[0] <= w and 0 <= p[1] <= h):
                raise BallLabelError(f"frame {f}: point must be [x, y] inside the image")
            point = (float(p[0]), float(p[1]))
        elif state != "HIDDEN" and e.get("point") is not None:
            raise BallLabelError(f"frame {f}: only HIDDEN has a point")
        others = tuple(_box(b, w, h, f"frame {f} other ball") for b in e.get("other_balls", []))
        out[f] = BallLabel(f, state, box, point, others)
    if not out:
        raise BallLabelError("no labelled frames")
    return out


def summary(labels: Dict[int, BallLabel]) -> dict:
    counts = {s: sum(1 for x in labels.values() if x.state == s) for s in STATES}
    return {"frames_labelled": len(labels), "states": counts,
            "other_balls": sum(len(x.other_balls) for x in labels.values())}


def load_frozen(label_dir: Path) -> Tuple[dict, Dict[int, BallLabel]]:
    """Load data/ball_labels/<clip>/v<N>: the record must still match the sha256 written when it was frozen."""
    label_dir = Path(label_dir)
    raw = (label_dir / "ball_labels_frozen.json").read_bytes()
    frozen = json.loads((label_dir / "FROZEN.json").read_text())
    if hashlib.sha256(raw).hexdigest() != frozen["record_sha256"]:
        raise BallLabelError(f"{label_dir}: ball labels changed after freezing")
    rec = json.loads(raw)
    return rec, validate_record(rec)


def frames_of(labels: Dict[int, BallLabel]) -> List[int]:
    return sorted(labels)
