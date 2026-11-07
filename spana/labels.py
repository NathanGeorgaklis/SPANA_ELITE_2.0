"""Human ground truth in MOT 1.1 format: parsing and validation (no inference, no tuning).

Columns: frame (1-based video frame), id, x, y, w, h, not_ignored (1 player / 0 ignore region), class, visibility
(1 = fully visible, 0.5 = occluded). Rows with not_ignored = 0 are ignore regions (crowds, bench): never scored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Box = Tuple[float, float, float, float]


class LabelError(ValueError):
    pass


@dataclass
class GroundTruth:
    players: Dict[int, Dict[str, Box]] = field(default_factory=dict)       # 0-based frame -> {player id: box}
    occluded: Dict[int, set] = field(default_factory=dict)                 # 0-based frame -> occluded ids
    ignore: Dict[int, List[Box]] = field(default_factory=dict)             # 0-based frame -> ignore regions

    @property
    def frames(self) -> List[int]:
        return sorted(set(self.players) | set(self.ignore))


def parse_mot(text: str, width: int, height: int, allowed_frames: Sequence[int]) -> GroundTruth:
    """Parse and validate. allowed_frames: 0-based video frame indices that were labelled (marked done)."""
    allowed = set(allowed_frames)
    gt = GroundTruth()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 9:
            raise LabelError(f"line {n}: expected 9 columns, got {len(parts)}")
        try:
            fr, pid = int(parts[0]), int(parts[1])
            x, y, w, h = (float(v) for v in parts[2:6])
            keep, cls, vis = int(parts[6]), int(parts[7]), float(parts[8])
        except ValueError as e:
            raise LabelError(f"line {n}: {e}") from None
        f0 = fr - 1
        if f0 not in allowed:
            raise LabelError(f"line {n}: frame {fr} was not a labelled (done) frame")
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise LabelError(f"line {n}: invalid box")
        if x < -0.5 or y < -0.5 or x + w > width + 0.5 or y + h > height + 0.5:
            raise LabelError(f"line {n}: box outside the {width}x{height} image")
        if keep not in (0, 1) or cls != 1 or vis not in (0.5, 1.0):
            raise LabelError(f"line {n}: not_ignored must be 0/1, class 1, visibility 1 or 0.5")
        box = (x, y, x + w, y + h)
        if keep == 0:
            gt.ignore.setdefault(f0, []).append(box)
            continue
        if pid < 1:
            raise LabelError(f"line {n}: player id must be >= 1")
        frame = gt.players.setdefault(f0, {})
        if str(pid) in frame:
            raise LabelError(f"line {n}: id {pid} appears twice on frame {fr}")
        frame[str(pid)] = box
        if vis == 0.5:
            gt.occluded.setdefault(f0, set()).add(str(pid))
    for f0 in allowed:              # a done frame with no players is valid (empty truth), keep it explicit
        gt.players.setdefault(f0, {})
    return gt
