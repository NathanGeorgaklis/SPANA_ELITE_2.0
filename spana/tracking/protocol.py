"""Minimal tracker interface (tracker bench v1, 2026-11-20).

Any multi-player tracker, ours or a third-party candidate, can be driven frame by frame through `Tracker.update` on
the same detections. The production pipeline (`spana.cli track`) still drives `MultiPlayerTracker` directly; this
protocol exists so candidates can be compared on identical input (bench/) without touching detection or identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..records import Detection


@dataclass(frozen=True)
class FrameInput:
    """One processed video frame. `embeddings`, when given, has one row per detection (same order)."""
    index: int                              # source video frame index, strictly increasing
    t: float                                # seconds (PTS)
    image: Optional[np.ndarray] = None      # BGR frame; trackers that need pixels (camera motion) refuse None
    embeddings: Optional[np.ndarray] = None


@dataclass(frozen=True)
class TrackOutput:
    """A confirmed track observed on this frame, on exactly one input detection."""
    track_id: str
    detection_index: int
    box: Tuple[float, float, float, float]
    identity_certain: bool = True           # False = the tracker itself flags the identity as uncertain


class Tracker(Protocol):
    name: str

    def update(self, detections: Sequence[Detection], frame: FrameInput) -> List[TrackOutput]:
        """Advance one frame; return the confirmed tracks observed on it (at most one per detection)."""
        ...


def check_outputs(outputs: Sequence[TrackOutput], detections: Sequence[Detection]) -> None:
    """Interface contract: unique track ids, each on a distinct, existing detection whose box it reports."""
    ids = [o.track_id for o in outputs]
    dets = [o.detection_index for o in outputs]
    if len(set(ids)) != len(ids):
        raise ValueError("a track id appears twice on one frame")
    if len(set(dets)) != len(dets):
        raise ValueError("two tracks claim the same detection")
    for o in outputs:
        if not 0 <= o.detection_index < len(detections):
            raise ValueError(f"track {o.track_id}: detection index {o.detection_index} out of range")
        if not np.allclose(o.box, detections[o.detection_index].box, atol=1e-3):
            raise ValueError(f"track {o.track_id}: box differs from its detection")
