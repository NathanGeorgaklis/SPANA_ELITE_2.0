"""Plain data records passed between layers. No logic beyond (de)serialisation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class CameraStatus(str, Enum):
    OK = "OK"            # inter-frame similarity transform estimated with enough inliers
    UNKNOWN = "UNKNOWN"  # could not be estimated; no compensation applied
    CUT = "CUT"          # shot change; motion continuity across this frame is meaningless
    FIRST = "FIRST"      # first frame of the video


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    REMOVED = "REMOVED"


class PositionSource(str, Enum):
    OBSERVED = "OBSERVED"    # matched to a detection on this frame
    PREDICTED = "PREDICTED"  # motion-model prediction (unobserved <= predict_max_s)
    HELD = "HELD"            # last position, camera-compensated, velocity frozen (unobserved > predict_max_s)


class EndReason(str, Enum):
    TENTATIVE_EXPIRED = "TENTATIVE_EXPIRED"
    LOST_EXPIRED = "LOST_EXPIRED"
    CAMERA_CUT = "CAMERA_CUT"
    CAMERA_MOTION_UNKNOWN = "CAMERA_MOTION_UNKNOWN"
    LEFT_FRAME = "LEFT_FRAME"


@dataclass(frozen=True)
class Detection:
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2, source pixels
    score: float
    class_id: int

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.box, dtype=float)


@dataclass(frozen=True)
class CameraObservation:
    status: CameraStatus
    affine: Tuple[Tuple[float, float, float], Tuple[float, float, float]]  # prev -> current, source pixels
    inliers: int
    hist_distance: Optional[float]

    @staticmethod
    def identity(status: CameraStatus, hist_distance: Optional[float] = None, inliers: int = 0) -> "CameraObservation":
        return CameraObservation(status, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), inliers, hist_distance)

    def matrix(self) -> np.ndarray:
        return np.asarray(self.affine, dtype=float)


@dataclass(frozen=True)
class TrackObservation:
    """One track on one frame, as emitted by the multi-player tracker."""
    track_id: str
    state: TrackState
    source: PositionSource
    box: Tuple[float, float, float, float]
    detection_index: Optional[int]   # index into the frame's detection list when OBSERVED
    detection_score: Optional[float]
    contested: bool                  # another track was a near-equal candidate for this detection
    time_since_observed_s: float
    observed_frames: int             # number of OBSERVED frames so far
    epoch: int                       # continuity epoch (increments at every cut or unknown camera motion)
    max_overlap_iou: float           # largest IoU with any other live track's box on this frame
    overlap_with: Optional[str]      # that track's id (None when max_overlap_iou == 0)
    gaps_bridged: int                # unobserved gaps this track bridged by motion re-match (S4)
    last_gap_s: float                # duration of the most recent bridged gap (0 if none)


@dataclass
class FrameTracks:
    frame_index: int
    time_s: float
    camera: CameraObservation
    tracks: List[TrackObservation] = field(default_factory=list)
    ended: List[Tuple[str, EndReason]] = field(default_factory=list)
    decisions: List[dict] = field(default_factory=list)   # association evidence, for the decision log
