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
    ACTIVE = "ACTIVE"          # observed on this frame
    OCCLUDED = "OCCLUDED"      # unobserved and covered by another observed player's box
    LOST = "LOST"              # unobserved, not explained by an occluder
    TERMINATED = "TERMINATED"  # ended on this frame (reported in FrameTracks.ended, never emitted as live)


class PositionSource(str, Enum):
    OBSERVED = "OBSERVED"    # matched to a detection on this frame
    PREDICTED = "PREDICTED"  # motion-model prediction (unobserved <= predict_max_s)
    HELD = "HELD"            # last position, camera-compensated, velocity frozen (unobserved > predict_max_s)


class EndReason(str, Enum):
    UNCONFIRMED_MISSED = "UNCONFIRMED_MISSED"   # an unconfirmed track stayed unobserved beyond rematch_max_s
    UNCONFIRMED_TIMEOUT = "UNCONFIRMED_TIMEOUT"  # an unconfirmed track could not confirm within unconfirmed_max_s
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
    """One live track on one frame, as emitted by the multi-player tracker."""
    track_id: str
    state: TrackState
    confirmed: bool                  # False = new uncertain track (not yet observed over confirm_s, or nested)
    source: PositionSource
    box: Tuple[float, float, float, float]
    detection_index: Optional[int]   # index into the frame's detection list when ACTIVE
    detection_score: Optional[float]
    confidence: float                # track confidence, see tracker docstring (uncalibrated, not a probability)
    contested: bool                  # another candidate was within contest_margin for this association
    age_frames: int                  # frames since the track was created
    missed_frames: int               # consecutive frames unobserved (0 when ACTIVE)
    time_since_observed_s: float
    observed_frames: int
    epoch: int                       # continuity epoch (increments at every cut or unknown camera motion)
    occluder: Optional[str]          # covering track when OCCLUDED
    max_overlap_iou: float           # largest IoU with any other live track's box on this frame
    overlap_with: Optional[str]
    recoveries: int                  # gaps bridged after LOST/OCCLUDED (motion-exclusive or appearance-distinct)
    last_gap_s: float                # duration of the most recent bridged gap (0 if none)
    possible_predecessors: Tuple[str, ...]  # unobserved tracks this one may continue (recovery was refused)


@dataclass(frozen=True)
class HistoryEntry:
    frame_index: int
    time_s: float
    state: TrackState
    source: PositionSource
    box: Tuple[float, float, float, float]
    detection_score: Optional[float]


@dataclass(frozen=True)
class TrackletSummary:
    track_id: str
    epoch: int
    first_frame: int
    last_frame: int                  # last frame the track was live
    last_observed_frame: int
    observed_frames: int
    confirmed: bool
    end_reason: Optional[str]        # None if still live at the end of the run
    recoveries: int
    possible_predecessors: Tuple[str, ...]


@dataclass
class FrameTracks:
    frame_index: int
    time_s: float
    camera: CameraObservation
    tracks: List[TrackObservation] = field(default_factory=list)
    ended: List[Tuple[str, EndReason]] = field(default_factory=list)
    decisions: List[dict] = field(default_factory=list)   # association evidence, for the decision log
