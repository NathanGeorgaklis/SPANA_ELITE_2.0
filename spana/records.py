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


class IdentityState(str, Enum):
    """Orthogonal to the lifecycle state: whether the track's identity is still trusted."""
    CERTAIN = "CERTAIN"
    IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"   # a rival track fitted the same evidence and appearance could not
                                                # separate them; held until clean appearance evidence resolves it


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


class TargetState(str, Enum):
    """Per-frame answer of the target-identity layer: where is the user's selected player?"""
    TARGET = "TARGET"        # confident: this box is the selected player
    CANDIDATE = "CANDIDATE"  # likely match awaiting the user's confirmation (spana.target >= 2.0.0)
    LOST = "LOST"            # not visible or not found
    ABSTAIN = "ABSTAIN"      # unsure: no answer rather than a possibly wrong one


class TargetReason(str, Enum):
    """Why the target layer gave its answer on a frame (spana.target 1.0.0)."""
    TRACK_OBSERVED = "TRACK_OBSERVED"        # TARGET: followed track observed, confirmed, identity CERTAIN
    NOT_CONFIRMED = "NOT_CONFIRMED"          # ABSTAIN: followed track not yet confirmed by the tracker
    IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"  # ABSTAIN: followed track is IDENTITY_UNCERTAIN now
    AFTER_UNCERTAIN = "AFTER_UNCERTAIN"      # ABSTAIN: followed track was uncertain earlier (target.resume_after_uncertain false)
    TRACK_OCCLUDED = "TRACK_OCCLUDED"        # ABSTAIN: followed track hidden behind another player
    TRACK_LOST = "TRACK_LOST"                # LOST: followed track unobserved, not explained by an occluder
    TRACK_ENDED = "TRACK_ENDED"              # LOST: followed track ended (any EndReason, incl. CAMERA_CUT)
    RELINK_PREDECESSOR = "RELINK_PREDECESSOR"  # CANDIDATE (2.0.0): the only new track the tracker links to the lost one
    REENTRY_REID = "REENTRY_REID"            # CANDIDATE (2.4.0): returning track offered by its re-ID signature
    USER_VOUCHED = "USER_VOUCHED"            # TARGET (2.2.0): track uncertain only by an appearance doubt (no rival),
                                             #   and the user tapped it during that same doubt episode
    PLAYER_CONFIRMED = "PLAYER_CONFIRMED"    # TARGET (2.10.0): track IDENTITY_UNCERTAIN, but the player confirmed IS_ME
                                             #   in this very uncertainty episode (player_confirmation, not in conflict)
    GALLERY_REID = "GALLERY_REID"            # TARGET (2.10.0): track IDENTITY_UNCERTAIN, episode cleared for the target
                                             #   by re-ID against the human-confirmed gallery (calibrated margin)
    PLAYER_REJECTED = "PLAYER_REJECTED"      # ABSTAIN (2.10.0): the player said NOT_ME for this track in this episode
    CONFIRMATION_CONFLICT = "CONFIRMATION_CONFLICT"  # ABSTAIN (2.10.0): a NOT_ME contradicts the target's own CERTAIN
                                             #   answer on this track: not applied, held uncertain, re-confirmation


class TeamLabel(str, Enum):
    """Per-segment answer of the team-classification layer (spana.teams 1.0.0). TEAM_A / TEAM_B are per-match names
    of the two kit clusters: no home / away meaning, nothing carried across matches."""
    TEAM_A = "TEAM_A"
    TEAM_B = "TEAM_B"
    GOALKEEPER = "GOALKEEPER"  # v1: no team side (needs pitch context)
    REFEREE = "REFEREE"
    UNKNOWN = "UNKNOWN"


class TeamReason(str, Enum):
    """Why a segment is UNKNOWN (spana.teams 1.0.0). A decided label carries no reason."""
    TOO_FEW_CLEAN_FRAMES = "TOO_FEW_CLEAN_FRAMES"    # fewer than team.min_segment_frames / min_segment_s of evidence
    LOW_VOTE_SHARE = "LOW_VOTE_SHARE"                # inside a team cluster, but the per-frame votes are not decisive
    COLOUR_OUTLIER = "COLOUR_OUTLIER"                # (1.1.0) far from both teams, roles off: never the nearest team
    COLOUR_OUTLIER_NO_ROLE = "COLOUR_OUTLIER_NO_ROLE"  # far from both teams, and no detector role-hint majority
    ROLE_HINT_CONFLICT = "ROLE_HINT_CONFLICT"        # far from both teams, both role hints present
    MATCH_MODEL_ABSTAIN = "MATCH_MODEL_ABSTAIN"      # the match's two-team colour model abstained
    OUTSIDE_SEGMENT = "OUTSIDE_SEGMENT"              # (label_at only) frame not in any segment: track not confirmed,
                                                     #   inside an uncertainty episode, or not a track of the document


class BallState(str, Enum):
    """Per-frame answer of the ball layer (spana.ball 1.1.0)."""
    OBSERVED = "OBSERVED"    # one confirmed ball hypothesis is observed on this frame
    PREDICTED = "PREDICTED"  # the selected ball is unobserved for <= ball.predict_max_s: extrapolated position
    LOST = "LOST"            # no ball answer: none seen, gap too long, or camera continuity broken
    AMBIGUOUS = "AMBIGUOUS"  # more than one confirmed hypothesis could be the ball: no answer rather than a guess


class BallReason(str, Enum):
    SINGLE_CONFIRMED = "SINGLE_CONFIRMED"        # OBSERVED: the only confirmed hypothesis observed
    CONTINUING = "CONTINUING"                    # OBSERVED: the selected ball, observed again (rivals -> contested)
    SHORT_GAP = "SHORT_GAP"                      # PREDICTED: selected ball unobserved <= predict_max_s
    NO_CONFIRMED_BALL = "NO_CONFIRMED_BALL"      # LOST: no confirmed hypothesis observed, nothing to extrapolate
    GAP_TOO_LONG = "GAP_TOO_LONG"                # LOST: selected ball unobserved > predict_max_s
    PREDICTED_OUTSIDE = "PREDICTED_OUTSIDE"      # LOST: the extrapolated position left the picture
    CAMERA_DISCONTINUITY = "CAMERA_DISCONTINUITY"  # LOST: CUT / UNKNOWN camera: every hypothesis was dropped
    RIVAL_CANDIDATES = "RIVAL_CANDIDATES"        # AMBIGUOUS: >= 2 confirmed observed, none the selected ball
    SELECTED_HIDDEN_RIVAL = "SELECTED_HIDDEN_RIVAL"  # AMBIGUOUS: selected ball briefly hidden, another one observed
    SPARSE_OBSERVATIONS = "SPARSE_OBSERVATIONS"  # LOST (spana.ball 1.1.0): the only candidate ball is observed on
    #   fewer than ball.min_seen_share of its recent frames (a flickering false object, or a ball barely seen)


class BallEndReason(str, Enum):
    UNCONFIRMED_EXPIRED = "UNCONFIRMED_EXPIRED"  # unconfirmed and unobserved for confirm_window_s
    LOST_EXPIRED = "LOST_EXPIRED"                # confirmed and unobserved for lost_max_s
    CAMERA_DISCONTINUITY = "CAMERA_DISCONTINUITY"


@dataclass(frozen=True)
class Detection:
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2, source pixels
    score: float
    class_id: int
    multi_body: bool = False  # box may contain more than one person (see postprocess.flag_multi_body)

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
    # Per-axis std (source px) of the camera displacement at a point, measured from how far ALL tracked background
    # features disagree with the fitted transform. 0 = no extra uncertainty (identity, or a perfectly agreeing fit).
    motion_std_px: float = 0.0

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
    identity: IdentityState = IdentityState.CERTAIN
    identity_rivals: Tuple[str, ...] = ()   # tracks this one could not be told apart from (while uncertain)
    uncertain_episodes: int = 0             # times the track has entered IDENTITY_UNCERTAIN
    identity_doubt: bool = False            # uncertain because its own appearance disagreed on an exclusive
                                            #   continuation (no rival track to name); tracks 2.2.0
    resolved_by: Optional[str] = None       # rule that cleared the latest episode: "APPEARANCE" (colour) or "REID"
                                            #   (every rival cleared by the calibrated re-ID margin); None if none;
                                            #   tracks 2.3.0


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


@dataclass(frozen=True)
class BallFrame:
    """The ball layer's answer on one frame (spana.ball 1.1.0)."""
    frame_index: int
    time_s: float
    camera_status: CameraStatus
    state: BallState
    reason: BallReason
    hypothesis: Optional[str]                    # selected hypothesis id (OBSERVED / PREDICTED), else None
    center: Optional[Tuple[float, float]]        # source px (OBSERVED / PREDICTED), else None
    diameter: Optional[float]                    # source px (OBSERVED / PREDICTED), else None
    confidence: Optional[float]                  # uncalibrated: detection score, decayed while PREDICTED
    contested: bool                              # OBSERVED while another confirmed hypothesis was also observed
    candidates: int                              # ball candidates the view gave on this frame
    confirmed_live: int                          # confirmed hypotheses alive after this frame
    decisions: List[dict] = field(default_factory=list)
    # spana.ball 1.3.0: every hypothesis observed on this frame (confirmed or not, selected or not), for the offline
    # ball path: (hypothesis id, centre x, centre y, detection diameter, detection score), sorted by id
    sightings: Tuple[Tuple[str, float, float, float, float], ...] = ()
    # the ball pass's inter-frame camera transform (previous frame -> this frame); identity unless the camera is OK
    camera_affine: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


@dataclass(frozen=True)
class BallHypothesisSummary:
    hypothesis: str
    first_frame: int
    last_observed_frame: int
    observed_frames: int
    confirmed_frame: Optional[int]               # None = never confirmed
    end_frame: Optional[int]                     # None = alive at the end of the run
    end_reason: Optional[str]
