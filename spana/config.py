"""Typed, validated configuration.

Every tunable value lives in a YAML file. The loader fails closed: a missing key, an unknown key,
a wrong type or an out-of-range value raises ConfigError. No value has a code default.

Units convention:
  * times are seconds (converted to frames with `frames_for(seconds, fps)`);
  * lengths are fractions of a body height (``*_h``) or of the frame height/diagonal (``*_frac``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Type, TypeVar, get_args, get_origin, get_type_hints

import yaml

CONFIG_SCHEMA_VERSION = "2.13.0"   # 2.1.0: + target section; 2.2.0: + target relink keys; 2.3.0: + tracking.exit_hold_s;
#   2.4.0: + tracking appearance doubt link (+ self-consistent resolution), extent-change link, low-score recovery,
#   target.tap_vouches_doubt / resume_after_doubt_resolved / review_min_gap_s; 2.5.0: + reid section, target re-entry
#   offer keys; 2.6.0: + target.reentry_position_gate / reentry_mutual; 2.7.0: + target.reentry_position_* score;
#   2.8.0: + reid.set_* (crop sets), target.reentry_signature / reentry_wait_samples / reentry_withdraw;
#   2.9.0: + target.reentry_exit_edge / reentry_exit_edge_max_frac; 2.10.0: + target.offer_hold_s;
#   2.11.0: + tracking.rival_eligible_only; 2.12.0: + reid_resolve section (re-ID evidence for IDENTITY_UNCERTAIN),
#   target.resume_after_reid; 2.13.0: + tracking.ambiguous_spawn_uncertain, crossing_uncertain, crossing_min_ioa


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DetectionConfig:
    score_low: float          # detections below this are discarded
    score_high: float         # detections at/above this may start tracks and match in stage 1
    nms_iou: float            # class-wise NMS IoU
    min_box_height_frac: float  # min box height as a fraction of frame height
    tiling_enabled: bool        # add overlapping high-resolution tiles to the full-frame pass
    tile_size_frac: float       # tile side as a fraction of frame height
    tile_overlap_frac: float    # overlap between neighbouring tiles, fraction of the tile side
    tile_edge_margin_frac: float  # tile boxes this close to an internal tile edge are truncated -> dropped
    union_member_ioa: float     # union-box rule: a member lies inside the union box by this share of its area
    union_member_max_iou: float # ... and two members overlap each other by at most this IoU (distinct people)
    multibody_member_ioa: float   # multi-body flag: a confident detection lies inside the box by this share of its area
    multibody_member_score: float # ... and scores at least this
    multibody_min_extent: float   # ... and the box is at least this many times wider or taller than it

    def validate(self) -> None:
        _pos("detection.tile_size_frac", self.tile_size_frac)
        _unit("detection.tile_overlap_frac", self.tile_overlap_frac)
        _unit("detection.tile_edge_margin_frac", self.tile_edge_margin_frac)
        _unit("detection.union_member_ioa", self.union_member_ioa)
        _unit("detection.union_member_max_iou", self.union_member_max_iou)
        _unit("detection.multibody_member_ioa", self.multibody_member_ioa)
        _unit("detection.multibody_member_score", self.multibody_member_score)
        if not self.multibody_min_extent > 1.0:
            raise ConfigError("detection.multibody_min_extent must be > 1")
        if self.tile_overlap_frac >= 1.0 or self.tile_edge_margin_frac >= self.tile_overlap_frac:
            raise ConfigError("detection: need tile_edge_margin_frac < tile_overlap_frac < 1")
        _unit("detection.score_low", self.score_low)
        _unit("detection.score_high", self.score_high)
        _unit("detection.nms_iou", self.nms_iou)
        _unit("detection.min_box_height_frac", self.min_box_height_frac)
        if self.score_low > self.score_high:
            raise ConfigError("detection.score_low must be <= detection.score_high")


@dataclass(frozen=True)
class CameraConfig:
    analysis_width_px: int        # frames are downscaled to this width for motion/cut analysis (compute only)
    max_features: int
    feature_quality: float        # goodFeaturesToTrack qualityLevel (relative corner strength)
    feature_min_distance_frac: float  # min corner spacing as a fraction of the analysis-frame width
    min_inliers: int
    ransac_reproj_frac: float     # RANSAC threshold as a fraction of the analysis-frame diagonal
    cut_hist_distance: float      # Bhattacharyya distance on HSV histograms above which a frame is a cut
    max_log_scale_rate_per_s: float  # plausible zoom speed, |log(scale)| per second
    scale_noise_log: float           # per-estimate |log(scale)| noise allowance on top of the rate
    disagreement_noise_scale: float  # camera motion std = this x per-axis RMS disagreement of tracked features

    def validate(self) -> None:
        _pos_int("camera.analysis_width_px", self.analysis_width_px)
        _pos_int("camera.max_features", self.max_features)
        _unit("camera.feature_quality", self.feature_quality)
        _pos("camera.feature_min_distance_frac", self.feature_min_distance_frac)
        _pos_int("camera.min_inliers", self.min_inliers)
        _pos("camera.ransac_reproj_frac", self.ransac_reproj_frac)
        _unit("camera.cut_hist_distance", self.cut_hist_distance)
        _pos("camera.max_log_scale_rate_per_s", self.max_log_scale_rate_per_s)
        _pos("camera.scale_noise_log", self.scale_noise_log)
        if not self.disagreement_noise_scale >= 0:
            raise ConfigError("camera.disagreement_noise_scale must be >= 0")


@dataclass(frozen=True)
class KalmanConfig:
    pos_std_h: float       # measurement / process std of position, in body heights
    vel_std_h_per_s: float  # process std of velocity, body heights per second
    init_vel_std_h_per_s: float

    def validate(self) -> None:
        _pos("kalman.pos_std_h", self.pos_std_h)
        _pos("kalman.vel_std_h_per_s", self.vel_std_h_per_s)
        _pos("kalman.init_vel_std_h_per_s", self.init_vel_std_h_per_s)


@dataclass(frozen=True)
class TrackingConfig:
    low_score_min_iou: float       # S2 (low-score detections) minimum IoU with the predicted box
    gate_base_h: float             # centre gate radius at zero elapsed time, body heights
    max_speed_h_per_s: float       # plausible max player speed (gate growth), body heights / s
    gate_max_h: float              # absolute cap on the gate radius; also the distance-cost scale
    gate_chi2: float               # Kalman gate: max squared Mahalanobis distance of the centre (2 dof)
    max_height_ratio: float        # det/pred height ratio must lie in [1/r, r]
    iou_weight: float              # cost = iou_weight*(1-IoU) + (1-iou_weight)*dist_h/gate_max_h
    contest_margin: float          # a rival within this cost margin marks the match CONTESTED
    confirm_s: float               # observed span required to confirm a new (uncertain) track
    confirm_nested_ioa: float      # ... and its box must not lie inside another observed box by this share
    unconfirmed_max_s: float       # an unconfirmed track older than this is terminated
    rematch_max_s: float           # LOST track: longest gap bridged by motion (exclusive pairs only)
    occlusion_min_ioa: float       # unobserved track is OCCLUDED when this share of its box is covered by an
                                   #   observed track's box
    occlusion_max_s: float         # longest OCCLUDED gap that may be recovered (appearance-distinct only)
    max_accel_h_per_s2: float      # long-gap recovery: plausible change of velocity while hidden, body heights/s^2
    predict_max_s: float           # after this unobserved time position is HELD (velocity frozen)
    lost_max_s: float              # unobserved track TERMINATED after this time
    exit_hold_s: float             # a confirmed track whose prediction left the image is held (LOST, velocity frozen,
                                   #   camera-compensated) this long, recoverable by the long-gap rules; then LEFT_FRAME
                                   #   (0 = end on leaving, as before 2026-11-18)
    duplicate_iou: float           # unmatched det overlapping an observed box above this IoU does not spawn
    duplicate_ioa: float           # ... or lying inside an observed box by this share of its own area
    confidence_window_s: float     # window of the track confidence (see tracker docstring)
    history_s: float               # per-track history kept in memory (API), seconds
    identity_resolve_s: float      # IDENTITY_UNCERTAIN ends after clean appearance evidence favours the track this long
    appearance_doubt_link: bool    # continue a track through an appearance veto on an exclusive motion link,
                                   #   flagged IDENTITY_UNCERTAIN (identity_doubt) instead of ending it
    appearance_doubt_min_iou: float  # ... only when the detection overlaps the predicted box by at least this IoU
    doubt_resolve_self_consistent: bool  # a doubt (no rival) may also clear on crops consistent with each other
                                   #   while the track stays exclusive, for identity_resolve_s; adopts that appearance
    extent_change_link: bool       # S3c: continue a recent confirmed track through a box that changed extent
                                   #   (part of the body hidden / re-emerging): nested, one edge kept, exclusive
    extent_edge_max_h: float       # ... the kept (top or bottom) edge moves at most this many body heights
    extent_max_ratio: float        # ... detection / predicted height within [1/r, r] (wider than max_height_ratio)
    low_score_recovery: bool       # S4: a confirmed track unobserved <= rematch_max_s may recover on a low-score
                                   #   detection overlapping its prediction by >= low_score_min_iou
    rival_eligible_only: bool      # S3c / S4: only a track that could take a detection on this frame (observed on the
                                   #   previous frame, or still recoverable) counts as another claimant / RIVAL_TRACK
    ambiguous_spawn_uncertain: bool  # a track spawned naming >= 2 possible predecessors starts IDENTITY_UNCERTAIN
                                   #   (its predecessors are its rivals) instead of CERTAIN
    crossing_uncertain: bool       # a confirmed track whose box holds a similar-looking confirmed track's body
                                   #   (crossing_min_ioa) marks both IDENTITY_UNCERTAIN, contested or not
    crossing_min_ioa: float        # ... share of the other track's box (matched detection, else prediction) inside it
    crossing_hidden_rivals: bool   # ... the other track may also be unobserved (its prediction; live or recoverable),
                                   #   not only observed on its own detection on this frame

    def validate(self) -> None:
        for n in ("low_score_min_iou", "iou_weight", "contest_margin", "duplicate_iou", "duplicate_ioa",
                  "occlusion_min_ioa", "confirm_nested_ioa", "appearance_doubt_min_iou", "crossing_min_ioa"):
            _unit(f"tracking.{n}", getattr(self, n))
        for n in ("gate_base_h", "max_speed_h_per_s", "gate_max_h", "gate_chi2", "confirm_s", "unconfirmed_max_s", "rematch_max_s",
                  "occlusion_max_s", "max_accel_h_per_s2", "predict_max_s", "lost_max_s", "confidence_window_s", "history_s",
                  "identity_resolve_s"):
            _pos(f"tracking.{n}", getattr(self, n))
        if self.max_height_ratio < 1.0:
            raise ConfigError("tracking.max_height_ratio must be >= 1")
        _pos("tracking.extent_edge_max_h", self.extent_edge_max_h)
        if self.extent_max_ratio < self.max_height_ratio:
            raise ConfigError("tracking.extent_max_ratio must be >= tracking.max_height_ratio")
        if self.gate_base_h > self.gate_max_h:
            raise ConfigError("tracking.gate_base_h must be <= tracking.gate_max_h")
        if self.confirm_nested_ioa > self.duplicate_ioa:
            raise ConfigError("tracking.confirm_nested_ioa must be <= tracking.duplicate_ioa")
        if self.confirm_s > self.unconfirmed_max_s:
            raise ConfigError("tracking.confirm_s must be <= tracking.unconfirmed_max_s")
        if not self.rematch_max_s <= self.occlusion_max_s <= self.lost_max_s:
            raise ConfigError("tracking: need rematch_max_s <= occlusion_max_s <= lost_max_s")
        if self.predict_max_s > self.lost_max_s:
            raise ConfigError("tracking.predict_max_s must be <= tracking.lost_max_s")
        if not 0.0 <= self.exit_hold_s <= self.lost_max_s:
            raise ConfigError("tracking: need 0 <= exit_hold_s <= lost_max_s")


@dataclass(frozen=True)
class AppearanceConfig:
    enabled: bool
    min_crop_height_px: int        # absolute floor: a colour histogram needs pixels (documented exception)
    torso_top_frac: float          # torso band, fractions of the box height from the top
    torso_bottom_frac: float
    torso_width_frac: float        # central share of the box width
    hue_bins: int
    sat_bins: int
    tau_s: float                   # time constant of the track's appearance running average
    clean_max_iou: float           # appearance is updated only when the box overlaps no other track above this
    min_similarity: float          # recovery after occlusion needs at least this similarity to the track
    min_margin: float              # ... and must beat every other track's similarity by this margin

    def validate(self) -> None:
        _pos_int("appearance.min_crop_height_px", self.min_crop_height_px)
        for n in ("torso_top_frac", "torso_bottom_frac", "torso_width_frac", "clean_max_iou", "min_similarity",
                  "min_margin"):
            _unit(f"appearance.{n}", getattr(self, n))
        if self.torso_top_frac >= self.torso_bottom_frac:
            raise ConfigError("appearance.torso_top_frac must be < torso_bottom_frac")
        _pos_int("appearance.hue_bins", self.hue_bins)
        _pos_int("appearance.sat_bins", self.sat_bins)
        _pos("appearance.tau_s", self.tau_s)


@dataclass(frozen=True)
class TargetConfig:
    """Target identity (following the user's selected player)."""
    resume_after_uncertain: bool   # after the followed track was IDENTITY_UNCERTAIN, answer again once it is
                                   #   CERTAIN (true) or abstain for the rest of the track (false, fail closed)
    relink_max_gap_s: float        # a relink candidate must be born within this time of the lost track's last
                                   #   observation (spana.identity.follower)
    relink_auto_confirm: bool      # promote a CANDIDATE to TARGET without the user's confirmation (measurement only)
    tap_vouches_doubt: bool        # a user's tap on a track that is uncertain ONLY by an appearance doubt (no rival)
                                   #   answers TARGET (USER_VOUCHED) for the rest of that doubt episode
    resume_after_doubt_resolved: bool  # an uncertainty episode that was only ever a no-rival doubt does not taint
                                   #   the followed track: it is answered again once the tracker resolves the doubt
    review_min_gap_s: float        # after-match review list: only gaps at least this long (0 = every gap)
    reentry_offer: bool            # offer a returning track by re-ID signature after a loss (needs track embeddings)
    reentry_max_gap_s: float       # ... born at most this long after the lost track's last observation
    reentry_min_similarity: float  # ... cosine similarity of the lost track's last and the candidate's first signature
    reentry_min_margin: float      # ... and above every other candidate's similarity by at least this
    reentry_position_gate: bool    # ... and reachable from where the lost player was (camera-carried, speed bound)
    reentry_mutual: bool           # ... and the lost track is also the candidate's best match among lost tracks
    reentry_position_weight: float # combined = appearance + weight * position fit (0 = appearance only)
    reentry_position_base_h: float # position fit: reach = base_h + speed_h_per_s * gap (body heights) ...
    reentry_position_speed_h_per_s: float
    reentry_position_time_scale_s: float  # ... times exp(-gap / time_scale_s)
    reentry_position_max_gap_s: float     # position carries no information beyond this gap
    reentry_position_max_pan_frac: float  # ... nor when the camera moved the scene more than this share of the width
    reentry_signature: str         # "mean": one averaged signature per track end; "set": quality-weighted best-pair
                                   #   matching of the crop sets (spana.track_embeddings 1.1.0)
    reentry_wait_samples: int      # set only: a candidate is judged once it has this many crops up to the current
                                   #   frame, from those crops only (0 = all its crops, also later ones)
    reentry_withdraw: bool         # withdraw a re-ID offer when the rule, re-run on the evidence so far, no longer
                                   #   offers that track
    reentry_exit_edge: bool        # a player who left the picture is only offered tracks born at the side they left by
    reentry_exit_edge_max_frac: float  # ... within this share of the frame width / height of that side
    offer_hold_s: float            # keep an offer whose track is briefly hidden (OCCLUDED / LOST, still CERTAIN) this
                                   #   long after its last usable frame (0 = drop at once, the behaviour before)
    resume_after_reid: bool        # answer again after an IDENTITY_UNCERTAIN episode that the calibrated re-ID margin
                                   #   cleared for every rival (track resolved_by == REID); others stay as above

    def validate(self) -> None:
        _pos("target.relink_max_gap_s", self.relink_max_gap_s)
        if not (isinstance(self.review_min_gap_s, (int, float)) and self.review_min_gap_s >= 0):
            raise ConfigError("target.review_min_gap_s must be >= 0")
        _pos("target.reentry_max_gap_s", self.reentry_max_gap_s)
        if not -1.0 <= self.reentry_min_similarity <= 1.0:
            raise ConfigError("target.reentry_min_similarity must be a cosine in [-1, 1]")
        if not 0.0 <= self.reentry_min_margin <= 2.0:
            raise ConfigError("target.reentry_min_margin must be in [0, 2]")
        if self.reentry_position_weight < 0:
            raise ConfigError("target.reentry_position_weight must be >= 0")
        for n in ("reentry_position_base_h", "reentry_position_speed_h_per_s", "reentry_position_time_scale_s",
                  "reentry_position_max_gap_s"):
            _pos(f"target.{n}", getattr(self, n))
        _unit("target.reentry_position_max_pan_frac", self.reentry_position_max_pan_frac)
        if self.reentry_signature not in ("mean", "set"):
            raise ConfigError("target.reentry_signature must be mean or set")
        _pos_int("target.reentry_wait_samples", self.reentry_wait_samples, allow_zero=True)
        _unit("target.reentry_exit_edge_max_frac", self.reentry_exit_edge_max_frac)
        if not (isinstance(self.offer_hold_s, (int, float)) and self.offer_hold_s >= 0):
            raise ConfigError("target.offer_hold_s must be >= 0")


@dataclass(frozen=True)
class ReidConfig:
    """Re-ID signatures per track (spana.identity.reid). The model and its crop convention come from the model
    record; these choose which observations are embedded."""
    samples: int                   # observations embedded at each end of a track (first / last signature)
    sample_min_gap_s: float        # ... at least this far apart in time
    min_crop_height_px: int        # absolute floor: smaller boxes carry too few pixels (training crops >= 40 px)
    clean_max_iou: float           # ... and overlap no other observed box by more than this
    set_samples: int               # crop sets (1.1.0): crops kept at each end of a track ...
    set_min_gap_s: float           # ... at least this far apart
    set_min_height_px: int         # ... at least this tall (looser than min_crop_height_px; quality weighted)
    set_max_iou: float             # ... overlapping other observed boxes by at most this (looser than clean_max_iou)

    def validate(self) -> None:
        _pos_int("reid.samples", self.samples)
        _pos_int("reid.set_samples", self.set_samples)
        _pos("reid.set_min_gap_s", self.set_min_gap_s)
        _pos_int("reid.set_min_height_px", self.set_min_height_px)
        _unit("reid.set_max_iou", self.set_max_iou)
        if self.set_min_height_px > self.min_crop_height_px or self.set_max_iou < self.clean_max_iou:
            raise ConfigError("reid.set_* must be at least as loose as min_crop_height_px / clean_max_iou")
        _pos("reid.sample_min_gap_s", self.sample_min_gap_s)
        _pos_int("reid.min_crop_height_px", self.min_crop_height_px)
        _unit("reid.clean_max_iou", self.clean_max_iou)


@dataclass(frozen=True)
class ReidResolveConfig:
    """Re-ID evidence for clearing IDENTITY_UNCERTAIN (spana.tracking.tracker; calibration records/reid_calibration_dev_*,
    records/reid_margin_floor_dev_2026-11-19). The model comes from a pinned model record (spana.identity.reid)."""
    enabled: bool                  # off by default: the tracker then behaves exactly as without this section
    model_record: str              # models/<name>.json (ONNX path + sha256 + preprocessing), relative to the repo
    model_sha256: str              # the margin below was calibrated for exactly this ONNX file
    margin: float                  # sim(query, own reference) - sim(query, rival reference) needed to clear a rival;
                                   #   the opposite margin is evidence against
    ref_update_s: float            # a CERTAIN track's reference embedding is refreshed at most this often

    def validate(self) -> None:
        _pos("reid_resolve.margin", self.margin)
        _pos("reid_resolve.ref_update_s", self.ref_update_s)
        if not (len(self.model_sha256) == 64 and all(c in "0123456789abcdef" for c in self.model_sha256)):
            raise ConfigError("reid_resolve.model_sha256 must be a sha256 hex digest (the calibrated model)")


@dataclass(frozen=True)
class OutputConfig:
    box_decimals: int
    score_decimals: int

    def validate(self) -> None:
        _pos_int("output.box_decimals", self.box_decimals, allow_zero=True)
        _pos_int("output.score_decimals", self.score_decimals, allow_zero=True)


@dataclass(frozen=True)
class SpanaConfig:
    schema_version: str
    detection: DetectionConfig
    camera: CameraConfig
    kalman: KalmanConfig
    tracking: TrackingConfig
    appearance: AppearanceConfig
    target: TargetConfig
    reid: ReidConfig
    reid_resolve: ReidResolveConfig
    output: OutputConfig

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigError(f"config schema_version {self.schema_version!r} != {CONFIG_SCHEMA_VERSION!r}")
        for f in fields(self):
            v = getattr(self, f.name)
            if hasattr(v, "validate"):
                v.validate()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


T = TypeVar("T")


def _build(cls: Type[T], data: Any, where: str) -> T:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    missing = [n for n in names if n not in data]
    unknown = [k for k in data if k not in names]
    if missing:
        raise ConfigError(f"{where}: missing keys {missing}")
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    kwargs = {}
    for n in names:
        t, v, path = hints[n], data[n], f"{where}.{n}" if where else n
        kwargs[n] = _value(t, v, path)
    return cls(**kwargs)


def _value(t: Any, v: Any, path: str) -> Any:
    """One typed value. Supports the scalar types, nested dataclasses and homogeneous tuples (``Tuple[X, ...]``,
    written as YAML lists)."""
    if get_origin(t) is tuple:
        args = get_args(t)
        if len(args) != 2 or args[1] is not Ellipsis:  # pragma: no cover - guarded by the dataclass definitions
            raise ConfigError(f"{path}: only Tuple[X, ...] is supported")
        if not isinstance(v, list):
            raise ConfigError(f"{path}: expected a list")
        return tuple(_value(args[0], item, f"{path}[{i}]") for i, item in enumerate(v))
    if dataclasses.is_dataclass(t):
        return _build(t, v, path)
    if t is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ConfigError(f"{path}: expected a finite number")
        return float(v)
    elif t is int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(f"{path}: expected an integer")
        return v
    elif t is bool:
        if not isinstance(v, bool):
            raise ConfigError(f"{path}: expected true/false")
        return v
    elif t is str:
        if not isinstance(v, str):
            raise ConfigError(f"{path}: expected a string")
        return v
    else:  # pragma: no cover - guarded by the dataclass definitions above
        raise ConfigError(f"{path}: unsupported type {t}")


def config_from_dict(data: Mapping[str, Any]) -> SpanaConfig:
    cfg = _build(SpanaConfig, data, "")
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> SpanaConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh))


def frames_for(seconds: float, fps: float) -> int:
    """Seconds -> whole frames (at least 1). fps must be a validated positive value."""
    if not (fps > 0 and math.isfinite(fps)):
        raise ConfigError(f"invalid fps {fps!r}")
    return max(1, int(round(seconds * fps)))


def _unit(name: str, v: float) -> None:
    if not 0.0 <= v <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1], got {v}")


def _pos(name: str, v: float) -> None:
    if not v > 0.0:
        raise ConfigError(f"{name} must be > 0, got {v}")


def _pos_int(name: str, v: int, allow_zero: bool = False) -> None:
    if v < 0 or (v == 0 and not allow_zero):
        raise ConfigError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} integer, got {v}")


def typed_from_dict(cls: Type[T], data: Any, where: str = "") -> T:
    """Build any config dataclass with the same fail-closed rules as the main config (used by other config files)."""
    return _build(cls, data, where)
