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
from typing import Any, Mapping, Type, TypeVar, get_type_hints

import yaml

CONFIG_SCHEMA_VERSION = "2.0.0"


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
    duplicate_iou: float           # unmatched det overlapping an observed box above this IoU does not spawn
    duplicate_ioa: float           # ... or lying inside an observed box by this share of its own area
    confidence_window_s: float     # window of the track confidence (see tracker docstring)
    history_s: float               # per-track history kept in memory (API), seconds
    identity_resolve_s: float      # IDENTITY_UNCERTAIN ends after clean appearance evidence favours the track this long

    def validate(self) -> None:
        for n in ("low_score_min_iou", "iou_weight", "contest_margin", "duplicate_iou", "duplicate_ioa",
                  "occlusion_min_ioa", "confirm_nested_ioa"):
            _unit(f"tracking.{n}", getattr(self, n))
        for n in ("gate_base_h", "max_speed_h_per_s", "gate_max_h", "gate_chi2", "confirm_s", "unconfirmed_max_s", "rematch_max_s",
                  "occlusion_max_s", "max_accel_h_per_s2", "predict_max_s", "lost_max_s", "confidence_window_s", "history_s",
                  "identity_resolve_s"):
            _pos(f"tracking.{n}", getattr(self, n))
        if self.max_height_ratio < 1.0:
            raise ConfigError("tracking.max_height_ratio must be >= 1")
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
        if dataclasses.is_dataclass(t):
            kwargs[n] = _build(t, v, path)
        elif t is float:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ConfigError(f"{path}: expected a finite number")
            kwargs[n] = float(v)
        elif t is int:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ConfigError(f"{path}: expected an integer")
            kwargs[n] = v
        elif t is bool:
            if not isinstance(v, bool):
                raise ConfigError(f"{path}: expected true/false")
            kwargs[n] = v
        elif t is str:
            if not isinstance(v, str):
                raise ConfigError(f"{path}: expected a string")
            kwargs[n] = v
        else:  # pragma: no cover - guarded by the dataclass definitions above
            raise ConfigError(f"{path}: unsupported type {t}")
    return cls(**kwargs)


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
