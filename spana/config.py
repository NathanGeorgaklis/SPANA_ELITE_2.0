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

CONFIG_SCHEMA_VERSION = "1.0.0"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DetectionConfig:
    score_low: float          # detections below this are discarded
    score_high: float         # detections at/above this may start tracks and match in stage 1
    nms_iou: float            # class-wise NMS IoU
    min_box_height_frac: float  # min box height as a fraction of frame height

    def validate(self) -> None:
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
    max_scale_change: float       # |log(scale)| above which an estimate is rejected as implausible

    def validate(self) -> None:
        _pos_int("camera.analysis_width_px", self.analysis_width_px)
        _pos_int("camera.max_features", self.max_features)
        _unit("camera.feature_quality", self.feature_quality)
        _pos("camera.feature_min_distance_frac", self.feature_min_distance_frac)
        _pos_int("camera.min_inliers", self.min_inliers)
        _pos("camera.ransac_reproj_frac", self.ransac_reproj_frac)
        _unit("camera.cut_hist_distance", self.cut_hist_distance)
        _pos("camera.max_scale_change", self.max_scale_change)


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
    low_score_min_iou: float       # stage-2 (low-score detections) minimum IoU
    gate_base_h: float             # centre gate radius at zero elapsed time, body heights
    max_speed_h_per_s: float       # plausible max player speed (gate growth), body heights / s
    gate_max_h: float              # absolute cap on the gate radius
    max_height_ratio: float        # det/pred height ratio must lie in [1/r, r]
    iou_weight: float              # cost = iou_weight*(1-IoU) + (1-iou_weight)*dist/gate
    contest_margin: float          # a rival within this cost margin marks the match CONTESTED
    confirm_s: float               # observed span required to confirm a tentative track
    tentative_max_miss_s: float    # tentative track removed after this unobserved time
    predict_max_s: float           # after this unobserved time position is HELD (velocity frozen)
    lost_max_s: float              # LOST track removed after this unobserved time
    duplicate_iou: float           # unmatched high det overlapping a matched track above this does not spawn

    def validate(self) -> None:
        for n in ("low_score_min_iou", "iou_weight", "contest_margin", "duplicate_iou"):
            _unit(f"tracking.{n}", getattr(self, n))
        for n in ("gate_base_h", "max_speed_h_per_s", "gate_max_h", "confirm_s", "tentative_max_miss_s",
                  "predict_max_s", "lost_max_s"):
            _pos(f"tracking.{n}", getattr(self, n))
        if self.max_height_ratio < 1.0:
            raise ConfigError("tracking.max_height_ratio must be >= 1")
        if self.gate_base_h > self.gate_max_h:
            raise ConfigError("tracking.gate_base_h must be <= tracking.gate_max_h")
        if self.predict_max_s > self.lost_max_s:
            raise ConfigError("tracking.predict_max_s must be <= tracking.lost_max_s")


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
