"""Configuration and licensing policy for the detection dataset (config/roboflow.yaml).

Same fail-closed loader as config/default.yaml: no code defaults; missing, unknown or out-of-range keys are rejected.

Policy (enforced by `authorize`, used by every tool that moves frames anywhere):
  * EVAL is never uploaded or processed by these tools (it needs Nathan's explicit approval, outside this code).
  * A match must be listed in `licensed_matches`; each match belongs to exactly one split (no per-image split).
  * Sending frames to a third party (Roboflow upload, hosted inference) additionally needs
    `third_party_processing_confirmed: true` on that match.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import yaml

from ..config import ConfigError, _pos, _pos_int, _unit, typed_from_dict

DATASET_CONFIG_SCHEMA_VERSION = "1.0.0"
SPLITS = ("train", "dev", "eval")
UPLOADABLE_SPLITS = ("train", "dev")


class PolicyError(PermissionError):
    """A request the licensing / split policy forbids. Always fatal; never caught to continue."""


@dataclass(frozen=True)
class RoboflowSection:
    workspace: str
    project: str
    api_key_env: str          # name of the environment variable holding the private API key (never the key)
    train_split_name: str     # Roboflow split that receives SPANA TRAIN matches
    dev_split_name: str       # Roboflow split that receives SPANA DEV matches (EVAL has none on purpose)
    upload_retries: int

    def validate(self) -> None:
        for n in ("workspace", "project", "api_key_env", "train_split_name", "dev_split_name"):
            if not getattr(self, n).strip():
                raise ConfigError(f"roboflow.{n} must be non-empty")
        if self.train_split_name == self.dev_split_name:
            raise ConfigError("roboflow: train and dev must map to different Roboflow splits")
        _pos_int("roboflow.upload_retries", self.upload_retries, allow_zero=True)


@dataclass(frozen=True)
class SamplingSection:
    interval_s: float            # one regular frame every interval_s seconds
    max_frames_per_video: int    # hard cap per video; excess is thinned evenly (deterministic)
    analysis_step_s: float       # motion / cut signals are computed on frames this far apart
    analysis_width_px: int       # compute-only downscale for the signals
    motion_threshold: float      # mean absolute grey-level change (0..1) between analysis frames = heavy motion
    cut_hist_distance: float     # HSV-histogram Bhattacharyya distance above which a frame starts a new shot
    event_min_gap_s: float       # extra (motion / cut) frames are at least this far apart
    jpeg_quality: int

    def validate(self) -> None:
        _pos("sampling.interval_s", self.interval_s)
        _pos_int("sampling.max_frames_per_video", self.max_frames_per_video)
        _pos("sampling.analysis_step_s", self.analysis_step_s)
        _pos_int("sampling.analysis_width_px", self.analysis_width_px)
        _unit("sampling.motion_threshold", self.motion_threshold)
        _unit("sampling.cut_hist_distance", self.cut_hist_distance)
        _pos("sampling.event_min_gap_s", self.event_min_gap_s)
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError("sampling.jpeg_quality must be in [1, 100]")
        if self.analysis_step_s > self.interval_s:
            raise ConfigError("sampling.analysis_step_s must be <= sampling.interval_s")


@dataclass(frozen=True)
class BaselineSection:
    model_id: str                # pretrained detector, TEST USE ONLY (baseline measurement, never shipped)
    api_url: str                 # inference server; a host in local_api_hosts keeps frames on this machine
    confidence: float
    iou: float
    timeout_s: float
    class_map: Tuple[str, ...]   # "model_label:spana_class"; labels not listed are dropped and counted
    render_samples: int          # annotated frames written for visual review
    histogram_bins: int          # confidence histogram bins over [0, 1]

    def validate(self) -> None:
        if not self.model_id.strip():
            raise ConfigError("baseline.model_id must be non-empty")
        u = urlparse(self.api_url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ConfigError("baseline.api_url must be an http(s) URL")
        _unit("baseline.confidence", self.confidence)
        _unit("baseline.iou", self.iou)
        _pos("baseline.timeout_s", self.timeout_s)
        _pos_int("baseline.render_samples", self.render_samples, allow_zero=True)
        _pos_int("baseline.histogram_bins", self.histogram_bins)


@dataclass(frozen=True)
class LicensedMatch:
    match_id: str
    split: str                              # the one split this match belongs to
    permission: str                         # where the written permission is recorded
    third_party_processing_confirmed: bool  # the licence covers processing by a third party (e.g. Roboflow)


@dataclass(frozen=True)
class PathsSection:
    manifest_dir: str    # upload manifests (records; committed)
    frames_dir: str      # extracted frames and dry-run manifests (run outputs; not committed)
    baseline_dir: str    # baseline detections, summaries and annotated samples (run outputs; not committed)


@dataclass(frozen=True)
class DatasetConfig:
    schema_version: str
    roboflow: RoboflowSection
    classes: Tuple[str, ...]
    split: str                   # the split the inference / baseline tools operate on; only "dev" is accepted
    sampling: SamplingSection
    baseline: BaselineSection
    local_api_hosts: Tuple[str, ...]
    licensed_matches: Tuple[LicensedMatch, ...]
    paths: PathsSection

    def validate(self) -> None:
        if self.schema_version != DATASET_CONFIG_SCHEMA_VERSION:
            raise ConfigError(f"dataset config schema_version {self.schema_version!r} != "
                              f"{DATASET_CONFIG_SCHEMA_VERSION!r}")
        for f in fields(self):
            v = getattr(self, f.name)
            if hasattr(v, "validate"):
                v.validate()
        if not self.classes or len(set(self.classes)) != len(self.classes) or any(not c.strip() for c in self.classes):
            raise ConfigError("classes must be a non-empty list of distinct names")
        if self.split != "dev":
            raise ConfigError(f"split must be 'dev' (got {self.split!r}): baselines are measured on DEV only; "
                              "EVAL needs Nathan's explicit approval")
        seen_labels = set()
        for entry in self.baseline.class_map:
            label, sep, target = entry.partition(":")
            if not sep or not label or target not in self.classes:
                raise ConfigError(f"baseline.class_map entry {entry!r}: expected 'model_label:<one of classes>'")
            if label in seen_labels:
                raise ConfigError(f"baseline.class_map: label {label!r} mapped twice")
            seen_labels.add(label)
        ids = [m.match_id for m in self.licensed_matches]
        if len(set(ids)) != len(ids):
            raise ConfigError("licensed_matches: a match_id is listed twice (a match belongs to one split)")
        for m in self.licensed_matches:
            if not m.match_id.strip() or not m.permission.strip():
                raise ConfigError("licensed_matches: match_id and permission must be non-empty")
            if m.split not in SPLITS:
                raise ConfigError(f"licensed_matches[{m.match_id}].split must be one of {SPLITS}")

    def class_map(self) -> dict:
        return dict(e.split(":", 1) for e in self.baseline.class_map)

    def roboflow_split(self, split: str) -> str:
        if split == "train":
            return self.roboflow.train_split_name
        if split == "dev":
            return self.roboflow.dev_split_name
        raise PolicyError(f"split {split!r} has no Roboflow split: EVAL is never uploaded")

    def is_local_api(self) -> bool:
        return urlparse(self.baseline.api_url).hostname in self.local_api_hosts

    def to_dict(self) -> dict:
        return asdict(self)

    def section_sha256(self, name: str) -> str:
        blob = json.dumps(asdict(getattr(self, name)), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def dataset_config_from_dict(data) -> DatasetConfig:
    cfg = typed_from_dict(DatasetConfig, data)
    cfg.validate()
    return cfg


def load_dataset_config(path: str | Path) -> DatasetConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return dataset_config_from_dict(yaml.safe_load(fh))


def authorize(cfg: DatasetConfig, match_id: str, split: str, *, third_party: bool) -> LicensedMatch:
    """Fail closed unless `match_id` may be used in `split`. `third_party`: frames leave this machine."""
    if split == "eval":
        raise PolicyError("refused: split=eval. EVAL frames are never uploaded or processed by these tools "
                          "(EVAL needs Nathan's explicit approval)")
    if split not in UPLOADABLE_SPLITS:
        raise PolicyError(f"refused: unknown split {split!r}; expected one of {UPLOADABLE_SPLITS}")
    entry = next((m for m in cfg.licensed_matches if m.match_id == match_id), None)
    if entry is None:
        raise PolicyError(f"refused: match_id {match_id!r} is not in licensed_matches of the dataset config. "
                          "Add it only with a written permission record")
    if entry.split != split:
        raise PolicyError(f"refused: match_id {match_id!r} is registered for split {entry.split!r}, not {split!r}. "
                          "A match never contributes frames to more than one split")
    if third_party and not entry.third_party_processing_confirmed:
        raise PolicyError(f"refused: the licence for {match_id!r} is not confirmed to cover third-party processing "
                          "(licensed_matches[].third_party_processing_confirmed is false)")
    return entry
