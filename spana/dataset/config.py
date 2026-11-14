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
from typing import Optional, Tuple
from urllib.parse import urlparse

import yaml

from ..config import ConfigError, _pos, _pos_int, _unit, typed_from_dict

DATASET_CONFIG_SCHEMA_VERSION = "1.4.0"
SPLITS = ("train", "dev", "eval")
UPLOADABLE_SPLITS = ("train", "dev")
ROBOFLOW_SPLITS = ("train", "valid", "test")
RESIZE_FORMATS = ("Stretch to", "Fill (with center crop) in", "Fit within", "Fit (reflect edges) in",
                  "Fit (black edges) in", "Fit (white edges) in")
# Version augmentations SPANA allows: geometric only. Anything that changes pixel colours (hue, saturation, brightness,
# exposure, grayscale, contrast, camera gain, ...) is refused: kit colour is used later for team classification.
GEOMETRIC_AUGMENTATIONS = ("flip_horizontal",)


class PolicyError(PermissionError):
    """A request the licensing / split policy forbids. Always fatal; never caught to continue."""


@dataclass(frozen=True)
class RoboflowSection:
    workspace: str
    project: str
    api_key_env: str          # name of the environment variable holding the private API key (never the key)
    api_url: str              # REST API base (project checks, uploads)
    project_type: str         # expected project type; uploads refuse any other
    train_split_name: str     # Roboflow split that receives SPANA TRAIN matches
    dev_split_name: str       # Roboflow split that receives SPANA DEV matches (EVAL has none on purpose)
    upload_retries: int
    upload_timeout_s: float
    rate_limit_retries: int     # HTTP 429: retry this many times (Retry-After honoured, else rate_limit_wait_s x attempt)
    rate_limit_wait_s: float
    rate_limit_max_wait_s: float   # longest single wait; must cover the server's Retry-After window

    def validate(self) -> None:
        for n in ("workspace", "project", "api_key_env", "project_type", "train_split_name", "dev_split_name"):
            if not getattr(self, n).strip():
                raise ConfigError(f"roboflow.{n} must be non-empty")
        if self.train_split_name == self.dev_split_name:
            raise ConfigError("roboflow: train and dev must map to different Roboflow splits")
        _pos_int("roboflow.upload_retries", self.upload_retries, allow_zero=True)
        _pos("roboflow.upload_timeout_s", self.upload_timeout_s)
        _pos_int("roboflow.rate_limit_retries", self.rate_limit_retries, allow_zero=True)
        _pos("roboflow.rate_limit_wait_s", self.rate_limit_wait_s)
        _pos("roboflow.rate_limit_max_wait_s", self.rate_limit_max_wait_s)
        u = urlparse(self.api_url)
        if u.scheme != "https" or not u.hostname:
            raise ConfigError("roboflow.api_url must be an https URL")


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
class AuditSection:
    phash_hamming_max: int     # two frames are near-duplicates when their 64-bit pHashes differ in <= this many bits
    page_size: int             # images per search page
    fetch_timeout_s: float
    search_attempts: int       # the search index lags writes: a listing shorter than its total is retried
    search_wait_s: float

    def validate(self) -> None:
        _pos_int("audit.search_attempts", self.search_attempts)
        if self.search_wait_s < 0:
            raise ConfigError("audit.search_wait_s must be >= 0")
        if not 0 <= self.phash_hamming_max <= 64:
            raise ConfigError("audit.phash_hamming_max must be in [0, 64]")
        _pos_int("audit.page_size", self.page_size)
        _pos("audit.fetch_timeout_s", self.fetch_timeout_s)


@dataclass(frozen=True)
class SplitsSection:
    assignable: Tuple[str, ...]   # Roboflow splits a TRAIN-licensed match may be assigned to (whole match only)
    excluded_tag: str             # near-duplicate frames get this tag; never deleted; versions filter it out
    move_method: str              # "none" = refuse moves; "delete_reupload" = remove from Roboflow, upload again with
                                  #   the new split (identical, sha-verified bytes; restore file written first)
    probe_max: int                # --probe may move at most this many frames (to test a move method first)
    verify_attempts: int          # post-move checks are retried this often (search lags writes) before failing
    verify_wait_s: float
    batch_timeout_s: float        # read timeout of the batch create / delete calls (hundreds of images per call)

    def validate(self) -> None:
        _pos("splits.batch_timeout_s", self.batch_timeout_s)
        if not self.assignable or any(s not in ROBOFLOW_SPLITS for s in self.assignable):
            raise ConfigError(f"splits.assignable must be a non-empty subset of {ROBOFLOW_SPLITS}")
        if "test" in self.assignable:
            raise ConfigError("splits.assignable must not contain 'test' (held-out testing is EVAL's job)")
        if not self.excluded_tag.strip():
            raise ConfigError("splits.excluded_tag must be non-empty")
        if self.move_method not in ("none", "delete_reupload"):
            raise ConfigError("splits.move_method must be 'none' or 'delete_reupload' (plain re-upload does not move "
                              "splits: probed 2026-11-13)")
        _pos_int("splits.probe_max", self.probe_max)
        _pos_int("splits.verify_attempts", self.verify_attempts)
        if self.verify_wait_s < 0:
            raise ConfigError("splits.verify_wait_s must be >= 0")


@dataclass(frozen=True)
class BlurGateSection:
    """Variance of the Laplacian on a grey copy resized to a fixed width (so the score does not depend on the
    frame's resolution). A frame scoring below the minimum is tagged and never pre-labelled."""
    analysis_width_px: int
    laplacian_var_min: float           # 0 disables the gate
    excluded_tag: str

    def validate(self) -> None:
        _pos_int("prelabel.blur.analysis_width_px", self.analysis_width_px)
        if self.laplacian_var_min < 0:
            raise ConfigError("prelabel.blur.laplacian_var_min must be >= 0")
        if not self.excluded_tag.strip():
            raise ConfigError("prelabel.blur.excluded_tag must be non-empty")


@dataclass(frozen=True)
class GrassFilterSection:
    """Pre-labels only (never the runtime detector): a box whose bottom-centre point lies outside the frame's grass
    mask is dropped and logged. Sizes are fractions of the frame height / area (resolution independent)."""
    enabled: bool
    hsv_lower: Tuple[int, ...]         # OpenCV HSV (H 0..179, S/V 0..255)
    hsv_upper: Tuple[int, ...]
    close_frac: float                  # closing kernel (fills lines, players, shadows) as a fraction of frame height
    min_component_frac: float          # grass components smaller than this fraction of the frame area are ignored
    convex_hull: bool                  # fill each grass component to its convex hull: players standing on the pitch
                                       # (holes / notches in the grass, incl. occluded players) stay inside
    dilate_frac: float                 # final dilation as a fraction of frame height (feet stand on the grass edge)
    frame_edge_margin_frac: float      # a box whose bottom edge is within this fraction of the frame height of the
                                       # frame's bottom border is cut by the frame: its foot point is not observed,
                                       # so it is never dropped on grass evidence (kept, reason logged)
    apply_to: Tuple[str, ...]          # classes the filter applies to

    def validate(self) -> None:
        for n in ("hsv_lower", "hsv_upper"):
            v = getattr(self, n)
            if len(v) != 3 or not (0 <= v[0] <= 179 and 0 <= v[1] <= 255 and 0 <= v[2] <= 255):
                raise ConfigError(f"prelabel.grass.{n} must be [H 0..179, S 0..255, V 0..255]")
        if any(lo > hi for lo, hi in zip(self.hsv_lower, self.hsv_upper)):
            raise ConfigError("prelabel.grass.hsv_lower must be <= hsv_upper per channel")
        for n in ("close_frac", "min_component_frac", "dilate_frac", "frame_edge_margin_frac"):
            _unit(f"prelabel.grass.{n}", getattr(self, n))


@dataclass(frozen=True)
class BallTilingSection:
    """Optional second pass for the ball class only: overlapping tiles at native resolution, mapped back to frame
    coordinates and merged with the full-frame ball boxes by NMS (one box per ball). Players are never tiled."""
    enabled: bool
    tile_px: int
    overlap_frac: float                # overlap between neighbouring tiles, fraction of tile_px
    merge_iou: float                   # ball boxes overlapping more than this are one ball (highest score kept)

    def validate(self) -> None:
        _pos_int("prelabel.ball_tiling.tile_px", self.tile_px)
        if not 0.0 <= self.overlap_frac < 1.0:
            raise ConfigError("prelabel.ball_tiling.overlap_frac must be in [0, 1)")
        _unit("prelabel.ball_tiling.merge_iou", self.merge_iou)


@dataclass(frozen=True)
class BallDiagnosticSection:
    """--diagnose-ball: raw ball scores before any class threshold (read-only; nothing is written to Roboflow)."""
    request_confidence: float          # server-side floor for the diagnostic request only
    top_k: int                         # scores logged per frame and pass
    tile_px_candidates: Tuple[int, ...]  # tile sizes measured besides the full frame (overlap: ball_tiling)

    def validate(self) -> None:
        _unit("prelabel.ball_diagnostic.request_confidence", self.request_confidence)
        _pos_int("prelabel.ball_diagnostic.top_k", self.top_k)
        for t in self.tile_px_candidates:
            _pos_int("prelabel.ball_diagnostic.tile_px_candidates[]", t)


@dataclass(frozen=True)
class PrelabelSection:
    model_id: str                      # pretrained general detector (commercial-use licence); "<project>/<version>"
    model_licence: str                 # recorded in every prelabel log
    api_url: str
    request_confidence: float          # sent to the server; must be <= every class threshold
    iou: float
    timeout_s: float
    class_map: Tuple[str, ...]         # "model_label:spana_class"; unmapped labels are dropped and counted
    class_thresholds: Tuple[str, ...]  # "spana_class:min_score", one per auto-labelled class
    tag: str                           # tag added to every pre-labelled frame
    superseded_tags: Tuple[str, ...]   # older pre-label tags; such frames are only touched by --migrate (backup first)
    ball_class: str                    # the class the ball-only tiling pass and diagnostic act on
    ball_max_size_frac: float          # a ball box whose longer side exceeds this fraction of the frame height is
                                       # dropped and logged (static overlays / logos); a missed ball beats a wrong one
    sample_size: int                   # largest run allowed without the explicit full-run confirmation
    min_interval_s: float              # pause between frames (Roboflow rate limits the platform API)
    add_to_dataset: bool               # false: pre-labels wait in their batch until reviewed
    blur: BlurGateSection
    grass: GrassFilterSection
    ball_tiling: BallTilingSection
    ball_diagnostic: BallDiagnosticSection

    def validate(self) -> None:
        if self.model_id.count("/") != 1 or any(not part for part in self.model_id.split("/")):
            raise ConfigError("prelabel.model_id must look like '<project>/<version>'")
        if not self.model_licence.strip() or not self.tag.strip():
            raise ConfigError("prelabel.model_licence and prelabel.tag must be non-empty")
        u = urlparse(self.api_url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ConfigError("prelabel.api_url must be an http(s) URL")
        _unit("prelabel.request_confidence", self.request_confidence)
        _unit("prelabel.iou", self.iou)
        _pos("prelabel.timeout_s", self.timeout_s)
        _pos_int("prelabel.sample_size", self.sample_size)
        if self.min_interval_s < 0:
            raise ConfigError("prelabel.min_interval_s must be >= 0")
        if not 0.0 < self.ball_max_size_frac <= 1.0:
            raise ConfigError("prelabel.ball_max_size_frac must be in (0, 1]")
        if self.tag in self.superseded_tags:
            raise ConfigError("prelabel.tag must not be one of prelabel.superseded_tags")
        for sec in (self.blur, self.grass, self.ball_tiling, self.ball_diagnostic):
            sec.validate()


@dataclass(frozen=True)
class ReviewSection:
    reviewed_tag: str     # a human sets this tag on a frame after checking / correcting its labels

    def validate(self) -> None:
        if not self.reviewed_tag.strip():
            raise ConfigError("review.reviewed_tag must be non-empty")


@dataclass(frozen=True)
class BallStaticFlag:
    """A ball box that stays at the same place in the scene between two consecutive sampled frames of one match
    while the camera moves (camera motion estimated from background features): likely a spare ball."""
    tag: str
    max_gap_s: float               # the two frames are at most this far apart (metadata timestamp_s)
    max_shift_frac: float          # residual between the camera-carried ball centre and the next frame's ball box
                                   # centre, fraction of frame height
    min_camera_shift_frac: float   # the camera moved the ball's position by at least this fraction of frame height
    max_features: int              # ORB features per frame (outside annotated boxes)
    min_inliers: int               # fewer consistent matches: no camera estimate, no flag
    ransac_reproj_frac: float      # RANSAC inlier distance, fraction of frame height
    box_pad_frac: float            # annotated boxes are padded by this fraction of frame height before masking

    def validate(self) -> None:
        _pos("review_flags.ball_static.max_gap_s", self.max_gap_s)
        for n in ("max_shift_frac", "min_camera_shift_frac", "ransac_reproj_frac", "box_pad_frac"):
            _unit(f"review_flags.ball_static.{n}", getattr(self, n))
        _pos_int("review_flags.ball_static.max_features", self.max_features)
        _pos_int("review_flags.ball_static.min_inliers", self.min_inliers)


@dataclass(frozen=True)
class BallOffpitchFlag:
    tag: str                       # ball box whose centre is outside the grass mask (prelabel.grass settings)

    def validate(self) -> None:
        pass


@dataclass(frozen=True)
class BallTinyFlag:
    tag: str
    min_size_frac: float           # a ball box whose longer side is below this fraction of the frame height

    def validate(self) -> None:
        _unit("review_flags.ball_tiny.min_size_frac", self.min_size_frac)


@dataclass(frozen=True)
class PlayerEdgeFlag:
    tag: str
    band_frac: float               # foot point within this fraction of the frame height of the grass mask's outer
                                   # boundary (or outside the mask)

    def validate(self) -> None:
        _unit("review_flags.player_edge.band_frac", self.band_frac)


@dataclass(frozen=True)
class DenseFlag:
    tag: str
    pair_iou_min: float            # two person boxes overlap "heavily" from this IoU
    min_pairs: int                 # frames with at least this many such pairs are flagged

    def validate(self) -> None:
        _unit("review_flags.dense.pair_iou_min", self.pair_iou_min)
        _pos_int("review_flags.dense.min_pairs", self.min_pairs)


@dataclass(frozen=True)
class ReviewFlagsSection:
    """Review-priority tags for pre-labelled frames (tools/roboflow_review_flags.py). Tags only: annotations are
    never changed. A flag orders human review; it does not say a box is wrong."""
    ball_static: BallStaticFlag
    ball_offpitch: BallOffpitchFlag
    ball_tiny: BallTinyFlag
    player_edge: PlayerEdgeFlag
    dense: DenseFlag
    write_batch_size: int          # tag writes per batch; the run log is saved after every batch (resume point)
    min_interval_s: float          # pause between tag writes

    def flags(self) -> dict:
        """{flag name: tag}."""
        return {n: getattr(self, n).tag for n in ("ball_static", "ball_offpitch", "ball_tiny", "player_edge", "dense")}

    def validate(self) -> None:
        for name, tag in self.flags().items():
            if not tag.strip() or tag.startswith(("match:", "split:")):
                raise ConfigError(f"review_flags.{name}.tag must be non-empty and not a match: / split: tag")
            getattr(self, name).validate()
        _pos_int("review_flags.write_batch_size", self.write_batch_size)
        if self.min_interval_s < 0:
            raise ConfigError("review_flags.min_interval_s must be >= 0")


@dataclass(frozen=True)
class VersionSection:
    auto_orient: bool
    resize_width: int
    resize_height: int
    resize_format: str
    augmentations: Tuple[str, ...]    # subset of GEOMETRIC_AUGMENTATIONS; colour-changing ones are refused

    def validate(self) -> None:
        for n in ("resize_width", "resize_height"):
            v = getattr(self, n)
            if not 1 <= v <= 2048:
                raise ConfigError(f"version.{n} must be in [1, 2048]")
        if self.resize_format not in RESIZE_FORMATS:
            raise ConfigError(f"version.resize_format must be one of {RESIZE_FORMATS}")
        bad = [a for a in self.augmentations if a not in GEOMETRIC_AUGMENTATIONS]
        if bad:
            raise ConfigError(f"version.augmentations {bad} refused: only {GEOMETRIC_AUGMENTATIONS} are allowed "
                              "(no colour-changing augmentation; kit colour matters for team classification)")


@dataclass(frozen=True)
class PreflightSection:
    """Read-only checks before a version is generated (tools/roboflow_preflight.py)."""
    split_matches: Tuple[str, ...]        # "<roboflow split>:<match_id>": the only matches a split may hold, and
                                          #   each listed pair must be present
    ball_small_px: float                  # report the share of ball boxes whose short side is below this (source px)
    ball_size_buckets_px: Tuple[float, ...]   # ascending short-side bucket edges (source px) for size reports
    ball_quantile: float                  # this quantile of the ball short side, scaled to the version size ...
    ball_min_version_px: float            # ... must be at least this many pixels, else the version is refused

    def validate(self) -> None:
        pairs = [e.partition(":") for e in self.split_matches]
        if not pairs or any(not sep or s not in ROBOFLOW_SPLITS or not m.strip() for s, sep, m in pairs):
            raise ConfigError(f"preflight.split_matches entries must be '<one of {ROBOFLOW_SPLITS}>:<match_id>'")
        mids = [m for _, _, m in pairs]
        if len(set(mids)) != len(mids):
            raise ConfigError("preflight.split_matches: a match is listed twice (a match belongs to one split)")
        if any(s == "test" for s, _, _ in pairs):
            raise ConfigError("preflight.split_matches must not use 'test' (held-out testing is EVAL's job)")
        _pos("preflight.ball_small_px", self.ball_small_px)
        _pos("preflight.ball_min_version_px", self.ball_min_version_px)
        b = self.ball_size_buckets_px
        if not b or any(v <= 0 for v in b) or list(b) != sorted(set(b)):
            raise ConfigError("preflight.ball_size_buckets_px must be ascending, distinct and > 0")
        if not 0.0 <= self.ball_quantile <= 1.0:
            raise ConfigError("preflight.ball_quantile must be in [0, 1]")

    def expected(self) -> dict:
        """{match_id: roboflow split}"""
        return {m: s for s, _, m in (e.partition(":") for e in self.split_matches)}


@dataclass(frozen=True)
class PathsSection:
    manifest_dir: str    # upload manifests (records; committed)
    frames_dir: str      # extracted frames and dry-run manifests (run outputs; not committed)
    baseline_dir: str    # baseline detections, summaries and annotated samples (run outputs; not committed)
    audit_dir: str       # dataset audits (records; committed; no pixels)
    split_dir: str       # split-assignment manifests (records; committed)
    prelabel_dir: str    # pre-label logs (records; committed; no pixels)
    render_dir: str      # annotated pre-label renders for visual review (run outputs; not committed)
    image_cache_dir: str  # verified copies of project images (run outputs; not committed)
    prelabel_backup_dir: str  # annotation backups written before any pre-label is replaced (records; committed)
    review_dir: str      # review-flag run logs (records; committed; no pixels)
    version_dir: str     # preflight records, label snapshots and version manifests (records; committed; no pixels)


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
    audit: AuditSection
    splits: SplitsSection
    prelabel: PrelabelSection
    review: ReviewSection
    review_flags: ReviewFlagsSection
    version: VersionSection
    preflight: PreflightSection
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
        targets = set()
        for entry in self.prelabel.class_map:
            label, sep, target = entry.partition(":")
            if not sep or not label or target not in self.classes:
                raise ConfigError(f"prelabel.class_map entry {entry!r}: expected 'model_label:<one of classes>'")
            targets.add(target)
        if len({e.partition(":")[0] for e in self.prelabel.class_map}) != len(self.prelabel.class_map):
            raise ConfigError("prelabel.class_map: a model label is mapped twice")
        thresholds = self.prelabel_thresholds()
        if set(thresholds) != targets:
            raise ConfigError(f"prelabel.class_thresholds must give exactly one threshold per auto-labelled class "
                              f"{sorted(targets)}, got {sorted(thresholds)}")
        if any(self.prelabel.request_confidence > t for t in thresholds.values()):
            raise ConfigError("prelabel.request_confidence must be <= every prelabel.class_thresholds value")
        tags = [self.splits.excluded_tag, self.prelabel.tag, self.review.reviewed_tag, self.prelabel.blur.excluded_tag,
                *self.prelabel.superseded_tags, *self.review_flags.flags().values()]
        if len(set(tags)) != len(tags):
            raise ConfigError("splits.excluded_tag, prelabel.tag, prelabel.superseded_tags, prelabel.blur.excluded_tag, "
                              "review.reviewed_tag and the review_flags tags must all differ")
        if self.prelabel.ball_class not in targets:
            raise ConfigError(f"prelabel.ball_class {self.prelabel.ball_class!r} must be an auto-labelled class")
        unknown = [c for c in self.prelabel.grass.apply_to if c not in targets]
        if unknown:
            raise ConfigError(f"prelabel.grass.apply_to {unknown}: not auto-labelled classes")
        ids = [m.match_id for m in self.licensed_matches]
        if len(set(ids)) != len(ids):
            raise ConfigError("licensed_matches: a match_id is listed twice (a match belongs to one split)")
        for m in self.licensed_matches:
            if not m.match_id.strip() or not m.permission.strip():
                raise ConfigError("licensed_matches: match_id and permission must be non-empty")
            if m.split not in SPLITS:
                raise ConfigError(f"licensed_matches[{m.match_id}].split must be one of {SPLITS}")

    def prelabel_tags(self) -> Tuple[str, ...]:
        """Every pre-label tag, current and superseded: such a frame holds machine labels until it is reviewed."""
        return (self.prelabel.tag, *self.prelabel.superseded_tags)

    def excluded_tags(self) -> Tuple[str, ...]:
        """Tags that keep a frame out of pre-labelling and out of any version."""
        return (self.splits.excluded_tag, self.prelabel.blur.excluded_tag)

    def prelabel_class_map(self) -> dict:
        return dict(e.split(":", 1) for e in self.prelabel.class_map)

    def prelabel_thresholds(self) -> dict:
        out = {}
        for entry in self.prelabel.class_thresholds:
            cls, sep, val = entry.rpartition(":")
            try:
                t = float(val)
            except ValueError:
                raise ConfigError(f"prelabel.class_thresholds entry {entry!r}: expected 'class:number'") from None
            if not sep or cls not in self.classes or not 0.0 <= t <= 1.0 or cls in out:
                raise ConfigError(f"prelabel.class_thresholds entry {entry!r}: expected '<class>:<0..1>', once per class")
            out[cls] = t
        return out

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


def match_of(cfg: DatasetConfig, tags, metadata: dict) -> Tuple[Optional[LicensedMatch], Optional[str]]:
    """(licensed match, None) when a project image may be used at all, else (None, reason). Used by pre-labelling,
    split assignment and version generation: the allow-list applies to every step, not only to upload."""
    match_tags = sorted(t[len("match:"):] for t in tags if t.startswith("match:"))
    split_tags = sorted(t[len("split:"):] for t in tags if t.startswith("split:"))
    if not match_tags:
        return None, "no match:<id> tag"
    if len(match_tags) > 1:
        return None, f"several match tags {match_tags}"
    mid = match_tags[0]
    if "eval" in split_tags:
        return None, "tagged split:eval (EVAL is never used here)"
    md = (metadata or {}).get("match_id")
    if md is not None and md != mid:
        return None, f"metadata match_id {md!r} disagrees with tag match:{mid}"
    entry = next((m for m in cfg.licensed_matches if m.match_id == mid), None)
    if entry is None:
        return None, f"match {mid!r} is not on the licensed_matches allow-list"
    if entry.split == "eval":
        return None, f"match {mid!r} is registered as EVAL"
    # A split tag names the licence split, or (TRAIN-licensed matches only) the Roboflow split the whole match was
    # assigned to: one of splits.assignable, which never contains "test". Anything else fails closed.
    allowed = {entry.split} | (set(cfg.splits.assignable) if entry.split == "train" else set())
    if split_tags and (len(split_tags) != 1 or split_tags[0] not in allowed):
        return None, f"split tags {split_tags} disagree with the allow-list split {entry.split!r}"
    if not entry.third_party_processing_confirmed:
        return None, f"match {mid!r}: third-party processing not confirmed"
    return entry, None
