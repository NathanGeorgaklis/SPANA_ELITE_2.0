# INTERFACES

Taken from the code as of 2026-11-14. Records: `spana/records.py`. Artifacts: `spana/artifacts.py`
(canonical JSON, validated, atomic, never overwritten; any change to keys/types/units/meaning bumps the version).

## Video input — `spana/video.py`, `spana/provenance.py`
- In: video path (optional `--video-sha256` guard). Out: `VideoMeta` (incl. sha256, PTS timing) + BGR frames.
- Provenance block in every artifact: git state, package versions, config sha256, input hashes, argv.

## Player detection — `spana/detection/`
- Interfaces (`base.py`): `PlayerDetector.detect(frame_bgr) -> List[Detection]` (persons only);
  `Detector.detect(frame_bgr) -> List[Detection]` + `class_names` (generic multi-class, e.g. Roboflow backend).
- `Detection`: `box` (x1,y1,x2,y2 source px), `score`, `class_id`, `multi_body`.
- Artifact out: **`spana.detections` 1.3.0** — top level `schema, schema_version, provenance, video, model,
  detection_config, frame_stride, frames`; frame `{frame, t, detections}`; row
  `[x1, y1, x2, y2, score, class_id, multi_body(0/1)]`.
- CLI: `python -m spana.cli detect --video --config --model-record --out [--start --stop --stride]`.

## Player tracking — `spana/tracking/`
- In: `spana.detections` (must match video sha256 and current `detection_config`) + video frames.
- Artifact out: **`spana.tracks` 2.1.0** — top level `schema, schema_version, provenance, video,
  detections_sha256, frames, tracklets`; frame `{frame, t, camera{status, affine, inliers, hist_distance},
  tracks[], ended[[track_id, EndReason]]}`; track keys: `track_id, state, confirmed, source, box,
  detection_index, detection_score, confidence, contested, age_frames, missed_frames, time_since_observed_s,
  observed_frames, epoch, occluder, max_overlap_iou, overlap_with, recoveries, last_gap_s,
  possible_predecessors, identity, identity_rivals, uncertain_episodes`; tracklet keys: `track_id, epoch,
  first_frame, last_frame, last_observed_frame, observed_frames, confirmed, end_reason, recoveries,
  possible_predecessors`.
- Side output: decision log `*.decisions.jsonl` (every association, refusal, spawn, end).
- CLI: `python -m spana.cli track --video --config --detections --out`.

### State enums (`spana/records.py`)
| enum | values |
|---|---|
| `TrackState` | ACTIVE, OCCLUDED, LOST, TERMINATED (only via `ended`) |
| `IdentityState` (per track, orthogonal to TrackState) | CERTAIN, IDENTITY_UNCERTAIN |
| `PositionSource` | OBSERVED (ACTIVE only), PREDICTED, HELD (OCCLUDED/LOST only) |
| `CameraStatus` | OK, UNKNOWN, CUT, FIRST |
| `EndReason` | UNCONFIRMED_MISSED, UNCONFIRMED_TIMEOUT, LOST_EXPIRED, CAMERA_CUT, CAMERA_MOTION_UNKNOWN, LEFT_FRAME |

## Evaluation (supporting) — `spana/evaluation.py`, `spana/labels.py`, `tools/benchmark.py`
- In: tracks + detections artifacts, frozen MOT labels (`data/labels/<clip>/v1`). Out: `Report` metrics JSON.
- Match rules: IoU 0.5, NESTED_V1, NESTED_V2 (default).

## Re-ID labels (supporting) — `spana/reid_labels.py`
- Schemas `spana.reid_match` 1.1.0, `spana.reid_identities` 1.0.0. Splits TRAIN / DEV / EVAL (EVAL needs approval).
- Side labels: A, B, GK_A, GK_B, REF, OTHER (labels only; no classifier consumes them).

## Detection dataset (supporting) — `spana/dataset/`
- Roboflow splits train/valid/test; SPANA splits train/dev/eval (eval never uploaded).
- Upload manifest frame statuses: uploaded, duplicate, failed, not_uploaded.

## Not yet defined
| layer | inputs / outputs | state enum |
|---|---|---|
| target identity | NOT DEFINED | NOT DEFINED |
| ball tracking | NOT DEFINED | NOT DEFINED |
| team classification | NOT DEFINED | NOT DEFINED |
| pitch context | NOT DEFINED | NOT DEFINED |
| possession/touches | NOT DEFINED | NOT DEFINED |
| event segmentation | NOT DEFINED | NOT DEFINED |
| event classification | NOT DEFINED | NOT DEFINED |
| outcomes | NOT DEFINED | NOT DEFINED |
| statistics | NOT DEFINED | NOT DEFINED |
| clips | NOT DEFINED | NOT DEFINED |
