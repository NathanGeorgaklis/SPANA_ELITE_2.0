# STATUS

As of 2026-11-14 (commit dfa8557 + docs). Full suite: `python -m pytest tests -q` → 214 passed, 3 skipped.
DEV numbers below come from `records/`; nothing here has been scored on EVAL.

| Layer | State | Module path | Tests | Notes |
|---|---|---|---|---|
| player detection | PARTIAL | `spana/detection/` (`onnx_backend.py`, `postprocess.py`, `tiling.py`, `roboflow_backend.py`, `base.py`) | `tests/test_detection_camera.py`, `tests/test_merge_detections.py`, `tests/test_multi_body.py`, `tests/test_roboflow_dataset.py` | Pinned pretrained COCO YOLOX-s, person class only (`models/yolox_s.json`, provisional). Full frame + tiles, union suppression, multi-body flag. FRESH_02 recall 0.971 (NESTED_V1). Football detector (player/goalkeeper/referee/ball) not trained: Roboflow dataset prep + pre-labelling in progress (`spana/dataset/`, `tools/roboflow_*.py`). YOLOX inference test skipped without weights. |
| player tracking | PARTIAL | `spana/tracking/` (`tracker.py`, `kalman.py`, `camera.py`), `spana/appearance.py` | `tests/test_tracker.py`, `tests/test_camera_uncertainty.py`, `tests/test_identity_uncertain.py`, `tests/test_unconfirmed_low_score.py`, `tests/test_cli_e2e.py`, `tests/test_foundation.py` | Generic multi-player tracker with camera compensation and per-track IDENTITY_UNCERTAIN. DEV FRESH_02/05/07: 0 silent swaps (NESTED_V2). Fragmentation remains the largest error (e.g. FRESH_07 47 fragments). Nested-box fix (M2) deferred pending Nathan's approval. |
| target identity | NOT BUILT | none | none | No selection/following of the user's player. Re-ID model is design only (`records/design/REID_MODEL_DESIGN.md`); TRAIN label tooling exists (`spana/reid_labels.py`, `tools/reid_pack.py`, `tools/freeze_reid_labels.py`). |
| ball tracking | NOT BUILT | none | none | Only dataset tooling handles the ball class (ball tiling / size filter in `spana/dataset/prefilter.py`, `tools/roboflow_ball_sheet.py`). |
| team classification | NOT BUILT | none | none | Torso colour histogram (`spana/appearance.py`) is used for track association only. Side labels (A/B/GK/REF) exist in re-ID labels only. |
| pitch context | NOT BUILT | none | none | Grass mask in `spana/dataset/prefilter.py` filters pre-labels only; no pitch model / homography. |
| possession/touches | NOT BUILT | none | none | |
| event segmentation | NOT BUILT | none | none | |
| event classification | NOT BUILT | none | none | |
| outcomes | NOT BUILT | none | none | |
| statistics | NOT BUILT | none | none | |
| clips | NOT BUILT | none | none | |

Supporting (not pipeline layers): `spana/video.py`, `spana/provenance.py`, `spana/artifacts.py`, `spana/config.py`,
`spana/evaluation.py` + `tools/benchmark.py` (scoring vs frozen labels), `spana/labels.py`, `spana/cli.py`.

## Open failures
- No failing tests. Skipped: YOLOX inference (weights not fetched), two Playwright labeller UI tests (no browser).
- Tracking, DEV: fragmentation / ID switches (FRESH_02 33, FRESH_05 20, FRESH_07 48 ID switches, NESTED_V1;
  `records/unconfirmed_low_score_2026-11-08/RECORD.md`).
- Tracking, DEV: one-directional duplicate refusal (M2: full-body box enclosing a partial-body track spawns a
  duplicate). Diagnosed, fix deferred (`records/partial_body_2026-11-08/RECORD.md`).
- Tracking, DEV: IDENTITY_UNCERTAIN often never resolves for same-kit players (FRESH_07: 0 resolved, 8.1 % of
  confirmed observed track-frames uncertain; `records/identity_one_sided_2026-11-08/RECORD.md`).
- Detection, DEV: label boxes looser than detector boxes, so IoU 0.5 scoring under-reports recall
  (`records/benchmark_FRESH_02_v1/BENCHMARK_RECORD.md`).
- README pipeline table is stale (artifact versions listed as 1.1.0 / 2.0.0; code is detections 1.3.0, tracks 2.1.0).

## Hardcoded values / magic numbers found
- `spana/tracking/camera.py:40` cut-detection HSV histogram bins `[32, 32]` (`camera.cut_hist_distance` depends on it).
- `spana/tracking/tracker.py:626` confirmation needs `observed_frames >= 2` in addition to `tracking.confirm_s`.
- `spana/evaluation.py:216` `dup_ioa=0.6` default; `spana/evaluation.py:41` `HEIGHT_BINS_PX` (reporting only).
- `tools/benchmark.py:23` `IGNORE_IOA = 0.5` (scoring convention); `tools/roboflow_ball_sheet.py:22` `CROP_FRAC = 0.09`.
- Numerical guards only (`_EPS = 1e-9`, `_INVALID = 1e6`): not thresholds.
- No player/team/clip/frame-specific logic found in `spana/`. Match ids appear only in `config/roboflow.yaml`
  (licence allow-list) and DEV clip names only in config comments.

## Next task (suggested)
Target identity (layer 3 of the README order): define the target-identity interface and states
(e.g. TARGET / CANDIDATE / LOST / ABSTAIN) in `docs/INTERFACES.md`, then build a minimal consumer of
`spana.tracks` 2.1.0 that follows a user-selected track and abstains on IDENTITY_UNCERTAIN, end of epoch or
refused recovery. Alternative if detection is the priority: train/evaluate the football detector once the
Roboflow pre-label review is complete.
