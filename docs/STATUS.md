# STATUS

As of 2026-11-14 (main fb7a1fd = detector v1 merged, PR #2, + this change). Full suite: `python -m pytest tests -q`
→ 275 passed, 3 skipped (Python 3.13, pinned requirements.lock.txt).
DEV numbers below come from `records/`; nothing here has been scored on EVAL.

| Layer | State | Module path | Tests | Notes |
|---|---|---|---|---|
| player detection | PARTIAL | `spana/detection/` (`onnx_backend.py`, `postprocess.py`, `tiling.py`, `roboflow_backend.py`, `base.py`) | `tests/test_detection_camera.py`, `tests/test_merge_detections.py`, `tests/test_multi_body.py`, `tests/test_roboflow_dataset.py`, `tests/test_detector_eval.py`, `tests/test_detection_interface.py`, `tests/test_dev_bench.py` | Football detector v1 trained (`nathans-workspace-9brdc/spana-elite-2-rfdetr-medium-t1`, RF-DETR medium, dataset version 2), adapter `RoboflowDetector.from_detector_config` (config `detector`, confidence / iou 0.50 untuned). Scored only on the M001 valid split (table below), which shares a broadcaster with training: generalisation unknown. DEV benchmark tooling built (`tools/dev_bench.py`), not yet run. Pipeline CLI still uses pinned COCO YOLOX-s, person only (`models/yolox_s.json`); FRESH_02 recall 0.971 (NESTED_V1). YOLOX inference test skipped without weights. |
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

## Detector v1 (M001 valid, 295 frames, confidence 0.50, IoU 0.50; `records/detector_v1/RECORD.md`)
mAP50 0.922, mAP50-95 0.730. Not a generalisation measure: M001 shares a broadcaster with the training matches.

| class | AP50 | AP50-95 | precision | recall | labels |
|---|---|---|---|---|---|
| player | 0.970 | 0.813 | 0.940 | 0.952 | 2383 |
| goalkeeper | 0.830 | 0.647 | 0.905 | 0.642 | 134 |
| referee | 0.967 | 0.785 | 0.978 | 0.929 | 239 |
| ball | 0.920 | 0.674 | 0.939 | 0.884 | 225 |

Known weaknesses:
- Goalkeeper recall 0.64 (86 / 134): 23 goalkeepers boxed as player, 25 missed.
- 118 false player boxes (background row of the confusion matrix). Some are off the pitch (e.g. coaching staff in
  the foreground of f0014544). The off-pitch share has not been counted yet: the DEV benchmark scoring will count it
  with the grass mask, for analysis only.
- Ball recall 0.20 under 8 px short side (1 / 5); 0.81 at 8-12 px, >= 0.92 from 12 to 24 px.

DEV benchmark (`tools/dev_bench.py`, project `spana-elite-dev-bench`, 40 frames per allow-listed DEV match,
`bench:v1`, `prelabel:detector-v1`): project created (empty, private); sampling + pre-label NOT yet run (needs the
local DEV clips and the API key); labels to be reviewed by Nathan before any scoring script is written or run.

## Open failures
- No failing tests. Skipped: YOLOX inference (weights not fetched), two Playwright labeller UI tests (no browser).
- Detection -> tracking, not yet wired (docs/INTERFACES.md): the tracker ignores `class_id` (balls would be tracked
  as players); the detections artifact does not carry class names; the adapter's 0.50 cut leaves the tracker's
  low-score round (`detection.score_low` 0.10, default.yaml) with nothing.
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
1. Run `tools/dev_bench.py` on each allow-listed DEV match (dry run first), then Nathan reviews the benchmark labels.
2. After his OK: scoring script (per match / class AP50, P, R; ball recall by size bucket; off-pitch false positives
   via the grass mask) and DEV vs M001 side by side in `records/detector_v1/RECORD.md`.
3. Then the tracking adapter for the football detector (class filter, class names in the artifact, one threshold
   story), per the detection -> tracking interface.
