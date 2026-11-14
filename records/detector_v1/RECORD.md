# Detector v1 (2026-11-14)

## Data
- Preflight (read-only): `records/roboflow_version/preflight_20261114T082808Z.json`, passed. Frozen labels
  `labels_20261114T082808Z.json` (sha256 in the version manifest).
- Dataset version 2 (`records/roboflow_version/version_2.json`): 578 train (TRAIN_M002, TRAIN_M003), 295 valid
  (TRAIN_M001), 2 platform test frames (TRAIN_M001 f0000000, f0000144; never scored). 1280x1280
  "Fit (black edges) in", auto-orient, excluded tags filtered, no baked augmentation.
- Version 1 (578 / 297 / 0) exists on Roboflow and is unused: Roboflow refuses to train a version with fewer than
  2 test images (`insufficient_split_images`). Nathan moved 2 valid frames to test in the UI;
  `version.platform_test_frames: 2` allows exactly that and nothing else in test.
- Ball boxes (875 frames, source pixels): short side median 13 px, 5th percentile 9 px, 68.2 % under 16 px.
- No EVAL data anywhere in this step. The valid split is a held-out TRAIN match, not SPANA DEV.

## Training
- Roboflow training `9b0b72177c37bc0cd3aa` on version 2: `rfdetr-medium`, COCO checkpoint, input 1280, 100 epochs
  (ran to the end), online augmentation horizontal flip p = 0.5 only. 08:37 to 10:23 UTC = 106 min, about 3.5
  credits at the published 30 GPU-minutes per credit.
- Model id `nathans-workspace-9brdc/spana-elite-2-rfdetr-medium-t1` (config `detector.model_id`).
- Roboflow's own figures for the run: mAP50 91.67, precision 91.4, recall 87.5.

## Evaluation on valid (295 frames, TRAIN_M001), confidence 0.50, IoU 0.50, nothing tuned
`eval_nathans-workspace-9brdc_spana-elite-2-rfdetr-medium-t1_20261114T103752Z.json`; tool `tools/detector_eval.py`.

mAP50 0.9215, mAP50-95 0.7297.

| class | AP50 | AP50-95 | precision | recall | labels |
|---|---|---|---|---|---|
| player | 0.970 | 0.813 | 0.940 | 0.952 | 2383 |
| goalkeeper | 0.830 | 0.647 | 0.905 | 0.642 | 134 |
| referee | 0.967 | 0.785 | 0.978 | 0.929 | 239 |
| ball | 0.920 | 0.674 | 0.939 | 0.884 | 225 |

Confusion (rows: label, columns: prediction):

| | player | goalkeeper | referee | ball | missed |
|---|---|---|---|---|---|
| player | 2268 | 8 | 2 | 0 | 105 |
| goalkeeper | 23 | 86 | 0 | 0 | 25 |
| referee | 4 | 0 | 221 | 0 | 14 |
| ball | 1 | 0 | 0 | 199 | 25 |
| background | 118 | 1 | 4 | 13 | - |

Ball recall by short side (identical at IoU 0.50 and 0.25): 0-8 px 1/5; 8-12 px 46/57 (0.81); 12-16 px 88/92
(0.96); 16-24 px 49/53 (0.92); >= 24 px 15/18 (0.83).

20 annotated frames (5 worst ball, 5 worst player, 10 evenly spaced): `runs/detector_v1/<model>/frames/`
(not committed). Worst player frame f0014544 (14 errors): motion-blurred pan; the blurred group in the goal mouth
is missed and two coaching staff in the foreground are boxed as player.

## Adapter
- `RoboflowDetector.from_detector_config` reads the `detector` section (model id, confidence 0.50, iou 0.50).
- Smoke test, one DEV frame (DEV_G1_SYDNEY_UNITED_APIA, FRESH_02 frame 120, 1920x1080, hosted inference):
  14 player, 1 referee, 1 ball, 0 goalkeeper; no unknown labels. Not scored.
- DEV matches on the allow-list: `records/dev_permissions/DEV_PERMISSIONS_2026-11-14.md`.

## Not done (waiting for Nathan's review)
- No threshold tuning, no retraining.
