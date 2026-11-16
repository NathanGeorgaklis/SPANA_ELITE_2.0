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

## DEV generalisation benchmark (66 reviewed frames; not yet scored)
- Roboflow project `spana-elite-dev-bench` created 2026-11-14 (private, object detection, own annotation group
  `spana-elite-dev-bench`, 0 images). No version is ever generated from it. The training version gate refuses any
  frame tagged `bench:v1`.
- Tool `tools/dev_bench.py`: 40 frames per allow-listed DEV match (DEV_G1_SYDNEY_UNITED_APIA,
  DEV_G2_NWS_SPIRIT_SD_RAIDERS; not Game 3, never EVAL), selected with the same sampling, blur and near-duplicate
  rules, pre-labelled with detector v1 at detector.confidence / iou 0.50 and no filters, tagged `match:<id>`,
  `bench:v1`, `prelabel:detector-v1`.
- Source footage (Nathan, 2026-11-15): the full DEV match videos, not the FRESH clips. The FRESH clips kept only 19
  (G1) and 30 (G2) usable frames, below frames_per_match 40; the rules were not loosened. Full-match permission
  confirmed (`records/dev_permissions/DEV_PERMISSIONS_2026-11-14.md`). Only 1280x720 copies were used: a first
  640x360 download was dry-run only and never uploaded.

  | match | video | source resolution | fps | length | sha256 (prefix) |
  |---|---|---|---|---|---|
  | DEV_G1_SYDNEY_UNITED_APIA | `Sydney_United_APIA.mp4` | 1280x720 | 30 | 130.9 min | c9f1dd9b3c25 |
  | DEV_G2_NWS_SPIRIT_SD_RAIDERS | `NWS_Spirit_SD_Raiders.mp4` | 1280x720 | 30 | 142.1 min | 527f2bdc47eb |

- Selection (dry runs 20261115T093253Z / T093307Z, reviewed by Nathan on contact sheets):

  | match | sampled (300 cap) | blur | near-duplicate | selected | removed by Nathan | uploaded | errors |
  |---|---|---|---|---|---|---|---|
  | G1 | 300 | 0 | 17 | 40 | 2 (frames 122400, 127410) | 38 | 0 |
  | G2 | 300 | 0 | 29 | 40 | 2 (frames 130818, 134700) | 38 | 0 |

  Removed: footage of other matches shown inside the broadcast at half-time (not on the allow-list). Kept as they
  are, at Nathan's decision: countdown / table / stats graphics, half-time interview and presentation close-ups,
  team huddles, stoppages (G1 frames 0, 7500, 18840, 31920, 116640, 229260; G2 frames 0, 8400, 17820, 31560, 61440,
  68460, 102780, 127200, 138240, 241740, 247680). The upload runs reproduced the reviewed dry runs exactly
  (`--reviewed-dry-run`); exclusions and reason are in the manifests.
- Uploaded 2026-11-15: 76 frames in `spana-elite-dev-bench` (38 per match), each tagged `match:<id>`, `split:dev`,
  `bench:v1`, `prelabel:detector-v1`, checked in Roboflow; `spana-elite` unchanged (900 images). Manifests:
  `records/dev_bench/DEV_G1_SYDNEY_UNITED_APIA__bench_v1__20261115T100734Z.json`,
  `records/dev_bench/DEV_G2_NWS_SPIRIT_SD_RAIDERS__bench_v1__20261115T102105Z.json`.
- Pre-label boxes (detector v1, 0.50 / 0.50, unfiltered; NOT scores): G1 player 424, referee 64, ball 12,
  goalkeeper 10; G2 player 170, referee 116, ball 21, goalkeeper 11. The G2 referee count is high for one match
  (yellow / green kits): check referee boxes during review.
- Label review (Nathan, done by 2026-11-16): boxes edited by hand from the detector-v1 pre-labels and checked frame
  by frame. Roboflow names the review jobs "Auto Label" (labeller AUTOLABEL) because the pre-labels were saved by
  the API; no other model's labels were used. Class totals over the 76 frames after review: player 713, referee 73,
  ball 44, goalkeeper 34 (pre-labels: 594 / 180 / 33 / 21).
- Approved 66, rejected 10. The 10 rejected frames are dropped from the benchmark (Nathan: not needed for tracking).
  They are not deleted and stay outside the Dataset, untagged: G1 frames 0, 7500, 18840, 116640; G2 frames 0, 8400,
  17820, 127200, 138240, 247680 (countdown / table / stats graphics, half-time close-ups).
- 2026-11-16: the 66 approved frames were accepted into the `spana-elite-dev-bench` Dataset (approved status only,
  split kept `valid`; G1 34, G2 32) and tagged `reviewed` (66 / 66 tag writes succeeded; checked: 66 tagged, none
  outside the Dataset). The benchmark is the 66 frames tagged `reviewed`. `spana-elite` unchanged (900).
- Not frozen and not scored yet. Next: freeze the 66 labels, then the scoring script, then DEV vs M001 here.
- Bias to note when scoring: reviewed labels start from detector v1's own boxes, so a detector miss the reviewer
  does not add is never counted. Review must add missed objects, not only correct the boxes that were drawn.
