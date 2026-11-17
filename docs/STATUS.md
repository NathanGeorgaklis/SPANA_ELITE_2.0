# STATUS

As of 2026-11-17 (branch identity/target-recovery-v1, on main f647048).
- Full suite: `python -m pytest tests -q` → 344 passed, 4 skipped (container, Python 3.13).
- Target / re-ID / guard tests (53) also passed on Nathan's Mac.
DEV numbers below come from `records/`; nothing here has been scored on EVAL.

| Layer | State | Module path | Tests | Notes |
|---|---|---|---|---|
| player detection | PARTIAL | `spana/detection/` (`onnx_backend.py`, `postprocess.py`, `tiling.py`, `roboflow_backend.py`, `base.py`) | `tests/test_detection_camera.py`, `tests/test_merge_detections.py`, `tests/test_multi_body.py`, `tests/test_roboflow_dataset.py`, `tests/test_detector_eval.py`, `tests/test_detection_interface.py`, `tests/test_dev_bench.py` | Football detector v1 trained (`nathans-workspace-9brdc/spana-elite-2-rfdetr-medium-t1`, RF-DETR medium, dataset version 2), adapter `RoboflowDetector.from_detector_config` (config `detector`, confidence / iou 0.50 untuned). Scored only on the M001 valid split (table below), which shares a broadcaster with training: generalisation unknown. DEV benchmark uploaded and pre-labelled (76 frames, 2 DEV matches), waiting for Nathan's label review; not scored. Pipeline CLI still uses pinned COCO YOLOX-s, person only (`models/yolox_s.json`); FRESH_02 recall 0.971 (NESTED_V1). YOLOX inference test skipped without weights. |
| player tracking | PARTIAL | `spana/tracking/` (`tracker.py`, `kalman.py`, `camera.py`), `spana/appearance.py` | `tests/test_tracker.py`, `tests/test_camera_uncertainty.py`, `tests/test_identity_uncertain.py`, `tests/test_unconfirmed_low_score.py`, `tests/test_cli_e2e.py`, `tests/test_foundation.py` | Generic multi-player tracker with camera compensation and per-track IDENTITY_UNCERTAIN. DEV FRESH_02/05/07: 0 silent swaps (NESTED_V2). Fragmentation remains the largest error (e.g. FRESH_07 47 fragments). Nested-box fix (M2) deferred pending Nathan's approval. |
| target identity | PARTIAL | `spana/identity/follower.py` | `tests/test_target_follow.py`, `tests/test_target_relink.py` | Follows one tapped track (selection `track_at_frame` 1.0.0) into `spana.target` 2.0.0 (TARGET / CANDIDATE / ABSTAIN / LOST). Fails closed (v1 rules). Relink v1: after a loss, proposes CANDIDATE on the tracker's unique predecessor link (gap ≤ `target.relink_max_gap_s` 0.5); TARGET only after the user confirms (`confirm_candidate` 1.0.0) or with `relink_auto_confirm` (off). DEV, 4 clips (`records/target_relink_v1_2026-11-16`): correct-time 0.509 without confirmation (= v1, wrong 1 frame) → **0.597 with a simulated user (wrong 2 frames: +1 from a tracker identity change after a correct confirmation)**; auto-confirm 0.597, wrong 3. Ceiling 0.853. Recovery inputs (`spana.target` 2.1.0): confirm / reject offers, tap to re-select, gap list. **Simulated after-match reviewer: correct-time 0.779, wrong 2 frames (the two known ones), but about 6.3 user actions per followed player-minute** (`records/target_recovery_v1_2026-11-17`). Long re-entries unsolved. Re-ID model is design only (`records/design/REID_MODEL_DESIGN.md`). |
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

DEV benchmark (`tools/dev_bench.py`, project `spana-elite-dev-bench`): uploaded 2026-11-15 from the full DEV match
videos at 1280x720. 76 frames: 38 per match (40 selected, 2 per match removed by Nathan as footage of other
matches), tagged `match:<id>`, `split:dev`, `bench:v1`, `prelabel:detector-v1`; `spana-elite` unchanged. Manifests in
`records/dev_bench/`, details in `records/detector_v1/RECORD.md`. Reviewed by Nathan (hand-edited): 66 approved
(G1 34, G2 32) accepted into the bench Dataset and tagged `reviewed`; 10 rejected graphics / close-up frames dropped
(kept, untagged). Labels frozen (`data/labels/dev_bench/bench_v1/`, sha256 088678a9…) and scored 2026-11-16
(`tools/dev_bench_score.py`):

| | M001 valid | DEV (66 fr.) | DEV G1 | DEV G2 |
|---|---|---|---|---|
| mAP50 | 0.922 | 0.701 | 0.693 | 0.736 |
| player recall | 0.952 | 0.804 | 0.922 | 0.597 |
| goalkeeper recall | 0.642 | 0.412 | 0.200 | 0.714 |
| referee precision | 0.978 | 0.367 | 0.641 | 0.216 |
| ball recall | 0.884 | 0.705 | 0.550 | 0.833 |

Main DEV failure: class confusion from kit colour (91 of 708 players boxed as referee, 85 of them the yellow-kit
team in G2; goalkeepers boxed as referee in G1). Players are found but given the wrong class. Off-pitch false positives
are rare on DEV (1 of 33). Details: `records/detector_v1/RECORD.md`.

## Open failures
- No failing tests. Skipped: YOLOX inference (weights not fetched), two Playwright labeller UI tests (no browser).
- Detection -> tracking: wired and run 2026-11-16 (`PersonView`, `detect --backend roboflow`, detections 1.4.0).
  On FRESH_02 / FRESH_05 the football detector's person view beats YOLOX on every tracking metric (NESTED_V2):
  - detection recall 0.973 → 0.980 / 0.989 → 0.998;
  - duplicate detections 126 → 17 / 84 → 3;
  - ID switches 28 → 25 / 22 → 8;
  - fragments 28 → 24 / 21 → 8.
  Details: `records/person_view_2026-11-16/RECORD.md`.
- Local football detector (option A, 2026-11-16): `detect --backend local`, torch on the Apple GPU, 0.44 s/frame
  end to end, no hosted calls. On FRESH_02 / FRESH_05 tracking equals hosted (ID switches 24 / 7, duplicate
  detections 17 / 1; `records/local_inference_2026-11-16/RECORD.md`). The CLI default backend is still onnx (YOLOX)
  and needs no extra arguments; for tracking work, use `--backend local`.
- Tracking, DEV: fragmentation / ID switches (FRESH_02 33, FRESH_05 20, FRESH_07 48 ID switches, NESTED_V1;
  `records/unconfirmed_low_score_2026-11-08/RECORD.md`).
- Tracking, DEV: one-directional duplicate refusal (M2: full-body box enclosing a partial-body track spawns a
  duplicate). Diagnosed, fix deferred (`records/partial_body_2026-11-08/RECORD.md`).
- Tracking, DEV: IDENTITY_UNCERTAIN often never resolves for same-kit players (FRESH_07: 0 resolved, 8.1 % of
  confirmed observed track-frames uncertain; `records/identity_one_sided_2026-11-08/RECORD.md`).
- Detection, DEV: label boxes looser than detector boxes, so IoU 0.5 scoring under-reports recall
  (`records/benchmark_FRESH_02_v1/BENCHMARK_RECORD.md`).
- Target identity, DEV: 30 tapped players followed with 0 wrong player-frames.
  - Time without an answer: F02 0.50, F05 0.28. Main causes: the track ends while the player is still in the
    picture (F02 340, F05 116 player-frames), then IDENTITY_UNCERTAIN (F02 100).
  - `records/target_follow_v1_2026-11-16/RECORD.md`.
- Re-ID headroom, DEV (4 clips): 108 losses of a player's track + 13 tracker identity switches / parallel tracks.
  - Only the tracker's own link within 0.5 s is safe (44 / 44).
  - Position + colour agreement picked 4 real wrong players in same-kit clusters.
  - `records/reid_headroom_F07_F08_2026-11-16/RECORD.md`.
- Long run, DEV G2 10-minute window: the tracker emitted an occluder that was not live on its frame (fixed on
  `tracking/occluder-live-fix`; DEV clip tracks unchanged).
  - Follower: 0 wrong answers, but 0 answered time for the chosen player without user taps (his track was
    uncertain, then ended on a camera pan after 7.3 s).
- Target relink, DEV: with confirmation, FRESH_02 player 9 gets 1 wrong TARGET frame (276). Track T000045 was
  confirmed correctly at frame 270, then changed player; it names two predecessors. Auto-confirm adds a wrong
  second-hop relink (FRESH_02 player 15 → player 16's track).
- Target identity, DEV F07: 1 wrong player-frame. A two-person box at a crossing, unflagged by the detector and the
  tracker.
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
1. Long-play capability (`records/window_f05_2026-11-18`, 10-minute DEV window of match G2).
   - Following one player unattended lasts one track. Median confirmed track life is 2.8 s on broadcast footage
     (camera pans: LEFT_FRAME; crowds: LOST_EXPIRED), so about one user action every 3 s.
   - Priorities:
     (a) re-identification across pans / re-entries (SPANA re-ID model and/or team classification);
     (b) tracker: keep tracks alive across pans, fewer LOST_EXPIRED in crowds. The exit hold was tried
         (`records/exit_hold_2026-11-18`): no net gain without a stronger appearance signal; off by default.
   - Re-measure track life and actions per minute on the same window after each.
2. Cut the after-match review effort (6.3 actions per followed player-minute on the DEV clips,
   `records/target_recovery_v1_2026-11-17`). Gap-length threshold / top-N per minute.
3. Relink v1 accepted as built (Nathan, 2026-11-16). Watch confirmed tracks that named several predecessors.
4. Remaining tracking errors (F07 38 / F08 51 fragments).
