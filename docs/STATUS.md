# STATUS

As of 2026-11-19 (branch identity/reentry-exit-side = identity/reid-signatures-v2 + evaluation/offer-verdict-offered-track,
on main 5c7dac5).
- Full suite: `python -m pytest tests -q` → 489 passed, 6 skipped (container, Python 3.13, without torch: the re-ID
  training tests skip).
- Target / re-ID / guard tests (53) also passed on Nathan's Mac.
DEV numbers below come from `records/`; nothing here has been scored on EVAL.

| Layer | State | Module path | Tests | Notes |
|---|---|---|---|---|
| player detection | PARTIAL | `spana/detection/` (`onnx_backend.py`, `postprocess.py`, `tiling.py`, `roboflow_backend.py`, `base.py`) | `tests/test_detection_camera.py`, `tests/test_merge_detections.py`, `tests/test_multi_body.py`, `tests/test_roboflow_dataset.py`, `tests/test_detector_eval.py`, `tests/test_detection_interface.py`, `tests/test_dev_bench.py` | Football detector v1 trained (`nathans-workspace-9brdc/spana-elite-2-rfdetr-medium-t1`, RF-DETR medium, dataset version 2), adapter `RoboflowDetector.from_detector_config` (config `detector`, confidence / iou 0.50 untuned). Scored only on the M001 valid split (table below), which shares a broadcaster with training: generalisation unknown. DEV benchmark uploaded and pre-labelled (76 frames, 2 DEV matches), waiting for Nathan's label review; not scored. Pipeline CLI still uses pinned COCO YOLOX-s, person only (`models/yolox_s.json`); FRESH_02 recall 0.971 (NESTED_V1). YOLOX inference test skipped without weights. |
| player tracking | PARTIAL | `spana/tracking/` (`tracker.py`, `kalman.py`, `camera.py`), `spana/appearance.py` | `tests/test_tracker.py`, `tests/test_camera_uncertainty.py`, `tests/test_identity_uncertain.py`, `tests/test_unconfirmed_low_score.py`, `tests/test_cli_e2e.py`, `tests/test_foundation.py` | Generic multi-player tracker with camera compensation and per-track IDENTITY_UNCERTAIN (`spana.tracks` 2.2.0: + `identity_doubt`). DEV 4 clips (local detector): 120 fragments, 121 ID switches, 6 transfers, 0 silent (NESTED_V2). Fragmentation remains the largest error. Where ByteTrack kept one track (76 of 120 fragments) ours broke mostly at the identity gate (44; 25 of them the torso-colour veto on the track's own next box), then low-score (7), LOST timeout (4), M2 nested (4), other (17) (`records/continuity_attribution_2026-11-18`). Appearance doubt link (`tracking.appearance_doubt_link`, S3b): continues such tracks flagged IDENTITY_UNCERTAIN: fragments 120 → 108, silent transfers 0, but reviewer correct-time 0.778 → 0.698 because doubts rarely resolve and the follower abstains on them. Self-consistent doubt resolution (`tracking.doubt_resolve_self_consistent`, option (a)): 5 more doubts cleared, fragments 106. With the follower's tap vouch (b): 0.783 correct, 2 wrong, but actions 6.27 / min = today, so the pass bar (fewer actions) is not met; **all off by default** (`records/doubt_resolution_2026-11-18`). Nested-box fix (M2) deferred pending Nathan's approval. Tests also `tests/test_appearance_doubt_link.py`. |
| target identity | PARTIAL | `spana/identity/follower.py` | `tests/test_target_follow.py`, `tests/test_target_relink.py` | Follows one tapped track (selection `track_at_frame` 1.0.0) into `spana.target` 2.0.0 (TARGET / CANDIDATE / ABSTAIN / LOST). Fails closed (v1 rules). Relink v1: after a loss, proposes CANDIDATE on the tracker's unique predecessor link (gap ≤ `target.relink_max_gap_s` 0.5); TARGET only after the user confirms (`confirm_candidate` 1.0.0) or with `relink_auto_confirm` (off). DEV, 4 clips (`records/target_relink_v1_2026-11-16`): correct-time 0.509 without confirmation (= v1, wrong 1 frame) → **0.597 with a simulated user (wrong 2 frames: +1 from a tracker identity change after a correct confirmation)**; auto-confirm 0.597, wrong 3. Ceiling 0.853. Recovery inputs (`spana.target` 2.1.0): confirm / reject offers, tap to re-select, gap list. **Simulated after-match reviewer: correct-time 0.779, wrong 2 frames (the two known ones), but about 6.3 user actions per followed player-minute** (`records/target_recovery_v1_2026-11-17`). Long re-entries unsolved. Re-ID model: pilot trained on TRAIN_M001 only (`tools/reid_crops.py`, `tools/reid_train.py`, `config/reid_train.yaml`); DEV rival test 0.888 vs shirt colour 0.902, but 2–4× colour's true accepts at 1 % false accepts on hard pairs; not used by the product (`records/reid_pilot_v1_2026-11-18`). Colour + re-ID combined (DEV eval only, `tools/dev/cue_fusion.py`): pooled TAR @ 0.5 % FAR on hard pairs 0.172 vs re-ID 0.146 / colour 0.061, but below colour on FRESH_05 (0.717 vs 1.000) → not promising under the decision rule (`records/reid_fusion_dev_2026-11-18`). |
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
- Target identity: `target.tap_vouches_doubt` (`spana.target` 2.2.0, reason USER_VOUCHED) lets a tap answer on a
  doubt-only track for that episode; built and validated, off by default (no action saving on DEV,
  `records/doubt_resolution_2026-11-18`). `target.resume_after_doubt_resolved` (`spana.target` 2.3.0): a resolved
  no-rival doubt no longer taints the track. With doubt link + (a): unaided 0.509 → 0.547 (1 wrong frame), reviewer
  0.783 / 6.27 actions with the vouch or 0.753 / 5.92 without; pass bar not met, off
  (`records/resume_after_doubt_2026-11-18`).
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
- Tracking vs Roboflow ByteTrack, DEV (`records/bytetrack_compare_2026-11-18`): on the same detections ByteTrack has
  less than half our fragments / ID switches and longer tracks, but all 14 of its identity transfers are silent; a
  follower on it shows the wrong player 6.3 % of the time (8.1 % with the after-match reviewer) vs ours 0.03 % /
  0.06 %. Not adopted; its continuity (low-score second association, motion-only lost pool) is worth porting with
  our uncertainty flags.
- Tracking continuity, DEV (`records/continuity_attribution_2026-11-18`): of the 76 fragments ByteTrack bridged, 44
  are identity-gate refusals; 25 of those are the colour veto on a continuing track's own next box (IoU 0.46–0.99,
  similarity 0.36–0.70). The doubt link removes 12 fragments with 0 silent transfers, but only 3 of 24 doubts
  resolve (the frozen colour model never catches up) and the follower cannot use a doubted track even after a tap:
  reviewer correct-time 0.778 → 0.698, actions 6.27 → 5.50 per min. Shipped off; needs (a) a no-rival doubt
  resolution rule, or (b) a user tap counting as evidence on a doubted track, or (c) stay off.
- Config schema is 2.4.0 on this branch; 2.3.0 is used by the unmerged `tracking/exit-hold-v1` /
  `identity/target-render-v1` and by Nathan's local inference / profiling keys: reconcile when merging.

## Hardcoded values / magic numbers found
- `spana/tracking/camera.py:40` cut-detection HSV histogram bins `[32, 32]` (`camera.cut_hist_distance` depends on it).
- `spana/tracking/tracker.py:626` confirmation needs `observed_frames >= 2` in addition to `tracking.confirm_s`.
- `spana/evaluation.py:216` `dup_ioa=0.6` default; `spana/evaluation.py:41` `HEIGHT_BINS_PX` (reporting only).
- `tools/benchmark.py:23` `IGNORE_IOA = 0.5` (scoring convention); `tools/roboflow_ball_sheet.py:22` `CROP_FRAC = 0.09`.
- Numerical guards only (`_EPS = 1e-9`, `_INVALID = 1e6`): not thresholds.
- No player/team/clip/frame-specific logic found in `spana/`. Match ids appear only in `config/roboflow.yaml`
  (licence allow-list) and DEV clip names only in config comments.

## Next task (suggested)
1. Long-play capability (`records/window_f05_2026-11-18`, 10-minute DEV window of match G2; re-measured with the
   extent link in `records/extent_link_window_2026-11-18`).
   - Following one player unattended lasts one track; tracks end mostly at camera pans (LEFT_FRAME) and in crowds
     (LOST_EXPIRED), so the user acts about every few seconds on broadcast footage.
   - Review actions on the DEV clips by cause (`records/review_min_gap_2026-11-18` reviewer run, 81 actions):
     track lost 24 (13 offer confirmations), identity uncertain at a crossing 20, left the picture 18, occluded 12.
   - Priorities:
     (a) re-identification across pans / re-entries and at crossings. Built for testing (Nathan's option 1): re-entry
         OFFERS by re-ID v2 (`target.reentry_offer`, `spana.cli embed`, `spana.target` 2.4.0;
         `records/reentry_reid_offers_2026-11-18`): at 0.75 / 0.10 offers right 16, wrong 1, undecided 2 on DEV;
         reviewer 0.776 / 2 wrong / 5.57 actions per min (no offers: 0.778 / 2 / 5.64). Nathan tests in the app.
         Automatic re-entry links ruled out (`records/reid_reentry_eval_2026-11-18`): 79 labelled re-entries, best
         match right 78 %; at ~40–50 % coverage about 1 pick in 10 is wrong; 0-wrong settings cover < 20 % with a
         >= 18 % bound. Crossing resolution by re-ID likewise not built. Next for re-ID: more TRAIN identities
         (TRAIN_M003) and/or team classification. Two-way (mutual) best match now on (`target.reentry_mutual`,
         margin 0; `records/reentry_position_mutual_2026-11-18`): offers 16 right / 0 wrong / 1 undecided (was
         16 / 1 / 2), offline 33 / 2 (was 28 / 3); the camera-carried position bound changed nothing (off).
         Position is now a SCORE (`target.reentry_position_weight` 0.5; `records/reentry_position_score_2026-11-18`,
         `spana.target` 2.5.0 with a reentry_log): offline rank-1 wrong 17 -> 13 (6 short-gap fixes, 2 new), offer
         rule right 33 -> 36 / wrong 2; in the follower neutral (offers 15 / 0 / 1 vs 16 / 0 / 1; RE-SCORED
         2026-11-19 by who the offered box is, `records/offer_verdict_2026-11-19`: 15 right / 1 wrong, reviewer
         0.777 / 2 / 5.78 vs 0.778 / 2 / 5.64 without offers; the wrong one was made while the player was off
         screen, found by Nathan in the FRESH_08 review videos). Pan / long
         returns untouched by design. Option 2 (better signatures, `records/reid_signatures_v2_2026-11-19`, branch
         identity/reid-signatures-v2, not on main): quality-weighted crop sets (`spana.track_embeddings` 1.1.0)
         matched best-pair (`target.reentry_signature: set`, floor 0.65): by who the offered box is, 18 right /
         3 wrong (all wrong offers made while our player was off screen). Not for main as is.
         Wait-for-evidence + withdraw built but off (withdrew right offers, reviewer 0.762). A
         re-entry OFFER costs the reviewer one action, like a tap: it eases the action but does not lower actions
         per minute; only re-ID that acts without the user (auto-link at a strict margin, or crossing resolution)
         would, and both touch the 2026-11-16 rule that relinks need the user. Nathan to choose;
     (b) tracker: keep tracks alive across pans, fewer LOST_EXPIRED in crowds. The exit hold was tried
         (`records/exit_hold_2026-11-18`): no net gain without a stronger appearance signal; off by default.
2. Relink v1 accepted as built (Nathan, 2026-11-16). Watch confirmed tracks that named several predecessors.
3. Tracking-error history on the 4 DEV clips (for reference):
   Remaining tracking errors (F07 38 / F08 51 fragments): fewer fragments mean fewer gaps and fewer review actions.
   ByteTrack shows how much continuity is possible on the same detections (F07 11 / F08 22 fragments) and what it
   costs without flags (silent swaps) (`records/bytetrack_compare_2026-11-18`). Causes attributed
   (`records/continuity_attribution_2026-11-18`): identity gate 44, other 17 (height-ratio gate 8, motion gate 7),
   low-score 7, LOST timeout 4, M2 nested 4. The colour-veto part: doubt link + (a) self-consistent resolution +
   (b) tap vouch measured (`records/doubt_resolution_2026-11-18`): 106 fragments, 0 silent, reviewer 0.783 / 2 wrong,
   but 6.27 actions per min (= today): a doubt costs one tap like a fragment did. Resume after a no-rival doubt
   (`records/resume_after_doubt_2026-11-18`): unaided 0.547, reviewer 0.783 / 6.27 or 0.753 / 5.92; pass bar not
   met, all off (Nathan may relax the bar to "no more actions"). Doubt path closed for now. Gate / low-score
   breaks diagnosed (`records/gate_break_diagnosis_2026-11-18`): the 15 gate breaks are partial-body extent
   changes (box shrinks to 0.59–0.67 behind an occluder, or grows 1.13–1.57× on re-emerging, so the centre gate
   fails); this overlaps the deferred M2 decision. 4 low-score breaks are short-gap re-associations S4 refuses
   because of the low score. Proposed, not built:
   - A, extent-change continuation (15; needs Nathan, certain or flagged?);
   - B, low-score short-gap recovery (4).
   Both built and measured (`records/extent_and_low_score_2026-11-18`; `tracking.extent_change_link`,
   `tracking.low_score_recovery`): fragments 120 → 115 / 116 / 114 (A / B / both), silent 0, but flagged
   transfers 7 / 7 / 8 (knock-on crossings, not on linked tracks) and reviewer correct 0.768 / 0.769 / 0.761. Pass
   bar not met: both OFF. After the code review (`826d41f`: gap links recorded as recoveries with S4's motion rule;
   the linked box replaces the Kalman extent) A v2: fragments 110, reviewer 0.779 / 2 wrong / 6.06 actions, unaided
   0.532, silent 0, but 7 transfers (the same flagged knock-on FRESH_07 crossing). Bar revised by Nathan to "silent
   0 and no transfer on a linked track": **A is ON by default** (`c427922`; B off). 10-minute DEV window
   (`records/extent_link_window_2026-11-18`): 1,804 -> 1,747 track starts (−3 %), uncertainty after a link 7.0 % vs
   8.0 % baseline, uncertain share flat; wrong frames outside the labelled 26 s unmeasurable.
   Review list (`target.review_min_gap_s` 1.0, `records/review_min_gap_2026-11-18`): reviewer 0.778 correct,
   2 wrong, 5.64 actions per followed min (was 6.06 listing every gap; 6.27 before the extent link). Only 6 of 87
   actions were on gaps < 1 s: the review effort is in long gaps (track ends at pans, crowds, re-entries).
   Across these attempts fewer fragments barely reduced review effort: longer tracks collect uncertainty and the
   reviewer still spends about one action per gap. The remaining lever is item 1 (a).
