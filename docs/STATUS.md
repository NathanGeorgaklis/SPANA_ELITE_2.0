# STATUS

As of 2026-11-22: main 94b1451. Full suite on Nathan's Mac at 94b1451: 708 passed, 13 skipped.

## Next tasks (2026-11-22)
1. **Ball: scoring tool BUILT (branch `evaluation/ball-score-v1`, not merged); next: score DEV on the Mac.**
   - DEV ball labels v1 frozen in `data/ball_labels/<clip>/v1` (Nathan, 0.1 s, 131 frames each):
     FRESH_02 129 seen / 2 hidden; FRESH_05 130 / 1; FRESH_07 124 / 7; FRESH_08 54 seen / 3 hidden / 74 out.
     No UNSURE, no other balls.
   - Score by CENTRE distance, not box IoU: FRESH_05 boxes are drawn ~1.5x the ball (centres within 1-4 px of the
     ball view on a spot check), so label box size is not truth for diameter.
   - Scorer built 2026-11-22: `tools/ball_score.py` + `config/ball_score.yaml` (hit radius 0.01 x frame height,
     set before any scoring; `records/ball_score_v1_2026-11-22`). Report only: coverage, wrong answers with the 95 %
     bound (OBSERVED / PREDICTED / all), hit distances, wrong frames. Tested on synthetic outputs and on a perfect
     tracker over the real frozen labels (coverage 1, 0 wrong on all 4 clips). NO real ball output scored yet.
   - Next (Mac): `detect --backend local --ball-out` + `spana.cli ball` on each DEV clip, covering every labelled
     frame (stride 1, or a stride dividing the label step), then
     `python -m tools.ball_score --clip data/ball_labels/<clip>/v1 <clip>.ball.json ... --out ball_score_dev.json`.
     Then set a pass bar and fit the `ball` priors on DEV (separate task).
   - FIRST DEV SCORE 2026-11-23 (`records/ball_score_dev_2026-11-23`): pooled coverage seen 0.64, wrong OBSERVED
     32 / 310 (bound 0.136); FRESH_05 / 07 good (0 / 2 wrong OBSERVED), FRESH_02 / 08 bad. Cause: sporadically
     detected false objects (heads / boots / spots) get confirmed and kept alive; they are seen on 25-37 % of their
     answered frames vs 63-94 % for the real ball. Ball output NOT fit for touches until fixed.
   - Fix built 2026-11-23 on `ball/seen-share-v1` (Nathan chose the seen-share rule; pass bar recorded first in
     DECISIONS 2026-11-23: 0 wrong OBSERVED pooled, FRESH_05 / 07 seen coverage down <= 0.05). Config 2.19.0
     (`ball.min_seen_share` 0.5 over `seen_share_window_s` 1.0), `spana.ball` 1.1.0. MEASURED 2026-11-23: FAILS the
     bar (5 wrong OBSERVED pooled, was 32; all wrong answers 122 -> 23; coverage F05 0.938 / F07 0.758 pass; F02
     0.27 -> 0.16). Not adopted. Next: diagnose the 5 remaining wrong frames (`records/ball_seen_share_2026-11-23`).
   - Player-aware step 1 built 2026-11-23 on `ball/player-aware-v1` (Nathan approved the ball layer reading player
     tracks; pass bar in DECISIONS 2026-11-23, recorded first): `spana.cli ball --tracks`, config 2.20.0
     (`ball.feet_zone_h` 0.20, `ball.reacquire_max_d` 3.0), `spana.ball` 1.2.0. MEASURED 2026-11-23: FAILS the bar
     (wrong OBSERVED pooled 5 -> 2, both FRESH_02; FRESH_07 coverage 0.734 < 0.740; F07 / F08 now 0 wrong OBSERVED).
     Not adopted (`records/ball_player_aware_2026-11-23`). Step 2 ("with player X" memory) not built yet.
   - Step 1 NARROWED 2026-11-23 (DECISIONS; last tracker-tuning round on these clips): block only on-body AND far
     rejoins, and starts on a body. MEASURED 2026-11-23: FAILS the bar on FRESH_02 only (3 wrong OBSERVED, all
     FRESH_02; FRESH_05 / 07 / 08: 0 wrong of 234, coverage 0.938 / 0.758 / 0.333). Pooled wrong OBSERVED 32 -> 3,
     all wrong answers 122 -> 19, seen coverage 0.636 -> 0.586. FRESH_08 coverage drop (0.463 -> 0.333) not
     diagnosed. Tracker tuning on these clips STOPPED. FRESH_02 = detector limit (a false object at a player's feet
     scoring like its small ball): detector data, not tracker rules. Nothing merged; merge decision with Nathan.
   - BALL PATH v1 built 2026-11-23 on `ball/path-v1` (Nathan's design decision: decide afterwards from both ends,
     never guess; DESIGN + pass bar committed before code, `records/ball_path_v1_2026-11-23`). Two passes:
     `spana.cli ball` (now `spana.ball` 1.3.0 with every sighting) then `spana.cli ball-path` (`spana.ball_path`
     1.0.0, config 2.21.0 `ball_path`): only stretches >= 0.5 s seen on >= 50 % of their frames are the ball, split at
     holes > 0.4 s and at far rejoins (amendment 1); overlapping stretches = CONFLICT; gaps recorded from both ends;
     no predicted positions. Replaces the seen-share rule (off) and the "with player X" memory (not built).
     MEASURED 2026-11-23 on the 4 clips: PASSES (0 wrong SEEN on each: 0/36, 0/123, 0/99, 0/60 on FRESH_08's
     every-frame labels; pooled 0/318, bound 0.94 %; coverage 0.279 / 0.946 / 0.798 / 0.370, at or above every earlier
     version). Reviewed twice, all findings fixed (code frozen at 3ab9829). UNSEEN CLIP (FRESH_03, labelled every
     1/30 s after the freeze, scored once, 2026-11-23): FAILS the bar, 4 wrong SEEN of 314 (1.3 %, bound 2.9 %),
     coverage 0.847. The four design clips overstated reliability. Settings not changed. Diagnosed: 3 of the 4 are the ball
     (label OUT marked too early; labels v2 frozen with the reason), 1 is a real near-rejoin error on the last frame.
     Re-score on v2: 1 / 314 (bound 1.5 %), coverage 0.839; weaker evidence than the v1 score (correction prompted by
     the disagreements). All 55 misses on the clip are detector misses. "0 wrong" is not met on unseen footage; a
     replacement bar (error budget on unseen clips; strict bar at event level) is NOT set yet (Nathan). MERGED to
     main 2026-11-23 as an improvement, not as meeting its bar (DECISIONS).
   - BALL BAR from 2026-11-23 (Nathan): 95 % upper bound on wrong SEEN <= 2 %, on unseen clips only, pooled. The path
     is on the edge of it (FRESH_03: 2.9 % on labels v1, 1.5 % on v2); more unseen clips needed.
   - BALL TILING built 2026-11-23 on `detection/ball-tiling-v1` (config 2.22.0, OFF by default): ball-only tiled
     detector pass, two frame-height tiles on 16:9, ~3x detection time. MEASURED 2026-11-23: FAILS the adoption
     rule, stays OFF. Pooled coverage 0.687 -> 0.706 (needed 0.737); FRESH_05 / 07 / 03 gain 0.02-0.06 with no new
     wrong answer, but FRESH_02 gets 42 wrong SEEN of 62 (coverage 0.279 -> 0.155) and FRESH_08 17 of 83. It also
     shows the ball path does not stop a steadily detected false object: its 0 wrong on the design clips depended on
     the untiled detector's false positives being sparse (`records/ball_tiling_v1_2026-11-23`). Not merged. Next
     lever: detector data (small balls, hard negatives), not resolution.
   - TILING FALSE-POSITIVE DIAGNOSIS (CODE TASK 2026-11-23, branch `ball/tiling-fp-diagnostic`, analysis only):
     `tools/diagnostics/ball_tiling_fp.py` + `config/ball_fp_diag.yaml` built and tested (what the tiled detector
     mistook for the ball on FRESH_02 / FRESH_08: source pass, tile overlap, new vs untiled, persistence, contact
     sheets, generic categories). RUN 2026-11-24 (`docs/BALL_TILING_FP_FINDINGS.md`): NOT a tiling / merge bug (146
     accepted false detections, all from one tile pass, none merged, none in the overlap, none at a seam, all
     reproduced); all 146 are NEW with tiling; 5 false objects on FRESH_02 and 1 on FRESH_08 lasting 0.6-2.8 s at
     density 0.63-0.97, not separable from the real ball by duration, density, size or score. The path's "brief
     false alarm" assumption fails for them. Categories (Nathan, from the contact sheets): equipment/cone, ONE physical
     object in 5 stretches / 129 frames across nearly all of FRESH_02 (a persistent false object), and
     crowd/background, 1 object / 17 frames (FRESH_08). Hard negatives to mine from TRAIN
     footage only: cones first, then crowd/background. No pipeline change; tiling stays off; not merged.
   - DETECTOR RETRAINING PLAN written 2026-11-24 (`docs/DETECTOR_BALL_RETRAINING_PLAN.md`, branch
     `detection/ball-retraining-plan`): a PROPOSAL, nothing built or decided. TRAIN footage only: more small-ball
     examples (mined from the path's holes / gaps), hard negatives (the tiled detector's own false ball detections on
     TRAIN footage; look-alikes left unboxed), and training at the scales used at inference; same recipe, runs v2a /
     v2b; checks on valid, DEV bench, the 5 DEV ball clips (untiled and tiled), the diagnosed objects, and an unseen
     clip against the 2 % ball bar. APPROVED 2026-11-24 with one change: ONLY THE MATCH BALL is boxed (spare balls
     get no box; `has-spare-ball` tag; version 2 audited for boxed spare balls first). Adoption rule recorded in
     DECISIONS before any work. Budget ~520 frames; a further TRAIN match is available if needed.
   - RETRAINING TASK 1 DONE 2026-11-24 (`records/detector_retrain_task1_2026-11-24`): `tools/detector_mine.py`
     (TRAIN-only miner: missed balls and tiled-only objects from video windows; spare-ball audit; scale report).
     Scale check: training is 1280 x 720 at 1.0x, ball 9 / 13 / 23 px; FRESH_02's ball reaches the detector at ~8 px
     (below the trained range), FRESH_08's at ~11.5 px (inside it: size does not explain FRESH_08). Spare-ball audit:
     31 ball boxes in 30 frames of version 2 reviewed by Nathan 2026-11-25: 30 match ball, 1 spare ball (TRAIN_M001
     f0020508, valid, second box; to be removed in task 3; the match ball in that frame is already boxed). NEXT: task 2 on Nathan's Mac (audit sheets, mining run on the
     three TRAIN videos, yield report). Nothing labelled or trained.
   - RETRAINING TASK 2 MINED AND REVIEWED 2026-11-28 (`records/detector_retrain_task2_2026-11-25`): the three TRAIN
     videos are 10-minute clips (3000-3600 s); the full broadcasts are no longer on Nathan's Mac. Mined every 30 s
     (`config/detector_mine.yaml`): 232 objects, 493 pictures, all reviewed by Nathan per picture. Match ball 348,
     nothing there 49, crowd/background 43, shin/sock 19, equipment/cone 14, boot/foot 11, spare ball 6, head 2,
     unclear 1. Only crowd/background reaches `min_per_category` 40; equipment/cone (the FRESH_02 failure) is THIN.
     AT THE PLAN'S STOP POINT: Nathan decides about more TRAIN footage. The answers were checked
     against the mined pictures (`detector_mine review-apply`; M001 uses `review_M001_v2.txt`). Nathan
     re-downloaded the three full TRAIN broadcasts (next: mine them) and offered two more matches, both DEV-only by
     their permissions (`records/dev_permissions/DEV_PERMISSIONS_2026-11-28.md`): Game 2 again, and a NEW DEV match,
     DEV_G4_MACARTHUR_RAMS_WESTERN_RANGERS (allow-listed as dev; unused, so a source of unseen clips for task 6). Nothing labelled or trained.
     Next lever for coverage: tiling for the football detector. Touches v1 (unmerged) reads `spana.ball` OBSERVED: it must be switched to the path's SEEN frames, and
     its `possession` section becomes config 2.22.0 on merge.
   - CORRECTION 2026-11-23: FRESH_08 re-scored on Nathan's every-frame labels (ball labels v2, 391 frames; on his
     Mac, not yet pushed): 2 wrong OBSERVED of 59 (frames 361-362, ball out of the picture), not 0; coverage 0.352;
     77 of 162 visible frames have no answer. The every-3rd-frame labels missed the 2 errors, so FRESH_05 / 07's
     "0 wrong" may hide the same.
     If touches v1 merges after this, its `possession` section becomes config 2.21.0.
2. **Possession / touches v1: review and merge decision.**
   - The work is on `origin/possession/touches-v1` (160e1b0, not merged; another session). It has QA on scripted
     scenes only; no DEV touch labels exist.
   - It claims config 2.18.0, which main now uses for the `team` section. On merge it must take the next free
     version (2.19.0), the same rule as the ball and team merges.
   - It needs scored ball tracks first (task 1) to mean anything on DEV.
3. **Team classification v1.1: Nathan's coverage review.**
   - On main, answers TEAM_A / TEAM_B / UNKNOWN only. DEV held out 0 wrong of 2,052, bound 0.146 %.
   - Coverage: FRESH_02 0.67, FRESH_05 0.84, FRESH_07 0.69, FRESH_08 0.37.
   - Nothing consumes `spana.teams` yet. Linking's team input stays unwired until decided.
   - Roles stay off until pitch context exists (`records/team_score_v11_2026-11-22`).
4. **Target identity: parked** at the reviewer baseline of 0.879 correct, 0 wrong, 6.48 actions / min, until re-ID
   separates same-kit teammates.
   - `tracking.appearance_doubt_link` stays off, and transitive predecessors are not built (DECISIONS 2026-11-22).
   - The remaining effort is contact track changes, a player-tracking matter with no generic fix found
     (`records/contact_continuity_diag_2026-11-22`).
5. **Re-ID:** v2 stays, and the adoption rule is unchanged (all tests held out). The lever is more TRAIN identities
   for same-kit separation.
6. **Housekeeping:**
   - `origin/infra/modal-reid-train` 7dd0286 (a docs note: run Modal with `--detach`) is not merged. Merge it or
     drop it.
   - `~/SPANA_ELITE_2.0` needs `git pull` (its local `main` is behind origin).

## Log (entries since 2026-11-20; the 2026-11-20 snapshot first, newer entries below it)

Snapshot as of 2026-11-20 (branch identity/reid-merge-v1 = main 74eb1a0 + the re-ID resolution work; PR #13; main also has
the memoised re-entry rule, records/reentry_cache_2026-11-20, and records/fresh08_followup_2026-11-20).
- Full suite: `python -m pytest tests -q` → 539 passed, 5 skipped (container, Python 3.13, with torch + transformers for the re-ID tests).
- Full suite on Nathan's Mac, branch tracking/tracker-bench-v1-gtalink (2026-11-21, project venv): 604 passed, 12 skipped; bench venv: harness + summary tests 15 passed.
- Full suite on Nathan's Mac, branch tracking/tracker-bench-v1-botsort (2026-11-20): 546 passed, 6 skipped; the BoT-SORT bench tests run end to end only with `trackers` on PYTHONPATH (8 passed).
- Target / re-ID / guard tests (53) also passed on Nathan's Mac.
- Tracklet linking v1 merged to main (2026-11-20, Nathan's Mac): `tests/test_tracklet_linking.py` 19 passed; full suite
  565 passed, 6 skipped.
- Transfer flag rule merged to main (2026-11-20, Nathan's Mac; full suite 572 passed, 6 skipped). Silent-transfer counts
  below that predate it use the OLD rule; under the merged rule our tracker has 3 silent of 7 (see Next task 0).
- Silent-transfer fix merged to main (2026-11-20, Nathan approved; config 2.13.0, both rules on): DEV silent
  3 -> 0, wrong frames 1 / 2 -> 0 / 0, but reviewer correct 0.783 -> 0.680 (`records/silent_transfer_fix_2026-11-20`).
  Full suite 583 passed, 6 skipped.
- Re-ID episode resolution MERGED to main (2026-11-20, Nathan approved; config 2.14.0, `reid_resolve` OFF by default,
  follower guard kept; stays off until re-ID separates same-kit teammates AND references / clean crops reach the episodes): STOP condition met. Re-ID v2 separates no same-look pair on FRESH_07 / 08 at the calibrated margin (0 of 66 clean
  rival tests; FRESH_02 50 / 129, 0 wrong); margin not lowered. On: 2 / 2 resolutions right, 0 silent, 0 / 0 wrong,
  0 transfers on linked tracks, unaided 0.508 -> 0.513, reviewer 0.680 -> 0.680 (follower guard). 112 of 126 episodes
  refused: NO_REFERENCE 49 (39 ambiguous spawns whose predecessors were never CERTAIN), NO_CLEAN_CROP 29, NOT_OBSERVED
  24, BELOW_MARGIN 7 (`records/reid_episode_resolution_2026-11-20`). Full suite 596 passed, 6 skipped.
- Re-ID v4 DEV evaluation (2026-11-21, branch `identity/reid-v4-dev-eval`; nothing adopted, config unchanged): v4
  (Modal, M001-M003, v2 / v3 loss without the teammate triplet) pinned in `models/spana_reid_v4.json`. FAILS held out:
  re-ID alone 384 resolved, bound 1.23 % (v2 1,323 / 0.36 % on the same tables); clean floored 164 / 1.81 % (v2 644 /
  0.46 %); FRESH_07 held out no margin. Rival accuracy 0.924 vs 0.915. Same-kit: clean same-look rival tests right at
  0.517 30 / 427 vs v2 62 / 427 (0 wrong). v2 stays (`records/reid_v4_dev_eval_2026-11-21`).
- Re-ID v5 = v4 design, teammate-only triplet (2026-11-21, branch `identity/reid-v5-dev-eval`; code identity/reid-v4
  on main 8343082 which now has the Modal app; nothing adopted, config unchanged): FAILS the all-tests held-out rule
  (1,725 resolved, 6 wrong, bound 0.69 % vs v2 0.36 %), PASSES the clean floored check better than v2 (794 / 0 /
  0.38 % vs 644 / 0 / 0.46 %); same-kit clean same-look tests right at 0.517 53 / 427 vs v2 62 / 427. v2 stays
  (`records/reid_v5_dev_eval_2026-11-21`). Re-ID training now runs on Modal from main (`tools/modal_reid.py`).
- Ball tracker v1 MERGED to main (2026-11-22, Nathan; config 2.15.0 `ball` section, dataset config 1.8.0, `spana.ball` 1.0.0,
  `spana.cli ball`, `detect --ball-out`). Full suite in the container: 635 passed, 14 skipped, 2 failed (BoT-SORT bench
  only: `supervision` not installed, as on main). UNMEASURED on footage: every `ball` value is a prior; next = DEV
  ball-track labels, then score and fit on DEV only.
- Demo render MERGED to main (2026-11-22, Nathan): `tools/target_render.py --ball` overlay + real-time playback for
  strided runs. DEV FRESH_05 at stride 3 (local ONNX CPU): ball OBSERVED 244 / PREDICTED 15 / LOST 2 of 261 frames
  (unscored: no DEV ball labels); player 6 with simulated review 60 / 62 labelled frames correct, 0 wrong. Visual only.
- DEV ball labelling tool (2026-11-22): `tools.label_pack --kind ball` + `tools/ball_labeller.html` +
  `tools.freeze_ball_labels`, schema `spana.ball_labels` 1.0.0 (INTERFACES). `tests/test_ball_labels.py` incl. headless
  Chrome session. NO ball labels exist yet: next = Nathan labels FRESH_02/05/07/08 at 0.1 s, freeze, then the ball
  tracker scoring tool (DEV only).
- Player confirmation MERGED to main (2026-11-22, Nathan accepted range scope; config 2.16.0, spana.target 2.10.0):
  the player's IS_ME / NOT_ME on a track, with conflict rule, human-confirmed gallery, undo and logging
  (`spana/identity/confirmation.py`, `spana.cli crops`). An IS_ME answers only the confirmed frames
  (`target.confirmation_scope: range`); whole-episode scope rejected (51 wrong frames on DEV vs 0; DECISIONS
  2026-11-22). DEV, correct taps: silent 0, wrong 0 / 0, 0 transfers on linked tracks, reviewer correct 0.680 -> 0.870,
  ABSTAIN 735 -> 295, actions 4.11 -> 7.52 / min. 5 % wrong taps: 1 drawn, through (1 wrong frame, no evidence
  existed); 30 %: 8 / 22 blocked, 14 through (27 wrong frames), all contained to the tapping player. Gallery re-ID
  (off) cleared 0 episodes. Undo exact (30 / 30). `records/player_confirmation_2026-11-20`. Full suite on Nathan's Mac after the merge: 663 passed, 12 skipped.
- Re-ID adoption rule KEPT (2026-11-22, Nathan): all DEV rival tests, held out (2026-11-18); not changed to the clean
  check after v5. v2 stays, `config/default.yaml` unchanged. Evidence-only record of the leave-one-TRAIN-match-out run
  of the teammate-triplet design (`identity/reid-v4` 066650f, Mac-trained, not in `models/`) now on main
  (`records/reid_v4_2026-11-21`): floored clean DEV check held out 274 resolved / 0 wrong, bound 1.09 % -> FAIL
  (FRESH_07 no floor). That branch's training code / config / DECISIONS rows were not brought over.
- Cue eval deterministic (2026-11-22, branch `evaluation/cue-eval-deterministic`): `tools/dev/appearance_cue_eval.py`
  orders rival tests by sorted player pair, not set iteration; DEV reruns under two hash seeds give byte-identical
  tables / report / calibration, and calibration equals the v5 record (v2 exact, v5 within 2.2e-16)
  (`records/cue_eval_deterministic_2026-11-22`). Table hashes from now on differ from earlier records (new order).
- Episode references diagnosis (2026-11-22, branch `identity/episode-references-diag`, analysis only, no code):
  NO_REFERENCE 49: 79 of 100 missing references belong to predecessors that were themselves ambiguous spawns, never
  CERTAIN (circular chain); 17 CERTAIN but never clean, 4 never confirmed. Following chains to their roots would make
  17 / 43 spawn episodes evaluable (true player among candidates 16 / 17). NO_CLEAN_CROP 29: real contact (IoU
  mostly > 0.2); the uncertain track is observed a median 5 of 59 frames while its player continues on the rival or
  a new spawn: first failing layer player tracking. Proposed (awaiting Nathan): transitive predecessors for
  ambiguous spawns, off by default (`records/episode_references_diag_2026-11-22`).
- Stretch confirmation / reviewer policy (2026-11-22, Nathan: option 1; no product change): the simulated reviewer
  stretches a confirmation over unobserved frames, never over a two-player box, and skips stretches < 2 labelled frames
  (`config/reviewer_policy.yaml`). New DEV baseline 0.879 correct, 0 wrong, 6.48 actions / min (was 0.870 / 0 / 7.52;
  5 % wrong taps: 2 wrong frames, own taps). 4.1 / min not reachable: the effort is 56 track changes (player tracking).
  `records/stretch_confirm_2026-11-22`.
- Contact-continuity diagnosis (2026-11-22, branch `tracking/contact-continuity-diag`, analysis only): 103 labelled
  track changes on DEV. First failing gate: appearance veto 36 (motion passed, clean box, stale colour model of the
  old track), TAKEN_BY_OTHER 15, hidden-track refusal 10, motion / extent gates 19, low-score box 7, detector miss 4,
  old track on another player's box 11. The existing `tracking.appearance_doubt_link` (off) measured: changes 103 ->
  100, 0 silent, 0 wrong, +1 flagged transfer, reviewer 0.879 -> 0.885 at 6.55 / min. Recommendation: no tracking
  task now (`records/contact_continuity_diag_2026-11-22`). Decision (Nathan): the doubt link stays OFF; merged to
  main as a record. Target identity parked at 0.879 / 0 wrong until re-ID separates same-kit teammates.
- Team classification v1 design (2026-11-22, branch `team/classification-v1-design`, no code): see the layer row;
  scoring on DEV needs a new frozen side label per FRESH GT id (`records/design/TEAM_CLASSIFICATION_V1_DESIGN.md`).
- Team classification v1 built (2026-11-22, branch `team/classification-v1`): `spana.teams` 1.0.0, config 2.17.0;
  unscored until the DEV side labels exist (done: labels frozen, scored and merged; see below).
- DEV team side labels (2026-11-22, branch `evaluation/team-labels-dev`): contact sheets made for the 73 FRESH GT ids
  (`tools/team_label_sheet.py`, `config/team_labels.yaml`; sheets and `<clip>.sides.json` templates in
  `~/SPANA_team_labels/`, not in the repo). Labelled by Nathan and FROZEN 2026-11-22 to
  `data/labels/<clip>/teams_v1/`: 73 ids. G1: A 9, B 8, REF 1. G2: A 6, B 5, GK_A 1. G3 (FRESH_07 + FRESH_08, one A / B):
  A 19, B 19, GK_B 1, REF 4. In G3, A = light-blue kit and B = red. The existing GT v1 is untouched.
- Team classification v1 scored (2026-11-22, branch `evaluation/team-score-v1`, priors, nothing tuned or adopted):
  **FAILS acceptance on DEV.** Pooled held out: 16 wrong of 2,194 answered frames, bound 1.11 % > 0.5 %. All 16 are
  GOALKEEPER on one red player's track (FRESH_08 T000049: colour 0.48 from its team, detector "goalkeeper" 70 %). No
  TEAM_A / TEAM_B swap on any DEV frame. Coverage: FRESH_02 0.72, FRESH_05 0.89, FRESH_07 0.69, FRESH_08 0.46.
  Roles: 3 of 7 ids answered right, 0 role ids wrong; the G3 referees stay UNKNOWN (no detector hint). TRAIN check
  (team sides only): M002 / M003 0 wrong; M001 686 / 11,861 wrong (clusters overlap, separation 0.44). Options for
  Nathan in `records/team_score_dev_2026-11-22`.
- Team classification v1.1 (2026-11-22, Nathan: option 3 + option 2): roles OFF (`team.roles_enabled`), match
  abstains when the kit clusters overlap (`team.min_separation_radius_ratio` 1.0, stated prior); config 2.18.0,
  `spana.teams` 1.1.0. DEV held out: 0 / 2,052, bound 0.146 %: PASS. No DEV team answer changed. TRAIN_M001
  abstains (CLUSTERS_OVERLAP); M002 / M003 0 wrong. Merged to main with the label and score branches
  (`records/team_score_v11_2026-11-22`).
DEV numbers below come from `records/`; nothing here has been scored on EVAL.

| Layer | State | Module path | Tests | Notes |
|---|---|---|---|---|
| player detection | PARTIAL | `spana/detection/` (`onnx_backend.py`, `postprocess.py`, `tiling.py`, `roboflow_backend.py`, `base.py`) | `tests/test_detection_camera.py`, `tests/test_merge_detections.py`, `tests/test_multi_body.py`, `tests/test_roboflow_dataset.py`, `tests/test_detector_eval.py`, `tests/test_detection_interface.py`, `tests/test_dev_bench.py` | Football detector v1 trained (`nathans-workspace-9brdc/spana-elite-2-rfdetr-medium-t1`, RF-DETR medium, dataset version 2), adapter `RoboflowDetector.from_detector_config` (config `detector`, confidence / iou 0.50 untuned). Scored only on the M001 valid split (table below), which shares a broadcaster with training: generalisation unknown. DEV benchmark uploaded and pre-labelled (76 frames, 2 DEV matches), waiting for Nathan's label review; not scored. Pipeline CLI still uses pinned COCO YOLOX-s, person only (`models/yolox_s.json`); FRESH_02 recall 0.971 (NESTED_V1). YOLOX inference test skipped without weights. |
| player tracking | PARTIAL | `spana/tracking/` (`tracker.py`, `kalman.py`, `camera.py`), `spana/appearance.py` | `tests/test_tracker.py`, `tests/test_camera_uncertainty.py`, `tests/test_identity_uncertain.py`, `tests/test_unconfirmed_low_score.py`, `tests/test_cli_e2e.py`, `tests/test_foundation.py`, `tests/test_silent_transfer_fix.py` | Generic multi-player tracker with camera compensation and per-track IDENTITY_UNCERTAIN (`spana.tracks` 2.2.0: + `identity_doubt`). Silent-transfer fix ON (merged 2026-11-20): ambiguous spawns start uncertain and nested crossings of similar-looking tracks mark both uncertain; DEV silent 3 -> 0, wrong 0 / 0, reviewer correct 0.783 -> 0.680 (`records/silent_transfer_fix_2026-11-20`). DEV 4 clips (local detector): 120 fragments, 121 ID switches, 6 transfers, 0 silent (NESTED_V2). Fragmentation remains the largest error. Stale tracks past their recovery window no longer block S3c / S4
(`tracking.rival_eligible_only` ON, `records/rival_eligible_2026-11-19`): fragments 110 -> 105, ID switches 112 -> 107,
same 7 flagged transfers, 0 silent, reviewer 0.778 / 2 / 5.64 -> 0.783 / 2 / 5.50 (FRESH_08 player 10 no longer split at 99 / 104; still split at 148 by the colour veto: doubt link re-measured, still fails
the bar; 10-minute window check passed, `records/fresh08_followup_2026-11-20`).
Root cause (`records/veto_root_cause_2026-11-20`, analysis only): an appearance learned from motion-blurred crops during a fast
pan, frozen 2.5 s in a crowd; across DEV 9 of 22 wrong own-box vetoes had pan-learned models, and lowering the threshold
is unsafe (2 right vetoes at 0.62 / 0.67). The two-way re-entry check prevented 6 wrong offers per 1 right withheld. Where ByteTrack kept one track (76 of 120 fragments) ours broke mostly at the identity gate (44; 25 of them the torso-colour veto on the track's own next box), then low-score (7), LOST timeout (4), M2 nested (4), other (17) (`records/continuity_attribution_2026-11-18`). Appearance doubt link (`tracking.appearance_doubt_link`, S3b): continues such tracks flagged IDENTITY_UNCERTAIN: fragments 120 → 108, silent transfers 0, but reviewer correct-time 0.778 → 0.698 because doubts rarely resolve and the follower abstains on them. Self-consistent doubt resolution (`tracking.doubt_resolve_self_consistent`, option (a)): 5 more doubts cleared, fragments 106. With the follower's tap vouch (b): 0.783 correct, 2 wrong, but actions 6.27 / min = today, so the pass bar (fewer actions) is not met; **all off by default** (`records/doubt_resolution_2026-11-18`). Nested-box fix (M2) deferred pending Nathan's approval. Tests also `tests/test_appearance_doubt_link.py`. |
| tracklet linking (tracking post-processing) | BUILT, on main (separate stage, not in a default pipeline) | `spana/tracking/linking.py`, `config/tracking/linking.yaml` 1.1.0, `spana.cli link` | `tests/test_tracklet_linking.py` | Offline stage after the tracker, same `spana.tracks` contract + link log (`records/tracklet_linking_v1_2026-11-20`). Uncertainty / rival appearance clearance OFF (`allow_uncertain_clearance`, decision 2026-11-20; with it on, FRESH_02 player 8 was linked to teammate 10). DEV 4 clips, clearance off: fragments 105 -> 85, ID switches 107 -> 88, 0 transfers on linked tracks, 0 silent, wrong-player time unchanged (1 / 2 frames, same frames), correct time unaided 0.535 -> 0.606, reviewer 0.783 -> 0.785 at 5.50 -> 4.18 actions per min. |
| target identity | PARTIAL | `spana/identity/follower.py`, `spana/identity/confirmation.py` | `tests/test_target_follow.py`, `tests/test_target_relink.py`, `tests/test_player_confirmation.py` | Player confirmations (merged 2026-11-22, spana.target 2.10.0): range-scoped IS_ME / NOT_ME, conflict rule, gallery, undo; DEV reviewer 0.870 / 0 wrong (`records/player_confirmation_2026-11-20`). Follows one tapped track (selection `track_at_frame` 1.0.0) into `spana.target` 2.0.0 (TARGET / CANDIDATE / ABSTAIN / LOST). Fails closed (v1 rules). Relink v1: after a loss, proposes CANDIDATE on the tracker's unique predecessor link (gap ≤ `target.relink_max_gap_s` 0.5); TARGET only after the user confirms (`confirm_candidate` 1.0.0) or with `relink_auto_confirm` (off). DEV, 4 clips (`records/target_relink_v1_2026-11-16`): correct-time 0.509 without confirmation (= v1, wrong 1 frame) → **0.597 with a simulated user (wrong 2 frames: +1 from a tracker identity change after a correct confirmation)**; auto-confirm 0.597, wrong 3. Ceiling 0.853. Recovery inputs (`spana.target` 2.1.0): confirm / reject offers, tap to re-select, gap list. **Simulated after-match reviewer: correct-time 0.779, wrong 2 frames (the two known ones), but about 6.3 user actions per followed player-minute** (`records/target_recovery_v1_2026-11-17`). Long re-entries unsolved. Re-ID model: pilot trained on TRAIN_M001 only (`tools/reid_crops.py`, `tools/reid_train.py`, `config/reid_train.yaml`); DEV rival test 0.888 vs shirt colour 0.902, but 2–4× colour's true accepts at 1 % false accepts on hard pairs; not used by the product (`records/reid_pilot_v1_2026-11-18`). Colour + re-ID combined (DEV eval only, `tools/dev/cue_fusion.py`): pooled TAR @ 0.5 % FAR on hard pairs 0.172 vs re-ID 0.146 / colour 0.061, but below colour on FRESH_05 (0.717 vs 1.000) → not promising under the decision rule (`records/reid_fusion_dev_2026-11-18`). |
| ball tracking | PARTIAL (v1 built, UNMEASURED) | `spana/ball/tracker.py`, `spana/detection/ball_view.py` | `tests/test_ball_tracker.py` | v1 2026-11-19: ball view of the football detector from the same detector pass (`detect --ball-out`, `detector.ball_classes`), ball tracker over several hypotheses with camera compensation, speed-bound gate, confirmation (3 obs / 0.30 s), fail-closed answer OBSERVED / PREDICTED / LOST / AMBIGUOUS (`spana.ball` 1.0.0, `spana.cli ball`, decision log). All `ball` config values are priors: **no DEV ball-track labels exist, so nothing is measured**. Synthetic tests only (pan, cut, kick, hidden ball, static false positive, two balls). |
| team classification | BUILT (v1.1: TEAM_A / TEAM_B / UNKNOWN; passes DEV acceptance) | `spana/teams/classifier.py`, `spana.teams` 1.1.0 (`validate_teams`), CLI `teams`, config `team` 2.18.0 (`roles_enabled: false`, `min_separation_radius_ratio: 1.0`) | `tests/test_teams.py`, `tests/test_team_labels.py`, `tests/test_team_score.py` | DEV held out (`records/team_score_v11_2026-11-22`): 0 wrong of 2,052 answered, bound 0.146 %, no role answers. Coverage FRESH_02 0.67, FRESH_05 0.84, FRESH_07 0.69, FRESH_08 0.37. Roles off until pitch context exists; referees / keepers are UNKNOWN. TRAIN: M001 abstains (clusters overlap), M002 / M003 0 wrong. No consumer yet. |
| pitch context | NOT BUILT | none | none | Grass mask in `spana/dataset/prefilter.py` filters pre-labels only; no pitch model / homography. |
| possession/touches | NOT BUILT | none | none | |
| event segmentation | NOT BUILT | none | none | |
| event classification | NOT BUILT | none | none | |
| outcomes | NOT BUILT | none | none | |
| statistics | NOT BUILT | none | none | |
| clips | NOT BUILT | none | none | |

Supporting (not pipeline layers): `spana/video.py`, `spana/provenance.py`, `spana/artifacts.py`, `spana/config.py`,
`spana/evaluation.py` + `tools/benchmark.py` (scoring vs frozen labels), `spana/labels.py`, `spana/cli.py`.
Re-ID training on Modal (cloud GPU): `tools/modal_reid.py` + `tools/modal_reid_plan.py`, config
`config/modal_reid.yaml` (A10G, 120 min per call, training stops at 105 and resumes on rerun), volume `spana-reid`;
auto-deployed by `.github/workflows/modal-deploy.yml` on main (secrets MODAL_TOKEN_ID / MODAL_TOKEN_SECRET set by
Nathan 2026-11-20). Not yet run on real crops; `tests/test_modal_reid.py`.

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

## Tracker bench v1, A-D harness (`bench/`, `config/tracking/`; run on DEV 2026-11-21, `records/tracker_bench_v1_harness_2026-11-21`)
- Candidates on the same frozen DEV detections: REF (ours), A ByteTrack and B BoT-SORT + CMC (Roboflow `trackers`
  2.7.0, Apache-2.0), C Deep-EIoU (no licence: EVAL-ONLY, benchmark use only), D GTA-Link offline linking on B and on
  C (MIT code). C / D use SPANA re-ID v2 embeddings. Licences in DECISIONS 2026-11-20. Bench venv only
  (`bench/requirements-bench.txt`); tests: `tests/test_bench_harness.py` (skip what needs bench packages elsewhere).
- **Recommended by the pre-registered rule: ours (REF)**, the only candidate with 0 unflagged wrong-player frames
  (73 tapped players). D on BoT-SORT: 254 (BoT-SORT alone 273; FRESH_07 / FRESH_08 unchanged), so GTA-Link does not close
  the silent-swap gap. ByteTrack 109, Deep-EIoU 152 (best IDF1 0.812, no licence). Ours has the lowest continuity
  (IDF1 0.674, coverage 0.626 vs 0.71-0.81). The BoT-SORT rejection is reproduced by this independent scorer (273 vs 272).

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
- Ball, DEV: not run on footage yet. Needs (1) a `detect --backend local --ball-out` + `ball` run on the 4 DEV clips
  on Nathan's Mac, (2) per-frame ball-centre labels on 1-2 DEV clips to score it (the 66 bench frames are too sparse
  for tracking). Camera motion is re-estimated in the ball pass (masking only ball boxes), not shared with tracking.
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
- Review video label (2026-11-21, `identity/persistent-target-label`, `records/persistent_target_label_2026-11-21`):
  `tools/target_render.py` labels the selected player `YOU` on every TARGET frame (track id as a small tag), so a
  track change at a re-entry no longer looks like lost identity. Display only. DEV before / after videos and
  re-entry verdicts NOT run (no footage in the cloud session); commands for Nathan's Mac are in the record.
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
- Tracker bench v1, DEV (`records/tracker_bench_v1_2026-11-20`): Roboflow `trackers` 2.7.0 BoT-SORT (published
  defaults, camera motion compensation on, bench-only install) on the same detections as ours / ByteTrack. Best
  continuity (fragments ours 105 / ByteTrack 49 / BoT-SORT 42; reviewer actions 5.50 / 2.23 / 1.81 per min) but
  20 identity transfers, all silent (ByteTrack 14 silent, ours 7 all flagged). Wrong-player time unaided 0.03 % /
  6.3 % / 7.9 %, with the reviewer 0.06 % / 8.2 % / 9.2 %. Not adopted; the adoption task was dropped (Nathan) and an
  uncertainty layer on BoT-SORT is not triggered, since BoT-SORT widens the silent-transfer gap.
  Cross-checked 2026-11-21 by the A-D harness (`records/tracker_bench_v1_harness_2026-11-21`): BoT-SORT 273 unflagged
  wrong frames there vs 272 here; GTA-Link on BoT-SORT 254.
- Tracking continuity, DEV (`records/continuity_attribution_2026-11-18`): of the 76 fragments ByteTrack bridged, 44
  are identity-gate refusals; 25 of those are the colour veto on a continuing track's own next box (IoU 0.46–0.99,
  similarity 0.36–0.70). The doubt link removes 12 fragments with 0 silent transfers, but only 3 of 24 doubts
  resolve (the frozen colour model never catches up) and the follower cannot use a doubted track even after a tap:
  reviewer correct-time 0.778 → 0.698, actions 6.27 → 5.50 per min. Shipped off; needs (a) a no-rival doubt
  resolution rule, or (b) a user tap counting as evidence on a doubted track, or (c) stay off.
- Config schema is 2.4.0 on this branch; 2.3.0 is used by the unmerged `tracking/exit-hold-v1` /
  `identity/target-render-v1` and by Nathan's local inference / profiling keys: reconcile when merging.

## Hardcoded values / magic numbers found
- Moved to config 2026-11-22 (`tracking/thresholds-to-config`, config 2.17.0, same values; synthetic tracker / camera
  outputs byte-identical): `camera.cut_hist_hue_bins` / `cut_hist_sat_bins` (were `[32, 32]` in
  `spana/tracking/camera.py`), `tracking.confirm_min_observed_frames` (was `observed_frames >= 2` in
  `spana/tracking/tracker.py`). Not yet re-run on DEV footage (expected identical tracks).
- `spana/evaluation.py:216` `dup_ioa=0.6` default; `spana/evaluation.py:41` `HEIGHT_BINS_PX` (reporting only).
- `tools/benchmark.py:23` `IGNORE_IOA = 0.5` (scoring convention); `tools/roboflow_ball_sheet.py:22` `CROP_FRAC = 0.09`.
- Numerical guards only (`_EPS = 1e-9`, `_INVALID = 1e6`): not thresholds.
- No player/team/clip/frame-specific logic found in `spana/`. Match ids appear only in `config/roboflow.yaml`
  (licence allow-list) and DEV clip names only in config comments.

## Earlier next-task lists (2026-11-19 and before; superseded by "Next tasks" at the top)
0. Ball v1 on DEV (2026-11-19): run it on the DEV clips, label ball centres on 1-2 DEV clips, then score and fit the
   `ball` priors on DEV only. Then team classification v1 (kit grouping per match from track history: A / B / GK /
   REF), which also gives re-ID a cue and works around the detector's kit-colour role confusion.
0. Transfer flagging fix (merged 2026-11-20; `records/benchmark_transfer_flagging_2026-11-20`). `tools/benchmark.py` now flags a transfer only if the track was
   IDENTITY_UNCERTAIN within 0.5 s (`config/benchmark.yaml`) and reports recovery- / link-caused columns. Re-score:
   our tracker 7 transfers, silent 0 -> **3** (FRESH_02 T000045 ambiguous spawn 9 -> 10; FRESH_07 T000018
   14 -> 15 -> 14, the follower's one wrong frame); same 3 in both linking runs (none link-caused); ByteTrack /
   BoT-SORT unchanged (rejection stands). Earlier "0 silent" claims superseded (DECISIONS 2026-11-20). Next: a
   tracking task for the 3 silent transfers (ambiguous spawn with no uncertain state; FRESH_07 T000018
   swap-and-return).
   DONE, merged to main 2026-11-20 (`records/silent_transfer_fix_2026-11-20`; Nathan accepted the cost). Two root
   causes:
   - Ambiguous spawns started CERTAIN. Now `tracking.ambiguous_spawn_uncertain`: they start IDENTITY_UNCERTAIN.
   - FRESH_07 was a crossing without a contest, not an assignment swap (margin 0.37). Each track gated only its own
     box, and one took a merged box holding the other's body. Now `tracking.crossing_uncertain`, containment
     >= 0.60 of an observed, similar-looking track.
   DEV: silent 3 -> 0, wrong frames 1 / 2 -> 0 / 0, h1 0 silent at 10 / 25 / 60 fps, linked 0 transfers on linked
   tracks.
   Cost: reviewer correct 0.783 -> 0.680, unaided 0.535 -> 0.508, actions 5.50 -> 4.11 / min (reviewer ABSTAIN
   frames 435 -> 735), links 21 -> 10. Recovering that time is owned by the re-ID resolution task (item 1 (c)):
   resolving same-look uncertain episodes.
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
         Exit side (`target.reentry_exit_edge` 0.05, `records/reentry_exit_side_2026-11-19`, `spana.target` 2.7.0):
         after the player left the picture only tracks born at the side they left by are offered (17 / 17 DEV
         returns within 0.011 of it). Follower: mean 15 / 0 (was 15 / 1), set 0.65 18 / 0 (was 18 / 3), no right
         offer lost; reviewer 0.777 / 2 / 5.71 (no offers: 0.778 / 5.64). Lookalikes in the picture untouched.
         Offer hold (`target.offer_hold_s`, `spana.target` 2.8.0, `records/offer_hold_2026-11-19`): ON at 0.2 s
         after re-measuring on main's new tracks (right offer frames 442 -> 450, wrong unchanged; player 10's offer
         now lasts 48-147). Merged main, DEV: re-ID offers 16 right / 0 wrong, unaided 0.536 / 1 wrong frame,
         reviewer 0.783 / 2 / 5.50.
         Wait-for-evidence + withdraw built but off (withdrew right offers, reviewer 0.762). A
         re-entry OFFER costs the reviewer one action, like a tap: it eases the action but does not lower actions
         per minute; only re-ID that acts without the user (auto-link at a strict margin, or crossing resolution)
         would, and both touch the 2026-11-16 rule that relinks need the user. Nathan to choose;
     (b) tracker: keep tracks alive across pans, fewer LOST_EXPIRED in crowds. The exit hold was tried
         (`records/exit_hold_2026-11-18`): no net gain without a stronger appearance signal; off by default.
     (c) re-ID evidence at crossings (IDENTITY_UNCERTAIN), merged onto main in `identity/reid-merge-v1`
         (config 2.12.0 `reid_resolve`, tracks 2.3.0 `resolved_by`, target 2.9.0 `resume_after_reid`; all OFF):
         - TRAIN labels: M002 v1 (107 player tracklets, 21 players; group-then-confirm, `confirmed_only`) and M003 v1
           (81 player tracklets, 19 players; direct flow). Re-ID v2 (M001 + M002) DEV rival 0.921 vs colour 0.902;
           v3 (+ M003) 0.933 (`records/reid_retrain_m001_m002_2026-11-18`, `records/reid_retrain_m001_m003_2026-11-19`).
         - Margin (`records/reid_calibration_dev_2026-11-18`): v2 alone passes held out (1 / 1,036, bound 0.46 %);
           0.517 chosen. v3 fails the floored clean check (286 / 0, bound 1.04 %; `records/reid_margin_floor_dev_2026-11-19`);
           its confident errors are crops of the wrong person (`records/reid_v3_wrong_answers_2026-11-19`). About 3
           more labelled DEV clips would let clean tests prove 0.5 %.
         - Merge DEV check (`records/reid_merge_dev_2026-11-19`): with `reid_resolve` off, tracks / follower / reviewer
           equal main exactly; on (+ `resume_after_reid`): tracks identical, 2 FRESH_02 episodes cleared by re-ID,
           unaided 0.532 -> 0.537, reviewer 0.777 -> 0.782, wrong frames 1 / 2 (same players), silent swaps 0,
           flagged transfers 7, actions 5.71 / min unchanged. Re-checked on main 24ccd58 (target 2.9.0, config 2.12.0):
           off = main; on: unaided 0.536 -> 0.540, reviewer 0.783 -> 0.787, wrong 1 / 2 (same players), silent 0,
           transfers 7, actions 5.50 unchanged. Still off by default (Nathan's decision).
         - Episode resolution (2026-11-20, merged to main, off by default,
           `records/reid_episode_resolution_2026-11-20`):
           - Built: a separate re-ID run (config 2.14.0, `reid_resolve.min_evidence_frames` 3). Colour that cannot tell
             same-look rivals apart no longer vetoes; ambiguous spawns are cleared by one predecessor beating the
             others by the margin. Every evidence frame / refusal is logged (`reid_evidence`). The follower resumes
             after REID only on a track attached while CERTAIN.
           - DEV on, against main 67df7ce: 0 silent, 0 / 0 wrong, 0 linked transfers; 2 REID resolutions, both right;
             unaided 0.508 -> 0.513, reviewer 0.680 -> 0.680, ABSTAIN 735 -> 735, uncertain 22 / 16 / 39 % unchanged.
           - STOP: re-ID v2 cannot separate same-kit teammates on G3 (same-player median 0.595 vs teammate proxy
             0.407; clean same-look rival tests at 0.517: F07 0 / 52, F08 0 / 14). Next: re-ID on more
             permissioned TRAIN footage; and references for spawned tracks (39 spawn episodes are blocked by
             predecessors with no trustworthy reference, which re-ID quality alone will not fix).
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
