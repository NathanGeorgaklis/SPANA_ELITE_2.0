# STATUS

As of 2026-11-20 (branch identity/reid-merge-v1 = main 74eb1a0 + the re-ID resolution work; PR #13; main also has
the memoised re-entry rule, records/reentry_cache_2026-11-20, and records/fresh08_followup_2026-11-20).
- Ball tracker v1 MERGED to main (2026-11-22, Nathan; config 2.15.0 `ball` section, dataset config 1.8.0, `spana.ball` 1.0.0,
  `spana.cli ball`, `detect --ball-out`). Full suite in the container: 635 passed, 14 skipped, 2 failed (BoT-SORT bench
  only: `supervision` not installed, as on main). UNMEASURED on footage: every `ball` value is a prior; next = DEV
  ball-track labels, then score and fit on DEV only.
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
- Re-ID v5 = v4 design, teammate-only triplet (2026-11-21, branch `identity/reid-v5-dev-eval`; code identity/reid-v4
  on main 8343082 which now has the Modal app; nothing adopted, config unchanged): FAILS the all-tests held-out rule
  (1,725 resolved, 6 wrong, bound 0.69 % vs v2 0.36 %), PASSES the clean floored check better than v2 (794 / 0 /
  0.38 % vs 644 / 0 / 0.46 %); same-kit clean same-look tests right at 0.517 53 / 427 vs v2 62 / 427. v2 stays
  (`records/reid_v5_dev_eval_2026-11-21`). Re-ID training now runs on Modal from main (`tools/modal_reid.py`).
