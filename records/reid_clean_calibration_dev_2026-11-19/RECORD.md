# Re-ID margin calibration on clean DEV rival tests, v2 and v3 (2026-11-19)

Scope:
- Nathan's go-ahead after `records/reid_v3_wrong_answers_2026-11-19`.
- **Tool change (8a66eed):**
  - DEV tables 1.1.0 record, for each rival test, whether any of its 3 crops is labelled occluded, and the largest
    IoU of any of them with another labelled player.
  - `reid_calibrate` (config 1.1.0) repeats the calibration on the clean tests, using
    `appearance.clean_max_iou` from `config/default.yaml` (0.05, the tracker's own rule).
- Same DEV clips, labels and tracks as before. EVAL not used. Run on Nathan's Mac. Product unchanged.
- **Sanity check:** the all-tests results reproduce `records/reid_calibration_dev_2026-11-18` (v2) and
  `records/reid_retrain_m001_m003_2026-11-19` (v3) exactly.
- **Caveat:** "clean" here uses overlap with *labelled players*; the tracker uses overlap with its *tracks*. Close,
  not identical.

## Clean tests
1,417 of 4,488 tests are clean: FRESH_05 38/53, FRESH_07 509/1,496, FRESH_02 486/1,329, FRESH_08 384/1,610.

## Calibration rule (unchanged), re-ID alone

| | v2 all | v3 all | **v2 clean** | **v3 clean** |
|---|---|---|---|---|
| margin fitted on all 4 clips | 0.481 | 0.476 | 0.199 | 0.120 |
| held out, pooled: resolved / wrong | 1,036 / 1 | 718 / 1 | 1,137 / 6 | 1,224 / 3 |
| held-out bound (≤ 0.5 % passes) | **0.46 % pass** | 0.66 % fail | 1.04 % fail | 0.63 % fail |
| FRESH_08 held out: margin fitted on the other 3 clips | 0.517 | 0.476 | 0.146 → 6 wrong | 0.108 → 3 wrong |

- Colour alone and colour + re-ID fail on clean tests too: held-out bounds 5.8 % and 1.7–1.8 %.

## Clean tests at fixed margins (re-ID alone, all 4 clips)

| margin | v2 resolved / wrong | v3 resolved / wrong |
|---|---|---|
| 0.517 (tracker today) | 424 / 0 | 517 / 0 |
| 0.476 | 511 / 0 | 613 / 0 (bound 0.49 %) |
| 0.40 | 667 / 0 | 774 / 0 |
| 0.30 | 909 / 0 | 1,002 / 0 |

- **Most confident wrong answer on a clean test:**
  - v2: −0.205 (FRESH_08, players 16 / 17);
  - v3: −0.220 (FRESH_08, players 16 / 17).

## Findings
1. **On clean crops neither model makes a confident mistake.** No clean test is answered wrong at any margin above
   0.22. The tracker's margin (0.517) and v3's (0.476) are more than twice that. All the confident errors in the
   earlier calibrations came from occluded or overlapped crops (`records/reid_v3_wrong_answers_2026-11-19`).
2. **The held-out failures on clean tests come from the calibration rule, not from the models.**
   - The rule picks the *lowest* margin at which the bound still holds.
   - On clips with no clean errors, nothing stops that scan, so it reaches 0.11–0.22. FRESH_08's few low-margin
     errors then count against it.
   - At the margins the tracker actually uses, the same clean tests give 0 wrong.
3. **The DEV set is too small to prove ≤ 0.5 % on clean tests at the tracker's margins.**
   - With 0 wrong, the bound needs about 600 resolutions.
   - Only v3 at 0.476 gets there (613 resolved, bound 0.49 %). That margin was fitted on the same clips (all-tests
     calibration), so it is not a held-out result.
4. **Under the decided rule (held-out pass, DECISIONS 2026-11-18), nothing changes:** the tracker keeps v2 at
   0.517, off.
   - Switching to v3 needs either Nathan's decision on the evidence above or a rule change.

## Options (Nathan to choose)
- **(a)** Switch the tracker's re-ID to v3 at margin 0.476 (still off by default) and re-run the end-to-end DEV check
  (`records/reid_resolve_dev_2026-11-18`).
  - Justification: 0 wrong on 613 clean resolutions, worst clean error −0.22.
  - Not a held-out pass: a DECISIONS change.
- **(b)** Change the held-out rule so the fitted margin cannot slide below where the fitting clips have evidence.
  For example, a floor equal to the all-tests margin. This is a DECISIONS change.
- **(c)** Keep v2; label more DEV clips so the clean set can prove it.
