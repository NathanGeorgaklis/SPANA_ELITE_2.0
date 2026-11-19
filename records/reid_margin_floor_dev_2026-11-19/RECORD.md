# Margin floor in the clean-test safety check; re-ID v3 re-checked on DEV (2026-11-19)

Scope:
- Nathan's CODE TASK, option (b) of `records/reid_clean_calibration_dev_2026-11-19`.
- **The clean-test check may never accept a margin below the all-tests margin:**
  margin = max(clean margin, all-tests margin).
  - The floor uses the same fitting clips and cue weights.
  - It applies to the final fit and to every held-out fit.
  - If either side is None, the result is None (nothing resolves).
- Config `reid_calibration` 1.2.0, `clean_subset.margin_floor: all_tests`; code c24cf06.
- **DEV tables reused unchanged** from the clean-calibration record. EVAL not used. No tracker runs (v3 failed, see
  below). Run on Nathan's Mac.

## Unchanged results
- The all-tests calibration of v2 and of v3 is byte-identical to before (HASHES.txt).
- The tracker's v2 margin 0.517 and its adoption basis are untouched.

## Re-ID v3 under the floored check (re-ID alone, DEV clean tests, 1,417)

| | value |
|---|---|
| clean margin (own fit, 4 clips) | 0.120 |
| all-tests margin (floor, 4 clips) | 0.476 |
| **final margin** = max | **0.476** |
| in sample: resolved / wrong / coverage | 613 / 0 / 43.3 % (bound 0.49 %) |
| **held out, pooled: resolved / wrong / coverage** | **286 / 0 / 20.2 %, bound 1.04 % → FAIL** (target ≤ 0.5 %) |
| worst wrong answer on a clean test | margin −0.220 → gap to the final margin **0.256** |

Held out per clip (fit on the other 3):

| clip | clean margin | floor | final | resolved / clean tests | wrong |
|---|---|---|---|---|---|
| FRESH_05 | 0.120 | 0.556 | 0.556 | 26 / 38 | 0 |
| FRESH_07 | 0.222 | **none** | none (fail closed) | 0 / 509 | 0 |
| FRESH_02 | 0.221 | 0.556 | 0.556 | 157 / 486 | 0 |
| FRESH_08 | 0.108 | 0.476 | 0.476 | 103 / 384 | 0 |

For comparison, v2 under the same check: final 0.481, held out 452 resolved, 0 wrong, bound 0.66 %. v2 would not
pass the clean check either. It stays adopted on its 2026-11-18 all-tests held-out pass, which the floor does not
change.

## Decision outcome
- **v3 fails, so v2 stays:** `config/default.yaml` is unchanged (v2, margin 0.517, `reid.enabled: false`). No
  end-to-end run.
- **The failure is a lack of evidence, not an error:** 0 wrong anywhere at the floored margins. The bound needs
  enough confident resolutions:
  - ≥ 598 with 0 wrong;
  - ≥ 949 with 1 wrong;
  - ≥ 1,259 with 2 wrong.

## Option (c): how many more labelled DEV clips
- **Need:** at least 598 − 286 = **312 more held-out resolutions**, with 0 wrong.
- **Yield of a new clip:**
  - Clips like FRESH_02 / 08 give about 435 clean tests each (by labelled duration and crowding), and about 30 % of
    them resolve held out at the floored margin (260 / 870): **about 130 per clip**.
- **Estimate:**
  - **About 3 more clips of FRESH_02 / 08 length (about 10 s, 18–22 labelled players), at 0 wrong.**
  - About 2 if the added clips also give FRESH_07 an all-tests floor when it is held out. It then contributes about
    0.3 × 509 ≈ 150.
  - **About 6** if one clean error appears (949 needed).
- **Assumptions:** this is an estimate from the current clips' rates. Clips with many contested rival pairs give more
  rival tests than quiet ones.
