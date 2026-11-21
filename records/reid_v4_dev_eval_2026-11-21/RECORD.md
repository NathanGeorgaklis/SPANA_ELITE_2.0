# Re-ID v4 vs v2 on DEV: same-kit rival test and held-out margin calibration (2026-11-21)

CODE TASK (Nathan), target identity. Branch `identity/reid-v4-dev-eval`, base main `d34082f`. DEV FRESH_02 / 05 /
07 / 08 frozen labels only; no EVAL; nothing adopted; `config/default.yaml` unchanged (v2, margin 0.517,
`reid_resolve.enabled: false`). Run on Nathan's Mac (`run_eval.sh`, outputs `~/spana-reid-v4-eval/out`, HASHES.txt).

**Outcome: v4 FAILS held out; v2 stays.** v4 ranks rivals slightly better without a threshold (pooled rival accuracy
0.924 vs v2 0.915), but at the strict operating point (<= 0.5 % wrong, exact 95 % bound, leave one clip out) it
resolves far fewer tests and its bound misses the target: re-ID alone 384 held-out resolutions / bound 1.23 % (v2
1,323 / 0.36 %); clean floored check 164 / 1.81 % (v2 644 / 0.46 %). It does not separate same-kit teammates better
than v2 (clean same-look rival tests resolved at 0.517: 30 / 427 vs v2 62 / 427, 0 wrong for both).

## Model
- `models/spana_reid_v4.json`: ONNX sha256 `2df55a9099e786f03f4a647566ad623ef1c121c96ee4e8a41be6474952a25a30`
  (`~/SPANA_modal/models/reid_v4.onnx`; copied to `models/spana_reid_v4/spana_reid.onnx`, git-ignored;
  `load_pinned` accepts it). The task text gave a 63-character string (one "4" missing at character 22); the
  64-character hash of the file, equal to the Modal export log, is pinned.
- Training: Modal run `runs/reid_v4`, `config/reid_train.yaml` 1.0.0 (sha256 d1f49554...), TRAIN M001 + M002 + M003,
  12 epochs (resumed after epoch 1). This is the v2 / v3 loss, NOT the teammate triplet of the v4 design (Nathan's
  option (a), 2026-11-21): in effect a re-run of the v3 recipe.

## Inputs (identical for both models)
- `tools/dev/appearance_cue_eval.py --dump-tables` on the 4 clips, with the main tracker's current tracks and
  decision logs (`~/spana-player-confirmation-run/base`) for the uncertain-pair annotations.
- The rival test set is larger than on 2026-11-19 (5,782 vs 4,488 tests; clean 1,917): the tables also cover the
  tracker's unresolved uncertain pairs, and today's tracks (silent-transfer fix) have more. FRESH_05 tables are
  byte-identical to `records/reid_clean_calibration_dev_2026-11-19` (v2); F02 / F07 / F08 differ by that test set.
  So both models are compared here on one set; v2's numbers below are not the ones of its 2026-11-18 adoption.
- `tools/dev/reid_calibrate.py`, `config/reid_calibration.yaml` 1.2.0 (target 0.5 %, confidence 0.95,
  `clean_subset.margin_floor: all_tests`, decision 2026-11-19).

## Held-out calibration (leave one clip out, pooled)

| | v2 | v4 |
|---|---|---|
| rival accuracy, pooled (threshold-free; colour 0.878) | 0.915 | 0.924 |
| re-ID alone, all tests: final margin | 0.469 | 0.661 |
| ... held out: resolved / wrong / bound | 1,323 / 1 / **0.36 % PASS** | 384 / 1 / **1.23 % FAIL** |
| re-ID alone, clean floored: final margin | 0.469 | 0.661 |
| ... held out: resolved / wrong / bound | 644 / 0 / **0.46 % PASS** | 164 / 0 / **1.81 % FAIL** |
| combined (colour + re-ID), all tests, held out | 2,705 / 7 / 0.49 % | 2,523 / 12 / 0.77 % FAIL |
| combined, clean floored, held out | 1,108 / 0 / 0.27 % | 1,020 / 0 / 0.29 % |
| colour alone (same for both), all tests, held out | 2,158 / 21 / 1.40 % FAIL | same |

Per held-out clip, re-ID alone, all tests (resolved / tests / wrong, margin fitted on the other 3):
- v2: F05 42 / 53 / 0 (0.469), F07 627 / 2,106 / 1 (0.482), F02 296 / 1,377 / 0 (0.517), F08 358 / 2,246 / 0 (0.517).
- v4: F05 36 / 53 / 0 (0.661), **F07 0 / 2,106 (no margin meets the target on the other 3 clips: fail closed)**,
  F02 166 / 1,377 / 1 (0.636), F08 182 / 2,246 / 0 (0.661).
- Clean floored: v2 F05 37, F07 330, F02 152, F08 125 (0 wrong); v4 F05 30, F07 0 (no floor), F02 80, F08 54.

## Same-kit teammates (`records/reid_episode_resolution_2026-11-20/distributions.py`; teammate proxy = mean
torso-colour similarity >= 0.70; clean rival tests; margin 0.517)

| | v2 | v4 |
|---|---|---|
| same player, median cosine | 0.595 | 0.616 |
| same-look different player (teammate proxy), median | 0.407 | 0.440 |
| gap (same - teammate) | 0.188 | 0.176 |
| clean same-look rival tests resolved right at 0.517 (of 427) | 62 | 30 |
| ... wrong | 0 | 0 |
| per clip right (F07 / F02 / F08; of 225 / 129 / 73) | 5 / 50 / 7 | 10 / 18 / 2 |
| median same-look rival margin, pooled | 0.249 | 0.267 |

v4 pulls same-player crops together but teammates as much; the separation the episode-resolution STOP needed
(records/reid_episode_resolution_2026-11-20) is not there.

## Remaining
- Nothing adopted; v4 is pinned only so it can be evaluated again. The v4 design's teammate loss
  (`train.teammate_triplet_weight`, on `identity/reid-v4`, uncommitted there) was not trained or tested here.
- Comparison caveat: v2's numbers on today's test set (bound 0.36 % / 0.46 %) differ from its 2026-11-18 adoption
  record because the test set changed with the tracks; v2's adoption is unchanged.
