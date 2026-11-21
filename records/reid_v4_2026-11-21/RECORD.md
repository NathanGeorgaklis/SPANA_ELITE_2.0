# Re-ID v4: teammate-only loss, leave-one-TRAIN-match-out (2026-11-21) — FAILS the gate, not used

Scope (Nathan's CODE TASK, approved as proposed):
- **Model:** v4 = v3's data (TRAIN M001 + M002 + M003, frozen labels; M003 v1 brought onto this branch from
  `origin/data/reid-train-m003-v1`, byte-identical) and v3's settings, plus **one new loss term**: a batch-hard
  triplet whose negatives are only same-kit teammates (same match and side, side from the TRAIN labels), weight
  `train.teammate_triplet_weight` 1.0 in `config/reid_train.yaml` 1.1.0 (a prior equal to `triplet_weight`, not
  tuned). The existing PK sampler (`same_team_share` 0.5) is kept.
- **Evaluation:** leave one TRAIN match out, 3 folds (`tools/reid_heldout.py`); final model on all 3 matches; its
  margin is the fold-pooled one, labelled **"borrowed"** (fitted on fold models, never measured on the final model).
- **Gate:** the 2026-11-19 floored-margin check on DEV clean rival tests (`tools/dev/reid_calibrate.py`,
  `config/reid_calibration.yaml` 1.2.0, unchanged), which v3 failed.
- **DEV:** report only, on exactly v2's tests (rival pairs replayed from `cue_eval_p2.json`). EVAL not used. No
  tracker run, no full-match run. `config/default.yaml` (v2, `reid_resolve` off) unchanged. Run on Nathan's Mac (M3,
  MPS), 12 epochs per model, 17–22 min each.

## Gate: the floored clean check (DEV, re-ID alone, 1,417 clean tests) — **FAIL**

| | v2 | v3 | **v4** |
|---|---|---|---|
| clean margin (own fit, 4 clips) | 0.199 | 0.120 | 0.139 |
| all-tests margin (floor) | 0.481 | 0.476 | 0.531 |
| final margin = max | 0.481 | 0.476 | **0.531** |
| in sample: resolved / wrong | — | 613 / 0 | 511 / 0 (bound 0.58 %) |
| **held out, pooled: resolved / wrong, bound** | 452 / 0, 0.66 % | 286 / 0, 1.04 % | **274 / 0, 1.09 % → FAIL** (≤ 0.5 %) |

Held out per clip (fit on the other 3): FRESH_05 0.531 → 33 / 38, 0 wrong; FRESH_07 **no floor (fail closed)** →
0 / 509; FRESH_02 0.566 → 155 / 486, 0 wrong; FRESH_08 0.531 → 86 / 384, 0 wrong.
All-tests calibration (not the gate): final 0.531, held out 661 / 1, bound 0.72 % (v2 1,036 / 1, 0.46 %).

**Outcome: v4 is recorded like v3 and stops here. It is not used anywhere** (no registry entry in `models/`, no
config change, weights only on the Mac, hashes in HASHES.txt). As with v3, the failure is lack of evidence, not a
confident error: 0 wrong on clean tests at the floored margins, but fewer resolutions than v3.

## TRAIN folds (each match held out, model trained on the other two)

| held-out match | crops / players | rank-1 | mAP | AUC same vs teammate | AUC same vs other side | teammate rival acc (clean) |
|---|---|---|---|---|---|---|
| M001 | 10,139 / 25 | 0.770 | 0.402 | 0.700 | 0.907 | 0.881 (0.896) |
| M002 | 5,968 / 21 | 0.889 | 0.557 | 0.791 | 0.979 | 0.944 (0.945) |
| M003 | 3,884 / 19 | 0.876 | 0.499 | **0.553** | 0.880 | 0.926 (0.928) |

**M003 fold vs v2** (v2 = M001 + M002, so M003 is held out for both; same crops, same protocol):

| on M003 | rank-1 | mAP | AUC same vs teammate | AUC vs other side | median same / teammate | rival acc (clean) |
|---|---|---|---|---|---|---|
| v2 | 0.878 | 0.502 | **0.566** | 0.888 | 0.713 / 0.683 | 0.932 (0.934) |
| v4 fold | 0.876 | 0.499 | **0.553** | 0.880 | 0.718 / 0.693 | 0.926 (0.928) |

The only difference between these two models is the teammate term, and **on the unseen match it does not separate
teammates better** (0.553 vs 0.566; all six numbers equal or slightly worse). On M003 same-player and teammate
similarities overlap almost completely (medians 0.718 / 0.693) for both models.

**Fold calibration ("borrowed" margin):** pooled out-of-fold teammate rival tests, floored rule: clean 0.813, all
tests 0.853 → **0.853**; in-sample 720 / 87,986 clean resolved (0.8 %), 0 wrong. Leave one fold out: M001 at 0.303
(fitted on M002 + M003) resolves 20,833 with **392 wrong (bound 2.0 %)**; M002 and M003 at 0.853 resolve 172 / 0.
Pooled 21,005 / 392, bound 2.03 % → fails. A margin fitted on two matches does not transfer to the third.

## DEV report (same tests as v2; report only)

**Same player vs same-look teammate AUC** (teammate proxy: torso-colour similarity ≥ 0.70, as the 2026-11-20 record):

| | method of the 0.70 (`distributions.py`, first 2,000 same-look pairs) | all pairs (`dev_auc.py`) |
|---|---|---|
| v2 | 0.696 (recorded 0.70) | 0.758 (F05 0.717, F07 0.762, F02 0.823, F08 0.758) |
| v4 | **0.689** | 0.771 (F05 0.740, F07 0.807, F02 0.837, F08 0.724) |

Neither method shows a material gain; FRESH_08 gets worse.

**Clean same-look rival tests per clip** (query of A vs references of A and B, all three crops clean):

| clip | tests | v2 @ 0.517 right / wrong | v4 @ 0.517 right / wrong | v4 @ borrowed 0.853 right / wrong | v4 median margin (v2) |
|---|---|---|---|---|---|
| FRESH_05 | 0 | — | — | — | — |
| FRESH_07 | 52 | 0 / 0 | 5 / 0 | 0 / 0 | 0.270 (0.191) |
| FRESH_02 | 129 | 50 / 0 | 35 / 0 | 3 / 0 | 0.437 (0.458) |
| FRESH_08 | 14 | 0 / 0 | 0 / 0 | 0 / 0 | 0.088 (0.121) |
| pooled | 195 | 50 / 0 | 40 / 0 | 3 / 0 | 0.383 (0.346) |

At its borrowed margin v4 resolves 3 of 195 same-look tests. Different-look clean tests at 0.853: 65 / 1,222 right,
0 wrong.

**A-or-B rival accuracy (all 4,488 tests):** v4 1.000 / 0.951 / 0.952 / 0.902 (F05 / F07 / F02 / F08), pooled
0.934; v2 0.921, v3 0.933.

## Findings
1. **The teammate-only loss did not make re-ID separate same-kit teammates.** On the held-out TRAIN match the v4 fold
   is level with or slightly below v2 (teammate AUC 0.553 vs 0.566). On DEV the same-look AUC is unchanged (0.689 vs
   0.696 by the 0.70 method), and FRESH_07 / 08 still resolve 5 / 0 of 66 clean same-look tests at 0.517.
2. **The gate fails for the same reason as v3:** 0 wrong, but too few held-out resolutions (274; ≥ 598 needed at
   0 wrong), and FRESH_07 held out still has no all-tests floor.
3. **TRAIN-fold calibration is not usable as a margin source:** M001 held out gets 0.303 from the other two folds and
   is 1.9 % wrong; the pooled "borrowed" margin (0.853) is so high it resolves almost nothing on DEV.
4. Training holdout checks (closed set, same players) were 0.975–0.989; the gap to the held-out-match numbers is the
   usual one: one model learns its training players, not same-kit players in general.

## Method notes
- Replay check: v2 re-scored through the replay path reproduces its 2026-11-19 DEV tables as multisets (FRESH_07's
  rival tests come out in a different order, from set iteration); the re-ID-alone and colour calibrations are
  identical, the combined one differs by 1 ulp.
- TRAIN rival tests: B = every same-side teammate; references are the latest crops of A and B ≥ 1.0 s before the
  query; clean = all three crops overlap no other labelled box above 0.05 (TRAIN crops carry no occlusion flag).
- Scripts: `run_all.sh` (training), `post.sh` (evaluation), `dev_tables.sh` (DEV replay), `dev_auc.py` (all-pairs
  AUC). Outputs: `*.summary.json` (fold reports without per-test arrays), `calibration_train_folds.json`,
  `calibration_dev_v4.json`, `dist_*.json`.
