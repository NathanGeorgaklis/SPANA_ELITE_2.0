# Shirt colour + re-ID combined: DEV comparison (2026-11-18), report only

Scope:
- Does combining the colour histogram (`hist`) with the SPANA re-ID pilot (`spana`) beat either alone at strict
  settings?
- DEV only (FRESH_02 / 05 / 07 / 08), same crops, pairs, rival tests and tracks as `records/reid_pilot_v1_2026-11-18`.
- No retraining: the pilot ONNX (`682d0a09…`) is unchanged. Nothing in `spana/` changed and no threshold entered
  the product. EVAL not used. Run on Nathan's Mac.

## Method
- **Combined option:** `tools/dev/cue_fusion.py`, via `tools/dev/appearance_cue_eval.py --fusion=config/reid_fusion.yaml`.
- **Normalisation:** each cue's score is z-normalised. The mean and spread come from every score of that cue in the
  *other three* matches; they are unlabelled, so no labels are used.
- **Rule:** combined = w · z(colour) + (1 − w) · z(re-ID).
- **Weight w, leave-one-match-out:** for match X, w is chosen on the other three matches from the grid 0, 0.1, …, 1.
  The objective is pooled TAR at 0.5 % FAR on hard pairs. Ties go to the w nearest the fixed 0.5.
  - Chosen w: FRESH_05 0.2, FRESH_07 0.2, FRESH_02 0.2, FRESH_08 0.4.
  - Match X is never used to fit or normalise itself (tested).
  - A secondary option `hist+spana@fixed` (w = 0.5, no fitting) is reported for context only.
- **Missing score:** abstain. The pair is never accepted and a rival test counts as not correct; never filled in.
  - Result: 0 abstentions in 96,733 test items. Every labelled crop had both scores.
- **Metrics:**
  - A-or-B (rival) accuracy.
  - TAR at FAR 1 % and 0.5 %: the share of same-player pairs accepted at the threshold that lets that share of
    different-player pairs through. Reported on hard pairs and on all pairs.
  - **Pooled:** one threshold over all four matches. A product margin would be one number too.
- **CIs:** 95 % bootstrap, 1,000 resamples, seed 0.
  - Items are resampled, stratified by match. The combined-minus-better-single difference is bootstrapped paired.
  - Items share crops and come from few player pairs, so these CIs are **optimistic**. This matters most where one
    player pair dominates (FRESH_05).
- **Adjacent check:** with fusion on, the main report is byte-identical to the pilot's (`eb4e54c8…`). The colour and
  re-ID numbers below are exactly those of `records/reid_pilot_v1_2026-11-18`.

## Results: TAR at 0.5 % FAR, hard pairs (the decision metric)

| | colour | re-ID | combined (LOMO w) | combined − better single (paired 95 % CI) | combined, w = 0.5 |
|---|---|---|---|---|---|
| FRESH_05 (53 rival tests, 1 hard player pair) | **1.000** [0.999, 1.000] | 0.519 [0.387, 0.640] | 0.717 [0.580, 0.810] | −0.283 [−0.420, −0.190] vs colour | 0.943 [0.921, 0.971] |
| FRESH_07 | 0.057 | 0.171 | **0.186** [0.156, 0.224] | +0.015 [−0.017, +0.036] | 0.160 |
| FRESH_02 | 0.030 | 0.170 | **0.177** [0.148, 0.209] | +0.007 [−0.004, +0.030] | 0.183 |
| FRESH_08 | 0.038 | 0.120 | **0.152** [0.133, 0.179] | +0.032 [+0.009, +0.061] | 0.142 |
| **pooled** | 0.061 [0.055, 0.069] | 0.146 [0.133, 0.163] | **0.172** [0.152, 0.190] | **+0.026 [+0.008, +0.039]** vs re-ID | 0.169 |

Other measures, pooled (95 % CI):

| | colour | re-ID | combined | combined, w = 0.5 |
|---|---|---|---|---|
| A-or-B accuracy | 0.902 [0.894, 0.911] | 0.888 [0.879, 0.896] | 0.918 [0.909, 0.925] | 0.932 [0.926, 0.939] |
| TAR @ 1 % FAR, hard | 0.099 | 0.191 | 0.234 [0.219, 0.248] | 0.235 |
| TAR @ 1 % FAR, all pairs | 0.080 | 0.198 | 0.212 [0.199, 0.225] | 0.235 |
| TAR @ 0.5 % FAR, all pairs | 0.049 | 0.137 | 0.148 [0.133, 0.164] | 0.175 |
| abstention rate | 0 | 0 | 0 | 0 |

- **A-or-B accuracy per match (colour / re-ID / combined):**
  - FRESH_05 1.000 / 0.962 / 0.962;
  - FRESH_07 0.914 / 0.909 / 0.938;
  - FRESH_02 0.889 / 0.867 / 0.898;
  - FRESH_08 0.899 / 0.883 / 0.913.
- **Partly blocked (query crop occluded), pooled, 550 rival tests:**
  - A-or-B accuracy: colour 0.849, re-ID 0.784, combined 0.864.
  - TAR @ 0.5 % FAR, pairs with an occluded crop: 0.014 / 0.034 / 0.032. Every cue is weak here.
- **Colour-hard player pairs** (colour rival accuracy < 0.85 on the pair, `report.colour_hard_pair_acc`):
  - FRESH_07: 10|12, 14|15, 5|6.
  - FRESH_02: 12|16, 13|14, 2|3.
  - FRESH_08: 10|13, 10|8, 11|9, 13|16, 13|17, 16|17.
  - FRESH_05: none.
  - Pooled over these 1,258 rival tests: colour 0.735, re-ID 0.816, combined 0.831.
  - FRESH_07 5|6: colour 0.697, re-ID 0.844, combined 0.861. This is a reported subset, not tuned for.
  - FRESH_08 10|8 stays hard for every cue (0.653 / 0.663 / 0.673).
  - Per-pair numbers for all options: `fusion_summary.json` → `rival_pair_accuracy`.

## Verdict (decision rule; nothing adopted)
**Not promising** under the agreed rule.
- **Pooled condition: met.** Combined TAR @ 0.5 % FAR on hard pairs is 0.172, above colour 0.061 and re-ID 0.146.
  The paired difference to re-ID is +0.026, CI [+0.008, +0.039].
- **Per-match condition: failed on FRESH_05.** Colour is perfect there (1.000). The combined option's CI upper
  bound 0.810 is below it.
  - The fixed w = 0.5 variant would fail too (CI upper 0.971 < 1.000).
  - FRESH_05's hard set is a single player pair (12|7) with 53 rival tests. The item-level CI cannot express how
    little one pair says, but the rule is applied as written.
- On the other three matches the combined option is never worse than the better single cue. Only FRESH_08's gain
  excludes zero.

## Findings
1. **Combining helps on average at strict settings, but not on every match.** Where colour is already decisive
   (FRESH_05), a combination dominated by the weaker re-ID score loses accepts. The weight fitted on other matches
   (0.2 colour) suits the matches where colour is weak.
2. **A per-match weight would need a signal that is available at run time**, for example how distinct the two kits
   or players' colours are. That is a design question, not a tuning knob, and is not tested here.
3. **The A-or-B test improves with the combination on 3 of 4 matches** (pooled 0.918, 0.932 at w = 0.5). Shirt
   colour alone stays the best single cue on that test.
4. **Re-ID is not ready for the product on this evidence.** That matches the decision rule. The pilot was trained on
   one match; the next lever is more TRAIN matches (M002 / M003), then this same comparison again.

## Files
- `fusion_summary.json`: all options, matches, subsets, CIs, weights, abstentions, per-pair accuracy and the verdict.
- `reid_fusion.config_snapshot.yaml`: the config used (= `config/reid_fusion.yaml` at 88a9632).
- `HASHES.txt`: model, inputs and full outputs on Nathan's Mac.
