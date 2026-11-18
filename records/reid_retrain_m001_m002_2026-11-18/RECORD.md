# Re-ID model v2: retrained on TRAIN_M001 + TRAIN_M002 (2026-11-18)

Scope:
- **TRAIN_M002 labels v1 frozen** from Nathan's cluster review, rule `confirmed_only`.
- **Crops cut;** the model retrained with the pilot's settings, unchanged (`config/reid_train.yaml`).
- **The same DEV comparisons as the pilot** (`records/reid_pilot_v1_2026-11-18`, `records/reid_fusion_dev_2026-11-18`).
- **Product unchanged.** EVAL not used. Everything ran on Nathan's Mac.

## TRAIN_M002 labels v1
- **Source:** the 8 s pack (414 tracklets), reviewed as clusters.
  - Nathan answered all 86 drawable tracklets in the audit, so no cluster label was frozen unconfirmed.
- **Player labels:** 107 player tracklets (86 audit answers + 21 single-tracklet clusters), 21 players, 24,006 boxes.
  - 195 rejected, 112 unsure (59 tracker-flagged).
- **Cluster quality (reported only):** 4 of 86 cluster labels differed from Nathan's answers: 4.7 %, exact one-sided
  95 % bound 10.3 %.
- **In git:** `data/reid/TRAIN/TRAIN_M002_EASTERN_SUBURBS/v1`, commit f785ee1 (branch data/reid-train-m002-v1, pushed
  from a scratch clone on Nathan's Mac with his approval; merged into identity/reid-retrain-v2).
  - FROZEN.json sha `1766d19d…bf41` equals the Mac copy.
  - `load_frozen(..., "train")` verifies every file hash; "tune" is refused.

## Training
- **Data:** M001 crops (unchanged) + M002 crops (5,968, 21 players): 46 identities, 13,109 train / 2,998 holdout crops.
- **Run:** 12 epochs, 17.5 min on the M3.
- **Holdout rival check** (closed-set): 0.978, vs 0.970 for the pilot.
- **Export:** ONNX; max difference from PyTorch 1.2e-6.

## DEV results (same crops, pairs, rival tests and tracks as the pilot)

**A-or-B (rival) accuracy:**

| clip | tests | colour | re-ID v1 (M001) | **re-ID v2 (M001 + M002)** | colour + re-ID v2 |
|---|---|---|---|---|---|
| FRESH_05 | 53 | 1.000 | 0.962 | **1.000** | 1.000 |
| FRESH_07 | 1,496 | 0.914 | 0.909 | **0.930** | 0.944 |
| FRESH_02 | 1,329 | 0.889 | 0.867 | **0.928** | 0.951 |
| FRESH_08 | 1,610 | 0.899 | 0.883 | **0.904** | 0.922 |
| pooled | 4,488 | 0.902 | 0.888 | **0.921** | **0.939** (0.945 at w = 0.5) |

**TAR at 0.5 % FAR, hard pairs (the decision metric of the fusion test):**

| | colour | re-ID v2 | colour + re-ID v2 | combined − better single (paired 95 % CI) |
|---|---|---|---|---|
| FRESH_05 | 1.000 | 0.991 (at 1 % FAR) | 0.997 [0.992, 1.000] | −0.003 [−0.008, 0.000] |
| FRESH_07 | 0.057 | 0.165 | 0.143 [0.124, 0.166] | −0.022 [−0.047, +0.006] |
| FRESH_02 | 0.030 | 0.164 | **0.200** | +0.036 [+0.008, +0.063] |
| FRESH_08 | 0.038 | 0.097 | **0.151** | +0.054 [+0.017, +0.088] |
| pooled | 0.061 | 0.133 | **0.207** [0.189, 0.236] | **+0.074 [+0.049, +0.098]** |

- **Fusion weights for colour**, fitted leave-one-match-out: 0.3 / 0.6 / 0.3 / 0.7 (FRESH_05 / 07 / 02 / 08).
- **Partly blocked queries** (pooled rival accuracy): colour 0.849, re-ID v2 0.847, combined 0.878.
- **Colour-hard player pairs** (pooled rival accuracy): colour 0.735, re-ID v2 0.851, combined 0.842.

## Findings
1. **One more labelled match moved re-ID past shirt colour on the A-or-B test:** pooled 0.921 vs 0.902. It is better
   on every clip, and FRESH_02 rose by 0.06. This supports the pilot's finding that it lacked identities, not
   capacity.
2. **The colour + re-ID combination is now "promising" under the agreed rule:**
   - pooled TAR at 0.5 % FAR on hard pairs is 0.207, above both single cues;
   - on no clip does the combination's CI fall below the better single cue.
   - FRESH_07 is marginal: the combination is 0.022 under re-ID alone, and its CI upper bound (0.166) only just
     reaches re-ID's 0.165.
   - Nothing is adopted: the rule only decides whether the combination is worth pursuing.
3. **Caveats:**
   - CIs resample single test items and are optimistic.
   - FRESH_05 rests on one player pair.
   - The fitted colour weight varies a lot between matches (0.3–0.7).
   - Re-ID v2's own strict-threshold TAR pooled is 0.133, not higher than v1's 0.146. The strict regime gains come
     from the combination.

## Next (for Nathan)
- Calibrate the combined score's margin on DEV for a false-resolution rate ≤ 0.5 %. Then try it in the tracker as
  extra evidence for IDENTITY_UNCERTAIN, behind a config flag, off by default.
- Success is judged end to end (silent swaps 0, wrong resolutions 0, uncertain track-frames down).
- More TRAIN matches (M003 has video and tracks ready) should keep improving re-ID alone.
