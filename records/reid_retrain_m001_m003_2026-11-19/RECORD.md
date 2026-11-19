# Re-ID model v3: retrained on TRAIN_M001 + M002 + M003 (2026-11-19)

Scope:
- TRAIN_M003 labels v1 frozen from Nathan's direct labelling.
- Crops cut; model retrained with the v2 settings, unchanged (`config/reid_train.yaml`).
- The same DEV comparison and margin calibration as v2 (`records/reid_retrain_m001_m002_2026-11-18`,
  `records/reid_calibration_dev_2026-11-18`), with the same tools and configs.
- Product unchanged: `config/default.yaml` still points at v2, margin 0.517. EVAL not used. All runs on Nathan's Mac.

## TRAIN_M003 labels v1
- **Review:** the 8 s pack (180 tracklets, 26 tracker-flagged excluded), rebuilt with re-ID v2 (59 clusters), labelled
  with the direct flow. All 154 queued tracklets answered.
- **Labels:** 81 player tracklets of 19 players, 15,396 boxes; 39 rejected; 34 unsure (+ 26 excluded).
- **Frozen:** `data/reid/TRAIN/TRAIN_M003_SYDNEY_OLYMPIC/v1` (f2c3218, branch data/reid-train-m003-v1, pushed from a
  scratch clone on the Mac), with `tools/freeze_reid_labels.py` at 41fbb75, video hash checked.

## Training
- **Data:** M001 + M002 crops unchanged; M003 3,884 crops (19 players).
  - 65 identities, 16,233 train / 3,758 holdout crops.
- **Run:** 12 epochs, 29 min on the M3 (MPS).
- **Holdout rival check (closed set):** 0.978, the same as v2 (0.978); this check is saturated.
- **ONNX export:** max difference from PyTorch 1.1e-6.

## DEV: A-or-B (rival) accuracy

| clip | tests | colour | re-ID v2 | **re-ID v3** | colour + re-ID v3 |
|---|---|---|---|---|---|
| FRESH_05 | 53 | 1.000 | 1.000 | **1.000** | 1.000 |
| FRESH_07 | 1,496 | 0.914 | 0.930 | **0.947** | 0.961 |
| FRESH_02 | 1,329 | 0.889 | 0.928 | **0.948** | 0.965 |
| FRESH_08 | 1,610 | 0.899 | 0.904 | **0.906** | 0.919 |
| pooled | 4,488 | 0.902 | 0.921 | **0.933** [0.927, 0.941] | **0.948** |

## DEV: TAR at 0.5 % FAR, hard pairs

| | colour | re-ID v2 | **re-ID v3** | colour + re-ID v2 | colour + re-ID v3 |
|---|---|---|---|---|---|
| FRESH_05 | 1.000 | 0.991 (1 % FAR) | 0.910 | 0.997 | 0.999 |
| FRESH_07 | 0.057 | 0.165 | **0.219** | 0.143 | 0.212 |
| FRESH_02 | 0.030 | 0.164 | **0.213** | 0.200 | 0.260 |
| FRESH_08 | 0.038 | 0.097 | **0.168** | 0.151 | 0.198 |
| pooled | 0.061 | 0.133 | **0.226** [0.201, 0.244] | 0.207 | **0.263** |

The fusion rule still says "promising" (no clip below the better single cue).

## DEV: margin calibration (target <= 0.5 % wrong resolutions, one-sided 95 % bound; held out = fit on the other 3 clips)

| | re-ID v2 | re-ID v3 |
|---|---|---|
| margin fitted on all 4 clips | 0.481 | 0.476 |
| in-sample: resolved / wrong of 4,488 | 1,154 / 1 (bound 0.41 %) | **1,413 / 2** (bound 0.44 %) |
| **held out, pooled: resolved / wrong** | 1,036 / 1, **bound 0.46 % (passes)** | 718 / 1, **bound 0.66 % (fails)** |
| held out, FRESH_07 | 456 resolved, 1 wrong (margin 0.481) | **no margin found** from the other 3 clips (0 resolved) |
| held out, FRESH_08 | 245 / 0 (margin 0.517) | 377 / 1 (margin 0.476) |
| colour + re-ID v3, held out | (v2: fails) | 2,356 / 11, bound 0.77 % (fails) |

## Findings
1. **v3 is the better model on every measure of accuracy.** Pooled rival accuracy rose to 0.933 (from 0.921). Strict
   TAR on hard pairs rose to 0.226 (from 0.133), and on every clip except FRESH_05, which rests on one player pair.
   A third labelled match helped again: identities, not capacity, are still the limit.
2. **v3 does not pass the safety calibration held out, so it is not swapped into the tracker.**
   - With FRESH_07 held out, the other three clips find no margin that meets the bound. The cause is a single
     high-margin wrong answer on FRESH_08: one error in ~400 resolutions is already above the 0.5 % target at 95 %.
   - Pooled held-out: 1 wrong in 718, bound 0.66 %.
   - v2's held-out pass (1 wrong in 1,036) was not much stronger. With about 4,500 DEV tests, the bound sits close
     to the target for both models; the verdict turns on one or two errors.
3. **The tracker setting (`reid` in `config/default.yaml`) is unchanged:** v2, margin 0.517, off by default. v3 is
   kept as a candidate.

## Next (proposal)
- More DEV rival tests (more labelled DEV clips), so the bound can separate the models instead of one error
  deciding.
- Or look at FRESH_08's high-margin wrong answer (which player pair, what the crops show) before deciding.
- More TRAIN matches still look worth it for accuracy.
