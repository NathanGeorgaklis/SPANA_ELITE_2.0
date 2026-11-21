# Re-ID v5 (the v4 design: teammate-only triplet) vs v2 on DEV (2026-11-21)

CODE TASK (Nathan), target identity. Code: `identity/reid-v4` e713f35 (teammate triplet, reid_train config 1.1.0),
rebased onto main 8343082 (the Modal app, merged from `infra/modal-reid-train` first: Nathan's choice). Evaluation
branch `identity/reid-v5-dev-eval`. DEV FRESH_02 / 05 / 07 / 08 frozen labels only; no EVAL; nothing adopted;
`config/default.yaml` unchanged. Same procedure and inputs as `records/reid_v4_dev_eval_2026-11-21` (`run_eval.sh`).

**Outcome: v5 FAILS the held-out rule v2 was adopted on; v2 stays.** Re-ID alone on all DEV rival tests, leave one
clip out: 1,725 resolved, **6 wrong**, bound **0.69 %** > 0.5 % (v2 1,323 / 1 / 0.36 %). Five of the six wrong
resolutions are FRESH_02, where the margin fitted on the other clips falls to 0.403. On the clean floored check
(decision 2026-11-19) v5 PASSES and beats v2 (794 / 0 / 0.38 % vs 644 / 0 / 0.46 %). Same-kit teammates are not
separated better: clean same-look rival tests right at 0.517 53 / 427 (v2 62 / 427), at each model's own margin
74 / 427 (v2 78 / 427), 0 wrong for both.

## Training (Modal A10G, `runs/reid_v5`, volume `spana-reid`)
- Fresh run (no checkpoint mixing with `runs/reid_v4`), crops M001 + M002 + M003 (3,758 holdout crops, as v4),
  `config/reid_train.yaml` 1.1.0 sha256 3ea10aec..., `teammate_triplet_weight` 1.0; 12 epochs in one call (10.7 min).
- TRAIN holdout rival check (closed set, not DEV): 0.884 -> 0.979 (epoch 9), 0.979 at 12 (v4 run: 0.978).
  The teammate term fell 0.419 -> 0.062.
- Export `models/reid_v5.onnx`, sha256 `1332759f155181cd74b0dc383e1c4f2e981cd91a44e77e432d1e8895cd511b4b` (volume =
  download), ONNX vs PyTorch max abs diff 8.0e-7. Pinned `models/spana_reid_v5.json` (`load_pinned` accepts it);
  weights in `models/spana_reid_v5/spana_reid.onnx`, git-ignored.

## Held-out calibration (`tools/dev/reid_calibrate.py`, `config/reid_calibration.yaml` 1.2.0; 5,782 rival tests,
clean 1,917; identical tests for both models; v4 from its record on the same tests, for reference)

| | v2 (baseline) | v5 (teammate triplet) | v4 (ref.) |
|---|---|---|---|
| rival accuracy, pooled (colour 0.878) | 0.915 | **0.928** | 0.924 |
| re-ID alone, all tests: final margin | 0.469 | 0.483 | 0.661 |
| ... held out: resolved / wrong / bound | 1,323 / 1 / **0.36 % PASS** | 1,725 / 6 / **0.69 % FAIL** | 384 / 1 / 1.23 % FAIL |
| re-ID alone, clean floored: held out | 644 / 0 / **0.46 % PASS** | 794 / 0 / **0.38 % PASS** | 164 / 0 / 1.81 % FAIL |
| combined (colour + re-ID), all tests, held out | 2,705 / 7 / 0.49 % | 3,062 / 18 / 0.87 % FAIL | 2,523 / 12 / 0.77 % |
| combined, clean floored, held out | 1,108 / 0 / 0.27 % | 1,006 / 0 / 0.30 % | 1,020 / 0 / 0.29 % |

Per held-out clip, re-ID alone (resolved / tests / wrong, margin fitted on the other 3):
- all tests: v2 F05 42 / 53 / 0 (0.469), F07 627 / 2,106 / 1 (0.482), F02 296 / 1,377 / 0 (0.517), F08 358 / 2,246 / 0
  (0.517); v5 F05 42 / 53 / 0 (0.483), F07 563 / 2,106 / 0 (0.554), **F02 621 / 1,377 / 5 (0.403)**, F08 499 / 2,246 / 1
  (0.487).
- clean floored: v2 F05 37, F07 330, F02 152, F08 125; v5 F05 34, F07 312, F02 295, F08 153 (0 wrong for both).
The 6 v5 errors are on all-tests rival tests (occluded or overlapping crops), which the tracker never uses as re-ID
evidence; on the clean tests it does use, v5 resolves more than v2 with 0 wrong. Under the adoption rule
(2026-11-18: all tests, held out) it still fails.

## Same-kit teammates (`records/reid_episode_resolution_2026-11-20/distributions.py`, teammate proxy = mean
torso-colour similarity >= 0.70, clean rival tests)

| | v2 | v5 |
|---|---|---|
| same player, median cosine | 0.595 | 0.595 |
| same-look different player (teammate proxy), median | 0.407 | 0.414 |
| clean same-look rival tests right at 0.517 (of 427), wrong | 62, 0 | 53, 0 |
| ... at the model's own final margin (v2 0.469, v5 0.483) | 78, 0 | 74, 0 |
| per clip right at 0.517 (F07 / F02 / F08 of 225 / 129 / 73) | 5 / 50 / 7 | 17 / 34 / 2 |
| median clean same-look rival margin | 0.249 | 0.290 |

The teammate loss moves FRESH_07 (5 -> 17) and the median same-look margin up, but loses FRESH_02 / 08; pooled
it resolves fewer same-kit tests at the margin. The episode-resolution STOP condition (re-ID that separates same-kit
teammates) is not met.

## Notes
- Reproducibility: `appearance_cue_eval.py` orders rival tests by set iteration (Python hash seed), so a rerun's
  table file differs byte-wise; here the v2 tables hold the same 5,782 tests as the v4 evaluation's and every
  calibration value agrees within 1e-9 (HASHES.txt). Not fixed (out of scope).
- `appearance_cue_eval.py` on this branch carries the v4 `pairs@` replay option; not used here.
- Remaining: nothing adopted. v5 is the first model to pass the clean floored check with more resolutions than v2,
  but it fails the all-tests held-out rule. Whether the adoption rule should be the clean check (the tracker's
  evidence) is a decision for Nathan, not made here.
