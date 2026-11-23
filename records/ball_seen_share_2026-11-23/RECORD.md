# Ball seen-share rule: a sparsely observed hypothesis is never the answer (2026-11-23)

Owner layer: ball tracking. Branch `ball/seen-share-v1` (from `evaluation/ball-score-v1` c338a99). DEV only.

## Why
First DEV score (`records/ball_score_dev_2026-11-23`): 32 / 310 wrong OBSERVED answers pooled. FRESH_02 and FRESH_08
were followed false objects (heads / boots / spots). Each was confirmed by a short run of detections and kept alive
by sporadic re-detections that restart `predict_max_s`. Such objects were seen on 25-37 % of their answered frames;
the real ball on 63-94 %.

## Change (Nathan chose the seen-share rule; values and pass bar recorded in DECISIONS before measuring)
- A confirmed hypothesis may be OBSERVED / PREDICTED only while observed on >= `ball.min_seen_share` (0.5) of its
  processed frames over the last `ball.seen_share_window_s` (1.0 s). Otherwise the answer is LOST
  SPARSE_OBSERVATIONS (logged with the share). A sparse hypothesis still counts as a rival (AMBIGUOUS): fail closed.
- `spana.ball` 1.1.0 (the new reason); 1.0.0 documents still validate. Config 2.19.0.
- `spana.cli ball` now checks only the ball-view keys (`score_min`, `nms_iou`) against the detections. Tracker
  settings can change without a detector rerun; a changed view is still refused.
- Side effect, by design: a ball followed for only a moment and then hidden goes LOST before `predict_max_s`, once
  its share drops below 0.5. Pinned by `test_young_ball_hidden_goes_sparse_before_predict_max`.

## Tests
`tests/test_ball_tracker.py`, 22 passed:
- a flickering false object is never answered;
- the real ball keeps the answer next to a flickering object;
- the young hidden ball goes sparse;
- the share comes from config;
- 1.1.0 / 1.0.0 validation;
- ball detections are reused unless the view changed.

`test_short_gap_predicted_then_lost` now follows the ball for > 1 s before the gap. Known pre-existing edge, not
changed: the tracker compares `gap > predict_max_s` without a tolerance, so the last predicted frame can depend on
float rounding.

Full suite: 723 passed, 15 skipped, 2 failed (`test_botsort_compare.py`: the bench-only `supervision` package is
not installed in the cloud container).

## DEV measurement (Nathan's Mac, 2026-11-23): FAILS the pass bar
`spana.cli ball` re-run on the same `*.balldet.json`, then `tools.ball_score`; files in `~/spana-ball-run/v2/`, not
committed. Numbers transcribed from the terminal.

| clip | coverage seen (before -> after) | coverage seen + predicted | wrong OBSERVED (95 % bound) | wrong all answers |
|---|---|---|---|---|
| FRESH_02 | 0.271 -> 0.163 | 0.395 -> 0.209 | 13 / 48 -> 2 / 23 (0.249) | 45 / 98 -> 7 / 36 |
| FRESH_05 | 0.938 -> 0.938 | 0.977 -> 0.977 | 0 / 122 -> 0 / 122 (0.024) | 2 / 130 -> 2 / 130 |
| FRESH_07 | 0.790 -> 0.758 | 0.919 -> 0.879 | 2 / 100 -> 2 / 96 (0.064) | 7 / 124 -> 6 / 118 |
| FRESH_08 | 0.426 -> 0.463 | 0.463 -> 0.519 | 17 / 40 -> 1 / 26 (0.170) | 68 / 95 -> 8 / 38 |
| pooled | 0.636 -> 0.600 | 0.725 -> 0.666 | 32 / 310 -> 5 / 267 (0.039) | 122 / 447 -> 23 / 322 |

Pass bar (DECISIONS 2026-11-23):
- 0 wrong OBSERVED pooled: 5, **FAIL**;
- FRESH_05 coverage >= 0.888: 0.938, pass;
- FRESH_07 coverage >= 0.740: 0.758, pass.

Not adopted, and the bar is not changed after the fact.

What changed:
- Wrong answers fall 122 -> 23.
- FRESH_08 phantoms are nearly gone, and FRESH_08 coverage rises: false objects no longer block the real ball.
- Cost: FRESH_02, whose small real ball is seen on ~63 % of its frames, drops below 0.5 often (coverage 0.27 ->
  0.16).
- The 5 remaining wrong OBSERVED frames (F02 2, F07 2, F08 1) are not yet diagnosed. F07's 2 match the 2 OBSERVED
  answers on HIDDEN labels of the first run.
