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

## DEV measurement: PENDING (Nathan's Mac)
Re-run `spana.cli ball` on the existing `~/spana-ball-run/*.balldet.json`, then `tools.ball_score`. Pass bar
(DECISIONS 2026-11-23):
- 0 wrong OBSERVED pooled;
- FRESH_05 OBSERVED coverage >= 0.888;
- FRESH_07 OBSERVED coverage >= 0.740.
