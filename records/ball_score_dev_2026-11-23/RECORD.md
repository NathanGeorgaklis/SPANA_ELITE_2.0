# Ball v1 on DEV: first score against the frozen ball labels, and why it fails (2026-11-23)

Owner layer: evaluation (ball). Run on Nathan's Mac:
- branch `evaluation/ball-score-v1` (efa41bd);
- `detect --backend local --runtime torch --device mps --ball-out`, then `spana.cli ball`, on the 4 DEV clips (stride
  1, `config/default.yaml`);
- then `tools.ball_score` (radius 0.01 x frame height = 10.8 px).

Files are in `~/spana-ball-run/` on the Mac, not committed; numbers are transcribed from the terminal output. No EVAL.
Nothing tuned.

## Score (DEV ball labels v1, 131 labelled frames per clip)
| clip | coverage seen | coverage seen + predicted | wrong among OBSERVED (95 % bound) | wrong among all answers |
|---|---|---|---|---|
| FRESH_02 | 0.271 | 0.395 | 13 / 48 (0.396) | 45 / 98 |
| FRESH_05 | 0.938 | 0.977 | 0 / 122 (0.024) | 2 / 130 |
| FRESH_07 | 0.790 | 0.919 | 2 / 100 (0.062) | 7 / 124 |
| FRESH_08 | 0.426 | 0.463 | 17 / 40 (0.567) | 68 / 95 |
| pooled | 0.636 | 0.725 | 32 / 310 (0.136) | 122 / 447 |

## What goes wrong
- Wrong positions are far from the ball, so this is not a radius effect:
  - FRESH_02: 45 frames, median 445 px (30-839);
  - FRESH_08: 20 frames, median 956 px;
  - FRESH_05: 2 near misses on PREDICTED frames (27-46 px);
  - FRESH_07: 5 PREDICTED (12-366 px) and 2 OBSERVED on HIDDEN labels.
- FRESH_08: 48 phantom frames (15 OBSERVED, 33 PREDICTED), frames 171-390, while the ball is out of the picture.
- By hypothesis (the object the tracker follows); seen share = OBSERVED / answered frames of that hypothesis:

| clip | hyp | right / wrong | seen share | median score | median diameter px |
|---|---|---|---|---|---|
| F02 | B000001 | 42 / 13 | 0.63 | 0.59 | 12.0 |
| F02 | B000004 | 9 / 32 | 0.37 | 0.41 | 12.4 |
| F05 | B000001 | 127 / 2 | 0.94 | 0.86 | 19.4 |
| F07 | B000001 | 114 / 7 | 0.82 | 0.85 | 16.4 |
| F08 | B000002 | 0 / 32 | 0.26 | 0.38 | 18.0 |
| F08 | B000001 | 25 / 6 | 0.72 | 0.83 | 17.3 |
| F08 | B000006 | 0 / 15 | 0.28 | 0.35 | 13.2 |
| F08 | B000008 | 0 / 15 | 0.25 | 0.49 | 15.9 |

- Reading:
  - Ball-class false positives (score >= `ball.score_min` 0.30) on non-balls get confirmed (3 observations in
    0.3 s).
  - Each sporadic re-detection restarts `predict_max_s`, so one false object alternates OBSERVED / PREDICTED for
    seconds.
  - When the real ball is not detected, the tracker answers the false object.
- How the hypotheses separate:
  - Seen share separates them: false 0.25-0.37, real 0.63-0.94.
  - Median score separates them narrowly: false 0.35-0.49, real 0.59-0.86 (F02's small ball scores lowest).
  - Diameter does not.
- The real-ball hypotheses still carry 13 (F02) / 6 (F08) wrong frames: likely the hypothesis jumping to a nearby
  object, since the gate allows ~15 diameters per 0.1 s. Not yet diagnosed.
- Small sample: 4 false and 4 real hypotheses. Any threshold must be set before it is measured, then measured on
  these clips.
