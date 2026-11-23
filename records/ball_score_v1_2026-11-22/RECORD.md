# Ball scorer v1: spana.ball against the frozen DEV ball labels (2026-11-22)

Owner layer: evaluation (ball). Branch `evaluation/ball-score-v1` from main `ecf527c`. DEV labels only; no EVAL; no
ball output from real footage was scored (none exists in the cloud session).

## Built
- `tools/ball_score.py`, conventions in `config/ball_score.yaml` (hit radius 0.01 x frame height = 10.8 px at 1080p;
  bound confidence 0.95). Set before any scoring. Choices are in DECISIONS 2026-11-22 (Nathan).
- Each labelled frame is judged by centre distance (label box centre; label box size is not truth):

| label \ tracker | OBSERVED | PREDICTED | LOST / AMBIGUOUS |
|---|---|---|---|
| VISIBLE | hit within the radius; else wrong spare ball (within the radius of an `other_balls` box) or wrong position | the same, in its own column | miss |
| HIDDEN | wrong (claims to see a hidden ball) | fine (distance to the annotator's point reported) | fine |
| OUT | wrong (phantom) | wrong (phantom) | fine |

  UNSURE frames are not scored.
- Reported per clip and pooled:
  - coverage: OBSERVED hits / VISIBLE, and OBSERVED + PREDICTED hits / VISIBLE;
  - wrong answers with the exact one-sided 95 % bound, for OBSERVED, PREDICTED and all answers;
  - hit distances (nearest rank, share of the frame height; in px when one resolution);
  - every wrong frame.
- Refused:
  - another video or frame size, or a ball document without video fields;
  - a labelled frame missing from the ball document;
  - the same video twice;
  - non-DEV labels.

## Tests
`tests/test_ball_score.py`, 14 passed:
- every cell of the table;
- the radius read from config, with the hit on the exact radius;
- spare-ball precedence;
- nearest-rank percentiles;
- each refusal;
- mixed-height pooling;
- the CLI, which never overwrites;
- a perfect tracker over each real frozen DEV label file: coverage 1, 0 wrong.

Mutation check: a PREDICTED answer on a hidden ball scored as wrong makes 3 tests fail.

Two independent reviews (correctness, conventions) found no bug or rule violation. All 6 minor findings were fixed:
- duplicate clips refused;
- clean refusal when video fields are missing;
- no hard-coded clip count;
- mixed-height pooling simplified and tested;
- spare-ball precedence tested;
- percentile convention documented and tested.

## Next (Nathan's Mac)
Run the ball pass on each DEV clip so that every labelled frame is covered, then run
`python -m tools.ball_score --clip data/ball_labels/<clip>/v1 <clip>.ball.json ... --out ball_score_dev.json`.
Then set the pass bar and fit the `ball` priors on DEV (separate task).
