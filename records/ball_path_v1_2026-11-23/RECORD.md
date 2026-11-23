# Ball path v1: build record (2026-11-23)

Owner layer: ball tracking. Branch `ball/path-v1`. Design, settings, pass bar and amendment 1: `DESIGN.md` (committed
before the code they describe: b0594d1; amendment 1 before any DEV measurement of the path).

## Built
- **Pass 1**, `spana.cli ball` -> `spana.ball` 1.3.0: per-frame `sightings` and `camera_affine`.
  - `sightings` lists every hypothesis observed on the frame, with `rejoin_d`.
  - `ball.min_seen_share` is 0 (off).
- **Pass 2**, `spana.cli ball-path` -> `spana.ball_path` 1.0.0 (`spana/ball/path.py`, `validate_ball_path`,
  config 2.21.0 `ball_path`): stretches, the confident test, the one-ball rule, per-frame SEEN / UNKNOWN, and gap
  records.
- `tools/ball_score.py` scores path documents (SEEN = OBSERVED; no PREDICTED).

## Checks done before any DEV run
- Tests: `tests/test_ball_path.py`, 23, plus the sighting tests in `tests/test_ball_tracker.py`. They drive the real
  tracker with scripted candidates.
  - Real error patterns discarded: 2-frame phantom, 5-frame glove / jersey, boot on 6 of 8 frames, 3 s flicker at
    ~25 %, false tail after a hole.
  - Kept: a steady ball, also before online confirmation; the real ball while a false object is selected; a kicked
    ball that goes on, with a gap record.
  - Also covered: holes, gaps, conflicts, each gap kind, camera cut and pan, the validator's recomputation and
    tamper refusals, the CLI, the scorer.
- Mutation checks: 10 rules removed in turn, all 10 caught by a test:
  - the duration test and the density test;
  - the hole split, the far-rejoin split and the camera split;
  - the one-ball rule;
  - the camera-break kind and the implausible kind;
  - the camera carry;
  - position only on sighted frames.
- Full suite: 759 passed, 15 skipped, 2 failed. The 2 are `test_botsort_compare.py`: the bench-only `supervision`
  package is not installed in the cloud container.
- Two independent code reviews: see "Reviews" below.

## DEV measurement: PENDING
