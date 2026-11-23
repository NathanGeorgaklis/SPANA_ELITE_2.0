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
- Tests: `tests/test_ball_path.py`, 37, plus the sighting / rejoin tests in `tests/test_ball_tracker.py`. They drive
  the real tracker with scripted candidates.
  - Real error patterns discarded: phantom (2 sightings, and 4 sightings = answered on 2 frames by pass 1), 5-frame
    glove / jersey, boot on 6 of 8 frames, 3 s flicker at ~25 %, false tail after a hole.
  - Kept: a steady ball, also before online confirmation; the real ball while a false object is selected; a kicked
    ball that goes on, with a gap record.
  - Structure: holes, gaps, trailing NO_BALL, conflicts, every gap kind and their precedence, camera CUT and
    UNKNOWN, adjacent segments at a cut, camera pan.
  - Exact boundaries of all four thresholds.
  - The validator alone and with the ball document; the CLI; the scorer.
  - One KNOWN LIMIT is pinned by a test: a near rejoin onto another object stays SEEN.
- Mutation checks: 23 changes, all 23 caught by a test.
  - 16 in `spana/ball/path.py`: each rule removed, each threshold moved off its boundary, the gap-kind order
    swapped, UNKNOWN camera ignored, a position kept on a hole frame.
  - 7 in `validate_ball_path`: no covering check, no reason-vs-cover check, a gap holding SEEN frames, gap frames
    unchecked, no video comparison, no recomputation, a position allowed off SEEN.
- Full suite: 773 passed, 15 skipped, 2 failed. The 2 are `test_botsort_compare.py`: the bench-only `supervision`
  package is not installed in the cloud container.

## Reviews (two independent, 2026-11-23; neither could run code, both read it)
**Correctness review.** It found no input that makes a frame SEEN without exactly one confident stretch sighting the
ball on it, and no crash in `ball_path` on degenerate inputs. Findings, all fixed:
1. `tools/ball_score.py` scored a path document after the standalone check only, and a hand-edited path could pass
   that check. Now a path is scored only with the ball document it was made from (`--clip LABEL_DIR PATH BALL`): it
   is fully recomputed, and it must carry the recorded `ball_path` settings (`--spana-config`).
2. The standalone check was strengthened:
   - SEEN / HOLE only where exactly that one confident stretch covers the frame;
   - CONFLICT exactly where two or more cover it;
   - a gap holds no SEEN frame;
   - a gap's `frames` / `duration_s` / kind must agree with the frames.
3. The full check now also requires the path and the ball document to name the same video.
4. Malformed gap ends, and a ball document without the needed `ball_config` keys, are refused cleanly instead of
   crashing.
5. The camera-break scan uses bisection (it was quadratic on long clips).

Noted, not changed: a single-cover run with no SEEN frame keeps reason HOLE; a gap's `frames` counts processed
frames.

**Test-quality and risk review.** No rule violation (no thresholds outside config, nothing guessed, nothing
clip-specific).
- Fixed:
  - tests added for UNKNOWN camera, gap-kind precedence, trailing NO_BALL, exact boundaries and adjacent segments;
  - the "6 of 8" fixture corrected (it was 7 of 8);
  - the `reacquire_max_d` config comment updated.
- Undisclosed risks it raised are now in `DESIGN.md` ("Risks added after the independent review"):
  - a near rejoin onto another object;
  - an undetected shot change;
  - replays;
  - persistent false objects;
  - duplicate detections of the real ball.

  None is fixed by v1.

Also added to the design by this review: a stretch is split at CUT / UNKNOWN camera frames. The code did it; the
design text did not say so.

## DEV measurement: PENDING
