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

## DEV measurement on the 4 clips (Nathan's Mac, 2026-11-23; `~/spana-ball-run/v5/`): PASSES the bar
- Code 3ab9829.
- `spana.cli ball --tracks` on the existing `*.balldet.json` / `*.tracks.json`, then `spana.cli ball-path`.
- `tools.ball_score --clip LABELS PATH BALL`: each path recomputed from its ball document, recorded settings checked.
- Labels: FRESH_02 / 05 / 07 v1, FRESH_08 v2 (every frame).
- Numbers pasted from the terminal.

| clip | frames scored | wrong SEEN (95 % bound) | coverage (SEEN hits / VISIBLE) | first run: coverage / wrong OBSERVED |
|---|---|---|---|---|
| FRESH_02 | 131 | 0 / 36 (0.080) | 0.279 | 0.271 / 13 |
| FRESH_05 | 131 | 0 / 123 (0.024) | 0.946 | 0.938 / 0 |
| FRESH_07 | 131 | 0 / 99 (0.030) | 0.798 | 0.790 / 2 |
| FRESH_08 | 391 | 0 / 60 (0.049) | 0.370 | (narrowed step 1 on the same v2 labels: 0.352 / 2) |
| pooled | 784 | 0 / 318 (0.0094) | 0.583 | |

Pass bar (0 wrong SEEN on EACH clip): met on all four. The path has no PREDICTED answers, so "wrong all" = "wrong
SEEN".

What this does and does not show:
- Coverage is at or above every earlier version on every clip, including the unfiltered first run.
  - Sightings before online confirmation now count.
  - A false object no longer blocks the real ball.
- FRESH_02 / FRESH_08 coverage stays low (0.28 / 0.37): the detector does not fire on the ball. That is now UNKNOWN,
  not wrong.
- These are the clips the design was shaped on.
- FRESH_02 / 05 / 07 are labelled every 3rd / 6th frame, which can hide short errors (it did on FRESH_08 v1).
- 0 / 36 on FRESH_02 is consistent with a true wrong rate of up to 8 %.
- The DESIGN's listed risks (near rejoin, undetected shot change, replays, persistent false objects) are not
  exercised unless they occur in these clips.

## Unseen clip (Nathan's Mac, 2026-11-23; `~/spana-ball-run/unseen/`): FAILS the bar
- Clip: FRESH_03_G2_2712_midfield_possession_switch_of_play (DEV, Game 2, 60 fps). Chosen before any tracker output
  on it existed; never used in the design.
- Labels: v1, every 2nd frame (1/30 s), 391 frames: VISIBLE 366, HIDDEN 13, OUT 12. Frozen and pushed (23ef2c1)
  BEFORE the detector / tracker was run on the clip.
- Code and settings unchanged since the freeze (3ab9829; `git diff 3ab9829 -- spana tools config` empty).
- Pipeline: `detect --backend local --ball-out` (stride 1), `track`, `ball --tracks`, `ball-path`.
- Scored ONCE with `tools.ball_score --clip LABELS PATH BALL`. Numbers pasted from the terminal.

| clip | frames scored | wrong SEEN (95 % bound) | coverage (SEEN hits / VISIBLE) |
|---|---|---|---|
| FRESH_03 | 391 | **4 / 314** (0.029) | 0.847 (310 / 366) |

Pass bar (0 wrong SEEN): **FAIL**. The result stands; the settings are not changed after it (DESIGN, pass bar).

- On footage the design was not shaped on, the path gave 4 wrong SEEN answers in 314 (1.3 %). The 0 / 318 on the four
  design clips overstated its reliability.
- What the 4 frames are (a false object, a near rejoin, near misses just outside the scoring radius) is NOT yet
  diagnosed.
- Ball path v1 is therefore NOT adopted as meeting the bar. Whether it is merged as an improvement over the tracker's
  own answer (first run: 32 wrong OBSERVED of 310 on the four clips) is Nathan's decision, to be recorded as that.

### The 4 wrong SEEN frames on the unseen clip (diagnosed 2026-11-23; `~/spana-ball-run/diag5.txt`; frames viewed by Nathan)
Path on FRESH_03: SEEN 629 frames, HOLE 132, GAP 20, CONFLICT 0; 3 confident stretches of 10; 2 gaps, both OPEN
(341 -> 349, 387 -> 401).

| frame | label v1 | path answer | what the frame shows (Nathan) |
|---|---|---|---|
| 380, 382, 386 | OUT (from frame 374) | SEEN at x 105-120, scores 0.57-0.82, same hypothesis as frame 372 (a hit at (177, 371), moving left ~7 px / frame), rejoin_d 1.9 after 7 unseen frames | the ball (also on 381, 385, 387): still in the picture near the left edge |
| 780 (last frame) | VISIBLE at (872, 360) | SEEN at (895, 354), 23.9 px away, score 0.51, rejoin_d 2.5 after 8 unseen frames | mostly on the player next to the ball, partly on the ball |

- Frames 380 / 382 / 386: the label is wrong, not the path. The ball was labelled OUT while still in the picture. A
  corrected label version (v2, with this reason) is needed. v1 is never edited, and the measured 4 / 314 stands as
  the result on v1.
- Frame 780: a real wrong position. It is the DESIGN's disclosed risk 1 (a near rejoin, `rejoin_d` 2.5 <=
  `reacquire_max_d` 3.0, onto a detection that is not centred on the ball).
- Caution: the label correction comes from inspecting only the frames where the path disagreed with the labels, which
  can only improve the score. A re-score on v2 is weaker evidence than the untouched score on v1.
- Coverage: of the 55 MISS frames (51 HOLE, 4 GAP), pass 1 had a sighting within the scoring radius of the label on
  0. All coverage lost on this clip is the detector not firing; none is the path's rules discarding a correct
  sighting.
