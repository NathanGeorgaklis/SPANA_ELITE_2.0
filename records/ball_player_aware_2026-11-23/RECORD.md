# Player-aware ball re-acquisition, step 1 (2026-11-23)

Owner layer: ball tracking. Branch `ball/player-aware-v1` (from `ball/seen-share-v1`). DEV only. Decision and pass
bar: DECISIONS 2026-11-23 (recorded in commit 3f9cbbe, before building).

## Diagnosis that set the rules (Nathan's Mac, `~/spana-ball-run/diag3.txt`, v2 ball output + new player tracks)
The 10 frames before each of the 5 wrong OBSERVED answers left after the seen-share rule. `y` = height of the point
inside the nearest observed player's box (0 top, 1 feet).

| clip, frames | before | the wrong answer |
|---|---|---|
| F02 29-33 | ball at a player's feet (y 0.92-0.94), last seen frame 24, unseen 25-28 | jumps 210 px to a detection on another player at y 0.47-0.48 (jersey), score 0.32-0.45, detected 5 frames running |
| F07 81 | ball at a player's feet (y 0.94-0.97), last seen 77, unseen 78-80 | one detection 250 px away on another player at y 0.73 (boot), score 0.42; not seen again |
| F07 308-309 | ball at a player's feet (y 0.91-0.99), last seen 301, unseen 302-307 | jumps 320 px to a detection on another player at y 0.57-0.58 (shorts), score 0.55-0.60, 2 frames |
| F08 330-334 | no ball (out of the picture) | a new hypothesis on a detection at y 0.29-0.31 (keeper's glove), score 0.35-0.66, 5 frames running |

- Every wrong answer is on a player's body at y 0.29-0.73; the real ball at feet is at y 0.91-0.99.
- Every re-acquisition is 13-25 diameters from the predicted position after 3-7 unseen frames.
- A planned rule (several detections running at the carrier's feet) was dropped: no error was at the feet, and the
  jersey was detected 5 frames running.

## Change
With `spana.cli ball --tracks` (boxes of the player tracks OBSERVED on the frame). Both rules apply only to a
hypothesis not observed on the previous processed frame:
- **Body rule** (`ball.feet_zone_h` 0.20): a candidate inside a player's box above the bottom 20 % cannot join the
  hypothesis and cannot start one. A ball followed frame to frame across a player is unaffected.
- **Far-rejoin rule** (`ball.reacquire_max_d` 3.0): a confirmed hypothesis rejoins only a candidate within 3
  diameters of its predicted position. A ball that reappears elsewhere starts a new hypothesis and is confirmed like
  any other (no silent jump).

Other changes: config 2.20.0; `spana.ball` 1.2.0 (`tracks_sha256`); decision log `blocked` / ON_BODY. Without
`--tracks` the behaviour is unchanged.

The 0.20 line was chosen after seeing the y values above, between 0.73 and 0.91. That is a weakness: the rules are
designed and measured on the same 4 clips.

## Tests
`tests/test_ball_tracker.py`, 30 passed. Full suite: 731 passed, 15 skipped, 2 failed (`test_botsort_compare.py`: the
bench-only `supervision` package is not installed in the cloud container). New tests:
- the jersey jump, the glove and the single far detection, each reproduced without players (the bug) and stopped
  with them;
- a ball back near its prediction rejoins at once;
- a ball followed across a body stays OBSERVED;
- a ball reappearing far away is confirmed as a new hypothesis;
- both settings are read from config;
- the CLI with `--tracks` (hash recorded; other video / uncovered frames refused).

## DEV measurement, step 1 (Nathan's Mac, 2026-11-23; `~/spana-ball-run/v3/`, not committed): FAILS the pass bar
`spana.cli ball --tracks` on the same `*.balldet.json` and the new `*.tracks.json`, then `tools.ball_score`. Numbers
transcribed from the terminal. Columns: first run -> seen-share rule (v2) -> player-aware step 1 (v3).

| clip | coverage seen | coverage seen + predicted | wrong OBSERVED (95 % bound, v3) | wrong all answers |
|---|---|---|---|---|
| FRESH_02 | 0.271 -> 0.163 -> 0.155 | 0.395 -> 0.209 -> 0.233 | 13 / 48 -> 2 / 23 -> 2 / 22 (0.259) | 45 / 98 -> 7 / 36 -> 3 / 34 |
| FRESH_05 | 0.938 -> 0.938 -> 0.915 | 0.977 -> 0.977 -> 0.946 | 0 / 122 -> 0 / 122 -> 0 / 119 (0.025) | 2 / 130 -> 2 / 130 -> 3 / 127 |
| FRESH_07 | 0.790 -> 0.758 -> 0.734 | 0.919 -> 0.879 -> 0.863 | 2 / 100 -> 2 / 96 -> 0 / 91 (0.032) | 7 / 124 -> 6 / 118 -> 4 / 116 |
| FRESH_08 | 0.426 -> 0.463 -> 0.407 | 0.463 -> 0.519 -> 0.463 | 17 / 40 -> 1 / 26 -> 0 / 22 (0.127) | 68 / 95 -> 8 / 38 -> 10 / 38 |
| pooled | 0.636 -> 0.600 -> 0.577 | 0.725 -> 0.666 -> 0.652 | 32 / 310 -> 5 / 267 -> 2 / 254 (0.025) | 122 / 447 -> 23 / 322 -> 20 / 315 |

Pass bar (DECISIONS 2026-11-23):
- 0 wrong OBSERVED pooled: 2 (both FRESH_02), **FAIL**;
- FRESH_05 coverage >= 0.888: 0.915, pass;
- FRESH_07 coverage >= 0.740: 0.734, **FAIL**.

Not adopted, and the bar is not changed after the fact.

- Gone: the F07 boot / shorts answers and the F08 glove (F07 and F08: 0 wrong OBSERVED).
- Not gone: 2 wrong OBSERVED on FRESH_02, not yet diagnosed (the body rule was expected to stop the frame 29-33
  jersey jump).
- Cost: about 3 correct OBSERVED frames each on FRESH_05 and FRESH_07; FRESH_08 coverage 0.463 -> 0.407.
