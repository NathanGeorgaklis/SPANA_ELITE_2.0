# Possession / touches v1: who is on the ball, per frame, fail closed (2026-11-22, synthetic only)

Owner layer: possession / touches (layer 8). Branch `possession/touches-v1`, from main `ef91010`. No tracker, ball,
follower or detector change. No DEV or EVAL data used: no footage in the cloud session, and no DEV touch labels
exist yet.

## Why
Target identity is parked at the reviewer baseline (0.879 correct, 0 wrong, 6.48 actions per followed minute).
Clips and stats only need the player right when they are involved with the ball. This layer answers who is on the
ball, so identity can later be measured at those moments (slice: touches -> DEV touch labels -> ball + touches on
DEV -> target identity at touch moments).

## What was built
- `spana/possession/touches.py`: per-frame answer from a tracks and a ball artifact of the same video and frames,
  and possessions (runs of CONTROL by one track). Rules: `docs/INTERFACES.md`, "Possession / touches".
- `spana.touches` 1.0.0 + `validate_touches` (`spana/artifacts.py`). The validator checks the state rules on the
  stored distances and recomputes the possessions from the frames. Given the source tracks and ball, it also
  recomputes every frame.
- `spana.cli touches --tracks --ball --config --out` (+ `.decisions.jsonl`).
- Config 2.18.0 `possession` section. Values are priors (`config/default.yaml` explains each):
  - `reach_h` 0.35;
  - `rival_margin_h` 0.15;
  - `max_ball_above_feet_h` 0.20;
  - `diameter_to_height_min` / `max` 0.06 / 0.25;
  - `bridge_s` 0.40.
- Deviations from the task text:
  - Two scalars `diameter_to_height_min` / `max` instead of one range key.
  - Candidates carry `usable`, so CONTROL / CONTESTED / TRACK_NOT_CERTAIN can be checked from the document alone.
  - A ball the ball tracker marks `contested` is UNKNOWN (BALL_CONTESTED).
  - A ball too small or large for the nearest player's height is its own reason (BALL_SCALE_MISMATCH), separate
    from BALL_AIRBORNE_SUSPECT.
  - A CONTESTED frame also breaks a possession: a challenge that the same player survives starts a new possession,
    not a bridged one.

## QA: scripted scenes (`python -m tools.touches_scenarios`; `tests/touches_synth.py`)
Validated tracks + ball artifacts per scene. The expected answer per frame is written by hand from the script, not
computed with the rule. Table: `scenarios_table.txt`.

| scene | what | result |
|---|---|---|
| dribble | one player with the ball | CONTROL from frame 1 (frame 0: track not yet confirmed -> UNKNOWN) |
| pass | A to B on the ground | CONTROL A 1-20, FREE 21-38, CONTROL B 39-59 |
| fifty_fifty | two players equally close | CONTESTED, never CONTROL |
| lofted | ball crosses a player's knees in the picture, in the air | FREE away, UNKNOWN BALL_AIRBORNE_SUSPECT near |
| behind | ball at the feet line in the picture, far behind the player | UNKNOWN BALL_SCALE_MISMATCH |
| occluded_dribble | ball hidden 0.2 s, then 0.6 s | UNKNOWN BALL_NOT_OBSERVED; 0.2 s bridged, 0.6 s splits |
| crossing_uncertain | same-kit crossing, both tracks IDENTITY_UNCERTAIN | UNKNOWN TRACK_NOT_CERTAIN, never CONTROL |
| hidden_nearest | the player on the ball is hidden (track PREDICTED) | UNKNOWN TRACK_NOT_CERTAIN |
| camera_cut | cut in the middle of a dribble | UNKNOWN CAMERA_DISCONTINUITY; two possessions |

Totals: 420 frames, 420 as scripted, **0 wrong CONTROL**, possessions 8 / 8 as scripted (crossing: not scripted).

Mutation check: with the "uncertain track" guard removed, 8 tests fail; with the airborne guard removed, 1 fails (in
the lofted scene the scale check also catches the small ball).

Tests: `tests/test_touches.py`, 15 passed. Full suite: 698 passed, 15 skipped (all environment: browsers, bench packages, weights), 2 failed. The 2 failures are
`test_botsort_compare.py`: the bench-only `supervision` package is not installed in the cloud container (same as on
main).

## Not done / limits
- Not run on DEV. Needs:
  - frozen DEV touch labels (next task, on Nathan's Mac);
  - ball v1 on the DEV clips;
  - tracks and ball artifacts of the same frames: the ball view must come from the same detector pass
    (`detect --ball-out`).
- Image space only. A ball in the air directly in front of a player's feet, at the right size, would still read
  as CONTROL. Only a pitch layer or a ball-height estimate removes that.
- PREDICTED ball frames never give CONTROL (stricter: some real touches become UNKNOWN). Revisit with DEV labels.
- On camera-UNKNOWN frames the tracker ends every track anyway, so treating them as a discontinuity costs nothing
  extra here.
