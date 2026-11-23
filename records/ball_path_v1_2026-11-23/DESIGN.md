# Ball path v1: decide afterwards, from both ends; never guess (design, 2026-11-23)

Owner layer: ball tracking. Branch `ball/path-v1` (from `ball/player-aware-v1`). Written and committed BEFORE any
code. Approved by Nathan 2026-11-23 (two-pass design; replaces the seen-share rule and the "with player X" memory).

## Why
`records/ball_score_dev_2026-11-23`, `ball_seen_share_2026-11-23`, `ball_player_aware_2026-11-23`:
- The online tracker must decide on each frame which object is the ball. Every false answer found so far was short
  or flickering:
  - a phantom seen for 2 frames;
  - a glove and a jersey seen for 5 frames;
  - a boot seen on 6 of 8 frames;
  - long-lived false objects seen on 25-37 % of their frames.
- The real ball in FRESH_05 / 07 is followed for hundreds of frames, seen on 82-94 % of them.
- 22 of the 24 wrong answers on FRESH_08 (every-frame labels) are forward predictions.
- While the tracker follows a false object, the real ball's detections are ignored.

Nathan's point: after losing the ball, wait for a confident lock instead of guessing. Knowing where it left and
where it arrived is enough; what happens in between need not be invented.

## Design
### Pass 1 (exists): `spana.cli ball` -> `spana.ball` 1.3.0
- Unchanged tracking of every ball-like hypothesis. With `--tracks`, the body rule stays.
- New per frame:
  - `sightings`: every hypothesis observed on the frame, as `{hypothesis, center, diameter, score}`. It includes
    hypotheses that are unconfirmed or not selected.
  - `camera_affine`: the ball pass's own inter-frame transform (identity when the camera status is not OK).
- The per-frame state / PREDICTED answer stays in the document for review, but pass 2 does not read it.
- `ball.min_seen_share` becomes 0.0 (off): pass 2 replaces it.

### Pass 2 (new): `spana.cli ball-path --ball B --config C --out P` -> `spana.ball_path` 1.0.0
Input: one `spana.ball` 1.3.0 document, read whole (uses the future). Pure and deterministic.
1. **Stretches.** A stretch is the sightings of one hypothesis, split wherever two consecutive sightings are more
   than `ball_path.max_hole_s` apart.
2. **Confident stretch.** Both must hold:
   - duration (last sighting time - first) >= `ball_path.min_stretch_s`;
   - density (sightings / processed frames from first to last) >= `ball_path.min_density`.

   Every other stretch is discarded and is never a ball answer.
3. **One ball.** On frames covered by two or more confident stretches the answer is UNKNOWN / CONFLICT for all of
   them. No stretch is preferred.
4. **Per-frame answer.**
   - SEEN: exactly one confident stretch covers the frame and has a sighting on it. Carries `center`, `diameter`,
     `score`, `stretch`.
   - UNKNOWN, with a reason and no position:
     - HOLE: inside the one covering stretch, not sighted;
     - CONFLICT;
     - GAP: between path segments;
     - NO_BALL: before the first or after the last path segment.
5. **Gap records.**
   - A path segment is a maximal run of frames covered by exactly one confident stretch.
   - Each gap between consecutive segments records `from {frame, t, center, diameter, stretch}` (the last SEEN frame
     before it), `to {...}` (the first SEEN frame after it), `frames`, `duration_s` and a `kind`.
   - `kind`, first match wins:
     - CAMERA_BREAK: a CUT / UNKNOWN camera frame in (from, to];
     - CONFLICT: a conflict frame inside the gap;
     - IMPLAUSIBLE: `from.center` carried through the camera transforms to the `to` frame is farther from
       `to.center` than `ball.gate_base_d + ball.max_speed_d_per_s x duration` diameters (the tracker's own speed
       bound, no new threshold; diameter = the larger of the two ends);
     - OPEN: otherwise. This is the only kind from which "left here, arrived there" may be inferred.
6. **No players.** Which player the ball left or arrived at is the possession layer's job (it will read SEEN
   frames); the ball path names no player. One geometry, in one place.

### Settings (config section `ball_path`, set here before any measurement)
| key | value | why |
|---|---|---|
| `max_hole_s` | 0.4 | = `ball.predict_max_s`, the existing bound on "briefly unseen" |
| `min_stretch_s` | 0.5 | about twice the longest dense false stretch seen so far (0.23 s) |
| `min_density` | 0.5 | the seen-share value recorded 2026-11-23 (false 0.25-0.37, real 0.63-0.94) |

These were chosen after seeing errors on the same 4 DEV clips. That is the design's known weakness, and the reason
for the unseen-clip test below.

## What it will not fix
- Frames where the detector does not fire: UNKNOWN, by design. FRESH_02 stays poor.
- A false object seen steadily for >= 0.5 s. The one-ball rule and IMPLAUSIBLE gaps catch some, not all.
- Anything that happens inside a gap (missed, never invented).

## Pass bar (recorded before building; Nathan, 2026-11-23)
- On EACH clip separately: 0 wrong SEEN answers. A wrong SEEN answer is a SEEN frame whose label is OUT or HIDDEN,
  or VISIBLE more than `hit_radius_frac_height` away (`tools/ball_score.py`).
- Labels: FRESH_02 / 05 / 07 v1, FRESH_08 v2 (every frame).
- Coverage (SEEN hits / VISIBLE) is reported per clip and does not decide: the detector limits it.
- Then one DEV clip the design has never seen, labelled every frame by Nathan AFTER this design and the code are
  frozen, scored exactly once with the same bar. Its result is reported whatever it is, and the settings are not
  changed after it.

## Verification plan
1. Unit tests on scripted sequences reproduce each real error (2-frame phantom, 5-frame glove / jersey, boot on 6
   of 8 frames, long flicker at 25 %) and show each discarded.
2. Further tests: holes, gaps, conflicts, implausible gaps, camera cuts and pans.
3. Mutation checks: each rule removed in turn must fail a test.
4. `validate_ball_path(doc, ball)` recomputes the whole document from the ball document and refuses any difference.
5. Two independent code reviews before any DEV run.
6. DEV run on the existing detector outputs (no detector rerun), then the unseen clip.
