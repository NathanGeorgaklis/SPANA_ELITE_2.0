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

## Amendment 1 (2026-11-23, during the build, BEFORE any DEV measurement of the path)
Found by a scripted test, not by DEV data.

**The hole in the design.** A stretch was "the sightings of one hypothesis, split at holes > max_hole_s". But the
tracker lets a lost hypothesis rejoin any candidate inside its speed gate (`gate_base_d + max_speed_d_per_s x gap`:
~60 diameters after 0.4 s). A real ball's stretch could therefore continue onto a different object after a short
hole. Those frames would be SEEN on the strength of the real part: a false tail, never judged on its own. This is
the FRESH_02 jersey jump in a form the body rule does not cover (off the body, or without `--tracks`).

**Change.**
- Pass 1: each sighting carries `rejoin_d`.
  - Null when the hypothesis was observed on the previous processed frame, or the sighting starts the hypothesis.
  - Otherwise the distance, in diameters, from the hypothesis' constant-velocity prediction (the association cost).
- Pass 2: a stretch is ALSO split before a sighting whose `rejoin_d` exceeds `ball.reacquire_max_d` (3.0).
  - Existing key, value recorded 2026-11-23, same meaning: after an unseen frame, how far from the prediction is
    "far". It is read from the ball document's own `ball_config`.
  - The part after the split is a new stretch and must be confident on its own. A false tail of a few frames is
    discarded; a kicked ball that goes on for >= min_stretch_s is kept, with a gap record between the two parts.

**Known cost (fail closed, never a wrong answer).**
- After a far reappearance, the first few sightings may belong to a new hypothesis before the old one's gate reaches
  them. That fragment is usually < min_stretch_s and is discarded: a few frames of coverage per far reappearance.
- Real-ball fragments between quick far rejoins that last < min_stretch_s are discarded too.

**Still not covered.** A hop between two candidates on consecutive frames (no unseen frame), within the per-frame
gate (~8 diameters), e.g. from the ball to a boot next to it.

**IMPLAUSIBLE gaps: weaker than the design implied.**
- A rejoin by the same hypothesis is inside the speed gate by construction.
- At `max_speed_d_per_s` 150 the bound exceeds the frame width after ~0.5 s for a small ball.
- So the kind will rarely fire with the default config. It is kept (harmless, and it does fire between different
  hypotheses over very short gaps), but it is not a safeguard to rely on.

Pass bar and the three `ball_path` settings are unchanged.

## Risks added after the independent review (2026-11-23, before any DEV measurement)
A SEEN answer can still be WRONG in these cases. None is fixed by v1.

1. **Near rejoin onto the wrong object.** After a short occlusion (the ball behind a player's legs for a few
   frames), a boot or sock within `reacquire_max_d` (3 diameters) of the predicted position is accepted as the same
   hypothesis: no far-rejoin split. The stretch then continues on that object.
   - If the real ball is itself a confident stretch at the same time, the frames are CONFLICT (fail closed).
   - Otherwise they are SEEN and wrong.
   - The same happens at the end of a stretch, when the ball leaves and the hypothesis lingers on something next to
     it.
2. **Undetected shot change.** Stretches split only on camera status CUT / UNKNOWN. A dissolve, wipe or fast zoom
   the camera layer reports as OK lets a stretch run through it.
3. **Replays, close-ups, picture-in-picture.** A replayed ball is a dense stretch of >= 0.5 s that is not the live
   ball.
4. **Persistent false objects** (a ball logo on a board, a spare ball at the touchline, a bright spot on the pitch).
   Such an object is a confident stretch for as long as it is visible.
   - While the real ball is not a confident stretch, the object is SEEN: wrong.
   - While the real ball is confident, every frame is CONFLICT: one such object can remove the real ball's coverage
     for the whole clip. That is a coverage failure, not a wrong answer.
5. **Two detections of the one real ball** (the ball and its motion blur) can form two overlapping stretches:
   CONFLICT on the real ball. Coverage lost, not wrong.
6. **A consecutive-frame hop at the START of a stretch** (a stretch that began on a false object before the ball came
   within its gate) is the same class as the "still not covered" hop in amendment 1.
7. **`min_stretch_s` 0.5 s comes from four clips.** A dense false stretch of 0.5-1 s (a boot carried by a running
   player) passes.

Items 1-4 cannot be tested on the four DEV clips unless they happen to occur there. The unseen clip is the first
real check, and one clip will not exercise all of them.
8. **Design text correction.** A stretch is also split at a CUT / UNKNOWN camera frame (the tracker ends every
   hypothesis there anyway). The first version of this document did not say so.

## How a path is scored (added after the review)
`tools/ball_score.py` scores a path only together with the ball document it was made from. It recomputes the whole
path and refuses any difference, and it refuses a path whose `ball_path` settings are not the recorded ones.

