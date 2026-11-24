# Detector retraining plan: ball recall and ball look-alikes (PROPOSAL, 2026-11-24)

Owner layer: player detection (the football detector's ball class). Status: APPROVED by Nathan 2026-11-24 with one
change (decision 2: only the match ball is boxed); see section 9 and DECISIONS 2026-11-24. Nothing has been built,
labelled or trained yet. Branch `detection/ball-retraining-plan`.

## 1. Why
What the ball work measured (`records/ball_score_dev_2026-11-23`, `ball_path_v1_2026-11-23`,
`ball_tiling_v1_2026-11-23`, `docs/BALL_TILING_FP_FINDINGS.md`):

| finding | evidence |
|---|---|
| The ball path finds the ball on 0.28-0.95 of visible-ball frames, by clip | FRESH_02 0.279, FRESH_08 0.370, FRESH_07 0.798, FRESH_03 0.839, FRESH_05 0.946 |
| Every miss examined is the detector not firing | FRESH_03: 55 of 55 missed frames had no sighting on the ball |
| The detector is weakest on small balls | detector v1, ball recall by short side: 0-8 px 1/5; 8-12 px 0.81 (valid), 0.62 (DEV bench); 12-16 px 0.96 / 0.88 |
| Showing it the ball larger (tiling) helps where it already works, and makes it call other things a ball | tiled: FRESH_05 / 07 / 03 +0.02 to +0.06 coverage, 0 new wrong; FRESH_02 42 of 62 SEEN wrong, FRESH_08 17 of 83 |
| Those false objects are steady, ball-sized and not a tiling bug | one piece of equipment (cone-like) across almost all of FRESH_02, one background object on FRESH_08; all from single tile passes; none in the untiled detections |
| The untiled detector also fires on body parts, briefly | jersey, shorts, boot, glove on FRESH_02 / 07 / 08 (`records/ball_player_aware_2026-11-23`) |
| No tracker or path rule separates a steady look-alike from the ball | accepted false objects lasted 0.6-2.8 s at density 0.63-0.97; the real ball's stretches: median 1.3-1.6 s |

So the detector's ball class has two limits, and both are in the detector:
1. **Recall on small balls.** This is what keeps coverage low.
2. **Discrimination.** It calls ball-sized round things a ball when they are shown large enough. This is what makes
   higher-resolution inference (tiling) unusable today, and tiling is the cheapest route to recall.

Fixing the second is what makes the first reachable.

One more observation, to verify in task 1. `config/roboflow.yaml` says the TRAIN frames are 1280 px wide, so the
detector learned the ball at its source size (median short side 13 px). The DEV clips are 1920 px wide and are
shrunk to 1280 for the detector, so their ball is about two thirds of that size (a 12 px ball becomes ~8 px, the
bucket where recall is worst). The detector is then being asked for a ball smaller than it was trained on. Tiles
bring it back to about the trained size (~14 px), and bring a 10.5 px cone to the same size.

## 2. Goal and non-goals
**Goal:** a detector v2 whose ball class
- finds small balls more often (recall on 8-12 px and below), and
- does not fire steadily on equipment, background objects or body parts, also at tile scale,

measured on DEV through the ball path, without lowering player / goalkeeper / referee quality.

**Non-goals:**
- no change to the class list (player, goalkeeper, referee, ball);
- no new "cone" class;
- no tracker or ball-path rule changes;
- no DEV or EVAL frame in any training or validation split;
- no change to the split of matches.

## 3. Data: TRAIN footage only
Sources: the three licensed TRAIN matches, split as today (`preflight.split_matches`): train = TRAIN_M002,
TRAIN_M003; valid = TRAIN_M001. The current dataset version 2 has 578 train and 295 valid frames.

DEV frames (FRESH_01-10, the DEV bench, the full DEV match videos) are never added. That includes the false-positive
crops of the diagnosis: they say WHAT to look for in TRAIN footage. The existing guards stay: the version gate
refuses `bench:v1` frames, and the ingest refuses any split but train / dev.

### 3a. More small-ball examples (positives)
- **What:** frames where the ball is small (short side under 12 px), blurred by motion, partly hidden at a player's
  feet, or in the air against the crowd.
- **How to find them:** run detector v1 and the ball path over the TRAIN match videos. Take frames inside the path's
  HOLE and GAP stretches, where a ball was seen shortly before and after but not on that frame. Those are frames the
  detector missed. A person then boxes the ball if it is visible.
- **Why this way:** it finds exactly the balls the detector cannot see, where sampling every 2 s finds mostly easy
  ones.

### 3b. Hard negatives (look-alikes the detector must learn are NOT a ball)
- **What**, in the generic categories of `config/ball_fp_diag.yaml`, most important first by today's evidence:
  1. equipment / cone;
  2. crowd / background;
  3. boot / foot, glove / hand, shin / sock, and jersey or shorts details: seen on the untiled detector, brief;
  4. head, pitch marking: not observed so far, included because they are ball-sized and round.
- **How to find them:** run detector v1 WITH tiling on over the TRAIN match videos and collect its ball detections
  that are not on a labelled ball. These are the detector's own mistakes at the scale that failed, so they are the
  most useful negatives. Reuse the diagnostic's contact-sheet and category code to review them.
- **How a negative is labelled:** the look-alike gets NO box. Every real ball in that frame MUST be boxed. A frame
  with an unboxed real ball would teach the detector that balls are background, so frames where the reviewer is not
  sure are excluded, not guessed.
- **Honest limit:** the ranking above rests on two physical objects in two DEV clips. Mine every category, and
  record how many of each the TRAIN footage actually yields.

### 3c. Training at the scales the detector is used at (no new labelling)
- **What:** for frames already labelled, add copies at the scales it meets at inference:
  - shrunk to about two thirds, padded to the frame size: what a 1920 px frame looks like to it today;
  - frame-height tile crops, enlarged: what a tile looks like.
- **Why:** today it is trained at one scale and used at two others. This is the only part of the plan that addresses
  small-ball recall without new human labels.
- **Rules:**
  - a copy stays in the same split as its source frame, so nothing leaks between train and valid;
  - copies are generated by a tool with settings in config.
- **Open:** whether Roboflow's version settings can do this, or the copies are generated before upload. Task 1
  answers that.

### 3d. How much
A proposal to size the labelling. The real numbers depend on what the TRAIN footage yields:

| set | train (M002, M003) | valid (M001) | new human labelling |
|---|---|---|---|
| today (version 2) | 578 | 295 | - |
| small-ball positives (3a) | about 200 | about 60 | box the ball |
| hard negatives (3b) | about 200, at least 40 per category found | about 60 | confirm "not a ball", box real balls |
| scale copies (3c) | generated | generated | none |

Valid grows in step with train, so the new failure cases can be measured on a match the detector never trained on.

## 4. Labelling protocol
Same flow as detector v1 (pre-label, human review, freeze, preflight), with these rules:
1. ONLY THE MATCH BALL is boxed as `ball` (Nathan, decision 2). A spare ball (at the touchline, by the goal, in
   someone's hands off the pitch, a warm-up ball) gets NO box: there is one real ball, and a detector that reports
   others throws off everything after it.
   - Every frame with a visible spare ball is tagged `has-spare-ball`, so the effect of these frames can be measured
     and they can be taken out again without relabelling.
   - Not sure which ball is the match ball: the frame is excluded.
2. A look-alike never gets a box and never a class of its own.
3. Not sure whether it is a ball: the frame is tagged excluded.
4. Each reviewed frame carries a tag for why it was mined (`mined:small-ball`, `mined:negative:<category>`), so
   results can be reported per category.
5. Labels are frozen and the preflight passes before any training run.

### 4a. The existing labels must follow the same rule (added with decision 2)
The detector-v1 pre-labels boxed spare balls: `records/roboflow_2026-11-12/RECORD.md` notes "about ten boxes on spare
balls" in TRAIN_M002 and a spare ball at a goalpost, some flagged `review:ball-static`. Whether the human review
removed them from dataset version 2 is not recorded. If any remain, version 3 would hold both "spare ball = ball" and
"spare ball = background". Task 1 therefore includes an audit of the existing ball boxes for spare balls (starting
from the `review:ball-static` frames), for Nathan to review. No label is changed without that review.

### 4b. Known cost of decision 2
A spare ball and the match ball look the same in a single frame; only where they are differs. Teaching "this ball is
background" can make the detector more timid on real balls that look like it, such as a match ball at rest before a
free kick or a ball near the touchline. Checks A and F measure that. If recall falls, the `has-spare-ball` frames can
be excluded from training instead of used as background, which removes the conflicting examples while still never
teaching the detector to report a spare ball.

## 5. Training
- Dataset version 3 = version 2 + the frames above. Same recipe as v1 (`rfdetr-medium`, COCO checkpoint, input 1280,
  100 epochs, horizontal flip only), so that a difference comes from the data.
- Two runs, so the effect of each part is known:
  - **v2a:** + small-ball positives + hard negatives (3a, 3b).
  - **v2b:** v2a + scale copies (3c).
- Cost per run, from the v1 record: about 106 minutes and about 3.5 Roboflow credits.

## 6. Evaluation and adoption rule (to be recorded in DECISIONS BEFORE training)
Proposed. The numbers in the last column are Nathan's to set.

| check | data | tool | v1 today | proposed rule for v2 |
|---|---|---|---|---|
| A. Ball recall by size | valid (M001), incl. the new valid frames | `tools/detector_eval.py` | 8-12 px 0.81; 0-8 px 1/5 | not lower in any size bucket; higher at 8-12 px |
| B. Background called ball | valid | same | 13 | not higher |
| C. Other classes | valid | same | recall: player 0.952, goalkeeper 0.642, referee 0.929 | no recall drop over 0.01 |
| D. Mined negatives | new valid negatives, per category | same | (v1 fires on them by construction) | reported per category |
| E. DEV bench | 66 frozen frames | `tools/dev_bench_score.py` | ball recall 0.705 | not lower |
| F. Ball path, untiled | 5 labelled DEV clips | `tools/ball_score.py` | pooled coverage 0.687, 1 wrong SEEN | coverage +0.05 or more; no clip with more wrong SEEN |
| G. Ball path, tiled | same 5 clips | same | 0.706, 60 wrong SEEN | the tiling adoption rule of DECISIONS 2026-11-23, re-applied |
| H. The diagnosed objects | FRESH_02, FRESH_08 | `tools/diagnostics/ball_tiling_fp.py` | 7 accepted false stretches (6 with SEEN frames; 2 physical objects) | none accepted |
| I. Unseen clip(s) | a DEV clip not used so far, labelled after training | `tools/ball_score.py` | - | the ball bar: bound on wrong SEEN <= 2 % |

- A-D say whether the detector learned what it was shown.
- E-H are regression checks. Those clips have all been looked at and no longer count as independent.
- I is the only independent evidence, and it decides adoption. FRESH_10 and FRESH_09 are unused; they are Game 3, so
  they run on the local backend only.

## 7. Risks
- **Three TRAIN matches.** Cones, equipment and background look-alikes may be rare in them. If mining yields too few
  of a category, the plan cannot fix that category without more licensed TRAIN footage. That would be visible after
  task 2, before any labelling effort is spent.
- **Same venues.** Learning three grounds' equipment and crowd may not carry over to others.
- **Tiny-ball label noise.** At under 8 px a reviewer's box is itself uncertain. Hence the exclude rule.
- **Negatives can cost recall.** Teaching "not a ball" for ball-sized round objects can make the detector more
  timid on real small balls. Checks A and F are there to catch that.
- **Reused DEV clips.** Every DEV ball clip so far has been used to diagnose. Only check I is clean.
- **Speed.** If v2 makes tiling usable, detection costs about three times as much (1.2 s per frame on an M3). That
  is a separate decision.

## 8. Order of work (one CODE TASK each)
1. **Dataset tooling.** TRAIN-only miners for 3a and 3b, with the split guard and the reuse of the diagnostic's
   contact sheets. A read-only check of the train / inference scale observation in section 1. Whether scale copies
   are done in Roboflow or before upload.
2. **Mining run and yield report (Nathan's Mac).** How many candidates per category the three TRAIN matches give.
   STOP point: if a category is nearly empty, decide about more TRAIN footage before labelling.
3. **Review and labelling (Nathan).** Then freeze, preflight, dataset version 3.
4. **Training:** v2a, then v2b.
5. **Evaluation:** checks A-H for v1, v2a and v2b, side by side.
6. **Unseen clip(s)** and the adoption decision (check I).

## 9. Decisions (Nathan, 2026-11-24)
| # | question | decision |
|---|---|---|
| 1 | Scope: retrain the existing detector on more TRAIN data; no new class, no new model family | YES |
| 2 | Box every real football as `ball`, including spare balls | NO: only the match ball is boxed. There is one real ball; a detector that picks up several, or the wrong one, throws everything off. Spare balls get no box (section 4, rule 1) |
| 3 | The adoption rule of section 6 | YES, as proposed (recorded in DECISIONS before any training) |
| 4 | Labelling budget of about 520 frames to review | YES |
| 5 | Another licensed TRAIN match if task 2 shows too few look-alikes | YES, available. It needs its permission recorded and its entry on the allow-list (`config/roboflow.yaml`) before any frame is used |
