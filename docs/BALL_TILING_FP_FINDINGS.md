# Tiled-detector ball false positives on FRESH_02 and FRESH_08: findings

Owner layer: ball tracking (detector diagnostics). Analysis only: no detector, tracker, ball-path or config-default
change; tiling stays OFF. Branch `ball/tiling-fp-diagnostic`.

## Scope and data
- Both clips are DEV (label files `split: DEV`; `records/dev_validation_2026-11-07`: "all DEV; no EVAL used or
  created"). No EVAL data.
- Inputs: the outputs of the tiled run of `records/ball_tiling_v1_2026-11-23` (code dcc0124), on Nathan's Mac in
  `~/spana-ball-run/tile/`:
  - `<clip>.balldet.json`: ball-view detections;
  - `<clip>.ball.json`;
  - `<clip>.path.json`.

  Plus the frozen ball labels (FRESH_02 v1, every 3rd frame; FRESH_08 v2, every frame) and the untiled ball-view
  detections of the first run (`~/spana-ball-run/<clip>.balldet.json`).
- The saved detections are merged boxes and do not say which pass produced them. The detector is therefore re-run
  on the accepted false-positive frames only (full frame + both tiles, each pass recorded). No full-clip run.
- DEV frames stay on the Mac (contact sheets are local files). They are not added to any training set. The
  categories below say what to mine from TRAIN footage only.

## Method (`tools/diagnostics/ball_tiling_fp.py`, `config/ball_fp_diag.yaml`)
- "Wrong" is the scorer's rule (`tools.ball_score.judge`, radius from `config/ball_score.yaml`). Tile geometry, edge
  margin and merge IoU are read from the run's own `ball_config`.
- Every stretch of the ball path is judged on the labelled frames where it sighted something: REAL / FALSE / MIXED /
  UNLABELLED.
- Accepted false detections = every SEEN frame of a confident FALSE stretch, plus the labelled wrong SEEN frames of
  a confident MIXED stretch. One false object = one such stretch.
- Recorded per accepted false detection:
  - frame, box, size, score;
  - source: full frame / tile_left / tile_right / merged;
  - whether it is in the tile overlap;
  - whether it is also in the untiled detections;
  - its object's persistence.
- Bug checks:
  - the re-run reproduces the saved box;
  - the merged score equals the best pass score;
  - no accepted box overlaps a pass box cut at a tile seam.
- Categories are assigned per false object from the contact sheets, from the generic list in
  `config/ball_fp_diag.yaml`.

## Results (Nathan's Mac, 2026-11-24; outputs in `~/spana-ball-run/tile_fp/`, not committed)
`inspect` on both clips: analysis of the saved run, plus a detector re-run on the accepted false-positive frames only
(129 + 17 frames). Tables pasted from the tool's output.

Contact sheets (local, on the Mac):
- `~/spana-ball-run/tile_fp/FRESH_02_G1_3000_midfield_buildup_passing_fp_sheet_01.jpg`
- `~/spana-ball-run/tile_fp/FRESH_08_G3_2103_attack_into_box_clearance_fp_sheet_01.jpg`

### Accepted false detections
| | FRESH_02 (labels v1) | FRESH_08 (labels v2) |
|---|---|---|
| accepted false detections (SEEN frames of false objects) | 129 | 17 |
| of them scored wrong SEEN on labelled frames | 42 | 17 |
| false objects (confident FALSE stretches with a SEEN frame) | 5 | 1 |
| source pass | tile_right 129 | tile_left 17 |
| merged from several passes | 0 | 0 |
| in the overlap of the two tiles | 0 | 0 |
| also in the untiled detections | 0 (all 129 new with tiling) | 0 (all 17 new with tiling) |
| box size, longer side, median / max (px) | 10.5 / 11.3 | 20.3 / 22 |
| score, median / max | 0.39 / 0.66 | 0.45 / 0.63 |

### Each accepted false object
| clip | # | frames | accepted (SEEN) frames | seconds | longest run of consecutive frames | density | category |
|---|---|---|---|---|---|---|---|
| FRESH_02 | 1 | 25-85 | 28 | 2.80 | 19 | 0.72 | equipment/cone |
| FRESH_02 | 2 | 157-163 | 7 | 0.97 | 16 | 0.63 | equipment/cone |
| FRESH_02 | 3 | 258-268 | 10 | 2.17 | 15 | 0.77 | equipment/cone |
| FRESH_02 | 4 | 284-319 | 35 | 1.17 | 25 | 0.97 | equipment/cone |
| FRESH_02 | 5 | 335-385 | 49 | 1.67 | 40 | 0.96 | equipment/cone |
| FRESH_08 | 1 | 313-331 | 17 | 0.60 | 16 | 0.89 | crowd/background |

FRESH_02 has 6 confident FALSE stretches but 5 objects: one accepted false stretch has no SEEN frame. Objects 1-3
have far fewer SEEN frames than their stretch length. Both are consistent with frames being CONFLICT (a false and
another confident stretch at once), which would also explain the fall of the real ball's coverage on FRESH_02
(0.279 -> 0.155). NOT verified: the tool does not print conflict counts.

### Persistence: accepted vs rejected false objects, and the real ball (median / max per group)
| clip | group | stretches | seconds | longest run | density | sightings |
|---|---|---|---|---|---|---|
| FRESH_02 | false, ACCEPTED by the path | 6 | 1.42 / 2.80 | 17.5 / 40 | 0.77 / 0.97 | 42 / 61 |
| FRESH_02 | false, REJECTED by the path | 16 | 0.017 / 0.20 | 1.5 / 4 | 1 / 1 | 1.5 / 6 |
| FRESH_02 | real ball, accepted | 4 | 1.57 / 3.10 | 28 / 69 | 0.79 / 1 | 44 / 74 |
| FRESH_02 | real ball, rejected | 2 | 0.37 / 0.73 | 2 / 3 | 0.65 / 1 | 4 / 7 |
| FRESH_08 | false, ACCEPTED by the path | 1 | 0.60 / 0.60 | 16 / 16 | 0.89 / 0.89 | 17 / 17 |
| FRESH_08 | false, REJECTED by the path | 37 | 0.033 / 1.13 | 1 / 9 | 1 / 1 | 2 / 17 |
| FRESH_08 | real ball, accepted | 2 | 1.30 / 1.30 | 27.5 / 31 | 0.83 / 0.85 | 33 / 34 |
| FRESH_08 | real ball, rejected | 2 | 0.30 / 0.40 | 8.5 / 10 | 0.96 / 1 | 9.5 / 12 |

- Why the rejected false objects were rejected:
  - FRESH_02: too short 16.
  - FRESH_08: too short 31, too short and too sparse 4, too sparse 2.
- Stretches with no labelled frame (not judged): FRESH_02 14 (labels every 3rd frame), FRESH_08 0.
- MIXED stretches accepted: 0 on both clips. No accepted stretch mixes the real ball and a false object on labelled
  frames.

### Is any of it a tiling / merge bug? No
| check (accepted false detections re-run) | FRESH_02 | FRESH_08 |
|---|---|---|
| re-run | 129 | 17 |
| saved box not reproduced | 0 | 0 |
| merged score above the best pass score | 0 | 0 |
| overlapping a pass box cut at a tile seam | 0 | 0 |

- Every accepted false detection comes from ONE tile pass, away from the overlap. None is a merge, so nothing was
  boosted by merging, and none is a seam artefact.
- The full-frame pass has none of them. The enlarged tile made the detector fire on these objects.
- This is detector discrimination, not a defect of the tiling or merge code.

### What it means for the ball path's "brief false alarm" assumption
- The path accepts a stretch of >= 0.5 s seen on >= 50 % of its frames.
- The accepted false objects lasted 0.60-2.80 s, with runs of 15-40 consecutive frames and density 0.63-0.97. The
  real ball's accepted stretches on the same clips: median 1.30-1.57 s, density 0.79-0.83.
- By duration, run length, density, size and score, the accepted false objects are not separable from the real
  ball. Excluding the longest (2.8 s) would need a minimum above the real ball's median stretch.
- The false objects the path rejected were brief (median 1-2 frames). That is where the assumption holds.

### Categories (labelled by Nathan from the contact sheets, 2026-11-24; generic categories only)
| clip | category | false objects | accepted false frames | scored wrong SEEN |
|---|---|---|---|---|
| FRESH_02 | equipment/cone | 5 | 129 | 42 |
| FRESH_08 | crowd/background | 1 | 17 | 17 |
| both | equipment/cone | 5 stretches = 1 physical object | 129 | 42 |
| both | crowd/background | 1 | 17 | 17 |
| both | boot/foot, head, glove/hand, shin/sock, pitch marking, other ball, unclear | 0 | 0 | 0 |

- The labels were given in the session, per object number on the sheets. The `category` column of the
  `.categories.csv` files on the Mac was not filled, so `summarise` was not run; the counts above combine these
  labels with the per-object table.
- The five FRESH_02 "objects" are ONE physical object (Nathan, 2026-11-24): the same piece of equipment, picked
  up in five separate stretches between frames 25 and 385 of the 391-frame clip. It is in the picture, and
  mistaken for the ball, for almost the whole clip. This is the DESIGN's "persistent false object" risk
  (`records/ball_path_v1_2026-11-23/DESIGN.md`, risks added after the review, item 4).
- So this diagnosis rests on TWO physical objects: one piece of equipment and one background object.
- Only the false objects the path ACCEPTED were cropped and labelled. The 53 rejected false stretches (16 + 37) were
  not categorised.

### Ranked hard-negative categories from this diagnosis (to mine from TRAIN footage only)
1. **equipment/cone**: one physical object, accepted in 5 stretches, 129 of 146 accepted false frames. Small
   (about 10.5 px), ball-sized, and detected steadily for up to 2.8 s at a time across nearly the whole clip.
2. **crowd/background**: one object, 17 of 146 frames. About 20 px, 0.6 s.

The ranking rests on two physical objects in two clips. It says these categories matter; it does not say how often
each occurs in general.

Separate evidence, from the UNTILED detector (`records/ball_seen_share_2026-11-23`,
`records/ball_player_aware_2026-11-23`): jersey / shorts (body), boot/foot and glove/hand were mistaken for the ball
on FRESH_02 / 07 / 08. Those were single frames or short runs. They are hard negatives too, but they are a different
failure from the steady objects above.

DEV frames from this diagnosis must not be added to any training set.
