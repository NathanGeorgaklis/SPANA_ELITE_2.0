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

## Results
PENDING: the diagnostic has to run on Nathan's Mac (the run outputs and the footage are there). Tables and
contact-sheet paths are added here from its output.
