# Detector retraining, task 1: dataset tooling, scale check, spare-ball audit candidates (2026-11-24)

Owner layer: player detection (dataset tooling). Branch `detection/ball-retraining-plan`. Plan and decisions:
`docs/DETECTOR_BALL_RETRAINING_PLAN.md`, DECISIONS 2026-11-24. Nothing labelled, uploaded or trained.

## 1. Scale check (read-only; the plan's "observation to verify")
Training side, from the frozen labels of dataset version 2 (`python -m tools.detector_mine scale --labels ...`):
875 frames, ALL 1280 x 720, so the detector (input 1280) was trained at 1.0x.

| split | ball boxes | short side p05 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|---|
| train | 414 | 9 | 11 | 13 | 16 | 23 |
| valid | 225 | 9 | 11 | 13 | 16 | 23 |

(641 ball boxes incl. the 2 platform-test frames; the percentiles above are of all 641: 9 / 11 / 13 / 16 / 23.)

DEV side. The DEV clips are 1920 x 1080: the full frame is shown at 0.667x, a frame-height tile at 1.185x. Ball size
is taken from the detector's own boxes (median hypothesis diameter, `records/ball_score_dev_2026-11-23`), because
the hand-drawn DEV labels are looser (about 1.3-1.5x) and the training labels came from detector pre-labels.

| clip | detector box at source (px) | at the detector, full frame | on a tile | coverage today |
|---|---|---|---|---|
| FRESH_02 | 12.0 | 8.0 | 14.2 | 0.279 |
| FRESH_07 | 16.4 | 10.9 | 19.4 | 0.798 |
| FRESH_08 | 17.3 | 11.5 | 20.5 | 0.370 |
| FRESH_05 | 19.4 | 12.9 | 23.0 | 0.946 |

- The plan's observation holds for FRESH_02 only: its ball reaches the detector at ~8 px, below the 5th percentile
  of training (9 px).
- It does NOT explain FRESH_08: 11.5 px is inside the trained range (p25 11), yet coverage is 0.370. FRESH_07 at
  10.9 px has 0.798. Something other than size limits FRESH_08; not diagnosed.
- On a tile every clip's ball is at or above the training median, FRESH_05's at the 95th percentile.
- Consequence for the plan, section 3c: shrunk copies target FRESH_02-type footage, tile-scale copies the range
  14-23 px. Scale copies will not fix FRESH_08-type misses.
- Caveat: the DEV sizes are medians of detected balls only. Balls the detector misses may be smaller.

## 2. Spare-ball audit of the existing labels: candidates
`python -m tools.detector_mine spare-audit` lists every ball box that is in a frame carrying the review flag
`ball_static` or `ball_offpitch` (`records/roboflow_review/review_flags_20261114T052034Z.json`), or in a frame with
more than one ball box.

- Frames with 0 / 1 / 2 ball boxes in version 2: 235 / 639 / 1. The review of 2026-11-14 left one frame with two
  balls (TRAIN_M001 f0020508, valid).
- Candidates: 31 ball boxes in 30 frames (`spare_ball_audit_candidates.md`):
  - 15 `ball_static`: 11 in TRAIN_M002, 4 in TRAIN_M003. They include M002 f0033108-f0034080, noted on 2026-11-12
    as a pale round object at the pitch-side fence and a spare ball at the goalpost.
  - 14 `ball_offpitch`: all TRAIN_M003.
  - 2 boxes in the two-ball frame.
- The contact sheets need the project image cache (`runs/roboflow_cache`, on Nathan's Mac). NOT yet reviewed. No
  label is changed without Nathan's review.

## 3. Mining tool (`tools/detector_mine.py`, `config/detector_mine.yaml`)
- TRAIN only: `mine` refuses any match not on the allow-list with split `train`, before the video is read. DEV and
  EVAL matches are refused. Local detector; frames stay on the machine.
- One pass over the match video (the reader decodes from frame 0, so windows are processed as they pass). Windows of
  6 s every 120 s, analysed at 15 frames per second whatever the video's rate: 45 windows and about 4,050 analysed
  frames for a 90-minute match. (TRAIN_M001 and M002 are 60 fps, M003 30 fps: `records/roboflow_uploads`.)
- Per frame the detector runs on the full frame and on each ball tile. Two ball paths per window come from the same
  passes: UNTILED (as the product runs) and TILED.
- Candidate objects:
  - `missed_ball`: a hole, or an OPEN gap of at most 1 s, in the untiled path. The ring marks the interpolated
    position; it is a hint, not a label.
  - `tiled_only`: a confident tiled stretch with no untiled answer on it. A real ball the full frame cannot see, or a
    look-alike.
  - `tiled_extends`: frames the tiles add to a ball the untiled path also has.
- Tiled objects are taken from each confident stretch's own sightings, not from SEEN frames. A look-alike in CONFLICT
  with the real ball has no SEEN frame at all (as on FRESH_02) and would otherwise be skipped. Found while writing
  the tests.
- Output per match: contact sheets per kind, a review CSV with one choice per object, and `<match>.mine.json`.
  - Choices: `match ball`, `spare ball`, `nothing there`, or a look-alike category.
  - Resumable per window. `--max-windows N` for a trial run.
- `yield` reports objects and frames per choice, and flags a look-alike category under 40 mined frames as thin.

## 4. Scale copies: where they are generated
Not verified whether a Roboflow version setting can shrink-and-pad or tile-crop. Decision for task 3: generate the
copies before upload with a tool (settings in config, a copy tagged with its source frame, same split as its
source). That is reproducible and independent of the platform's options.

## Tests
- `tests/test_detector_mine.py`, 9:
  - the facts above, read from the real frozen labels (875 frames at 1280 x 720, medians 13 px, 31 audit boxes in 30
    frames);
  - window layout;
  - the three candidate kinds on a scripted scene, including a look-alike in conflict with the real ball;
  - long gaps excluded;
  - TRAIN-only guard (DEV matches and unknown matches refused);
  - the `mine` command end to end on a generated video with a stand-in detector, resume, review file, yield report,
    invalid choice refused.
- The diagnostic's sheet rendering was split into reusable functions; its 10 tests are unchanged and pass.
- Full suite: 807 passed, 15 skipped, 2 failed (`test_botsort_compare.py`: the bench-only `supervision` package is
  not installed in the cloud container).

## Next: task 2 (Nathan's Mac)
The spare-ball audit sheets and the mining run on the three TRAIN match videos, then the yield report.

## Spare-ball audit: Nathan's review (2026-11-25, from the contact sheets on his Mac)
| result | boxes | which |
|---|---|---|
| match ball (box stays) | 30 | #1, #3-#31 (#5 confirmed separately: the first list had "#4" twice) |
| spare ball (box to be removed) | 1 | #2: TRAIN_M001_MORETON_CITY f0020508 (valid), box [1158, 695, 1215, 720], the second ball box of the one two-ball frame; its other box (#1) is the match ball |
| not a ball | 0 | |
| unsure | 0 | |
| not answered | 0 | |

- The existing labels already follow "only the match ball is boxed" except for one box, in the valid split.
- The `ball_static` and `ball_offpitch` flags did not find spare balls here: all 29 flagged boxes are the match ball
  (Nathan: every ball marked match ball was in play).
- The two-ball frame (TRAIN_M001 f0020508), settled with Nathan 2026-11-25: the ball in play is the one in box #1,
  which is boxed. The match ball he saw unringed in picture #2 is that same ball: each picture rings one box only.
  There is no third ball. The only change needed is removing box #2 (the spare ball).
- Not done: the box of #2 is still in the Roboflow project. It is removed in task 3, when dataset version 3 is
  prepared. The frozen labels of version 2 are not edited.
- Limit of this audit: it looked at flagged frames and the two-ball frame only. A spare ball boxed as the only ball
  in an unflagged frame would not be in the 31.
