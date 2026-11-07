# FRESH_02 benchmark against frozen DEV labels v1 (2026-11-08)

Roles: Failure Investigator, Player Detection Engineer, Multi-Player Tracking Engineer, QA / Regression Engineer.
No code changes to detection or tracking. No EVAL used.

- Labels: `data/labels/FRESH_02_G1_3000_midfield_buildup_passing/v1`, frozen and committed in `863ba6b` BEFORE any
  inference on FRESH_02. Annotator: nathan. 66 frames, 18 people, 1018 boxes, 207 ignore regions.
- Run: detect + track at commit `08152ac` (same detection and tracking code as `863ba6b`), 30 fps, all frames.
  Scored on the 66 labelled frames only; detections inside ignore regions removed.

## Official result (IoU >= 0.5, the pre-set rule)

| metric | value |
|---|---|
| detection recall | **0.727** (visible 0.735, occluded 0.444) |
| missed players | 278 / 1018 |
| duplicate detections | 131 |
| ID switches (player-centric) | 52 |
| identity transfers | 6 (3 unflagged) |
| fragments | 45 across 18 people |
| wrong recoveries | 0 |
| track coverage | 0.718 |

## Largest error: missed players. Cause: box convention, not the detector
- 271 / 278 misses have a confident detection (median score 0.87) centred on the player (centre offset 0.03 h)
  with IoU 0.3–0.5; 6 have no overlapping detection.
- The label boxes are looser than the detector's boxes: median 138 × 59 px vs 103 × 36 px. On matched pairs the
  detector box is 0.88 × the height and 0.74 × the width of the label box.
- Visual check (`missed_players_label_vs_detector.jpg`): the detector's boxes are tight and correct; the label
  boxes include margins. 2 of 8 samples also show label slips (a copied box not moved; one box over two players).
- Several remaining misses repeat the same label height across consecutive labelled frames (players 6, 1, 9):
  likely copied boxes that were not adjusted.

## Diagnostic only (IoU >= 0.3; NOT an official number, the rule was chosen after seeing results)
Recall 0.976 (24 missed), duplicate detections 122, ID switches 68, transfers 8 (3 unflagged), fragments 60,
wrong recoveries 0. With the box-convention artefact removed, **fragmentation is the largest real error**.
Of 68 track changes: 31 are the old track going unobserved with no decision logged (the tracker writes no
record when a continuing track simply gets no match); 9 LOW_SCORE recovery refusals; 9 NO_UNEXPLAINED_DETECTION;
9 AMBIGUOUS_WITH_HIDDEN_TRACK; 6 RIVAL_TRACK; others 4.

## Round 2 (2026-11-08): frozen rule NESTED_V1 + unmatched-track logging
- Rule `NESTED_V1` frozen in `aa59da5` before re-scoring (records/MATCHING_RULE_NESTED_V1.md).
- Logging of every unmatched existing track added in `7a12e4c`. Tracker behaviour proven identical: synthetic
  QA catalogue tracks sha256 equal to `aa59da5`; FRESH_02 frames and tracklets identical.

| metric (NESTED_V1) | value |
|---|---|
| detection recall | **0.971** (visible 0.977, occluded 0.741) |
| missed players | 30 / 1018 |
| duplicate detections | 123 |
| ID switches | 67 |
| identity transfers | 8 (3 unflagged) |
| fragments | **59** across 18 people |
| wrong recoveries | 0 |

Fragmentation causes (tools/dev/fragmentation_causes.py: first logged loss of the old track for each of the 67
player track changes): 23 old track still observed (the player moves to a newer track), 22 TAKEN_BY_OTHER
(passing detection taken by another track), 8 AMBIGUOUS_WITH_HIDDEN_TRACK refusals, 9 gate failures
(Mahalanobis / height / appearance / low score), 5 other.
Competing tracks: 23 / 31 traceable were born overlapping the old track while it was observed. At those births,
15 involve 2 labelled people in contact and 1 involves 3 (`fragmentation_competing_track_births.jpg`: players
in contact and player + referee). On labelled frames the detector output 115 boxes spanning ≥ 2 labelled
people (92 with score ≥ 0.5), against 150 labelled contact pairs.
First failing layer: detection output in contact situations (two-person boxes the union rule cannot remove,
because only one member has its own box).
