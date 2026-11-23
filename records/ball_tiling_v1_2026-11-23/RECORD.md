# Ball tiling v1 for the football detector (2026-11-23)

Owner layer: player detection (ball view). Branch `detection/ball-tiling-v1` (from main 921e0ba). Design, adoption
rule and the overlap correction: DECISIONS 2026-11-23 (recorded before building: 26656a4, and the correction before
any measurement).

## Why
On the unseen clip, all 55 frames where the ball was visible but unanswered were detector misses: no sighting within
the scoring radius, so nothing was discarded by the path's rules (`records/ball_path_v1_2026-11-23`). The football
detector resizes the 1920 px frame to 1280, so a 12 px ball is ~8 px.

## Built
- `spana/detection/ball_view.py`: with `ball.tile_size_frac` > 0 the view also runs the detector on square tiles.
  - Tile side = `tile_size_frac` x frame height.
  - Layout: the fewest evenly spaced tiles with at least `ball.tile_overlap_frac` (0.20) overlap. On 1920 x 1080 at
    1.0 that is two tiles at x = 0 and 840 (240 px overlap). The ball is then seen at ~1.19x instead of 0.67x.
  - Only the ball class is kept from tiles.
  - A tile box within `ball.tile_edge_margin_frac` (0.02 of the side, 22 px) of an INTERNAL tile edge is dropped: the
    other tile sees that ball whole.
  - Merged with the full-frame ball boxes by the view's NMS.
- Persons are never tiled: person detections and tracks are unchanged.
- Local backend only. Hosted inference refuses `tile_size_frac` > 0, since tiles would multiply third-party
  requests.
- Config 2.22.0. Default `tile_size_frac: 0.0` (OFF).
- The tile keys are part of the ball view (`VIEW_KEYS`).
- Cost: 3 detector passes per frame instead of 1.

## Tests
`tests/test_ball_tiling.py`, 15, with a stand-in detector that sees a small ball only on a tile:
- the tile layout;
- a ball found only with tiles, in frame coordinates;
- a ball in the overlap is one ball;
- a ball cut by an internal edge is taken from the other tile;
- frame-edge balls are kept;
- full-frame and tile detections merge;
- only the ball class comes from tiles;
- weak tile balls are dropped;
- tiling without a tile detector is refused (view and CLI);
- config validation.

Mutation checks: 7 changes, all 7 caught. Full suite: 788 passed, 15 skipped, 2 failed (`test_botsort_compare.py`:
the bench-only `supervision` package is not installed in the cloud container).

## Measurement plan (ONE setting, no sweep): `tiling_on.config.yaml` = default.yaml with `ball.tile_size_frac` 1.0
Five labelled clips, same labels as the path's results: FRESH_02 / 05 / 07 v1, FRESH_08 v2, FRESH_03 v2. Player tracks
reused (persons are not affected).

Baseline today (ball path on untiled detections, `records/ball_path_v1_2026-11-23`):

| clip | SEEN hits / VISIBLE | coverage | wrong SEEN |
|---|---|---|---|
| FRESH_02 | 36 / 129 | 0.279 | 0 |
| FRESH_05 | 123 / 130 | 0.946 | 0 |
| FRESH_07 | 99 / 124 | 0.798 | 0 |
| FRESH_08 (v2) | 60 / 162 | 0.370 | 0 |
| FRESH_03 (v2) | 313 / 373 | 0.839 | 1 |
| pooled | 631 / 918 | 0.687 | 1 |

ADOPTION RULE (DECISIONS): tiling goes on by default only if pooled coverage >= 0.737 (+0.05) AND no clip has more
wrong SEEN than in this table. None of these clips is unseen any more; an independent check needs a fresh clip.

## DEV measurement (Nathan's Mac, 2026-11-23; `~/spana-ball-run/tile/`): FAILS the adoption rule. Tiling stays OFF
- Code 052110d.
- `detect --backend local --ball-out` and `ball --tracks` with `tiling_on.config.yaml`; `ball-path` with
  `config/default.yaml`; player tracks reused.
- `tools.ball_score --clip LABELS PATH BALL` (each path recomputed).
- Detection took ~1.2 s / frame on an Apple M3 (8 GB): 3 passes.
- Numbers pasted from the terminal.

| clip | coverage: untiled -> tiled | wrong SEEN: untiled -> tiled (95 % bound, tiled) |
|---|---|---|
| FRESH_02 | 0.279 -> 0.155 | 0 / 36 -> 42 / 62 (0.775) |
| FRESH_05 | 0.946 -> 0.962 | 0 / 123 -> 0 / 125 (0.024) |
| FRESH_07 | 0.798 -> 0.855 | 0 / 99 -> 0 / 106 (0.028) |
| FRESH_08 (v2) | 0.370 -> 0.407 | 0 / 60 -> 17 / 83 (0.291) |
| FRESH_03 (v2) | 0.839 -> 0.887 | 1 / 314 -> 1 / 332 (0.014) |
| pooled | 0.687 -> 0.706 | 1 / 632 -> 60 / 708 (0.104) |

Adoption rule:
- pooled coverage >= 0.737: 0.706, **FAIL**;
- no clip with more wrong SEEN: FRESH_02 0 -> 42, FRESH_08 0 -> 17, **FAIL**.

Tiling is not adopted. `ball.tile_size_frac` stays 0.

Path summaries from the run log (tiled; frames / seen / hole / gap / no_ball / confident stretches of all):
- FRESH_05: 781 / 746 / 29 / 6 / 0 / 2 of 11;
- FRESH_07: 391 / 319 / 18 / 32 / 22 / 4 of 19;
- FRESH_08: 391 / 83 / 16 / 218 / 74 / 3 of 42;
- FRESH_03: 781 / 665 / 86 / 19 / 11 / 3 of 24.

No CONFLICT frames on any of these four.

What it shows:
- Where the detector already finds the ball (FRESH_05 / 07 / 03), tiling adds 0.02-0.06 coverage with no new wrong
  answer.
- On the two hard clips it produces wrong SEEN answers in bulk (FRESH_02: 42 of 62; FRESH_08: 17 of 83), and on
  FRESH_02 the real ball's coverage falls.
- NOT diagnosed: which objects, and why FRESH_02 coverage fell. Expected, unverified: enlarged non-balls are detected
  steadily enough to form confident stretches.
- The ball path does not stop a steadily detected false object (DESIGN risks 4 and 7). The 0 / 318 on the design
  clips depended on the untiled detector's false positives being brief or sparse. The path's protection is
  conditional on the detector, not absolute.
- The detector's limit on hard footage is ball / non-ball discrimination, not resolution. More pixels did not fix it.
  Detector data (small balls; boots, gloves, heads as hard negatives) is the remaining lever.
