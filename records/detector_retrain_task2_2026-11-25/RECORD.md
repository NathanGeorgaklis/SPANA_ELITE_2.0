# Detector retraining task 2: mining run (2026-11-25, in progress)

Footage: the three TRAIN clips only, each the 3000-3600 s window of its match (10 minutes), H.264 transcodes, on
Nathan's Mac under `Downloads/SPANA_reid/TRAIN_M00x/`. The full source broadcasts are no longer on the Mac.
Config: `records/ball_tiling_v1_2026-11-23/tiling_on.config.yaml`, detector `models/rfdetr_v1.json`, torch / mps.

## Run 1: `window_every_s: 120.0` (5 windows of 6 s per clip, 450 analysed frames per clip)

| Match | SEEN untiled | SEEN tiled | missed_ball | tiled_only | tiled_extends |
|---|---|---|---|---|---|
| TRAIN_M001_MORETON_CITY | 264 | 297 | 20 | 4 | 9 |
| TRAIN_M002_EASTERN_SUBURBS | 278 | 290 | 12 | 0 | 3 |
| TRAIN_M003_SYDNEY_OLYMPIC | 260 | 251 | 8 | 1 | 4 |
| Total | 802 of 1350 | 838 of 1350 | 40 | 5 | 16 |

61 candidate objects, not yet reviewed. That is far below the plan's budget (about 520 frames, at least
`min_per_category` 40 per look-alike category): at most 80 missed-ball frames and 21 possible look-alike objects.
Only 5 % of each clip was sampled (6 s in every 120 s), a spacing chosen for 90-minute matches.

## Change: `window_every_s: 30.0`

Samples 20 % of each clip (20 windows). Finished windows are reused, since every old window start is also a new one.
Decided before any candidate was reviewed. The plan's STOP point (more TRAIN footage) is judged after this run.

## Run 2: `window_every_s: 30.0` (20 windows per clip, 1800 analysed frames per clip)

| Match | SEEN untiled | SEEN tiled | missed_ball | tiled_only | tiled_extends |
|---|---|---|---|---|---|
| TRAIN_M001_MORETON_CITY | 1199 | 1255 | 57 | 8 | 19 |
| TRAIN_M002_EASTERN_SUBURBS | 1011 | 1079 | 56 | 4 | 18 |
| TRAIN_M003_SYDNEY_OLYMPIC | 1029 | 1062 | 47 | 3 | 20 |
| Total | 3239 of 5400 | 3396 of 5400 | 160 | 15 | 57 |

232 candidate objects, awaiting Nathan's review. Missed balls: up to 320 frames (`frames_per_gap` 2). Possible
look-alikes: 15 tiled-only objects plus whatever part of the 57 tiled-extends objects is not the match ball.
