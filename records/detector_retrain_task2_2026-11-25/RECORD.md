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

## Review (Nathan, 2026-11-26 to 2026-11-28)

Nathan answered per picture, in sheet order: `review_M001_raw.txt`, `review_M002_raw.txt`, `review_M003_raw.txt`, as
typed (short forms are mapped by `review_aliases` in `config/detector_mine.yaml`; "legs" is read as shin/sock,
"second ball" as spare ball, "unsure" as unclear). Every object of every match is answered, under its own kind, with no
more answers than the per-object picture limits. The exact per-picture count check needs each match's `.mine.json`
(on the Mac): `python -m tools.detector_mine review-apply --review FILE --match-id ID --out DIR`.

| choice | missed_ball | tiled_only | tiled_extends | pictures | note |
|---|---|---|---|---|---|
| match ball | 158 | 41 | 147 | 346 |  |
| spare ball | 0 | 6 | 0 | 6 |  |
| nothing there | 49 | 0 | 0 | 49 |  |
| boot/foot | 10 | 0 | 1 | 11 | THIN: under 40 |
| head | 2 | 0 | 0 | 2 | THIN: under 40 |
| glove/hand | 0 | 0 | 0 | 0 | THIN: under 40 |
| shin/sock | 9 | 6 | 4 | 19 | THIN: under 40 |
| pitch marking | 0 | 0 | 0 | 0 | THIN: under 40 |
| crowd/background | 8 | 29 | 6 | 43 |  |
| other ball | 0 | 0 | 0 | 0 | THIN: under 40 |
| equipment/cone | 7 | 7 | 0 | 14 | THIN: under 40 |
| unclear | 1 | 0 | 0 | 1 | excluded |

491 pictures, 232 objects. Per match, look-alike pictures: M001 68, M002 14, M003 8.

Reading:
- Small-ball positives: 346 pictures where the match ball was missed or only found with tiling. Enough for the plan.
- Hard negatives: 90 look-alike pictures in all, 43 of them crowd/background (the FRESH_08 failure type) and nearly
  all from M001. Equipment/cone (the FRESH_02 failure type) has 14. Every category but crowd/background is THIN.
- The plan's STOP point applies: more TRAIN footage is needed for the look-alike side before labelling.

## Check against the mined pictures (Nathan's Mac, 2026-11-28)

`review-apply`: M002 matches all 153 mined pictures, M003 all 149. M001 was one answer short on two objects, #28 and
#29 (tiled_extends, 6 pictures each, 5 answers each, all match ball). Nathan looked again: "#28 and #29 all match
ball". `review_M001_v2.txt` is the raw file with those two answers added; the raw file is kept as typed. Use v2.

Corrected totals: 493 pictures, 232 objects; match ball 348 (tiled_extends 149). Every other count is unchanged.

## Split of the clip yield (train = M002 + M003, valid = M001)

Mined frames stay in their match's split. Of the 90 look-alike pictures, 68 are from M001 (valid) and only 22 from
M002 + M003 (train): shin/sock 7, boot/foot 6, equipment/cone 4, crowd/background 3, head 1, unclear 1. So the
training side has almost no hard negatives from the clips; crowd/background reaches 40 only on the valid side.

## Full broadcasts (re-downloaded by Nathan, 2026-11-28)

The original downloads (1280 x 720) are gone; these are new downloads of the same matches as named by Nathan, at
1920 x 1080 and with different lengths, so they are different files from the ones the clips were cut from.

| Match | File (`Downloads/SPANA_reid/TRAIN_M00x/`) | Codec | Size | fps | Length | sha256 |
|---|---|---|---|---|---|---|
| TRAIN_M001_MORETON_CITY | `TRAIN_M001_MORETON_CITY_full.mp4` | av1 | 1920 x 1080 | 60 | 7054 s | `f4de17daf8a322ace13770bae28746bfdec19d1589a944bea239f7fa9bfd3023` |
| TRAIN_M002_EASTERN_SUBURBS | `TRAIN_M002_EASTERN_SUBURBS_full.mp4` | av1 | 1920 x 1080 | 60 | 6798 s (the first download: 188 min) | `0f6dbf0dd6435258ead32ca0b63fea56b90b5d4a1b7969529245685b7e5178ae` |
| TRAIN_M003_SYDNEY_OLYMPIC | `TRAIN_M003_SYDNEY_OLYMPIC_full.mp4` | h264 | 1920 x 1080 | 30 | 6941 s (the first download: 142 min) | `038b24491d04de01d38a27909a1a4c0c67f288717db98fa9d130b6290ca0aa74` |

Plan for them:
- Transcode to 1280 x 720 H.264 (the recipe of the clips plus `scale=1280:720`), so mined frames have the frame size
  and ball scale of dataset version 2 (all 1280 x 720) and OpenCV can read them.
- Mine at `window_every_s: 60.0`. Windows that overlap the already-labelled 3000-3600 s clip cannot be located by
  time (different files); near-duplicates of labelled frames are left to the dataset audit's pHash check in task 3.
- Nathan confirmed 2026-11-28: the same three matches at a larger size; he cut unused minutes from the start of each,
  which is why they are shorter than the first downloads.
