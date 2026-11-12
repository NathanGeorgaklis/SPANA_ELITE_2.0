# TRAIN_M001_MORETON_CITY: preparation status (2026-11-09): READY FOR LABELLING

## Match and rights
- Source: `Downloads/videoplayback (1).mp4`, the Moreton City Excelsior FQTV broadcast, 1280×720, 60 fps.
  sha256 `e8bd4d5fd8105a862ad8388f09473732b47481d6bda89ce10480ca18c810c519`.
- **Not a DEV match**: verified against G1, G2 and G3.
- **Manifest** (`Downloads/SPANA_reid/TRAIN_M001/TRAIN_M001_MORETON_CITY.reid_match.json`, schema 1.1.0): split
  TRAIN, rights holder Moreton City Excelsior. It validates.
- **Permission:** written permission dated 3 November 2026 covering commercial AI/ML development, including
  detection, tracking, re-ID, training, validation and testing. As stated by Nathan:
  - no minors in this match;
  - retention of derived data is covered;
  - continued model use after footage removal is covered.

## Window and video
- Window: 3000–3600 s of the source (50:00–60:00), live play throughout.
  - The stream-copy cut is AV1 (sha `2c61dd01f9503c7447571e59d76b190f686efbbca865596605d5b6cbf5807416`).
  - It was transcoded to H.264 because OpenCV here cannot decode AV1:
    `ffmpeg -c:v libdav1d -i CLIP -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -fps_mode passthrough -an`.
  - Result: all 36,170 frames, sha `2475f26c622cd91e6f23cbe15fa4fb11ae36ca8a63ed55be757e6dd719e04c58`. This is the manifest's `video_sha256`.
- Both clips are in `Downloads/SPANA_reid/TRAIN_M001/`.

## Detection and tracking (config/default.yaml, YOLOX-s, stride 4 = 15 Hz analysis)
- Detected in 4 chunks (0–9040, 9040–18080, 18080–27120, 27120–36170), about 1.4 h each, then merged with
  `tools/merge_detections.py`:
  - `TRAIN_M001.det.json` sha `cf12f80091db82f69390f3d5eeb1a47a141562d3edbbf54f19e88e118efcffbd`, 9,043 analysed frames.
- `TRAIN_M001.trk.json` sha `e2836bdf2da02251edb7a11b980ac9b11fb8b14d6440fcaff12bf9c39a3fbd6e`.

## Pack
- **Full proposal** (`--min-observed 3`): 4,716 tracklets, median 14 observed frames (< 1 s). Kept in the
  workspace only, not given to the labeller.
- **Labelling pack** (`--min-observed 15`, tracklets ≥ 1 s of observation):
  `Downloads/SPANA_reid/TRAIN_M001/TRAIN_M001_MORETON_CITY_reid_pack/`.
  - 2,354 tracklets; median 35 observed frames; 450 ≥ 5 s; 451 flagged by the tracker (identity contest).
  - tracklets sha `61b97d3a8101d404db3bbf196798da35d0e7e39157ab25833457701840e86b12` (equal on the Mac).
- **Shirt numbers** are legible in many thumbnails at this camera distance. They are recorded as an optional
  `shirt_number` per player.

Nothing has been trained. DEV and EVAL are not involved.

## Labels frozen (2026-11-10)
- Location: `data/reid/TRAIN/TRAIN_M001_MORETON_CITY/v1`. Annotator: nathan.
- Export `TRAIN_M001_MORETON_CITY_reid_v1.json` (Downloads, 307 KB). All freeze checks passed.

| item | count |
|---|---|
| tracklets | 2,354 |
| tracklets split by hand | 22 |
| parts | 2,378 |
| player parts | 837 |
| rejected | 937 |
| unsure | 604 |
| player boxes (MOT 1.1) | 40,076 |
| players | 25 |

- **Players:** 10 per side plus one keeper each, and 3 referees. 20 have a shirt number.
- **Observed frames per player:** median 1,606; min 184; max 4,624.
- **Split guard checked:** `load_frozen(..., "train")` accepts the labels and `"tune"` refuses them.

## Hash check on the Mac (2026-11-12)
- Every hash above was recomputed with `shasum -a 256` on the Mac copies and written out in full; before this
  they were recorded shortened (first 8 and last 3–5 hex characters).
- All files checked match:
  - source video;
  - AV1 window clip;
  - H.264 transcode (also equal to the manifest's `video_sha256`);
  - det.json and trk.json;
  - pack `tracklets.json`.
- On the Mac, det/trk are stored as `TRAIN_M001.det.json.xz` / `TRAIN_M001.trk.json.xz` in `TRAIN_M001/`.
  The hashes above are of the decompressed JSON.
