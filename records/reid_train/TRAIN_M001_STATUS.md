# TRAIN_M001_MORETON_CITY: preparation status (2026-11-09): READY FOR LABELLING

## Match and rights
- Source: `Downloads/videoplayback (1).mp4`, the Moreton City Excelsior FQTV broadcast, 1280×720, 60 fps.
  sha256 `e8bd4d5f…c519`.
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
  - The stream-copy cut is AV1 (sha `2c61dd01…7416`).
  - It was transcoded to H.264 because OpenCV here cannot decode AV1:
    `ffmpeg -c:v libdav1d -i CLIP -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -fps_mode passthrough -an`.
  - Result: all 36,170 frames, sha `2475f26c…4c58`. This is the manifest's `video_sha256`.
- Both clips are in `Downloads/SPANA_reid/TRAIN_M001/`.

## Detection and tracking (config/default.yaml, YOLOX-s, stride 4 = 15 Hz analysis)
- Detected in 4 chunks (0–9040, 9040–18080, 18080–27120, 27120–36170), about 1.4 h each, then merged with
  `tools/merge_detections.py`:
  - `TRAIN_M001.det.json` sha `cf12f800…fbd`, 9,043 analysed frames.
- `TRAIN_M001.trk.json` sha `e2836bdf…bd6e`.

## Pack
- **Full proposal** (`--min-observed 3`): 4,716 tracklets, median 14 observed frames (< 1 s). Kept in the
  workspace only, not given to the labeller.
- **Labelling pack** (`--min-observed 15`, tracklets ≥ 1 s of observation):
  `Downloads/SPANA_reid/TRAIN_M001/TRAIN_M001_MORETON_CITY_reid_pack/`.
  - 2,354 tracklets; median 35 observed frames; 450 ≥ 5 s; 451 flagged by the tracker (identity contest).
  - tracklets sha `61b97d3a…6b12` (equal on the Mac).
- **Shirt numbers** are legible in many thumbnails at this camera distance. They are recorded as an optional
  `shirt_number` per player.

Nothing has been trained. DEV and EVAL are not involved.
