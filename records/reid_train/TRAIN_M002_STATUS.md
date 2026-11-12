# TRAIN_M002_EASTERN_SUBURBS: preparation status (2026-11-11): READY FOR LABELLING

## Match and rights
- Source: `Downloads/videoplayback (2).mp4`, the FQTV broadcast of Eastern Suburbs v Brisbane City, night match,
  1280×720, 60 fps, 188 min. sha256 `b9f34c4a6ffbde2cb090664baaf691ec862d39ed1321adc8f99e762582523a3f`.
- **Not a DEV match**; also a different club and kit from TRAIN_M001.
- **Manifest** `TRAIN_M002_EASTERN_SUBURBS.reid_match.json` (schema 1.1.0, in
  `Downloads/SPANA_reid/TRAIN_M002/`): split TRAIN, rights holder Eastern Suburbs. It validates.
- **Permission:** written permission dated 4 November 2026 covering commercial AI/ML development, including
  detection, tracking, re-ID, training, validation and testing. As stated by Nathan:
  - no minors in this match;
  - retention of derived data is covered;
  - continued model use after footage removal is covered.

## Window and video
- Window: 3000–3600 s (50:00–60:00), live play throughout (six frames checked).
- AV1 stream-copy clip, sha `267e301025ca6459096847b9e469ba0b808a5f202ee7a519703ba6db712d54ba`, in `Downloads/SPANA_reid/TRAIN_M002/`.
- H.264 transcode, same command as TRAIN_M001: 36,300 frames, sha `62d91f7e1b4d1adbcdaaf4bcab2943f9aa40be436ac45cf33104ee1748b0707e`. This is the manifest's
  `video_sha256`.

## Detection and tracking (config/default.yaml, YOLOX-s, stride 4 = 15 Hz)
- 4 chunks (0–9080, 9080–18160, 18160–27240, 27240–36300), merged.
- `TRAIN_M002.det.json` sha `30d82a9f8d34bdffbb638aa66196d2ae5c5e6d7964f44cc4d079109436f0e470`.
- `TRAIN_M002.trk.json` sha `df13b9bf0aa725ac1518bcd047f008037a41287d0f9158acec7d46ed6c69d357`.

## Pack
- `--min-observed 15` (tracklets ≥ 1 s).
- Location: `Downloads/SPANA_reid/TRAIN_M002/TRAIN_M002_EASTERN_SUBURBS_reid_pack/`.
- 3,023 tracklets; median 41 observed frames; 800 ≥ 5 s; 424 flagged by the tracker.
- tracklets sha `f240c02f20a8f02a6545bbdcdf36a1844a48a9500e4fb0e52c449a0014c9be53` (equal on the Mac).

Nothing has been trained. DEV and EVAL are not involved.

## Hash check on the Mac (2026-11-12)
- Every hash above was recomputed with `shasum -a 256` on the Mac copies and written out in full; before this
  they were recorded shortened (first 8 and last 3–5 hex characters).
- All files checked match:
  - source video;
  - AV1 window clip;
  - H.264 transcode (also equal to the manifest's `video_sha256`);
  - det.json and trk.json;
  - pack `tracklets.json`.
- **Correction:** `TRAIN_M002.det.json` was previously recorded as `30d82a9f…f470`. The file's hash ends in
  `…36f0e470`, so the shortened suffix was a transcription error. The file itself is unchanged:
  `TRAIN_M002.trk.json` (which matches its record) stores `detections_sha256` =
  `30d82a9f8d34bdffbb638aa66196d2ae5c5e6d7964f44cc4d079109436f0e470`, equal to the file on the Mac.
