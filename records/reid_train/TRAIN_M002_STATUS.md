# TRAIN_M002_EASTERN_SUBURBS: preparation status (2026-11-11): READY FOR LABELLING

## Match and rights
- Source: `Downloads/videoplayback (2).mp4`, the FQTV broadcast of Eastern Suburbs v Brisbane City, night match,
  1280×720, 60 fps, 188 min. sha256 `b9f34c4a…23a3f`.
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
- AV1 stream-copy clip, sha `267e3010…54ba`, in `Downloads/SPANA_reid/TRAIN_M002/`.
- H.264 transcode, same command as TRAIN_M001: 36,300 frames, sha `62d91f7e…707e`. This is the manifest's
  `video_sha256`.

## Detection and tracking (config/default.yaml, YOLOX-s, stride 4 = 15 Hz)
- 4 chunks (0–9080, 9080–18160, 18160–27240, 27240–36300), merged.
- `TRAIN_M002.det.json` sha `30d82a9f…f470`.
- `TRAIN_M002.trk.json` sha `df13b9bf…d357`.

## Pack
- `--min-observed 15` (tracklets ≥ 1 s).
- Location: `Downloads/SPANA_reid/TRAIN_M002/TRAIN_M002_EASTERN_SUBURBS_reid_pack/`.
- 3,023 tracklets; median 41 observed frames; 800 ≥ 5 s; 424 flagged by the tracker.
- tracklets sha `f240c02f…be53` (equal on the Mac).

Nothing has been trained. DEV and EVAL are not involved.
