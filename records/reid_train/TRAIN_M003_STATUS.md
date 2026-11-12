# TRAIN_M003_SYDNEY_OLYMPIC: preparation status (2026-11-12): READY FOR LABELLING

## Match and rights
- Source: `Downloads/videoplayback.mp4`, a broadcast with the scoreboard reading SOL v MAR. 1280×720, 30 fps,
  142 min. sha256 `09aecc9b1ca2f64203864f25705618b03f35fa85cafd9610c51bdeecd585bfec`.
- **Not a DEV match.**
- **Manifest** `TRAIN_M003_SYDNEY_OLYMPIC.reid_match.json` (schema 1.1.0, in `Downloads/SPANA_reid/TRAIN_M003/`):
  split TRAIN, rights holder Sydney Olympic. It validates.
- **Permission:** written permission dated 4 November 2026 covering commercial AI/ML development, including
  detection, tracking, re-ID, training, validation and testing. As stated by Nathan:
  - no minors in this match;
  - retention of derived data is covered;
  - continued model use after footage removal is covered.

## Window and video
- Window: 3000–3600 s (50:00–60:00), live play throughout (frames checked).
- AV1 stream-copy clip, sha `17cc3e7a6924b02b5dea8a3c59a9cafe3f96505b53576b48e16af6ea307971ea`, in `Downloads/SPANA_reid/TRAIN_M003/`.
- H.264 transcode, same command as TRAIN_M001/M002: 18,149 frames at 30 fps, sha `3db10eb14e87a206f3968cb9530a86ae621d394bb97ba80ee95a530862a5da50`. This is the
  manifest's `video_sha256`.

## Detection and tracking (config/default.yaml, YOLOX-s, stride 2 = 15 Hz, as M001/M002)
- 4 chunks (0–4538, 4538–9076, 9076–13614, 13614–18149), merged.
- `TRAIN_M003.det.json` sha `a7033ee085a788f513b8d4939751e98649b5a0b371684be3420bc14b92995055`.
- `TRAIN_M003.trk.json` sha `f64a6234bef0d83b2f76d91b817e1761ec07751062416ce4e857155f1fbd8d1e`.

## Pack
- `--min-observed 15` (tracklets ≥ 1 s).
- Location: `Downloads/SPANA_reid/TRAIN_M003/TRAIN_M003_SYDNEY_OLYMPIC_reid_pack/`.
- 1,836 tracklets; median 42 observed frames; 449 ≥ 5 s; 318 flagged by the tracker.
- tracklets sha `9834a3d79d838192d9bc6cc89b5aa032b0073f7ad03974ed2f94849a3b5b88fa` (equal on the Mac).

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
