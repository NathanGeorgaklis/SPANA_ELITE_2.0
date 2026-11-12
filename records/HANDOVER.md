# Handover: cloud chat sessions → Claude Code on Nathan's Mac (2026-11-12)

Read CLAUDE.md first. This note is the current state; the dated folders in records/ hold the evidence.

## Where things stand
- **Tracking/identity (DEV):** silent identity swaps eliminated on DEV. Key changes:
  - IDENTITY_UNCERTAIN state with pairwise resolution (records/identity_*);
  - camera-motion uncertainty (records/camera_uncertainty_*);
  - multi_body detection flag (records/multibody_flag_*);
  - low-score association for confirmed tracks only (records/unconfirmed_low_score_*).
- **Benchmark:** default rule NESTED_V2 (records/MATCHING_RULE_NESTED_V2.md); `--rule NESTED_V1` kept.
- **Frozen DEV labels:** data/labels/FRESH_02, FRESH_05, FRESH_07, FRESH_08 (v1).
- **Re-ID model:** design only (records/design/REID_MODEL_DESIGN.md). Nothing trained.
- **Re-ID TRAIN labelling** (workflow: records/design/REID_LABELLING_HOWTO.md):
  - TRAIN_M001_MORETON_CITY: labels frozen, data/reid/TRAIN/TRAIN_M001_MORETON_CITY/v1.
  - TRAIN_M002_EASTERN_SUBURBS: pack delivered; Nathan labelling.
  - TRAIN_M003_SYDNEY_OLYMPIC: pack delivered; Nathan labelling.
  - Status, rights and hashes for each: records/reid_train/TRAIN_M00x_STATUS.md.

## Data on the Mac (not in Git)
- `~/Downloads/SPANA_reid/TRAIN_M00x/`: labelling pack, manifest (`*.reid_match.json`), AV1 window clip.
- `~/Downloads/SPANA_reid/TRAIN_M00x/work/`, unzipped from `TRAIN_M00x_work.zip`:
  - H.264 transcode (its sha is the manifest's video_sha256);
  - `TRAIN_M00x.det.json` and `TRAIN_M00x.trk.json`.
- Check each against the hashes in its status record before use. The transcode is not reproducible
  byte-for-byte on another ffmpeg build, so reuse it rather than re-transcoding.

## Pending
1. Freeze TRAIN_M002 / TRAIN_M003 re-ID labels when Nathan exports them (tools/freeze_reid_labels.py),
   then commit and push.
2. **Deferred until Nathan approves:** the nested-box fix (a full-body box enclosing a partial-body track;
   records/partial_body_*).
3. **Not yet:** re-ID model training. It needs a GPU machine and Nathan's go-ahead.

## Conventions learned
- **Source footage:** often AV1, which OpenCV cannot decode.
  - Cut the window with an ffmpeg stream copy.
  - Transcode with: `ffmpeg -c:v libdav1d -i CLIP -map 0:v:0 -c:v libx264 -preset veryfast -crf 18
    -pix_fmt yuv420p -fps_mode passthrough -an`.
  - Record both hashes.
- **Analysis rate:** 15 Hz. Use `--stride 4` for 60 fps footage and `--stride 2` for 30 fps.
- **Re-ID packs:** `tools.reid_pack --min-observed 15` (≥ 1 s at 15 Hz).
- **TRAIN rights:** a TRAIN match needs written rights in the manifest.
  - Ask Nathan every time about minors and retention.
  - Never invent rights details.
- **Separation:**
  - DEV is for development and tuning only.
  - EVAL is never touched without Nathan's explicit approval.
  - Labels are frozen before any inference is scored against them, and never edited.
