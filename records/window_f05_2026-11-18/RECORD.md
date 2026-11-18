# 10-minute DEV window of a full match (2026-11-18): can the system follow one player for long?

Scope:
- DEV match G2 (`DEV_G2_NWS_SPIRIT_SD_RAIDERS`, `NWS_Spirit_SD_Raiders.mp4`, 720p, 30 fps, 142 min file).
- Full-match footage is covered by the written permission (`records/dev_permissions/`).
- Processed on Nathan's Mac only; no frame left it.
- A 10-minute window, run at Nathan's request ("track a player, see if it is capable"). Not a full-match run.
  Nothing tuned. EVAL not used.

## Steps
1. **Locate FRESH_05 in the full video** (`tools/long_run.py locate`): it starts at frame 171,150 = 5,705.0 s
   (95:05).
   - Thumbnail difference 0.34, against a median of 38.7 in the search windows.
   - Confirmed by the clip frame 5 s later (0.28).
2. **Analysis rate check** (`long_run subsample` on FRESH_05's existing local detections, re-tracked and scored,
   NESTED_V2). The match video is 30 fps, so every frame is needed.

   | rate (clip 60 fps) | match equivalent | coverage | ID switches | target correct | wrong |
   |---|---|---|---|---|---|
   | 60 fps | - | 0.998 | 7 | 0.716 | 0 |
   | 30 fps | every frame | 0.993 | 8 | 0.682 | 0 |
   | 15 fps | every 2nd | 0.991 | 26 | 0.553 | 1 (+1 unflagged transfer) |
   | 10 fps | every 3rd | 0.991 | 19 | 0.380 | 0 |

3. **Window:** frames 170,250–188,250 (94:35–104:35), with FRESH_05 inside.
   - Detected with `detect --backend local` (torch, MPS) in two chunks of 9,000 frames (Mac background commands
     stop after 2 h): 60 + 60 min, about 0.4 s per frame including decoding.
   - Joined with `tools/merge_detections.py`.
4. **Tracking:** 18,000 frames in 3.6 min.
   - The first attempt failed validation (fail closed): frame 174,397 named an occluder that was not live.
   - Tracker fix (branch `tracking/occluder-live-fix`, `tests/test_occluder_live.py`): only a matched track that
     survives the frame may be an occluder.
   - Adjacent check: FRESH_02 / 05 / 07 / 08 tracks are identical frame for frame before and after the fix.
5. **Follower:** FRESH_05 player 6 (labelled in 62 of 66 frames), selected from the labels (`long_run select`) on
   the window's first FRESH_05 frame: track T000217 at frame 171,150.
   - Default follower: offers shown, nobody acts.
   - Then `long_run overlap` (the labelled 26 s), `long_run summary`, and the review video (`tools/target_render.py`).

## Results
**Player 6, no user action.**
- At the selection frame his track was already IDENTITY_UNCERTAIN (the tracker could not tell him from a rival),
  so the follower abstained.
- His track ended after 7.3 s, when he left the picture on a camera pan (the same moment as in FRESH_05).
- The follower then has no evidence: LOST for the remaining 9.5 min, and no offer (offers cover gaps up to
  0.5 s).
- Labelled 26 s: 37 abstain, 25 lost, 0 answers. TARGET time over 10 min: 0.0.
- Wrong answers: 0.

**The whole window (all tracks), as a measure of how much user help following needs.**

| measure | value |
|---|---|
| confirmed tracks started | 1,804 in 10.0 min (about 180 per minute) |
| confirmed track life | median 2.8 s (p25 2.1, p75 5.0, p90 8.8, max 33.6) |
| how confirmed tracks end | LOST_EXPIRED 992, LEFT_FRAME 793, CAMERA_MOTION_UNKNOWN 5, still live 14 |
| confirmed players observed per frame | median 8 |
| confirmed track starts per visible player per minute | about 22 |
| observed track-frames IDENTITY_UNCERTAIN | 16.7 % |

## Findings
1. **Not capable yet of following a player over long play without help.**
   - On broadcast footage a player's track lasts about 3 s (median) before a camera pan (LEFT_FRAME) or a
     crowded moment (LOST_EXPIRED) ends it.
   - With the current tracker, following one player would need a tap or confirmation about every 3 s.
   - That is several times the 6.3 actions per followed player-minute measured on the short DEV clips
     (`records/target_recovery_v1_2026-11-17`), which were labelled every 0.2 s and are short.
2. **Fail closed holds over 10 minutes:** no wrong answer.
   - The cost is silence: identity uncertainty (16.7 % of track time) and track ends leave the follower with
     nothing it may say.
3. **The limits are upstream of the follower:**
   - tracker fragmentation (track life), especially camera pans that push players out of the picture and back;
   - no re-identification for re-entries.
   - The relink rule (gaps up to 0.5 s) cannot bridge pans of several seconds by design.
4. **Long runs work operationally:** 10 min = 2 h of local detection on the M3 at 30 fps, 4 min of tracking, a
   124 MB tracks file.
   - A full 90 min would be about 18 h of detection and over 1 GB of tracks. The tools would need streaming
     output before that is practical on an 8 GB Mac.
   - Analysing fewer frames is not an option (rate check above).

## What this points to (for Nathan)
- **Re-identification across camera pans and re-entries:** the SPANA re-ID model, and/or team classification to
  prune candidates. This is the step that turns "a tap every 3 s" into occasional confirmations.
- **Tracker work on broadcast footage:** keep tracks alive across pans (camera-compensated prediction outside the
  picture), and reduce LOST_EXPIRED in crowds.
- **Measure the same window again after each of these:** track life and taps per minute are the yardsticks.

## Files
- `HASHES.txt` (all artifacts stay on Nathan's Mac in `runs/window_f05/` and `runs/stride_check/`).
