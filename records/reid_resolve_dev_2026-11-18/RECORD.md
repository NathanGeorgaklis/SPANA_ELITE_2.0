# Re-ID evidence for IDENTITY_UNCERTAIN, on DEV (2026-11-18)

Scope:
- `tracking/reid-resolve-v1` (568d743) on DEV FRESH_02 / 05 / 07 / 08, with the same local detections as every
  earlier DEV record.
- **Off:** `config/default.yaml` (reid.enabled false).
- **On:** reid.enabled true + target.resume_after_reid true.
  - Re-ID v2 `c269450e…`, margin 0.517 (the conservative value from `records/reid_calibration_dev_2026-11-18`,
    Nathan's go-ahead).
- No tuning. EVAL not used. Run on Nathan's Mac.

## What the code does
- **Tracker:** while a track is IDENTITY_UNCERTAIN, each clean, uncontested crop gets a re-ID verdict per rival:
  - cleared if sim(own reference) − sim(rival's reference at the contest) ≥ 0.517;
  - evidence against if ≤ −0.517;
  - otherwise no evidence.
- **Combining cues:** colour and re-ID are combined per pair. Either against resets the run; either clears. The
  existing 0.5 s sustained rule is unchanged.
- **`resolved_by`:** REID only when re-ID cleared every rival on every frame of the run, otherwise APPEARANCE.
  Recorded per track row (`spana.tracks` 2.2.0).
- **Follower:** answers again after an episode resolved by REID (`target.resume_after_reid`). Colour-cleared episodes
  still mean abstain for the rest of the track (`resume_after_uncertain: false`).
- **Off by default.** With it off, the tracks are frame-for-frame identical to the earlier runs on all 4 clips
  (frames and track summaries; only the new field is added).

## Results

| | re-ID off | re-ID on |
|---|---|---|
| tracks (frames, boxes, ids, states) | as before | **identical** to off (only `resolved_by` differs) |
| uncertain observed track-frames | 1,299 | 1,299 |
| episodes cleared | 4 (all APPEARANCE) | 4: **2 REID** (FRESH_02), 2 APPEARANCE |
| cleared episodes judged against labels | 3 right, 1 unknown | 3 right (both REID right), 1 unknown, **0 wrong** |
| follower, no user help: correct / wrong player-frames | 1,756 / 1 | **1,773 / 1** |
| follower + simulated reviewer: correct / wrong / actions | 2,685 / 2 / 90 | **2,702 / 2 / 90** |

The "off" follower numbers equal `records/target_recovery_v1_2026-11-17` exactly.

## Findings
1. **Safe:**
   - re-ID never changed an association: tracks are identical, so there are no new swaps, flagged or silent;
   - its 2 resolutions were right;
   - wrong-player time is unchanged.
   - The +17 correct frames come from the follower resuming after the 2 REID-cleared episodes on FRESH_02.
2. **The effect is small, because uncertain tracks rarely give a clean crop.**
   - Only 99 of 1,299 uncertain observed track-frames (7.6 %) are uncontested and overlap no other player by more
     than 5 % IoU (`appearance.clean_max_iou`).
   - FRESH_08 has 8 such crops in 515 uncertain frames.
   - Tracks are uncertain precisely while players are entangled, and often end before they separate.
   - Re-ID is limited by the evidence the tracker accepts, not by its accuracy.
3. **The re-ID references themselves are fine.** The two REID resolutions happened exactly where colour had cleared
   the same episodes before.

## Next (proposal)
- Measure re-ID on partly overlapped crops. The DEV rival test can be split by query overlap, and its labels mark
  occlusion. If a calibrated margin holds there at ≤ 0.5 % (held out), re-ID evidence could use crops up to that
  overlap, with its own config value. The colour rule stays clean-only.
- Ending fewer uncertain tracks before the players separate (tracker continuity) would also give re-ID more chances.
- Keep `reid.enabled` / `resume_after_reid` off by default until that measurement. Turning them on now is safe
  (results above) but worth little.
