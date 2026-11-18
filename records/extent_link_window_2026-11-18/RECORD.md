# Extent-change link switched on; check on the 10-minute DEV window (2026-11-18)

## Decision (Nathan, 2026-11-18)
- `tracking.extent_change_link` (fix A v2, commit `826d41f`) is ON by default (commit `c427922`).
- The pass bar's transfer condition is revised from "transfers <= 6" to two conditions: silent transfers 0, and no
  transfer on a track the new rule linked. A v2 meets every condition
  (`records/extent_and_low_score_2026-11-18`).
- `tracking.low_score_recovery` (B) stays off.
- The occluder-live fix (`393ddad`, branch `tracking/occluder-live-fix`) is cherry-picked, because long DEV windows
  need it to validate.

## Check 1: the 4 labelled DEV clips under the new default
- Re-tracked from the same detections with `c427922`.
- Benchmark and target_score are identical to A v2 on all 4 clips, so the occluder fix changes nothing there:
  - fragments 110;
  - transfers 7, silent 0;
  - unaided correct 0.532, 1 wrong frame;
  - reviewer 0.779, 2 wrong frames, 6.06 actions per followed minute.

## Check 2: 10-minute DEV window of match G2
**Scope:**
- Frames 170,250–188,250 (94:35–104:35), from `records/window_f05_2026-11-18`.
- The existing local-detector detections were re-tracked with A off and on. No new detection, EVAL not used.
- With A off the run is identical to the recorded window tracks (same decision-log hash).
- **What it cannot show:** the window is unlabelled except FRESH_05's 26 s. So wrong-player frames over 10 minutes
  cannot be measured here.
  - That labelled part is scored as the DEV clip FRESH_05 in check 1. There it is unchanged: 7 fragments, reviewer
    0.964, 0 wrong.
  - Over the rest of the window, the safety signal used is whether a link is followed by new identity uncertainty
    more often than an ordinary moment of tracking.

| measure | A off | A on |
|---|---|---|
| confirmed tracks started in 10 min | 1,804 | **1,747** (−3.2 %) |
| confirmed track starts per visible player-minute | 22.6 | **21.8** |
| confirmed track life, median (p75 / p90), s* | 1.37 (3.57 / 7.53) | 1.40 (3.73 / 7.93) |
| ends: LOST_EXPIRED / LEFT_FRAME | 992 / 793 | 952 / 777 |
| observed confirmed track-frames IDENTITY_UNCERTAIN | 17.7 % | 17.8 % |
| S3c links (continuing / across a gap) | — | 124 / 47 |

\* first frame to last observed frame. This is a different definition from the 2.8 s median of the 17 Nov record,
so compare only the two columns here.

**Safety proxy:**

| within 1 s after... | a new uncertainty episode on that track | the track ends |
|---|---|---|
| an S3c link (171) | 12 (7.0 %) | 34 (19.9 %) |
| any confirmed observed track-frame, A off | 8.1 % | 9.4 % |
| any confirmed observed track-frame, A on | 8.0 % | 9.5 % |

## Findings
1. **No sign of harm over 10 minutes.**
   - Links are not followed by new identity uncertainty more often than ordinary tracking (7.0 % vs 8.0 %).
   - The uncertain share of track time is flat (17.8 % vs 17.7 %).
   - On the labelled DEV clips nothing got worse, including wrong-player frames.
2. **The gain on broadcast footage is small:** 3 % fewer track starts (57 in 10 minutes).
   - Track ends are dominated by camera pans (LEFT_FRAME 777) and crowds (LOST_EXPIRED 952), which this rule does
     not address.
   - The "about 22 track starts per visible player-minute" problem remains.
3. **Linked tracks end soon more often** (19.9 % within 1 s vs 9.5 %). The rule fires where bodies are partly
   hidden, at occlusions and near the picture edge, where tracks end anyway. A link there does not give a long
   track.
4. **Unmeasured:** wrong-player frames outside the labelled 26 s. A labelled long window would be needed. That is
   label work for Nathan, or the reviewer's own use in the app.

## Status
- A stays ON.
- The evidence is consistent across the 4 labelled clips and the 10-minute window. The gain is modest; the next
  levers are review effort (show only gaps above a configured length) and track survival across camera pans
  (re-ID, pan-aware prediction).
