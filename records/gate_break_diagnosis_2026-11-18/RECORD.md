# Gate and low-score breaks that ByteTrack bridged: diagnosis (DEV, 2026-11-18)

Step 2 of the plan. Diagnosis only: nothing changed.

Scope:
- **Breaks:** the 22 in-scope fragments of `records/continuity_attribution_2026-11-18` caused by the height-ratio
  gate (8), the motion (Kalman) gate (6 + 1 with height) or a low-score detection (7). They are on today's tracks
  (FRESH_02 / 07 / 08; FRESH_05 has none).
- **Tool:** `tools/dev/gate_break_trace.py`. At the first frame where ByteTrack's detection for the player is not on
  our track A, it reports:
  - A's state;
  - the detection's size and position against A's last observed and predicted boxes;
  - nesting with other detections;
  - the tracker's logged reason;
  - what happened to the detection.

## Trace

| clip | player | frame | cause | A | gap (fr) | score | h / pred | h / last | centre move (h) | IoU pred | Kalman m² | then |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F02 | 10 | 385 | height | OCCLUDED | 1 | 0.74 | 0.62 | 0.61 | 0.20 | 0.50 | 6.7 | spawn |
| F02 | 15 | 175 | height | LOST | 1 | 0.54 | 0.60 | 0.61 | 0.21 | 0.44 | 8.4 | spawn |
| F02 | 16 | 376 | height | OCCLUDED | 1 | 0.83 | 0.67 | 0.74 | 0.13 | 0.64 | 2.2 | spawn |
| F07 | 12 | 23 | height | OCCLUDED | 1 | 0.81 | 0.64 | 0.71 | 0.14 | 0.47 | 4.2 | spawn |
| F08 | 10 | 95 | height | OCCLUDED | 2 | 0.80 | 0.61 | 0.66 | 0.27 | 0.44 | 2.7 | spawn |
| F08 | 5 | 380 | height | OCCLUDED | 1 | 0.83 | 0.59 | 0.70 | 0.22 | 0.41 | 3.3 | spawn |
| F08 | 9 | 45 | height | OCCLUDED | 1 | 0.62 | 0.66 | 0.65 | 0.23 | 0.55 | 3.2 | spawn |
| F08 | 11 | 1 | height (unconfirmed) | OCCLUDED | 1 | 0.87 | 1.52 | 1.52 | 0.26 | 0.44 | 5.1 | spawn |
| F02 | 4 | 111 | motion | LOST | 1 | 0.84 | 1.26 | 1.47 | 0.25 | 0.56 | 15.6 | spawn |
| F07 | 12 | 54 | motion + height | LOST | 1 | 0.90 | 1.57 | 1.55 | 0.29 | 0.54 | 16.9 | spawn |
| F07 | 14 | 187 | motion | LOST | 1 | 0.91 | 1.37 | 1.53 | 0.29 | 0.55 | 19.5 | spawn |
| F07 | 17 | 325 | motion | LOST | 1 | 0.87 | 1.13 | 1.35 | 0.18 | 0.64 | 9.5 | spawn |
| F07 | 4 | 321 | motion | LOST | 1 | 0.91 | 1.35 | 1.38 | 0.18 | 0.59 | 11.8 | spawn |
| F07 | 8 | 10 | motion | LOST | 1 | 0.78 | 1.23 | 1.37 | 0.22 | 0.61 | 11.1 | spawn |
| F07 | 8 | 140 | motion | LOST | 1 | 0.88 | 1.40 | 1.45 | 0.23 | 0.65 | 10.0 | spawn |
| F02 | 1 | 317 | low score | OCCLUDED | 11 | 0.43 | 0.78 | 0.77 | 0.24 | 0.72 | 0.1 | none |
| F02 | 10 | 176 | low score | OCCLUDED | 2 | 0.45 | 1.03 | 1.01 | 0.05 | 0.86 | 0.0 | none |
| F07 | 15 | 250 | low score | LOST | 3 | 0.35 | 0.98 | 1.02 | 0.15 | 0.75 | 0.0 | none |
| F08 | 20 | 183 | low score | LOST | 4 | 0.30 | 0.96 | 0.95 | 0.13 | 0.73 | 0.2 | none |
| F08 | 19 | 145 | low score (unconfirmed) | LOST | 1 | 0.48 | 1.04 | 1.05 | 0.06 | 0.83 | 0.2 | none |
| F08 | 19 | 193 | low score (unconfirmed) | OCCLUDED | 1 | 0.40 | 1.01 | 1.01 | 0.02 | 0.89 | 0.0 | none |
| F07 | 4 | 342 | low score (misfiled) | OCCLUDED | 3 | 0.72 | 0.48 | 0.41 | 0.32 | 0.25 | 14.1 | spawn |

Gates: `tracking.max_height_ratio` 1.5 (refuses below 0.667); `tracking.gate_chi2` 9.21.

## Findings
1. **The 15 gate breaks are one phenomenon: the player's box changes extent, not position.**
   - **Height gate (7 of 8):** the box shrinks to 0.59–0.67 of the prediction, one frame after the last
     observation, while A is OCCLUDED. Part of the body goes behind another player and the detector boxes only
     the visible part.
   - **Motion gate (7):** the box grows 1.13–1.57× (vs the prediction; 1.35–1.55× vs the last box) as the body
     re-emerges. The centre moves 0.18–0.29 body heights with the extent change, so the centre-based Kalman gate
     fails (m² 9.5–19.5 > 9.21) though the player barely moved.
   - In all 15, the detection then spawned a new track on the spot.
   - The 8th height case is an unconfirmed track whose box grew 1.52×.
2. **This is the same partial-body problem as M2.** Its fix is deferred until Nathan approves, because a part box
   is sometimes a different real person: the box next to an occluding player could be the occluder's own part box.
   A centre-and-height gate cannot tell a partial body from another person. A rule that accepts extent changes
   would need to keep one box edge fixed (bottom for re-emerging legs, top for hidden legs) and stay exclusive
   against the occluder.
3. **Low score (7) is two small groups:**
   - **4 short re-associations ByteTrack makes and we refuse:**
     - A was not observed for 2–4 frames (one case 11);
     - the detection fits the prediction well (IoU 0.72–0.86, m² ≈ 0);
     - its score is 0.30–0.45.
     - Our S2 only continues tracks observed on the previous frame, and S4 recovery never uses a low score.
   - **2 unconfirmed tracks** (F08 player 19) refusing a low-score box. That refusal is deliberate (DECISIONS
     2026-11-08, M1: an unconfirmed track took a partial-body low-score box).
   - **1 misfiled case:** the box ByteTrack used scored 0.72; it belongs with the partial-body group.

## Proposed fixes (not built)
- **A. Partial-body extent change (15 breaks; overlaps M2, needs Nathan's approval).** For a track observed within
  `rematch_max_s`, accept a high-score detection that fails only the height / Kalman gate, when all of these hold:
  - the two boxes are nested (the smaller lies in the larger by >= `confirm_nested_ioa`);
  - they share the stable edge (top or bottom within `gate_base_h`);
  - the pair is exclusive (no other track gates the detection; no other detection in the track's gate).

  Open question for Nathan: certain, or flagged IDENTITY_UNCERTAIN? Flagged costs a review tap per case, as the
  doubt path showed.
- **B. Low-score short-gap recovery (4 breaks).** Let S4 recover a confirmed LOST / OCCLUDED track within
  `rematch_max_s` on a low-score detection with IoU >= `low_score_min_iou` with the prediction, under the existing
  exclusivity rules. Unconfirmed tracks still never take low-score boxes.
  - Small and generic.
  - About 4 fragments of 120 on DEV.
