# Extent-change link (A) and low-score short-gap recovery (B) (DEV, 2026-11-18)

Built from `records/gate_break_diagnosis_2026-11-18`, as designed and approved by Nathan on 2026-11-18:
- A counts as certain if it is safe;
- the approval covers A only, not the M2 duplicate-spawn fix;
- confirmed tracks only;
- same pass bar as before.

Commit: `eecbc72` on `tracking/extent-and-low-score-v1`.

What was built (both switches off):
- **A, `tracking.extent_change_link`, new stage S3c.**
  - Who: a confirmed track observed within `rematch_max_s`.
  - What it may take: a remaining high-score detection whose box changed extent. It must be nested with the
    prediction (>= `confirm_nested_ioa`), keep its top or bottom edge (<= `extent_edge_max_h` 0.15 body heights),
    have a height ratio within `extent_max_ratio` 2.0, sit within the horizontal gate, and pass the appearance
    check.
  - Exclusive only: no other track may gate or extent-fit the box, and the track may have no other candidate.
  - Links are certain. 0.15 and 2.0 are priors, not fitted.
- **B, `tracking.low_score_recovery`, in S4.**
  - A confirmed track unobserved for at most `rematch_max_s` may recover on a low-score detection overlapping its
    prediction by >= `low_score_min_iou`.
  - All other S4 rules apply.
- **Tests:** `tests/test_extent_and_low_score.py` (11):
  - off breaks, on links (top or bottom edge kept);
  - no link when no edge is kept;
  - no link with two candidate boxes, or when another track could take the box;
  - unconfirmed tracks are excluded from both rules;
  - B recovers a short gap but not one longer than `rematch_max_s`.

  Full suite 426 passed, 6 skipped.

Scope:
- **Clips:** DEV FRESH_02 / 05 / 07 / 08, re-tracked on Nathan's Mac from the existing local-detector detections.
  Off = today's recorded tracks, identical on all 4 clips.
- EVAL not used.
- **Pass bar:**
  - reviewer correct >= 0.778;
  - actions < 6.27 per followed minute;
  - wrong frames <= 2 (unaided <= 1);
  - silent transfers 0;
  - transfers not above 6.

## Results

**Tracking (benchmark NESTED_V2, pooled):**

| | fragments | ID switches | transfers | silent | wrong recoveries | links / recoveries |
|---|---|---|---|---|---|---|
| today | 120 | 121 | 6 | 0 | 0 | 0 / 18 |
| A | 115 | 116 | **7** | 0 | 0 | 23 extent links |
| B | 116 | 117 | **7** | 0 | 0 | 46 recoveries |
| A + B | 114 | 115 | **8** | 0 | 0 | 20 / 41 |

Per clip, fragments today → A → B:

| clip | today | A | B |
|---|---|---|---|
| F02 | 24 | 24 | 22 |
| F05 | 7 | 8 | 7 |
| F07 | 38 | **30** | 37 |
| F08 | 51 | 53 | 50 |

**Following (3,449 labelled player-frames, 14.4 followed player-minutes):**

| | correct, no help | wrong, no help | correct, reviewer | wrong, reviewer | actions / min |
|---|---|---|---|---|---|
| today | 0.509 | 1 | 0.778 | 2 | 6.27 |
| A | 0.519 | 1 | 0.768 | 2 | 6.20 |
| B | 0.518 | 1 | 0.769 | 1 | 6.13 |
| A + B | 0.527 | 1 | 0.761 | 1 | 5.99 |

## Findings
1. **Neither fix meets the bar.**
   - Both raise flagged transfers above 6 (A 7, B 7, A + B 8; silent stays 0).
   - Reviewer correct time falls by about 1 point (0.761–0.769).
   - Actions fall slightly (5.99–6.20).
   - Unaided correct time rises slightly (0.518–0.527); wrong frames stay at 1 unaided, and fall from 2 to 1 with
     the reviewer for B and A + B.
2. **The extra transfers are not on linked tracks.** The new one under A (FRESH_07 T000037, frame 366) and under B
   (FRESH_02 T000033, frame 270) are on tracks that received no extent link or new recovery. Both are flagged
   IDENTITY_UNCERTAIN crossings: longer-lived neighbouring tracks met a crossing that, before, happened to a
   track that had already ended. So flagging A's links would not remove them.
3. **The links themselves behave.** 19 of A's 23 links continue on the next frame, and none ends within 0.5 s.
   FRESH_07 loses 8 fragments.
   - But FRESH_08 (the set-piece crowd) gains 2 fragments, and its reviewer correct time falls 0.689 → 0.640.
   - One changed association there re-routes later ones. I have not isolated the chain.
4. **For B,** the follower's IDENTITY_UNCERTAIN frames rise by 195 and AFTER_UNCERTAIN by 63 (all video frames,
   reviewer run). Tracks that survive more short gaps take part in more crossings, and the fail-closed follower
   then abstains for the rest of the track.
5. **Pattern across the four attempts** (doubt link, resume after doubt, A, B): each safely removes some
   fragments. But the follower then loses about as much answered time to the uncertainty that longer tracks
   collect, and the reviewer still needs about one action per gap. On these clips, fragment count alone is not the
   lever for review effort.

## Status
- All switches stay **off** under the agreed bar. The product is unchanged.
- Mechanisms, tests and logs (`extent_link`, S4 low-score `recovery`) remain behind config.
- Decision for Nathan:
  - B misses two conditions: transfers 7 > 6 (all flagged) and reviewer correct time 0.769 < 0.778. It lowers wrong
    frames with the reviewer (2 → 1) and actions (6.13). Shipping it would mean trading about 1 point of correct
    time for 1 wrong frame and 0.14 actions per minute fewer.
  - A should stay off until the FRESH_08 drop is understood.
