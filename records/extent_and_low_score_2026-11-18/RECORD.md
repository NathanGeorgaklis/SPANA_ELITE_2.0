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

## Update: code review and re-measure of A (commit `826d41f`)
An independent code review of `eecbc72` found that an S3c link across an unobserved gap (track LOST / OCCLUDED for
2–6 frames) was not recorded as a bridged gap and skipped S4's own-motion bound. Fixed:
- such a link must pass `_motion_consistent`;
- it is logged as a `recovery` (stage S3c, rule EXTENT_CHANGE) and counted in `recoveries` / `last_gap_s`.

While testing that fix, a second defect appeared, and it explains most of A's weak first result:
- the Kalman update averaged the old and new extents;
- so on the frame after a link neither box edge matched the prediction, and the track broke again.
- The linked box now replaces the box state (`BoxKalman.rebase_extent`, velocity kept).

Tests: 4 new cases (15 → 16 with the rebase case), each failing on the previous code; full suite 431 passed,
6 skipped. Re-tracked on the same detections (all-off again identical to the recorded tracks); B unchanged.

**Tracking:**

| | fragments | ID switches | transfers | silent | S3c links (continuing / across a gap) |
|---|---|---|---|---|---|
| today | 120 | 121 | 6 | 0 | — |
| A v1 (`eecbc72`) | 115 | 116 | 7 | 0 | 23 |
| **A v2** | **110** | **112** | 7 | 0 | 21 / 4 |
| A v2 + B | 109 | 111 | 8 | 0 | 20 / 2 |

**Following:**

| | correct, no help | wrong, no help | correct, reviewer | wrong, reviewer | actions / min |
|---|---|---|---|---|---|
| today | 0.509 | 1 | 0.778 | 2 | 6.27 |
| A v1 | 0.519 | 1 | 0.768 | 2 | 6.20 |
| **A v2** | **0.532** | 1 | **0.779** | 2 | **6.06** |
| A v2 + B | 0.540 | 1 | 0.772 | 1 | 5.85 |

Per clip, A v2: FRESH_07 fragments 38 → 29 and unaided 0.586 → 0.651; FRESH_08 no longer drops (51 fragments,
reviewer 0.688 vs 0.689).

**A v2 against the pass bar:**

| condition | result | met |
|---|---|---|
| reviewer correct >= 0.778 | 0.779 | ✓ |
| actions < 6.27 per min | 6.06 | ✓ |
| wrong frames <= 2 / unaided <= 1 | 2 / 1 | ✓ |
| silent transfers 0 | 0 | ✓ |
| transfers <= 6 | **7** | ✗ |

The one extra transfer:
- It is the same flagged FRESH_07 crossing as in v1 (track T000037, frame 366, IDENTITY_UNCERTAIN).
- Neither T000037 nor any rival of it received an S3c link.

A + B misses on correct time (0.772) and transfers (8).

**Status:** A and B stay OFF under the agreed bar. A v2 misses only on one flagged knock-on transfer. Nathan to
decide whether "transfers <= 6" may count only transfers on linked tracks, or only silent ones. In either case A v2
would ship.
