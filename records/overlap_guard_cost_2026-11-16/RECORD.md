# Overlap guard on the target follower (2026-11-16): cost measured before deciding

Scope: DEV only (FRESH_02, 05, 07, 08; frozen labels v1). Analysis; the follower is unchanged. EVAL not used.

## Why
- The only wrong answer on DEV so far is FRESH_07, player 14, frame 66: one box around two crossing players, on a
  track the tracker still marked CERTAIN and not contested (`records/reid_headroom_F07_F08_2026-11-16`).
- Nathan asked to measure a guard before deciding, with the cost reported separately for contested and near-ball
  moments. The rule he chose: overlap treated as identity uncertainty, with a minimum duration and/or a hold.

## What was run (Nathan's Mac, scratch extract of branch `identity/reid-headroom-v1` at 29470a8)
1. `tools/ball_pass.py` on the 4 clips: the local detector (torch, MPS), ball class only, at `detector.confidence`
   0.50, every frame.
   - About 160 s per 391-frame clip, 313 s for FRESH_05. Written to `runs/ball_pass/` (not committed).
   - Frames with a ball box: F02 85 / 391, F05 716 / 781, F07 290 / 391, F08 87 / 391.
   - DEV ball recall is about 0.70, so near-ball moments are under-counted.
2. `tools/guard_cost.py` on the same local tracks as `tools/target_score.py`, default config. It uses the same
   selection and the same v1 follower.

## Guard measured
- A TARGET answer becomes ABSTAIN once the followed box has overlapped another live track's box (the tracker's
  `max_overlap_iou`) at ≥ iou for at least min_s seconds, and stays ABSTAIN for hold_s seconds after.
- Grid: iou 0.2 / 0.3 / 0.4; min_s 0 / 0.1 / 0.2; hold_s 0 / 0.2.
- Evaluated on the players' labelled frames:
  - **cost** = correct answers turned into ABSTAIN;
  - **benefit** = wrong answers turned into ABSTAIN.
- Slices:
  - all;
  - contested (the followed track's association `contested` on that frame);
  - near the ball (ball box centre within 1 or 3 target-box heights of the target's foot point).

## Results, 4 clips pooled
v1 baseline over the 4 clips: 1756 correct, 1 wrong labelled player-frames. Of the correct ones, 1 is contested,
101 are within 1 height of the ball and 386 within 3 heights.

Cost columns give lost / baseline correct frames in that slice; "all" is the cost over all moments.

| iou | min_s | hold_s | TARGET time guarded | all | contested | near ball (1 h) | near ball (3 h) | wrong avoided (of 1) |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0 | 0 | 5.6 % | 99 (5.6 %) | 1 / 1 | 4 / 101 | 14 / 386 | 1 |
| 0.2 | 0 | 0.2 | 8.9 % | 161 (9.2 %) | 1 / 1 | 8 / 101 | 27 / 386 | 1 |
| 0.2 | 0.1 | 0 | 3.6 % | 60 (3.4 %) | 0 / 1 | 3 / 101 | 11 / 386 | 1 |
| 0.2 | 0.2 | 0 | 2.4 % | 39 (2.2 %) | 0 / 1 | 2 / 101 | 7 / 386 | 0 |
| 0.3 | 0 | 0 | 2.6 % | 42 (2.4 %) | 1 / 1 | 3 / 101 | 8 / 386 | 1 |
| 0.3 | 0 | 0.2 | 4.6 % | 79 (4.5 %) | 1 / 1 | 5 / 101 | 12 / 386 | 1 |
| 0.3 | 0.1 | 0 | 1.5 % | 18 (1.0 %) | 0 / 1 | 1 / 101 | 6 / 386 | 0 |
| 0.3 | 0.2 | 0 | 0.9 % | 14 (0.8 %) | 0 / 1 | 1 / 101 | 3 / 386 | 0 |
| **0.4** | **0** | **0** | **1.1 %** | **17 (1.0 %)** | 0 / 1 | 1 / 101 | 5 / 386 | **1** |
| 0.4 | 0 | 0.2 | 2.3 % | 38 (2.2 %) | 0 / 1 | 4 / 101 | 8 / 386 | 1 |
| 0.4 | 0.1 | 0 | 0.5 % | 5 (0.3 %) | 0 / 1 | 0 / 101 | 2 / 386 | 0 |
| 0.4 | 0.2 | 0 | 0.2 % | 2 (0.1 %) | 0 / 1 | 0 / 101 | 1 / 386 | 0 |

The rest of the grid (min_s 0.1 / 0.2 with hold 0.2) is in the full report.
- Each of those avoids the same wrong frames as its hold-0 row: 1 only at iou 0.2 with min_s 0.1.
- Each costs more than its hold-0 row, for example iou 0.2 / 0.1 s / hold 0.2: 100 correct lost.

Per clip at iou 0.4, min_s 0, hold 0 (correct lost / baseline correct, TARGET frames guarded / all TARGET frames):

| clip | correct lost | TARGET frames guarded | wrong avoided |
|---|---|---|---|
| FRESH_02 | 3 / 491 | 59 / 3029 | 0 |
| FRESH_05 | 0 / 322 | 6 / 3877 | 0 |
| FRESH_07 | 9 / 706 | 43 / 4314 | 1 |
| FRESH_08 | 5 / 237 | 25 / 1448 | 0 |

## Findings
1. **A minimum-duration rule ("held for several frames") catches the wrong frame only at a low threshold.**
   - The two-person box came at the very start of the overlap: IoU 0.24 at frame 64, 0.45 at frame 66.
   - With min_s 0.1 s it is caught only at iou 0.2, which costs 3.4 % of correct answers.
   - At iou 0.3 / 0.4 any minimum duration misses it.
   - The immediate guard (min_s 0) catches it at every iou up to 0.4; the frame's overlap was 0.45.
2. **Holding after the overlap roughly doubles the cost and buys nothing** on DEV (the one wrong frame is caught
   without the hold).
3. **Cheapest variant that catches it: iou 0.4, immediate, no hold.**
   - Cost: 1.0 % of correct answers (17 of 1756), 1.1 % of TARGET time.
   - The next cheapest that catch it are iou 0.3 immediate (2.4 %) and iou 0.2 with min_s 0.1 s (3.4 %).
   - Near the ball the cost is in proportion: 1 of 101 within 1 height, 5 of 386 within 3 heights.
   - Caution: the iou 0.4 threshold is fitted to a single event (0.45 against 0.4); it has no margin and no
     evidence beyond that one frame.
4. **Contested moments cannot be costed on DEV:** only 1 labelled TARGET frame is on a contested association.
   Contested matches and confident answers rarely meet, because the tracker's own uncertainty already keeps the
   follower quiet there.
5. **Near the ball the guard costs about what it costs elsewhere**, so it would not disproportionately blank the
   moments that matter most for clips. Ball recall about 0.7 means this is an estimate.

## Decision needed (Nathan)
- Adopt the immediate overlap guard (config: iou 0.4, min_s 0, hold 0), trading about 1 % of correct time for the
  one known wrong frame.
- Or keep v1 as it is and revisit when more wrong frames are known.
- The minimum-duration rule as proposed would have prevented the DEV wrong frame only at iou 0.2 (3.4 % cost).

## Files
- `HASHES.txt`: full report, compact summary, ball passes, input tracks (all on the Mac).
