# Re-entry offers after the player left the picture: only at the side they left by (DEV, 2026-11-19)

Owner layer: target identity. Branch `identity/reentry-exit-side` = option 2 (`identity/reid-signatures-v2`) + the
fixed offer verdict (`evaluation/offer-verdict-offered-track`, records/offer_verdict_2026-11-19). Diagnostic code
`072e06e`, rule `3bbae47`. DEV FRESH_02 / 05 / 07 / 08 only (same tracks and 1.1.0 embeddings as
records/reid_signatures_v2_2026-11-19); no EVAL, no new video runs.

## Why
Nathan's FRESH_08 review: player 4 left the picture and the "is this them?" box went to player 19. With the fixed
verdict, every wrong re-ID offer on DEV is of this kind: our player was off screen and a different player in view was
offered (main 1, option 2 at 0.65: 3).

## Diagnostic (`tools/dev/reentry_exit_side.py`)
Geometry (`spana.identity.reid.exit_side`, the same function the rule uses): the lost track's last box carried
through the per-frame camera transforms to the candidate's birth. Off screen = the track ended LEFT_FRAME by then, or
the carried centre is outside the picture. Exit side = the side the carried centre is beyond (else, for a LEFT_FRAME
end at the edge, the nearest side of the carried box). edge_dist = the candidate's birth box distance from that side,
as a share of the frame width (left / right) or height (top / bottom). A camera CUT / UNKNOWN in between = unknown.

| | count | edge_dist median | 75 % | 90 % | max |
|---|---|---|---|---|---|
| labelled off-screen returns: the true track | 17 | 0.000 | 0.000 | 0.004 | **0.011** |
| other candidates of those returns (another player) | 328 | 0.098 | 0.399 | 0.666 | 0.985 |

At 0.05: 17 / 17 true returns kept, 121 / 328 other candidates kept (207 ruled out).

Follower offers (fixed verdict):
- **Wrong offers, all off screen:** FRESH_08 player 4 → T000040 (player 19), edge_dist 0.974; FRESH_05 player 2 →
  T000014 (player 11), 0.980; FRESH_07 player 11 → T000075 (player 5), 0.623.
- **Right offers made while off screen:** all at edge_dist <= 0.0105 (FRESH_02 6; FRESH_05 4, 5, 6, 7, 8, 9, 10;
  FRESH_08 6, 12 (both offers)).
- Every other right offer was made while the player's carried position was still in the picture: the rule does not
  apply to them.

## Rule (`target.reentry_exit_edge` on, `reentry_exit_edge_max_frac` 0.05)
- When the followed player left the picture, a track born more than 0.05 of the frame from the exit side is not a
  candidate: not ranked, not counted for the margin. It is listed in the decision's `exit_excluded` (logged in
  `reentry_log`).
- No effect while the player is still in the picture, or after a camera cut (unknown).
- 0.05 = a margin over the largest true value (0.011), not fitted further.
- **`spana.target` 2.7.0:** target_config + the two keys; `reentry_log` + `exit_excluded`. With the tracks, the
  validator re-checks every re-ID offer against the exit side. Config 2.9.0.
- **Tests** (`tests/test_reentry_exit_side.py`, 11):
  - geometry: LEFT_FRAME, carried out by the camera, in picture, cut;
  - a far-side lookalike is not offered and the return at the exit is;
  - only far-side tracks → NO_CANDIDATE;
  - the threshold comes from config;
  - no effect in picture or after a cut;
  - follower log;
  - the validator refuses an offer that breaks the rule ("born 0.078 from the side the player left by");
  - an exit-side offer is still made;
  - config range.

  The position-score and older offer tests pin the rule off to isolate their own mechanism. Full suite 500 passed,
  6 skipped.

## Measured
**Follower** (fixed verdict; offers right / wrong / undecided; reviewer correct / wrong frames / actions per min):

| config | exit rule off | **exit rule on** |
|---|---|---|
| mean 0.75 (main's matching) | 15 / 1 / 0; reviewer 0.777 / 2 / 5.78 | **15 / 0 / 0; reviewer 0.777 / 2 / 5.71** |
| set 0.60 | 20 / 3 / 1; 0.777 / 2 / 5.99 | 20 / 1 / 1; 0.774 / 2 / 5.71 |
| **set 0.65 (branch default)** | 18 / 3 / 0; 0.777 / 2 / 5.92 | **18 / 0 / 0; 0.777 / 2 / 5.71** |
| set 0.70 | 12 / 2 / 0; 0.778 / 2 / 5.85 | 12 / 0 / 0; 0.778 / 2 / 5.71 |

- CANDIDATE frames on another player while ours was absent: 12 / 37 / 37 / 13 → **0** in every row.
- Unaided 0.532 / 1 wrong frame in every row.
- No right offer was lost at any floor.
- At 0.60 a new wrong offer appears (FRESH_08 player 13 → T000097, player 9): with the far-side track ruled out,
  another candidate became the best. That lookalike was in the picture, so this rule cannot catch it.

**Offline** (`tools/dev/reid_reentry_eval.py`, mutual rule, accepted right / wrong):

| floor | mean off → on | set off → on |
|---|---|---|
| 0.60 | 47 / 4 → 47 / 3 | 48 / 3 → 48 / 4 |
| 0.65 | 43 / 4 → 43 / 3 | 41 / 3 → 41 / 3 |
| 0.70 | 41 / 3 → 41 / 2 | 32 / 1 → 32 / 1 |
| 0.75 | 36 / 2 → 36 / 2 | 22 / 1 → 22 / 1 |

Rank-1 right is unchanged (60 / 77 mean, 68 / 84 set).

## Findings
1. **A player who leaves the picture comes back in at the side they left by:** 17 / 17 on DEV within 1.1 % of the
   frame. Every wrong re-ID offer on DEV broke this.
2. **The rule removes every wrong offer in the follower at main's matching and at option 2's 0.65 / 0.70, and loses
   no right offer.** CANDIDATE frames showing another player while ours is off screen: 0.
3. **What it cannot do:**
   - Lookalikes inside the picture are untouched. The remaining offline wrong picks are of that kind, and so is the
     new wrong offer at 0.60.
   - It says nothing after a camera cut.
4. **The reviewer still takes about the same number of actions as with no offers** (5.71 vs 5.64 per minute; 0.777
   vs 0.778 correct). Offers make an action a one-tap accept; they do not remove it.
5. **Small sample:** 17 off-screen returns and 3 wrong offers. The 0.05 limit is a prior with margin, not a fit.
