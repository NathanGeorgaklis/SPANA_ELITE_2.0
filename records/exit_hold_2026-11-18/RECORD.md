# Exit hold (2026-11-18): keep a track alive when its player leaves the picture

Scope: DEV only (FRESH_02 / 05 / 07 / 08, and the 10-minute window of DEV match G2 from
`records/window_f05_2026-11-18`). Nothing tuned beyond one prior. EVAL not used.

## Why
- On the 10-minute window, 793 of 1,804 confirmed tracks ended LEFT_FRAME: camera pans, players at the edge of
  the shot.
- Nathan's example: a player's track ended as he went out at the left edge (frame 171,711) and he came back
  there 1.2 s later on a new track. The camera had moved only 17 px.

## Change (branch `tracking/exit-hold-v1`, config 2.3.0)
- A confirmed track whose prediction leaves the image can be held for `tracking.exit_hold_s`:
  - state LOST, velocity frozen where it left, still camera-compensated;
  - recoverable in S4 under the existing long-gap rules (exactly one unexplained high-score detection, no
    rival, motion-consistent, appearance matching and distinct by `appearance.min_margin`);
  - then LEFT_FRAME.
- Unconfirmed tracks still end on leaving.
- Tests: `tests/test_exit_hold.py` (4): back within the hold keeps the track; back after the hold gives a new
  track; a same-kit player in view blocks the recovery; the default is off.

## Results (hold 1.5 s)
| | FRESH clips (4) | 10-minute window, before | 10-minute window, hold 1.5 s |
|---|---|---|---|
| tracking metrics (NESTED_V2) | identical | - | - |
| follower / reviewer (target_score) | identical | - | - |
| confirmed tracks | - | 1,804 | 1,811 |
| ended LEFT_FRAME / LOST_EXPIRED | - | 793 / 992 | 499 / 1,291 |
| recoveries (all) | - | 235 | 219 |
| recoveries of held exits | - | 0 | 2 |
| recoveries refused: RIVAL_TRACK | - | 2,231 | 2,881 |

Track life looks longer with the hold (median 2.8 → 3.2 s), but only because held frames count as life. It is not
a gain.

**Nathan's player** (T000250 in the new run): held from frame 171,713.
- On re-entry (frame 171,745, 1.13 s out) the recovery was refused, APPEARANCE_NOT_DISTINCT: similarity to his
  held track 0.78 (needed 0.70), but only 0.085 above the next track (needed 0.10). Likely a teammate in the same
  kit.
- A new track was spawned, which then blocked the recovery as RIVAL_TRACK.

## Findings
1. **Holding works mechanically, but the safety rules for long gaps reject almost every re-entry.**
   - Torso-colour histograms do not separate teammates (margin below 0.10).
   - During pans several players leave and re-enter through the same edge, so their held tracks are rivals of
     each other.
2. **Net effect on the window is negative:** 2 re-entries recovered, 16 other recoveries lost. Held tracks at the
   edge add about 650 rival refusals.
3. **Relaxing the margin or the rival rule would trade this for identity swaps.** That is not acceptable (fail
   closed).

## Decision taken here (tracking layer)
- `exit_hold_s` default 0.0 (off): behaviour exactly as before.
- The code stays: with a re-ID model that tells teammates apart, the hold becomes the way to bridge pans.
- Revisit when re-ID exists.

## Files
- Artifacts on Nathan's Mac, `runs/window_f05/` (`G2_w_hold15.trk.json`, sha256
  0d358becb3c8a4d4833ffa868cc2cbd40e1acd6ff2270d0461c276ab2c821f86) and `runs/exit_hold_check/` (the FRESH re-runs).
