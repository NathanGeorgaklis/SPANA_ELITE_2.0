# Target recovery v1 (2026-11-17): after-match review on 4 DEV clips

Scope: DEV only (FRESH_02, 05, 07, 08; frozen labels v1; local-detector tracks). Short clips, not a full match.
Nothing tuned. EVAL not used.

## What was run
- Code: commit 36ac9c2 (branch `identity/target-recovery-v1`), `spana.target` 2.1.0:
  - user actions `confirm_candidate`, `confirm_relink`, `reject_candidate`, `track_at_frame` (a tap
    re-selects);
  - the gap list;
  - `tools/target_score.py --simulate-reviewer`.
- Run on Nathan's Mac from a scratch extract of 36ac9c2 (`~/spana-recovery-run`); his project copy was not
  changed. 53 target / re-ID / guard tests passed there. Full suite in the container: 344 passed, 4 skipped.
- Three modes, default config:
  1. **default:** offers shown, nobody acts;
  2. **simulated user:** confirms each right offer on its first frame (as in `records/target_relink_v1_2026-11-16`);
  3. **simulated reviewer:** works through the gap list after the match, knowing the truth. It acts only on
     labelled frames (every 0.2 s). Per gap, in order:
     - confirm a right offer (`confirm_relink`) or decline a wrong one (`reject_candidate`);
     - otherwise tap the player (`track_at_frame`) on the gap's first labelled frame where they have an
       unambiguous track.

     This is the best case of the review UI (decisions 2026-11-16).

## Results (labelled player-frames, 4 clips pooled: 3449)
| mode | correct | **wrong** | abstain | lost | user actions | actions per followed player-minute |
|---|---|---|---|---|---|---|
| default | 1756 (0.509) | **1** | 0.087 | 0.345 (+ 0.050 offers) | 0 | 0 |
| simulated user | 2059 (0.597) | **2** | 0.105 | 0.288 | 16 accepts | 1.1 |
| **simulated reviewer** | **2685 (0.779)** | **2** | 0.120 | 0.092 | 66 taps, 22 accepts, 2 declines (90) | **6.3** |

Ceiling with perfect relinking alone: 0.853 (`records/reid_headroom_F07_F08_2026-11-16`).

Per clip, correct-time:

| clip | default | simulated user | simulated reviewer | reviewer actions (taps / accepts / declines) |
|---|---|---|---|---|
| FRESH_02 | 0.482 | 0.519 | **0.771** | 14 / 3 / 1 |
| FRESH_05 | 0.716 | 0.716 | **0.964** | 7 / 0 / 0 |
| FRESH_07 | 0.586 | 0.688 | **0.773** | 20 / 6 / 0 |
| FRESH_08 | 0.305 | 0.490 | **0.689** | 25 / 13 / 1 |

## Findings
1. **The after-match review is where most of the remaining time is recovered:** 0.509 → 0.779 correct-time, close
   to the 0.853 ceiling of perfect relinking.
   - FRESH_05, whose losses are all long re-entries that no offer covers, reaches 0.964.
   - Taps on the player recover what the tracker's links cannot.
2. **Wrong frames stay at 2, both already known:**
   - FRESH_07 player 14, frame 66: the two-person box at a crossing (v1).
   - FRESH_02 player 9, frame 276: the tracker's own track changing player after a correct acceptance
     (relink v1).

   The reviewer adds no new wrong frame. It declined the 2 offers that would have been wrong, including FRESH_02
   player 15's second hop, which caused the extra wrong frame under auto-confirm.
3. **Effort is the open problem:** 90 actions for 14.4 followed player-minutes, about 6 per minute or one every
   10 s.
   - If that rate held over a full 90-minute match, one player would need about 560 actions. That is too many for
     an after-match review.
   - The DEV clips were chosen for contact (duels, build-up, box clearances), so this rate may overstate a whole
     match. It has not been measured on longer footage.
4. **Not everything is recovered:** 171 gaps remain after the review (127 in the default mode; taps split gaps).
   - Most are short stretches where the followed track is uncertain or not yet confirmed (abstain 0.120), or
     where the player is not labelled.
   - Lost time falls from 0.345 to 0.092.
5. **No regression:** the default and simulated-user modes give exactly the counts of
   `records/target_relink_v1_2026-11-16`.

## Next (for Nathan)
- **Cut the review effort before building the review UI.**
  - Measure correct-time against the number of actions when the reviewer only handles gaps longer than a
    threshold, or the N longest gaps per minute. That threshold goes in config.
  - Measure on longer DEV footage (a full DEV half, if allowed) for a realistic rate per match.
- The tracker's fragmentation is still the source of most gaps. Fewer fragments mean fewer reviewer actions.

## Files
- `recovery_summary.json`: per mode, per clip, counts, actions, followed seconds, gaps left, relinks, wrong
  events.
- `HASHES.txt`.
