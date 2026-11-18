# Review only gaps above a length (DEV, 2026-11-18)

Next step 1 after `records/extent_link_window_2026-11-18`.
- **Question:** does the after-match review need fewer actions if it shows only longer gaps?
- **Expectation going in:** many actions are spent on short gaps that would resolve on their own
  (`records/resume_after_doubt_2026-11-18`).

Built (commit `fb23909`, branch `identity/review-min-gap`):
- `target.review_min_gap_s` and `spana.identity.follower.review_list`: the review lists only gaps without a TARGET
  answer lasting at least that long.
- `gap_duration_s` = end_t − start_t, so a one-frame gap is 0 s.
- The `spana.target` document keeps every gap (no schema change).
- `tools/target_score.py --simulate-reviewer` works through that list.
- Tests: `tests/test_review_list.py` (4). Full suite 437 passed, 6 skipped.

Scope:
- **Tracks:** today's default tracks of DEV FRESH_02 / 05 / 07 / 08 (extent link on), scored by the simulated
  reviewer with the minimum swept. No re-tracking, EVAL not used.
- **Population:** 3,449 labelled player-frames, 14.4 followed player-minutes.
- Unaided, the same tracks give 0.532 correct and 1 wrong frame.

| review_min_gap_s | correct | wrong frames | actions | actions / followed min |
|---|---|---|---|---|
| 0 (every gap) | 0.779 | 2 | 87 | 6.06 |
| 0.1 | 0.779 | 2 | 87 | 6.06 |
| 0.2 | 0.778 | 2 | 86 | 5.99 |
| 0.3 | 0.778 | 2 | 85 | 5.92 |
| 0.5 | 0.778 | 2 | 84 | 5.85 |
| 0.75 | 0.778 | 2 | 83 | 5.78 |
| **1.0** | **0.778** | 2 | **81** | **5.64** |
| 1.5 | 0.778 | 2 | 81 | 5.64 |
| 2.0 | 0.773 | 2 | 76 | 5.29 |
| 3.0 | 0.766 | 2 | 68 | 4.74 |

Per clip at 1.0 s, correct / actions (vs every gap):

| clip | every gap | 1.0 s |
|---|---|---|
| F02 | 0.769 / 19 | 0.767 / 18 |
| F05 | 0.964 / 7 | 0.964 / 7 |
| F07 | 0.776 / 22 | 0.775 / 21 |
| F08 | 0.688 / 39 | 0.687 / 35 |

## Findings
1. **The expectation was wrong: short gaps cost little.**
   - Only 6 of 87 actions are on gaps shorter than 1 s.
   - Almost all review effort goes to gaps of a second or more. These are track losses and re-entries, not brief
     doubts.
2. **1.0 s is a safe default.** Compared with listing every gap:
   - actions −7 % (6.06 → 5.64 per followed minute; −10 % vs today's 6.27 before the extent link);
   - correct time −0.001 (0.779 → 0.778, still >= the 0.778 bar);
   - wrong frames unchanged.

   Beyond 1.5 s, real gaps are skipped and correct time falls.
3. **Chosen value:** `review_min_gap_s` 1.0 (default), the largest value meeting the bar on this sweep. It is fitted
   on 4 DEV clips, so it may move with more data.
4. **The remaining review effort is in long gaps,** that is, tracks that end:
   - camera pans (LEFT_FRAME);
   - crowds (LOST_EXPIRED);
   - re-entries the relink rule cannot bridge.

   Next step 2 (track survival across pans / re-identification) is where the effort is.
