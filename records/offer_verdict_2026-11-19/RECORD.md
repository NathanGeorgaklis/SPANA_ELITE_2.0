# Offer verdict: judge the offered box, also while our player is off screen (DEV, 2026-11-19)

Owner layer: evaluation (`tools/target_score.py`). Code `3502196`, branch `evaluation/offer-verdict-offered-track`
(from main `5c7dac5`). DEV FRESH_02 / 05 / 07 / 08 only; no EVAL, no new video runs.

## Finding (Nathan's review of the FRESH_08 videos)
Player 4 leaves the picture at frame 49; the follower offered T000040 at frame 64 ("is this them?"). Nathan: the box
was the wrong player. T000040 is player 19 (his own followed track from frame 56; NESTED_V2 attribution on frame 66).
The scorer reported this offer as "undecided": `relink_verdict` and `offer_verdict` judged an offer only on labelled
frames where the FOLLOWED player is labelled, and player 4 was out of the picture. An offer made while our player is
off screen could therefore never be counted wrong - exactly the situation where wrong offers happen.

## Change
- `relink_verdict` / `offer_verdict`: the first labelled frame (from the proposal on) where the offered track's box is
  clearly attributed decides - right if it is our player, wrong if it is another labelled player - whether or not our
  player is labelled there. Nothing else in the scoring changed.
- New count `candidates_while_absent_on_other_player` (per player and pooled), next to the existing
  `answers_while_absent_on_other_player` (TARGET only).
- The simulated reviewer now declines such offers (one action) instead of skipping them.
- Test (`tests/test_target_relink.py`): player 1 leaves, the offered track is player 3 → verdict wrong (player 3),
  2 candidate frames on another player while absent, the reviewer declines it and shows no wrong frame. It fails on
  the old scorer and passes on the new one. Full suite 476 passed, 6 skipped.

## Re-scored: main's defaults (same tracks and 1.0.0 embeddings as records/reentry_position_score_2026-11-18)
| | unaided correct / wrong | re-ID offers right / wrong / undecided | predecessor offers r / w / u | CANDIDATE frames on another player while ours absent | reviewer correct / wrong / actions per min |
|---|---|---|---|---|---|
| main, old scorer | 0.532 / 1 | 15 / 0 / 1 | 10 / 0 / 2 | (not counted) | 0.777 / 2 / 5.71 |
| **main, fixed scorer** | 0.532 / 1 | **15 / 1 / 0** | 10 / 0 / 2 | 12 | 0.777 / 2 / **5.78** |
| no re-ID offers, fixed scorer | 0.532 / 1 | – | 10 / 0 / 2 | 0 | 0.778 / 2 / 5.64 |

- The one wrong re-ID offer is FRESH_08 player 4 → T000040 (player 19).
- The 2 undecided predecessor offers stay undecided: the offered box is never clearly attributed to anyone.
- With the fix, re-ID offers cost the reviewer slightly MORE actions than no offers (5.78 vs 5.64 per minute) for
  0.001 less correct time on DEV.

## Earlier records affected
"Undecided" offers in `records/reentry_reid_offers_2026-11-18`, `reentry_position_mutual_2026-11-18`,
`reentry_position_score_2026-11-18` and `reid_signatures_v2_2026-11-19` were judged by the old rule; some were wrong
offers. Main's current numbers are corrected above; option 2 is re-scored in its own task. Older configurations are not
re-run (superseded).
