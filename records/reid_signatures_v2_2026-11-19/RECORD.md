# Re-entry offers: better re-ID signatures (option 2, DEV, 2026-11-19)

Nathan: "start option 2". Option 1 (position as a score) was neutral in the follower; this tries better appearance
evidence for the re-entry offer. Owner layer: target identity. DEV only (FRESH_02 / 05 / 07 / 08), no EVAL, no full
match. Code `3ac7aba` (mechanism, defaults = old behaviour) and `b43a77e` (new defaults), branch
`identity/reid-signatures-v2`.

## Built
- **Crop sets, `spana.track_embeddings` 1.1.0** (`spana.identity.reid.select_set_samples`, `track_embeddings`).
  Per track end, up to `reid.set_samples` 10 crops, `set_min_gap_s` 0.2 s apart, under looser rules than the 1.0.0
  signature: height >= `set_min_height_px` 24 (was 40), overlap with other observed boxes <= `set_max_iou` 0.30 (was
  0.05); still confirmed, CERTAIN, uncontested. Each crop has a quality weight in (0, 1]:
  min(1, height / 40) × (1 at overlap <= 0.05, falling linearly to 0.1 at 0.30). Priors, not fitted.
  The 1.0.0 averaged signatures are still written; 1.0.0 documents still validate. The validator checks every crop
  frame is a confirmed, CERTAIN observation of its track, frames increasing, quality in (0, 1].
- **Best-pair matching** (`set_similarity`, `appearance`): each crop's best cosine in the other set, averaged with
  the crops' quality weights, both directions, mean of the two. `target.reentry_signature` = `mean` (before) or
  `set`. Used for the forward choice and the two-way (mutual) check in `reentry_choice`, and in
  `tools/dev/reid_reentry_eval.py` (`--signature`).
- **Wait for evidence** (`target.reentry_wait_samples`): a candidate is judged only from its crops up to the current
  frame, and only once it has that many.
- **Withdraw** (`target.reentry_withdraw`): each frame the rule is re-run on the evidence so far; an offer it no
  longer makes for that track is dropped (`dropped_reason` WITHDRAWN, logged in `reentry_log`); another track may
  be offered.
- **`spana.target` 2.6.0**: target_config + `reentry_signature, reentry_wait_samples, reentry_withdraw`; with
  embeddings the validator recomputes set similarities (candidate crops limited to the proposed frame when waiting);
  WITHDRAWN needs 2.6.0; `set` needs 1.1.0 embeddings. Config schema 2.8.0.
- Tests: `tests/test_reid_signatures_v2.py` (13: quality weight, one occluded crop does not sink the match,
  low-quality crops count less, wait, withdraw + document validates, policy off keeps the offer, mean ignores crop
  sets, tampered set similarity refused, WITHDRAWN needs 2.6.0, set needs crop sets, crop on an unconfirmed frame
  refused, bad quality refused, config checks); `tests/test_reid_embed.py` (crop sets from the real embed path,
  spacing, set similarity separates the two shirts). Full suite 488 passed, 6 skipped.

## Measured (DEV, same tracks as option 1: out/F0x_mdefault.trk.json; re-embedded as 1.1.0 on Nathan's Mac)
**Offline, 94 labelled re-entries (`tools/dev/reid_reentry_eval.py`, mutual rule, position weight 0.5, margin 0;
all crops, not causal):**

| | usable (lost track has evidence) | true track has evidence | rank-1 right / of |
|---|---|---|---|
| mean (before) | 79 | 67 | 60 / 77 (78 %) |
| set | 84 | 78 | 68 / 84 (81 %) |

Accepted offers right / wrong by floor (the scales differ: set similarities run lower):

| floor | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|---|---|
| mean | 47 / 4 | 43 / 4 | 41 / 3 | **36 / 2 (shipped)** | 24 / 1 | 19 / 0 |
| set | 48 / 3 | **41 / 3** | 32 / 1 | 22 / 1 | 10 / 1 | 4 / 0 |

At equal wrong: 1 wrong → set 32 right vs mean 24; 3 wrong → set 48 vs mean 41. At 0 wrong mean is better (19 vs 4).
The chosen set 0.65 is +5 right and **+1 wrong** offline vs the shipped mean 0.75.

**In the follower (`tools/target_score.py`, offers only for tracks usable at the time):**

| config | unaided correct / wrong | re-ID offers right / wrong / undecided | withdrawn (right) | reviewer correct / wrong / actions per min |
|---|---|---|---|---|
| mean 0.75 (= main, reproduced exactly) | 0.532 / 1 | 15 / 0 / 1 | – | 0.777 / 2 / 5.71 |
| set 0.75 | 0.532 / 1 | 5 / 0 / 0 | – | 0.777 / 2 / 5.57 |
| set 0.70 | 0.532 / 1 | 12 / 0 / 2 | – | 0.778 / 2 / 5.71 |
| **set 0.65 (new default)** | 0.532 / 1 | **18 / 0 / 3** | – | **0.777 / 2 / 5.71** |
| set 0.60 | 0.532 / 1 | 20 / 0 / 4 | – | 0.777 / 2 / 5.78 |
| set 0.75, wait 3 + withdraw | 0.532 / 1 | 5 / 0 / 1 | 2 (1) | 0.774 / 2 / 5.57 |
| set 0.70, wait 3 + withdraw | 0.532 / 1 | 11 / 0 / 4 | 4 (2) | 0.767 / 2 / 5.57 |
| set 0.65, wait 3 + withdraw | 0.532 / 1 | 17 / 0 / 5 | 6 (3) | 0.762 / 2 / 5.57 |
| set 0.60, wait 3 + withdraw | 0.532 / 1 | 17 / 0 / 8 | 6 (1) | 0.763 / 2 / 5.64 |

Committed default config (`b43a77e`) reproduces the set 0.65 row exactly.

Set 0.65 vs mean 0.75, offer by offer:
- New right offers (5): FRESH_02 player 6 (0.68), FRESH_05 players 4 (0.67) and 6 (0.69), FRESH_08 players 6 (0.73)
  and 9 (0.76).
- New undecided (2): FRESH_05 player 2, FRESH_07 player 11.
- Lost right offers (2): FRESH_02 player 4 (set 0.62 < 0.65; mean 0.79) and FRESH_08 player 14 (set 0.83 passes the
  floor; lost to the two-way / ranking check). Player 14 is one of the FRESH_08 review videos Nathan is watching.

## Findings
1. **Crop sets give evidence to more tracks**: 84 vs 79 usable re-entries, true track has evidence 78 vs 67. Rank-1
   right 81 % vs 78 %.
2. **Best-pair matching moves the accuracy / coverage curve up in the middle** (same wrong count, +7 to +8 right
   offline), not at the strict end (0 wrong: mean better).
3. **In the follower: 18 right / 0 wrong / 3 undecided vs 15 / 0 / 1; reviewer unchanged (0.777 / 2 / 5.71).**
   The offers make more review actions one-tap accepts, but do not reduce their number (same as option 1).
4. **Wait + withdraw hurts**: it withdrew right offers (3 of 6 at 0.65) when later crops of the true track looked
   different (turning, occlusion), and the reviewer then had to tap: correct 0.777 → 0.762. Both stay off.
5. **Offline the chosen floor has one more wrong pick (3 vs 2)**; the follower shows none. Offers remain CANDIDATE
   only, never auto-confirmed, so a wrong offer costs one decline. Small sample: single events decide each row.
6. Unchanged: tracker, tracks artifacts, benchmark (identity layer only); target state machine; mean mode reproduces
   main's target documents' scores exactly.
