# Safer re-entry offers: position check and two-way match (DEV, 2026-11-18)

Steps 1 and 2 of the plan to make re-ID re-links safer (`records/reid_reentry_eval_2026-11-18`). The offline check
found about 1 wrong pick in 10, 13 of 17 of them in the FRESH_08 crowd, and 10 of 17 with the player reappearing
within 0.2 s.

## Built (commit `7fd992a`, branch `identity/reentry-position-mutual`)
- **One shared rule.** `spana.identity.reid.reentry_choice` is now the single re-entry rule, used by the follower's
  offer and by `tools/dev/reid_reentry_eval.py`. With both new switches off it reproduces the shipped offers
  exactly: same target_score, unaided and with the reviewer.
- **`target.reentry_position_gate`.**
  - The lost player's last box centre is carried through the camera transforms to the candidate's birth frame.
  - The candidate must lie within `tracking.gate_base_h + tracking.max_speed_h_per_s × gap` body heights of it
    (the tracker's own speed bound; no new threshold).
  - Not judged across a cut, when the carried point leaves the picture, or without a frame size.
- **`target.reentry_mutual`.** The lost track must also be the candidate's best match, by `reentry_min_margin`, among
  the tracks that were lost when the candidate appeared. This applies within the window, to tracks with a signature
  and, with the gate, able to reach the candidate. Two lost players can then no longer claim the same returning
  track.
- **Tests:** `tests/test_reentry_offer.py` +3: mutual refuses a track another lost player matches better, mutual
  keeps a clear two-way match, and the reachability cases (pan, far, cut, out of picture, no frame size). Full suite
  468 passed, 6 skipped. Config schema 2.6.0.

## Offline (79 scorable labelled re-entries, `tools/dev/reid_reentry_eval.py`)
| rule | min sim / margin | accepted | right | wrong |
|---|---|---|---|---|
| plain (before) | 0.75 / 0.10 | 31 | 28 | 3 |
| plain | 0.75 / 0.00 | 48 | 43 | 5 |
| position | 0.75 / 0.10 | 31 | 28 | 3 |
| **mutual** | **0.75 / 0.00** | **35** | **33** | **2** |
| mutual | 0.75 / 0.10 | 15 | 15 | 0 |
| mutual | 0.70 / 0.00 | 41 | 38 | 3 |
| both | 0.75 / 0.00 | 35 | 33 | 2 |

## In the follower (4 DEV clips, same tracks)
| offer rule | offers right / wrong / undecided | reviewer correct | wrong frames | actions / min |
|---|---|---|---|---|
| before: plain 0.75 / 0.10 | 16 / 1 / 2 | 0.776 | 2 | 5.57 |
| **now: mutual 0.75 / 0.00** | **16 / 0 / 1** | **0.776** | 2 | 5.64 |
| mutual 0.75 / 0.10 | 9 / 0 / 1 | 0.777 | 2 | 5.64 |
| mutual 0.70 / 0.00 | 19 / 2 / 3 | 0.763 | 2 | 5.64 |
| mutual + position 0.75 / 0.00 | 16 / 0 / 1 | 0.776 | 2 | 5.64 |

## Findings
1. **The two-way match is the safety gain.**
   - Offline, at the same similarity floor, it accepts more right links than the old margin rule, with fewer wrong
     ones (33 / 2 vs 28 / 3).
   - In the follower it keeps all 16 right offers and removes the wrong one.
   - It replaces the margin. A margin on top halves coverage and was 0 wrong anyway.
   - **New default:** `reentry_mutual` on, `reentry_min_margin` 0.0, `reentry_min_similarity` 0.75.
2. **The position check, as built, does nothing.** In 9 of the 14 wrong picks checked, the true track appeared
   within 0.2 s, mostly within 0.2 body heights of where the player vanished. The wrong track appeared 0.1–2 s later.
   - With a sprint-speed bound, a later-born track can be "reachable" from almost anywhere, so the bound rules out
     very little.
   - Position is strong RELATIVE evidence (this candidate appeared right there, at once), not a hard limit.
   - Left in the code, off. A position-likelihood score that favours the earliest, nearest candidate is the next
     variant to try.
3. **Still not safe enough for automatic links.** The best setting has 2 wrong in 35 offline. The 2026-11-16 rule
   (the user confirms) stays.
4. **Actions:** 5.64 vs 5.57 per minute, the same within noise. The one wrong offer that was removed had been an
   extra (decline) action for the simulated reviewer, but one fewer offer also means one more tap.
