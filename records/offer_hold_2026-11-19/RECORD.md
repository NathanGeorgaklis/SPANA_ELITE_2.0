# Offer hold: keep an offer through a brief hide of its track (DEV, 2026-11-19) — built, OFF

Owner layer: target identity. Branch `identity/offer-hold` (on top of `identity/reentry-exit-side`). Mechanism
`0bb94a8`; measured with code `a209e5d`. DEV FRESH_02 / 05 / 07 / 08 only (same tracks / 1.1.0 embeddings as
records/reentry_exit_side_2026-11-19); no EVAL, no new video runs.

## Why
Nathan's FRESH_08 review, player 10: "the box is tracking them, then stays on the player but has the 'is this them?'
box and then loses the target even though the player remained in frame". The follower offered T000029 (right, player
10) from frame 49. On frame 94 T000029 was OCCLUDED for one frame, and the follower dropped the offer for good
(CANDIDATE_NOT_USABLE; a dropped re-ID offer is never repeated for that loss).

## Built (`target.offer_hold_s`)
- An offer (predecessor or re-ID) whose track is live, confirmed and CERTAIN, but OCCLUDED or LOST (predicted box),
  is kept for up to offer_hold_s after its last usable frame.
- While held:
  - no CANDIDATE is shown (the answer is the followed track's own state, no box);
  - it cannot be confirmed or withdrawn on those frames.
- When the track is usable again, the same offer is shown again.
- Uncertain, unconfirmed, ended, or hidden too long: dropped as before.
- 0 = the behaviour before.
- `spana.target` 2.8.0 (+ target_config.offer_hold_s); config 2.10.0.
- **Tests** (`tests/test_offer_hold.py`, 4):
  - a one-frame hide keeps a predecessor offer and its gap lists it once;
  - hidden 0.6 s > 0.5 s is dropped (0.5 s is kept: boundary checked);
  - uncertain while hidden is dropped;
  - a re-ID offer is kept through a one-frame hide.

  Three existing tests that check drop-at-once now pin offer_hold_s 0. Full suite 504 passed, 6 skipped.

## Measured (follower, fixed verdict; set 0.65 + exit rule)
| offer_hold_s | re-ID offers right / wrong / undecided | dropped (not usable) | CANDIDATE frames right / wrong / unverified | unaided | reviewer correct / wrong / actions per min |
|---|---|---|---|---|---|
| 0 (= before, reproduced) | 18 / 0 / 0 | 17 | 472 / 1 / 5 | 0.532 / 1 | 0.777 / 2 / 5.71 |
| 0.2 | 18 / 0 / 0 | 17 | 472 / 1 / 6 | 0.532 / 1 | 0.777 / 2 / 5.71 |
| 0.5 | 18 / 0 / 0 | 17 | 470 / 1 / 6 | 0.532 / 1 | 0.777 / 2 / 5.71 |
| 1.0 | 18 / 0 / 0 | 17 | 468 / 1 / 6 | 0.532 / 1 | 0.777 / 2 / 5.71 |

**Player 10 with 0.5 s:**
- The offer of T000029 now survives the one-frame hide: CANDIDATE 49–93, LOST 94, CANDIDATE again 95–98.
- T000029 itself is then lost from frame 99, while the player is in frame (the tracker split him: T000054 from 99,
  T000055 from 104). The offer is dropped and the target stays lost.

## Findings
1. **The hold does what it says** (player 10's offer is not lost to a one-frame hide), **but changes no DEV outcome**:
   same offers, same verdicts, same reviewer numbers. The longer the hold, the fewer right CANDIDATE frames are shown
   (2–4 frames), because a held offer blocks a new search.
2. **Player 10's loss is a tracking failure**, not a follower one: his track was split twice in 5 frames while he
   stayed in view. Fixed in the tracking layer (records/rival_eligible_2026-11-19).
3. **Decision: offer_hold_s 0 (off).** Kept and tested for when tracking improves.

## Addendum (same day): re-measured on merged main's tracks — ON at 0.2 s
After the merge (main `8a63422`, `tracking.rival_eligible_only` on), Nathan's review videos were re-rendered. Player 10's
offer still vanished at frame 94: the tracker now keeps T000029 on him, but the one-frame hide at 94 dropped the
offer. That is exactly the case the hold was built for, and the OFF decision above was measured on the old, split
tracks. Re-measured on main's re-tracked, re-embedded DEV clips (code `8a63422`, `run_mainhold.sh`):

| offer_hold_s | re-ID offers r / w / u | CANDIDATE frames right / wrong / unverified | unaided | reviewer |
|---|---|---|---|---|
| 0 (main as merged) | 16 / 0 / 0 | 442 / 1 / 5 | 0.536 / 1 | 0.783 / 2 / 5.50 |
| **0.2** | 16 / 0 / 0 | **450 / 1 / 6** | 0.536 / 1 | 0.783 / 2 / 5.50 |
| 0.5 | 16 / 0 / 0 | 448 / 1 / 6 | 0.536 / 1 | 0.783 / 2 / 5.50 |
| 1.0 | 16 / 0 / 0 | 446 / 1 / 6 | 0.536 / 1 | 0.783 / 2 / 5.50 |

- Player 10 (0.5 s, the same at 0.2 s): CANDIDATE T000029 on frames 48–93, 95–102 and 104–147, held through the
  one-frame hides at 94 and 103.
- No verdict, offer, wrong frame or reviewer number changed.
- **Decision: offer_hold_s 0.2 (ON).** Right offer frames +8, wrong unchanged. Longer holds are a little worse,
  because a held offer blocks a new search.

