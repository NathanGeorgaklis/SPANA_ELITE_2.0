# Stretch confirmation: can one action per stretch bring review effort below 4.1 actions / min? DEV (2026-11-22)

CODE TASK (Nathan), target identity. Branch `identity/stretch-confirm`, base main `ae01a89`. DEV FRESH_02 / 05 / 07 / 08,
the same tracks and crops as `records/player_confirmation_2026-11-20` (`~/spana-player-confirmation-run/base`,
HASHES.txt). Range scope kept. No EVAL. **The target is not reachable with stretch confirmations on DEV (section 3).** Nathan chose option 1
(section 5): no product change; the simulated reviewer's policy is the new baseline.

## 1. What a stretch already is
- `player_confirmation` 1.0.0 already carries `frame..frame_end`. A range must be one track, live throughout and in
  one uncertainty episode.
- Inside the range the follower answers TARGET PLAYER_CONFIRMED only on frames where the track is observed and
  confirmed. Occluded or lost frames stay ABSTAIN / LOST (fail closed).
- So "one action per stretch" needs no new event for a stretch on one track. What limited the ranges was the
  simulated reviewer: it ended a range at the first labelled frame it could not verify.

## 2. Where the 108 actions go (7.52 / min over 14.36 followed min; `why_actions.py`, `seq.py`, `sametrack.py`)
- **By kind:** 70 player confirmations, 27 taps, 10 relink confirmations, 1 rejection.
- **Why a confirmed range ended:**
  - 33: on the next labelled frame the track is not the player's observed box (the player is on another track);
  - 16: the box holds two players (NESTED_V2 ambiguous);
  - 16: clip end;
  - 2: another player on the track;
  - 3: gap end.
- **No range ended because a new uncertainty episode started.**
- **The effort is track changes.** The player moves to another track 56 times.
  - Only 8 of these go to the single spawn that names the previous track as a predecessor.
  - 31 go to a track with no recorded relation to the previous one.
  - The system cannot propose most hand-overs itself, so it cannot fold them into one approved stretch.
  - Only 10 actions in a row were on the same track. 7 were separated only by two-player boxes, 3 by a new episode.

## 3. Measured (`variants.py`, `frontier.py`; always-correct reviewer; `results.json`)

| reviewer | correct | wrong | unverified | ABSTAIN frames | actions | actions / min |
|---|---|---|---|---|---|---|
| today (range ends at the first unverifiable labelled frame) | 0.870 | 0 | 32 | 295 | 108 | 7.52 |
| V1: stretch continues over frames where the track is not observed | 0.870 | 0 | 34 | 292 | 108 | 7.52 |
| V2: V1 + over two-player boxes | 0.871 | 0 | 51 | 272 | 101 | 7.04 |
| V2 + skip stretches < 2 labelled frames | **0.881** | 0 | 51 | 263 | 90 | 6.27 |
| V2 + skip < 4 | 0.872 | 0 | 51 | 287 | 76 | 5.29 |
| V2 + skip < 8 | 0.859 | 0 | 51 | 307 | 69 | 4.81 |
| V2 + skip < 11 | 0.856 | 0 | 51 | 311 | 69 | 4.81 |
| V1 + skip < 2 / < 4 / < 8 | 0.879 / 0.868 / 0.852 | 0 | 34 | 287 / 317 / 339 | 93 / 77 / 68 | 6.48 / 5.36 / 4.74 |
| taps only (no confirmations, for reference) | 0.680 | 0 | | 735 | 59 | 4.11 |

- **0 wrong frames in every variant.**
- **Stretches on one track** save at most 7 actions (V2). A stretch spanning two episodes of one track would save
  3 more and needs an event change.
- **The rest is the effort bound** (skip short stretches). The best trade is skipping stretches under 2 labelled
  frames: fewer actions AND more correct, because the reviewer takes a later, longer stretch in the same gap.
- **Floor about 4.7 / min.** Taps, relink confirmations and the 1 rejection are 38 actions (2.6 / min), and the
  confirmations that remain are the long, valuable ones. Every variant below 5.3 / min loses correctness (0.86 vs
  0.87).
- **V2 answers on two-player boxes.** These are scored unverified, not wrong (32 -> 51 frames). A UI that lets a
  player confirm a box holding two people would need Nathan's call.

## 4. Options for Nathan (nothing built)
1. **Stretch reviewer V1 + skip < 2 labelled frames.** Correct 0.879, 0 wrong, 6.48 / min. No product change, no
   two-player-box answers. Misses the 4.1 target.
2. **Same with V2.** 0.881 / 6.27 / min; accepts answers on two-player boxes (unverified 51).
3. **Relax the target.** About 5.3 / min keeps today's correctness (0.872); below about 4.7 is not reachable on DEV.
4. **The lever below 4.1 is fewer track changes.** In contact the player's track is lost and they continue on the
   rival or a new spawn (`records/episode_references_diag_2026-11-22` section 2). That is a player-tracking task.

## 5. Decision and new baseline (2026-11-22, Nathan: option 1)
- **Adopted as the simulated reviewer's policy** (`config/reviewer_policy.yaml`, `tools/target_score.py
  --reviewer-policy`, default on):
  - a confirmation stretches over labelled frames where its track is not observed;
  - it never stretches over a two-player box (option 2 rejected: it breaks fail-closed);
  - stretches under 2 labelled frames get no action.
- **`--reviewer-policy earlier`** reproduces the reviewer of `records/player_confirmation_2026-11-20`.
- **DEV with the real tool** (`run_baseline.sh`, `baseline.json`, HASHES_baseline.txt):

| run | correct | wrong | unverified | ABSTAIN | actions / min | confirmations issued / applied / conflict |
|---|---|---|---|---|---|---|
| earlier policy, always-correct taps | 0.870 | 0 | 32 | 295 | 7.52 | 70 / 70 / 0 |
| **new policy, always-correct taps (baseline)** | **0.879** | **0** | 34 | 287 | **6.48** | 55 / 55 / 0 |
| new policy, 5 % wrong taps (`config/review_sim.yaml`) | 0.879 | 2 | 34 | 287 | 6.69 | 58 / 57 / 1 (3 wrong: 1 blocked, 2 through) |

- **Wrong taps:** the seeded draws fall on a different action sequence than in the 2026-11-20 run, which drew 1
  wrong tap (1 wrong frame). Here 3 are drawn, 2 go through, 2 wrong frames. Each wrong frame is the tapping
  player's own wrong tap.
