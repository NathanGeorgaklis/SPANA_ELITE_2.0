# Making a doubted track usable: user tap (b) vs self-consistent resolution (a) (DEV, 2026-11-18)

Follows `records/continuity_attribution_2026-11-18`. The appearance doubt link (`tracking.appearance_doubt_link`)
cut fragments 120 → 108 with 0 silent transfers. But the follower abstains on a doubted track even after a tap, so
the simulated reviewer's correct time fell 0.778 → 0.698. Decision 2026-11-18 (Nathan, on Claude's recommendation):
build and measure (b), with (a) as a comparison.

Scope:
- **Clips:** DEV FRESH_02 / 05 / 07 / 08, the same local-detector detections. No new detection runs, EVAL not used.
- **Pass bar (set before measuring)**, with the reviewer, against today's product:
  - correct time >= 0.778;
  - fewer review actions than 6.27 per followed minute;
  - wrong-player frames <= 2;
  - silent transfers 0.

## What was built
- **(b) `target.tap_vouches_doubt`** (`spana/identity/follower.py`, `spana.target` 2.2.0; commit `3980327`).
  - When the user taps a track that is uncertain only by an appearance doubt, the follower answers TARGET with a
    new reason, `USER_VOUCHED`.
    - "Only by an appearance doubt" means: tracks `identity_doubt`, no rival.
    - "Tap" means the selection or a re-selection.
  - The vouch lasts for the rest of that uncertainty episode.
  - It ends, fail closed, on a rival or on a new episode: ABSTAIN, then AFTER_UNCERTAIN.
  - A tap on a rival ambiguity or on a CERTAIN track vouches nothing.
  - The validator checks every vouched answer back to the tap and the tracks.
  - Tests: `tests/test_tap_vouches_doubt.py` (12).
- **(a) `tracking.doubt_resolve_self_consistent`** (`spana/tracking/tracker.py`; commit `b3b351a`).
  - A no-rival doubt may also clear when two things hold continuously for `identity_resolve_s`:
    - the crops agree with the look seen since the doubt began;
    - the track is exclusive: only its detection is in its gate, and no other track gates that detection.
  - The track then adopts that look.
  - Tests: 3 more in `tests/test_appearance_doubt_link.py`. One of them caught a bug: the resolving frame's doubt
    link re-opened a new episode. Fixed; the default path never hits it.
- **Defaults:** all switches off. Full suite 409 passed, 6 skipped.

## Results (4 DEV clips; 3,449 labelled player-frames, 14.4 followed player-minutes)

**Tracking (benchmark NESTED_V2, pooled):**

| | fragments | ID switches | transfers | silent | wrong recoveries |
|---|---|---|---|---|---|
| today (all off) | 120 | 121 | 6 | 0 | 0 |
| doubt link | 108 | 109 | 6 | 0 | 0 |
| doubt link + (a) | 106 | 107 | 6 | 0 | 0 |

**Following with the simulated after-match reviewer:**

| | correct time | wrong frames | no answer | actions / followed min |
|---|---|---|---|---|
| today (all off) | 0.778 | 2 | 0.211 | 6.27 |
| doubt link | 0.698 | 2 | 0.293 | 5.50 |
| doubt link + (b) | **0.783** | 2 | 0.207 | 6.34 |
| doubt link + (a) | 0.698 | 2 | 0.293 | 5.50 |
| doubt link + (a) + (b) | **0.783** | 2 | 0.207 | **6.27** |

**Without user help:**

| | correct time | wrong frames |
|---|---|---|
| today | 0.509 | 1 |
| doubt link | 0.512 | 1 |
| doubt link + (b) | 0.512 | 1 |
| doubt link + (a) + (b) | 0.512 | 1 |

Per clip, reviewer correct time, today → link + (a) + (b):

| clip | today | link + (a) + (b) |
|---|---|---|
| F02 | 0.771 | 0.770 |
| F05 | 0.964 | 0.964 |
| F07 | 0.773 | 0.773 |
| F08 | 0.689 | **0.711** |

Checks:
- (b) on today's tracks (link off) changes nothing: 0.509 / 0.778, identical.
- 937 reviewer answer frames were USER_VOUCHED, with no wrong-player frame added.

## Findings
1. **(b) removes the loss safely.** With the doubt link, reviewer correct time goes back from 0.698 to 0.783, just
   above today's 0.778. Wrong frames stay at 2 and silent transfers at 0.
2. **(b) does not cut review actions.** The reviewer still needs one tap per doubt, where it used to tap the
   re-spawned track: 6.34 per minute vs 6.27.
3. **(a) clears few doubts.** 5 of 24 doubted tracks were cleared by self-consistent crops, out of 11 resolutions in
   all; the other 6 cleared by matching the old look.
   - Fragments drop 2 more, with no new transfers.
   - Alone, (a) does not help following: a track that was ever uncertain stays AFTER_UNCERTAIN for the follower
     (`target.resume_after_uncertain` false).
   - With (b), (a) saves 1 action: 6.27 per minute, the same as today.
4. **Pass bar not met.** The best combination, link + (a) + (b), passes three conditions:
   - correct time 0.783 >= 0.778;
   - wrong frames 2;
   - silent transfers 0.

   Its actions equal today's (6.27) instead of falling. The 14 fewer fragments do not become fewer review actions,
   because a doubt costs one tap just as a fragment did.
5. **Nothing switched on.** All three stay off. The product and its numbers are unchanged. The mechanisms are tested
   and kept behind config.

## What would move review actions
- The remaining tap per doubt goes away only if a doubt can be answered without the user. Two ways:
  - (a) resolves a doubt and `target.resume_after_uncertain` lets the follower answer again after it. That policy
    was kept false on 2026-11-16 because rival resolutions can be wrong for same-kit players.
  - A narrower version would resume only after a no-rival doubt resolution. It is not built.
- Otherwise, fewer gaps must come from the other fragment causes:
  - height-ratio / motion gates: 15;
  - low-score continuation: 7.

## Files
- `HASHES.txt`: configs, tracks, benchmark and target_score outputs on Nathan's Mac.
