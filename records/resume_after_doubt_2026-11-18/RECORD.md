# Resume the follower after a no-rival doubt (DEV, 2026-11-18)

Step 1 of the plan after `records/doubt_resolution_2026-11-18`. There, a resolved doubt still left the followed track
AFTER_UNCERTAIN, so each doubt cost one review tap.

`target.resume_after_doubt_resolved` (`spana/identity/follower.py`, `spana.target` 2.3.0; commit `673127a`):
- An uncertainty episode whose every uncertain frame was a doubt with no rival does not taint the track. The answer
  is ABSTAIN while the doubt lasts, then TARGET once the tracker resolves it.
- Any rival frame in the episode taints as before (`resume_after_uncertain` stays false), and so does an episode the
  follower never saw.
- Tests: `tests/test_resume_after_doubt.py` (6). Full suite 415 passed, 6 skipped.

Scope:
- **Clips:** DEV FRESH_02 / 05 / 07 / 08, scored on tracks already made from the same detections. No new runs of any
  kind, EVAL not used.
- **Pass bar:** unchanged.
  - Reviewer correct time >= 0.778.
  - Actions < 6.27 per followed minute.
  - Wrong frames <= 2.
  - Silent transfers 0.

## Results (3,449 labelled player-frames, 14.4 followed player-minutes)

| tracks | follower | no help: correct / wrong | reviewer: correct / wrong / actions per min |
|---|---|---|---|
| today | today | 0.509 / 1 | 0.778 / 2 / 6.27 |
| today | resume (+ vouch) | 0.509 / 1 | 0.778 / 2 / 6.27 (identical) |
| doubt link | resume | 0.523 / 1 | 0.709 / 2 / 5.64 |
| doubt link | resume + vouch | 0.523 / 1 | 0.783 / 2 / 6.34 |
| doubt link + (a) | resume | **0.547 / 1** | 0.753 / 2 / **5.92** |
| doubt link + (a) | resume + vouch | **0.547 / 1** | **0.783 / 2** / 6.27 |

Silent transfers are 0 on every set of tracks (records above).

Per clip, no help, today → doubt link + (a) + resume:

| clip | today | link + (a) + resume |
|---|---|---|
| F02 | 0.482 | 0.507 |
| F05 | 0.716 | 0.716 |
| F07 | 0.586 | 0.586 |
| F08 | 0.305 | **0.386** |

## Findings
1. **Pass bar not met.** No combination has both the reviewer's correct time >= 0.778 and fewer actions:
   - without the vouch: fewer actions (5.92) but lower correct time (0.753);
   - with it: the correct time (0.783) but the same actions (6.27).
2. **Without user help it is a clear gain at no identity cost.** Doubt link + (a) + resume raises correct time
   0.509 → 0.547 (FRESH_08 0.305 → 0.386), still 1 wrong frame, 0 silent transfers. This is the follower's answer
   before anyone reviews.
3. **Why the reviewer number does not move:** the simulated reviewer taps the first labelled frame of every gap,
   including short doubts that would have resolved on their own a moment later. So it spends a tap where resume
   would have answered anyway. A reviewer who skips short gaps (old next-task 1, "only handle gaps above a
   configured length") is where this saving would show.
4. **All switches stay off.** The doubt path is closed for now under the agreed pass bar.

## Decision for Nathan
- Keep everything off (pass bar), or
- accept the unaided gain: doubt link + (a) + resume + vouch is 0.547 unaided and 0.783 with the reviewer, at the
  same 6.27 actions and the same wrong frames. That would mean relaxing the bar from "fewer actions" to "no more
  actions".
- The weaker-evidence part is (a). Without it: 0.523 unaided, 0.783 / 6.34 with the reviewer.
