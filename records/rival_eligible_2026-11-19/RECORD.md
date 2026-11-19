# Stale tracks no longer block a player's own continuation (tracking, DEV, 2026-11-19)

Owner layer: player tracking. Branches:
- `tracking/rival-track-log` (from main `5c7dac5`): diagnostic log field `33584c5`, audit tool `b126cbf`;
- `tracking/rival-eligible-only`: the rule `21885f4`, default on in this commit.

DEV FRESH_02 / 05 / 07 / 08 only: re-tracked from the existing local-detector detections
(`runs/person_view_compare/F0x_local.det.json`). No EVAL, no full match.

## Why (Nathan's FRESH_08 review, player 10)
"The box is tracking them, then … loses the target even though the player remained in frame." The tracker split
player 10 three times in 60 frames: T000012 → T000029 (frame 45) → T000054 (99) → T000055 (104).

## Diagnosis
**Tracker log + a replay probe** (`~/spana-continuity-run/probes/probe_extent99.py`, scratch):
- **Frame 99:** T000029 had been following a partial-body box. The player's full-body detection was a clean S3c
  extent-change fit:
  - ratio 1.72 (limit 2.0);
  - nested 0.785 (min 0.60);
  - kept edge 0.12 h (max 0.15);
  - centre 0.04 h.

  The link was refused only by S3c's exclusivity check ("another track gates this box"). The other track was
  **T000012, the player's own original track, lost 1.8 s earlier and no longer eligible to take any detection**
  (tracker log: NOT_ELIGIBLE_GAP). The detection spawned T000054.
- **Frame 104:** T000054's S4 recovery onto its own detection (0.03 h away, IoU 0.79) was refused RIVAL_TRACK. The
  rival (now logged) was T000012 again. T000055 was spawned.

**Audit** (`tools/dev/rival_track_audit.py`; S4 RIVAL_TRACK refusals now name the rival). This is a log-only
change: tracks are byte-identical to today's apart from provenance, and decisions are identical apart from the new
field, on all 4 clips.
- 131 RIVAL_TRACK refusals on DEV. Rival owner vs the refused track's owner:

  | rival owner | refusals |
  |---|---|
  | a different player | 60 (the refusal protects identity) |
  | the same player | 1 (player 10, frame 104) |
  | no labelled owner | 70 |
- Rival eligibility on that frame (tracker log stage):

  | rivals' stage | refusals |
  |---|---|
  | all rivals `not_eligible` | **12** (every one followed by a new track spawned on the detection) |
  | some rivals not eligible | 47 |
  | all rivals eligible | 72 |

  The 12 include FRESH_08 player 10 (frame 104) and player 19 (frame 142).

## Rule (`tracking.rival_eligible_only`)
- In S3c's exclusivity check and S4's RIVAL_TRACK refusal, a track counts as another claimant only if it could take
  a detection on this frame: observed on the previous frame, or still recoverable (`_recoverable`).
- A stale track past its recovery window no longer blocks; it cannot take the detection anyway.
- Hidden-rival refusal, identity uncertainty, appearance checks and all other S3c / S4 conditions are unchanged.
- Config 2.11.0 (2.8.0–2.10.0 are used by unmerged identity branches).
- **Tests** (`tests/test_rival_eligible.py`, 3):
  - a stale track no longer blocks a recovery: before, RIVAL_TRACK + a new track; with the rule, one bridged S3c
    recovery, same tracks;
  - a still-recoverable rival refuses exactly as before (identical decisions with the flag on or off);
  - on by default.

  Plus `tests/test_tracker.py` (the refusal names its rival). Identity safety asserted on every scene. Full suite
  479 passed, 6 skipped.

## Measured (re-tracked with the flag off = today's tracks byte-identical, and on)
| | off (today) | **on** |
|---|---|---|
| fragments (F02 / F05 / F07 / F08) | 110 (23 / 7 / 29 / 51) | **105** (23 / 7 / 28 / 47) |
| ID switches | 112 | **107** |
| identity transfers | 7, all flagged | 7, all flagged: the same 7 events (frame and players); none on a newly linked track |
| unflagged (silent) transfers | 0 | **0** |
| wrong recoveries | 0 | 0 |
| track coverage F07 / F08 | 0.962 / 0.932 | 0.967 / 0.943 |
| follower unaided correct / wrong frames | 0.532 / 1 | 0.536 / 1 |
| **simulated reviewer correct / wrong / actions per min** | 0.778 / 2 / 5.64 | **0.783 / 2 / 5.50** |

Follower without re-ID offers, scored with the fixed offer verdict (records/offer_verdict_2026-11-19); identical
with the old one. FRESH_02 and FRESH_05 are unchanged.

**Links that exist only with the rule:**
- FRESH_08: T000029 at frames 99 and 104 (player 10 now keeps one track); T000040 at 142 (player 19).
- FRESH_07: 9 S4 / S3c links. Some appear only renumbered, because fewer tracks were spawned.

**Pass bar (agreed 2026-11-18):** reviewer correct >= 0.778, fewer actions, wrong frames <= 2, silent transfers 0,
no transfer on a track the new rule linked. **All met → on by default.**

## Not done
- Not checked on the 10-minute DEV window (the extent link was).
- The follower and re-ID results on the identity branches (`identity/offer-hold` and before) were measured on
  today's tracks. They need re-tracked tracks and re-embedding once this is merged.
