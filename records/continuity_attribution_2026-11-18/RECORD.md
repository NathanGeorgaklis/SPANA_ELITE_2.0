# Why our tracks break where ByteTrack keeps one (DEV, 2026-11-18)

Question (Nathan): for every one of our fragments where ByteTrack kept one continuous track on the same player, why
did ours break? Then fix only the largest cause, generically, without silent identity transfers.

Scope:
- **Clips:** DEV FRESH_02 / 05 / 07 / 08. EVAL not used.
- **Inputs:** the existing local-detector detections (`runs/person_view_compare/F0x_local.det.json`), our tracks and
  decision logs from them, and the ByteTrack tracks of `records/bytetrack_compare_2026-11-18`. No new detection runs.
- **Re-tracking:** the same detections re-tracked on Nathan's Mac (tracking reads the video for camera motion and
  torso colour), code `009f73a` extracted to `~/spana-continuity-run/code`. With the new rule off, all 4 re-tracked
  artifacts are identical to the recorded ones (every frame, every track, every end).
- **Scorers:** `tools/benchmark.py` (NESTED_V2) and `tools/target_score.py` (default and `--simulate-reviewer`).

## 1. Attribution
Tool: `tools/dev/continuity_attribution.py` (DEV diagnostic).
- A **break** is one of the benchmark's own ID-switch events: on consecutive attributed labelled frames the player's
  track changes A -> B (NESTED_V2, ignore regions, ambiguous attributions skipped). It is a **fragment** when B is new
  to that player. This reproduces the benchmark exactly: 121 breaks, 120 fragments.
- **In scope:** ByteTrack attributes the same track X to that player at its last labelled frame before the break and
  its first after it.
- **Cause = first divergence.** Walk the frames of the break. On each frame where X observed a detection d, look at
  what our tracker did with d. The first frame where A does not own d decides, from the tracker's own decision log:

| cause | rule |
|---|---|
| LOST timeout | A unobserved and its re-association window closed (`NOT_ELIGIBLE_GAP`), or A already ended (`LOST_EXPIRED`, `UNCONFIRMED_MISSED`) |
| appearance / identity gate refusal | appearance veto on the pair; hidden-rival refusal (`match_refused`); recovery refused for `RIVAL_TRACK` / `CONTESTED_NEIGHBOUR` / `AMBIGUOUS_DETECTIONS` / `APPEARANCE_*` |
| low-score detection unmatched | d below `detection.score_high` and not taken (S2 refused, A unconfirmed, or A not observed on the previous frame) |
| duplicate spawn (M2 nested box) | A observed on another detection d', and d / d' nested (IoA >= `tracking.confirm_nested_ioa`) |
| other | motion gate (`mahalanobis_ok`), height-ratio gate, taken by another track, ... |

**Result: 76 of our 120 fragments are breaks that ByteTrack bridged** (FRESH_05: none).

| cause | FRESH_02 | FRESH_05 | FRESH_07 | FRESH_08 | total | share |
|---|---|---|---|---|---|---|
| **appearance / identity gate refusal** | 5 | 0 | 16 | 23 | **44** | 58 % |
| low-score detection unmatched | 2 | 0 | 2 | 3 | 7 | 9 % |
| LOST timeout | 1 | 0 | 2 | 1 | 4 | 5 % |
| duplicate spawn (M2 nested box) | 1 | 0 | 2 | 1 | 4 | 5 % |
| other | 5 | 0 | 8 | 4 | 17 | 22 % |
| **in scope / all fragments** | 14 / 24 | 0 / 7 | 30 / 38 | 32 / 51 | **76 / 120** | |

"Other" splits into the height-ratio gate (8), the motion gate (6 + 1 both), taken by another track (1) and a passing
candidate without a recovery attempt (1).

Inside the identity-gate cause (44):

| rule that refused | n | what it looked like |
|---|---|---|
| **appearance veto on a continuing track** | **25** | track observed on the previous frame, its own player's next box: IoU 0.46–0.99 with the prediction, torso-colour similarity 0.36–0.70 (< `appearance.min_similarity` 0.70); a new track spawns on that box one frame later |
| hidden-rival refusal (`AMBIGUOUS_WITH_HIDDEN_TRACK` 8, `HIDDEN_TRACK_EXPLAINS_BETTER` 1) | 9 | another confirmed player hidden nearby could explain the box |
| recovery refused: `RIVAL_TRACK` | 5 | another unmatched track also gates the box |
| recovery refused: `AMBIGUOUS_DETECTIONS` | 4 | two unexplained boxes in the gate |
| recovery refused: `CONTESTED_NEIGHBOUR` | 1 | |

Per-fragment table: `fragments_baseline.tsv` (clip, player, frames, tracks, cause, divergence frame, logged reason,
detection score, IoU, similarity).

Reading: the 19 non-veto identity refusals are real ambiguities: another person could have been the box. ByteTrack
linked them anyway, which is where its 14 silent swaps come from. The 25 vetoes are different: no rival, the same
body one frame later; only the colour histogram disagreed.

## 2. Fix (largest cause, its dominant rule only)
`tracking.appearance_doubt_link` (stage S3b in `spana/tracking/tracker.py`):
- A track observed on the previous frame and still unmatched may take a detection that the appearance veto alone
  refused, only when all of these hold:
  - that detection is the only one in its motion gate (appearance ignored);
  - it is a remaining high-score detection;
  - no other track, in any state, gates it;
  - its IoU with the prediction is >= `tracking.appearance_doubt_min_iou` (0.40 = `low_score_min_iou`, not tuned).
- **The link is never certain.** The track becomes IDENTITY_UNCERTAIN (`identity_doubt`, `spana.tracks` 2.2.0) with
  its appearance frozen. The doubt clears only when clean crops match the pre-doubt appearance
  (>= `min_similarity`) for `identity_resolve_s`.
- Every later safety check still applies to the link: hidden-rival refusal, multi-body contest, identity ambiguity.
- The 19 rival-based refusals are untouched.

Tests: `tests/test_appearance_doubt_link.py` (10):
- off reproduces the break;
- on continues one track, flagged, with no rival named;
- the doubt resolves only on the old appearance;
- the appearance is not learned from doubted crops;
- no link when another track also gates the box, or when the overlap is too loose;
- gradual colour change is unaffected;
- `spana.tracks` 2.2.0 validation, and 2.1.0 documents still validate.

Full suite 394 passed, 6 skipped.

## 3. Results (4 DEV clips, same detections)

**Tracking (benchmark NESTED_V2, pooled):**

| | current (link off) | link on |
|---|---|---|
| fragments | 120 | **108** |
| ID switches | 121 | **109** |
| identity transfers | 6 | 6 |
| silent (unflagged) transfers | **0** | **0** |
| wrong recoveries | 0 | 0 |

Per clip, fragments: F02 24 → 25, F05 7 → 7, F07 38 → 35, F08 51 → 41.

**Following one tapped player** (3,449 labelled player-frames, 14.4 followed player-minutes):

| | correct time | wrong-player frames | no answer | review actions / followed min |
|---|---|---|---|---|
| current, no user help | 0.509 | 1 | 0.433 | 0 |
| link on, no user help | 0.512 | 1 | 0.454 | 0 |
| current + simulated reviewer | **0.778** | 2 | 0.211 | 6.27 |
| link on + simulated reviewer | **0.698** | 2 | 0.293 | **5.50** |

Per clip, reviewer correct time, current → link on:

| clip | correct time | actions / min |
|---|---|---|
| F02 | 0.771 → 0.674 | 5.30 → 5.30 |
| F05 | 0.964 → 0.933 | unchanged |
| F07 | 0.773 → 0.758 | 6.33 → 5.60 |
| F08 | 0.689 → 0.501 | 8.76 → 6.96 |

## Findings
1. **The largest cause is the identity gate (44 of 76)**, and its largest rule is the torso-colour veto on a track's
   own next box (25). LOST timeouts (4) and M2 nested duplicates (4) are small on these clips.
2. **The fix keeps tracks together safely.**
   - It removes 12 fragments and 12 ID switches.
   - Transfers stay at 6, silent transfers at 0, wrong-player frames unchanged (1 unaided, 2 with the reviewer).
   - On the new tracks, 20 in-scope veto breaks remain (25 before; tracks differ, so not one-to-one). 19 were refused
     because another track also gated the box, 1 because the overlap was below the minimum: the fail-closed rule
     working. In-scope fragments: 76 → 65.
3. **But the doubt almost never clears.**
   - 24 tracks were doubt-linked; only 3 doubts resolved.
   - Many tracks are re-linked on frame after frame (one for 172 consecutive frames). The stored colour model
     disagrees with the player's current look and, frozen, can never catch up.
   - Yet transfers did not rise, so these tracks stayed on their players. The colour model is what is wrong here.
4. **The follower then cannot use those tracks.** It abstains on any IDENTITY_UNCERTAIN track, even after the user
   taps it (`spana/identity/follower.py`). Before, the veto spawned a new CERTAIN track that a tap could follow.
   - So reviewer correct time falls 0.778 → 0.698, although actions fall 6.27 → 5.50 per minute.
   - Unaided correct time is flat (0.509 → 0.512).
5. **Shipped OFF by default** (`config/default.yaml`, like `tracking.exit_hold_s`). The product numbers are therefore
   unchanged. The mechanism, its tests and `spana.tracks` 2.2.0 are in place.

## Decision needed (Nathan)
To turn the link on, one of these must make a doubted link usable without ever showing the wrong player silently:
- **(a) Tracking: a resolution rule for a doubt with no rival.**
  - Example: crops self-consistent for `identity_resolve_s` while the track stays exclusive (no other track gates
    it) the whole time.
  - This is weaker evidence than today's rule (match the old appearance). It is your call, not built.
- **(b) Target identity: a user's tap or confirmation on a doubted track counts as identity evidence from that frame
  on** (doubt only, never a rival ambiguity). This keeps the tracker strict and lets the reviewer's tap work again.
- **(c) Leave it off** and attack the next causes: height-ratio / motion gates (15), low-score continuation (7).

## Files
- `continuity_summary.json`: attribution counts (current and link on), tracking per clip, following per clip.
- `fragments_baseline.tsv`: the 76 in-scope fragments with their cause and evidence.
- `HASHES.txt`: inputs and outputs on Nathan's Mac.
