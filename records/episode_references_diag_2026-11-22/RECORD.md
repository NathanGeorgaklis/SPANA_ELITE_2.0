# Diagnosis: NO_REFERENCE (49) and NO_CLEAN_CROP (29) refused uncertain episodes, DEV (2026-11-22)

CODE TASK (Nathan), target identity. Diagnose first, no code change. Branch `identity/episode-references-diag`, base
main `b5c38a4`. DEV FRESH_02 / 05 / 07 / 08 frozen labels only. No EVAL, no full match, `reid_resolve` stays off in
`config/default.yaml`.

**Inputs.** The `reid_resolve`-on tracker run of `records/player_confirmation_2026-11-20` section 5
(`~/spana-player-confirmation-run/reid`, HASHES.txt). Its episode report has the same 126 episodes and refusal
counts as `records/reid_episode_resolution_2026-11-20`. The tracker is unchanged since then (`spana/tracking`
differs only by the bench-only `protocol.py`).

**Method.**
- `diagnose.py` rebuilds every track's re-ID reference from the tracks and decision logs. A reference is set on the
  first frame a track is OBSERVED, CERTAIN, confirmed, uncontested and clean (IoU <= `appearance.clean_max_iou` 0.05
  with every other observed box). A rival's or predecessor's reference is the snapshot taken at the contest or spawn.
- The rebuild agrees with the tracker's own `reid_evidence` rows on 807 of 807 frames: 695 "no own reference" and
  112 "own reference present".
- `chains_and_overlaps.py`, `closure_and_follow.py` and `unmatched_reasons.py` add the chain and overlap analysis.
  Who each box is comes from the labels (NESTED_V2, labelled frames only, a scoring convention).

## 1. NO_REFERENCE: 49 episodes, 100 missing references

- **Where:** 43 of 49 start at the track's spawn (ambiguous spawns: two or more possible predecessors).
- **Which reference is missing:** a predecessor's (86), the track's own (12), a rival's (2).
- **Why those tracks had no reference:**

| why | missing refs | meaning |
|---|---|---|
| BORN_UNCERTAIN_NEVER_CERTAIN | **79** | the reference-less track was itself an ambiguous spawn (first uncertain rule AMBIGUOUS_SPAWN), never CERTAIN |
| CERTAIN_NEVER_CLEAN | 17 | CERTAIN, confirmed and observed, but every such frame overlapped another box (IoU > 0.05) |
| CERTAIN_NEVER_CONFIRMED | 4 | CERTAIN for less than `tracking.confirm_s` / 2 frames, then uncertain |

- **Timing is not the issue.** None of the 100 tracks got a reference after the snapshot was taken.
- **Why the other players were never CERTAIN.** A track born from an ambiguous spawn starts IDENTITY_UNCERTAIN
  (decision 2026-11-20, silent-transfer fix). It becomes CERTAIN only when the episode is resolved. Colour cannot
  resolve same-look predecessors. Re-ID needs a reference for every predecessor, and those predecessors are
  themselves spawns with no reference. The chain is circular: each fragment in a crowd makes the next spawn
  unresolvable.
- **First failing layer: the identity evidence rule** (tracker `_reid_evidence`, PREDECESSOR). It treats a
  reference-less ambiguous-spawn predecessor as a dead end. The track's identity is nonetheless bounded: it is one
  of that predecessor's own predecessors, whose reference snapshots the tracker already holds. Upstream, the chains
  come from tracking fragmentation in contact (section 2).
- **Following chains (closure):** replace each such predecessor by its own predecessors, recursively, with a cycle
  guard.
  - 17 of the 43 spawn episodes then have a reference for every candidate; 14 of those have >= 3 clean evidence frames.
  - The labelled player is among the closure candidates in 16 of 17.
  - The exception is FRESH_07 T000053: player 4, while the candidates are players 5 and 7.
  - The direct rule already has this exposure: in 7 of 43 spawns the true player is not among the direct
    predecessors. Only the margin rule protects these cases.
  - The other 26 spawn episodes are still blocked by chain roots that were CERTAIN but never clean (26 roots) or
    never confirmed (11). These are crowd cases that the closure does not change.
- **Not spawns:** the 6 other episodes miss the track's own reference (born uncertain) or a rival's.

## 2. NO_CLEAN_CROP: 29 episodes

- **The crowd is real.** The tracker logged 367 evidence frames: 349 NOT_CLEAN and 18 CONTESTED.
  - The overlapping box belongs to another labelled player on 278 observed frames, and duplicates the same player
    on 91.
  - Episodes where duplicates dominate: 4 of 29, which is a detection issue.
  - Overlap IoU: <= 0.10 on 16 frames, 0.10-0.20 on 75, 0.20-0.40 on 190, > 0.40 on 93.
  - A looser clean threshold would gain almost nothing and would let mixed crops in. Not proposed.
- **The track is barely observed.** Median 5 observed frames of 59 episode frames.
- **The player usually lives on, on another track.** On the 161 labelled frames inside an episode where the
  uncertain track is unobserved:
  - 73: the player is on the rival's box (34 of them on a rival that names this track as a predecessor);
  - 62: on a track spawned after the episode began (49 of them name this track as a predecessor);
  - 1: on another older track;
  - 25: no box matched the player (16 occluded).
- **How the track ends:**
  - LOST_EXPIRED 14, UNCONFIRMED_MISSED 5, clip end 10.
  - First miss: TAKEN_BY_OTHER 9, appearance veto 8, motion / height gate 6, refused 1, passing candidate unused 1.
  - Most later frames are NOT_ELIGIBLE_GAP (430) and TAKEN_BY_OTHER (291).
  - 8 of the players were re-spawned on an ambiguous spawn within 3 s. That feeds the NO_REFERENCE chains in
    section 1.
- **References are not what blocks this group.** The own reference existed in 20 of 29 episodes, and every rival's
  reference in 8 of 29.
- **First failing layer: player tracking.** In contact, the uncertain track loses its player's box (taken by the
  rival or the merged box, or vetoed) and ends while the player continues on a new spawn. While that track is alive
  no clean crop of the player exists, so no re-ID rule can clear it. Re-associating the lost uncertain track would
  touch the fail-closed association rules. That needs its own tracking task and is not proposed here.

## 3. Proposed smallest generic fix (NOT built; awaiting Nathan's go-ahead)

**Transitive predecessors for ambiguous spawns** (re-ID evidence rule, `reid_resolve` only; new key e.g.
`reid_resolve.transitive_predecessors`, off by default, with `reid_resolve.enabled` staying false):

- **At spawn:** when a spawn names a predecessor that has no re-ID reference and is itself an unresolved ambiguous
  spawn, the new track's candidates become that predecessor's own predecessor snapshots, recursively.
  - The snapshots are already held in the predecessor's `identity_rivals`.
  - Cycles and duplicates are removed.
- **Unchanged:** the PREDECESSOR rule. One candidate must beat every other by the calibrated 0.517 on every frame
  of a run of at least 0.5 s and 3 frames. Colour can only veto. Any candidate without a reference still refuses
  (fail closed).
- **Not done:** resolving the new track never resolves the intermediate tracks. No margin, threshold, spawn,
  crossing or linker change.
- **Logging:** the chain is logged in `reid_evidence`, `predecessors_via`.
- **Owner layer:** the evidence rule lives in `spana/tracking/tracker.py` (player tracking), although it serves
  target identity. Nathan to confirm which layer owns the change.
- **Expected DEV effect (upper bound, not a prediction):** at most 17 more episodes become evaluable, 14 with >= 3
  clean frames. How many clear depends on same-kit separation, which v2 lacks at 0.517 (FRESH_07 / 08). It may
  clear few.
- **Measurement plan:** the run of `records/reid_episode_resolution_2026-11-20` with the key on. Acceptance as
  there: 0 silent transfers, 0 / 0 wrong frames, 0 transfers on linked tracks, every resolution judged against the
  labels.
- **NO_CLEAN_CROP:** no change proposed in the re-ID layer. Its fix is in tracking continuity in contact (section 2),
  which needs a separate task.
