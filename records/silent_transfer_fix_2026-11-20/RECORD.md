# Silent transfers: ambiguous spawns and crossings without a contest (2026-11-20)

CODE TASK (Nathan / chat), player tracking. Branch `tracking/silent-transfer-fix`, base main `a1e802d`. DEV only
(FRESH_02 / 05 / 07 / 08, the same cached local-detector detections `runs/person_view_compare/F0x_local.det.json`).
Scored with the 2026-11-20 transfer rule (`config/benchmark.yaml` unchanged, tolerance 0.5 s). No EVAL, no full match.

## Failures reproduced (main's tracker, re-run = stored tracks exactly)
1. **FRESH_02 T000045 (player 9 -> 10, f276).** Spawned at f265 naming two possible predecessors (T000014 OCCLUDED,
   T000036 just unmatched). It stayed CERTAIN until a contest at f294.
2. **FRESH_07 T000018 (14 -> 15 at f66, back at f72).** A crossing with T000015 (players 14 / 15, the same look,
   stored-appearance similarity 0.805). Over f63-66 T000018 took a merged box (55-61 px wide vs ~38) that held
   79-95 % of T000015's own box, and T000015 took low-score nested boxes in S2. No crop was clean. The labels gave
   the merged box to player 15 at f66. T000018 was never `contested` and stayed CERTAIN until f239.
3. **`tests/test_tracker.py` h1 same-look, 60 fps.** T000003 spawned at f72 naming T000001 and T000002 (both hidden,
   the same look), took the merged box, and its label went B -> A at f76. It was never uncertain.

## Root causes: two different bugs
- **Ambiguous spawn (1 and 3).** `_spawn` recorded `possible_predecessors` but never set the identity state. A
  track that could be any of several hidden players started CERTAIN, and only a later contest could make it
  uncertain (0.6 s late on FRESH_02, never in h1).
- **Crossing without a contest (2).** IDENTITY_UNCERTAIN was raised only for a *contested* match whose rival
  passed every gate. Both crossing tracks had established velocities, so each Kalman gate held only its own box,
  and S1 (high score) and S2 (low score) never compared the two tracks. **This was not an assignment swap.** The
  ungated cost of the swapped assignment exceeded the chosen one by 0.37 (f66) and 0.19 (f67) (contest_margin
  0.10). T000018 kept following player 14 by motion. The labels counted the merged box as player 15. So an
  assignment-margin trigger (the task's expected direction) would not have fired at any sensible threshold. The
  generic signal is containment: a similar-looking track's body inside the box.

## Change (config 2.13.0, `spana/tracking/tracker.py`)
- `tracking.ambiguous_spawn_uncertain` (on): a track spawned naming >= 2 `possible_predecessors` starts
  IDENTITY_UNCERTAIN, rivals = its predecessors (`uncertain_episodes` 1; decision `identity_uncertain` rule
  AMBIGUOUS_SPAWN). The track may BE one of them, so these rivals are judged together (rule PREDECESSOR). The
  episode clears when the clean crop is clearly unlike every predecessor but at most one
  (<= `min_similarity - min_margin`) for `identity_resolve_s`. It is evidence against when two or more are not
  clearly unlike, and no evidence otherwise. A predecessor with no stored appearance is never ruled out.
- `tracking.crossing_uncertain` (on), `crossing_min_ioa` 0.60 (= `confirm_nested_ioa`, a prior, not fitted),
  `crossing_hidden_rivals` (off): a confirmed track whose detection holds >= 0.60 of another confirmed track's
  detection on the same frame marks both IDENTITY_UNCERTAIN (rule CROSSING), contested or not. Appearance must
  fail to separate them: with a crop, the two similarities are within `min_margin`; without one, the stored
  appearances are not clearly different (>= `min_similarity`). Resolution is unchanged.
- `spana.tracks` contract unchanged (no key added). `identity_rivals` may now name predecessors of an ambiguous
  spawn. The decision log's `identity_uncertain` rows gain `rule` (CONTESTED_MATCH / CROSSING / AMBIGUOUS_SPAWN).

## DEV results (`compare_summary.json`; pooled over 4 clips; reviewer = simulated after-match reviewer)

| run | transfers | silent | fragments | wrong frames unaided / reviewer | correct unaided | correct reviewer | reviewer ABSTAIN frames | actions / min |
|---|---|---|---|---|---|---|---|---|
| main | 7 | 3 | 105 | 1 / 2 | 0.535 | **0.783** | 435 | 5.50 |
| **fix (default)** | 7 | **0** | 101 | **0 / 0** | 0.508 | **0.680** | 735 | 4.11 |
| spawn rule only | 7 | 2 | 103 | 1 / 1 | 0.535 | 0.728 | 587 | 4.32 |
| crossing rule only | 7 | 1 | 104 | 0 / 0 | 0.509 | 0.720 | 629 | 5.02 |
| fix + hidden rivals | 6 | 0 | 106 | 0 / 0 | 0.492 | 0.594 | 912 | 3.97 |
| main, linked (clearance off) | 7 | 3 | 85 | 1 / 2 | 0.606 | 0.785 | 429 | 4.18 |
| **fix, linked (clearance off)** | 7 | **0** | 92 | **0 / 0** | 0.543 | 0.681 | 733 | 3.62 |

- Silent 3 -> 0. FRESH_02 f276 now sits on a track IDENTITY_UNCERTAIN since its spawn. FRESH_07 f66 / f72 are
  flagged by CROSSING at f63 / 65 / 66 (rival inside 0.80 / 0.93 / 0.96, both observed).
- Wrong-player time: 1 -> 0 frames unaided and 2 -> 0 with the reviewer. FRESH_07 player 14's wrong frame is gone.
- **Cost:** correct time falls, with the reviewer 0.783 -> 0.680 (-0.103) and unaided 0.535 -> 0.508. Review
  actions fall 5.50 -> 4.11 / min, but this is **not** a gain. The reviewer's ABSTAIN frames rise 435 -> 735,
  because a tap cannot answer an uncertain track, and same-look episodes almost never resolve (4-5 resolutions
  on each of F02 / F07 / F08). Uncertain share of confirmed ACTIVE track frames: F02 0.081 -> 0.217, F07 0.052 -> 0.155,
  F08 0.124 -> 0.386 (F05 0 -> 0).
- Hidden rivals: counting an unobserved rival's prediction (`crossing_hidden_rivals: true`) caught the same
  transfers but cost 0.783 -> 0.594. Most of those marks (647 of 811) were a teammate hidden behind the track,
  which the OCCLUDED / recovery / hidden-rival rules already guard. Off by default.
- Linker re-run (clearance off) on the new tracks: 0 transfers on linked tracks, 0 link-caused, 0 silent. Links
  21 -> 10 (refusals IDENTITY_RIVAL 13 -> 79, UNCERTAIN_EPISODE 44 -> 97), fragments after linking 85 -> 92.
- h1 same-look: 0 silent at 10 / 25 / 60 fps (pinned test updated).

## Tests
`tests/test_silent_transfer_fix.py` (11): a real FRESH_07 fixture (`tests/data/fresh07_crossing_f40_90.json`, built
by `make_fixture.py`) reproduces the silent transfer with the crossing rule off and flags it with the rule on.
Also: unit checks of the crossing rule (same look marked both ways, different looks not, hidden rivals only
with the switch), and the ambiguous-spawn rule (born uncertain naming its predecessors at 10 / 25 / 60 fps, one
predecessor stays certain, distinct predecessors resolve, same look never). `tests/test_tracker.py` h1 now pins
0 silent. Full suite 583 passed, 6 skipped.

## Files
`run_bench.sh` (CFG / OUT env select a config / output; the ablations used scratch copies of default.yaml with one
switch flipped), `compare_summary.json`, `link_summary_off.json`, `default.config_snapshot.yaml`,
`make_fixture.py`, `HASHES.txt`. Outputs: `~/spana-silent-fix-run/final` (default; identical results to `out_obs`, run before a comment-only config edit), `spawn_only`,
`crossing_obs_only`, `out_hidden`, `crossing_only` (crossing with hidden rivals, no spawn rule), `out` (the first
run, = `out_hidden`).
