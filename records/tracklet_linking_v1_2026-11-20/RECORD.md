# Offline global tracklet linking v1 on DEV (2026-11-20)

CODE TASK (Nathan / chat): a new post-processing stage after the tracker that links tracklets over a whole clip to
reduce fragmentation and review load, with no silent identity transfer. **The production tracker is unchanged.**
Acceptance (set in the task): silent transfers = 0 and wrong-player time not worse than current; otherwise report and
do not merge. **EVAL not used. No full-match run. Run on Nathan's Mac.** Branch `tracking/global-tracklet-linking-v1`.

## Design (spana/tracking/linking.py, config/tracking/linking.yaml 1.0.0)
- In: a `spana.tracks` document (2.1.0 / 2.2.0 / 2.3.0) + the video (appearance) + the tracker's decision log.
  Out: the same contract and version (no field added), the decision log with linked ids renamed (+ `link` rows),
  and a link log `*.links.jsonl` (`spana.links` 1.0.0: header + one row per predecessor tracklet with any candidate:
  `kind` link / abstain, cost terms, similarity, displacement, margins both ways, reasons, top-5 candidates per side).
- A link A -> B is written as the tracker writes its own recovery: B's rows take A's id, the gap is A held (LOST /
  HELD, camera-carried box), `recoveries` + 1, `last_gap_s` = gap.
- Candidates: same epoch (no cut / unknown camera between), B born after A's last observation, gap <= 2.0 s,
  camera-carried foot displacement <= 0.35 h + 5.0 h/s x gap, height ratio <= 1.5 (all = the tracker's own values).
- Cost = gap / max_gap + motion (distance from A's camera-carried constant-velocity prediction / gate, <= 1) +
  (1 - torso-colour similarity; 1 if unknown) + team (only with team labels; none exist yet). Appearance sits behind
  an `Embedder` interface (`TorsoColourEmbedder` = `spana/appearance.py`, mean of 5 clean crops at each end).
- Fail closed: link only on a two-way best match beating every second best by margin 0.10 (all gate-passing
  tracklets compete, confirmed or not), both confirmed, similarity known and >= 0.70, no team conflict; across an
  uncertain episode or to an identity rival (also anywhere along a chain) only at similarity >= 0.85
  (`strict_min_similarity`); never when a merged row would be uncertain with only its own chain as rival.
- Never raises confidence: linked rows carry the chain's `uncertain_episodes` (a follower selected before an
  earlier episode stays tainted), keep their own identity state, and `confidence` = min(own, merged window value).
- All values were set from the tracker config / policy BEFORE the bench; nothing was tuned on DEV.

## Scope
- Clips: DEV FRESH_02 / 05 / 07 / 08 (3,449 labelled player-frames, 14.4 followed player-minutes).
- Input tracks: production tracker output of `records/tracker_bench_v1_2026-11-20` (main `d6511e2`; `spana/`,
  `config/default.yaml` and the scorers unchanged since), from the cached local-detector detections.
- Scorers: `tools/benchmark.py` NESTED_V2; `tools/target_score.py` (config/default.yaml) without and with
  `--simulate-reviewer`; pooled by `tools/dev/tracker_bench.py`. `tools/dev/link_bench.py` counts link-caused
  transfers (see below). Full run: `run_bench.sh`.

## Results (pooled, same detections; first three columns = tracker-bench-v1)

| | ours | ByteTrack | BoT-SORT | **ours + linking** |
|---|---|---|---|---|
| fragments | 105 | 49 | 42 | **72** |
| ID switches | 107 | 52 | 45 | **76** |
| identity transfers | 7 | 14 | 20 | 8 |
| of which flagged | 7 | 0 | 0 | 8 |
| of which silent | **0** | 14 | 20 | **0** |
| link-caused transfers (silent) | — | — | — | 1 (0) |
| correct time, no help | 0.535 | 0.665 | 0.696 | **0.625** |
| wrong-player time, no help | 1 frame (0.03 %) | 217 (6.3 %) | 272 (7.9 %) | **1 frame (0.03 %)** |
| correct time, reviewer | 0.783 | 0.823 | 0.825 | **0.791** |
| wrong-player time, reviewer | 2 frames (0.06 %) | 282 (8.2 %) | 316 (9.2 %) | **2 frames (0.06 %)** |
| review actions per followed min | 5.50 | 2.23 | 1.81 | **4.04** |

Per clip, fragments ours -> linked: F02 23 -> 18, F05 7 -> 7, F07 28 -> 15, F08 47 -> 32. Correct time no help:
F02 0.482 -> 0.546, F05 0.716 -> 0.716, F07 0.651 -> 0.756, F08 0.321 -> 0.470. Reviewer actions: F02 18 -> 15,
F05 7 -> 7, F07 20 -> 13, F08 34 -> 23. The wrong-player frames are the same frames as before (FRESH_07 player 14
no help and reviewer, FRESH_02 player 9 reviewer): no new wrong frame.

Link log: 35 links (F02 5, F05 0, F07 14, F08 16), 128 abstentions. Abstention reasons (an entry can have several):
UNCONFIRMED 89, NOT_MUTUAL_BEST 50, APPEARANCE_UNKNOWN 35, APPEARANCE_BELOW_MIN 35, AMBIGUOUS_SUCCESSORS 25,
UNCERTAIN_EPISODE 25, AMBIGUOUS_PREDECESSORS 12, IDENTITY_RIVAL 7. 14 of the 35 links used the strict appearance
clearance (uncertain episode or rival at similarity >= 0.85).

**How link-caused transfers are counted.** `tools/benchmark.py` calls a transfer flagged when its track has
`recoveries > 0`; every linked track has one, so a wrong link would never show there as silent. `link_bench.py`
therefore takes every transfer event present in the linked run but not in the baseline (same frame, from, to) as
link-caused, and counts it silent unless the track was IDENTITY_UNCERTAIN (now or since its last attribution), which
is what the follower acts on.

## Findings
1. **Acceptance met:** silent transfers 0 (link-caused included) and wrong-player time unchanged (1 / 2 frames, the
   same frames). Fragments 105 -> 72 (-31 %), unaided correct time +9.0 points, review actions 5.50 -> 4.04 per min
   (-27 %), reviewer correct time +0.8 points.
2. **One link is wrong, and it is flagged, not silent.** FRESH_02 T000010 (player 8) -> T000033 (player 10) at frame
   178, gap 1 frame, similarity 0.896, margins 0.19 / 0.21. It crossed an uncertain episode and passed only through
   the strict clearance (0.896 >= 0.85). The transfer at frame 186 is IDENTITY_UNCERTAIN, so the follower abstains
   (player 8: 0 wrong frames). The same-kit limit of torso colour (exit-hold record 2026-11-18) is what lets it
   through. The 2026-11-18 bar for the extent-change link ("no transfer on a track the new rule linked") would count
   this as a failure; the task's acceptance does not.
3. The strict clearance carries 14 of the 35 links, and one of those 14 is the wrong one. Fewer links without it
   was not measured (no tuning on DEV).
4. Most abstentions involve an unconfirmed tracklet (89): short partial-body / duplicate tracks gate-pass next to
   the real continuation. That is fail-closed by design (they compete in the margin).

## Files
`compare_summary.json` (all four trackers, per clip and pooled), `link_transfers.json`, `F0x_linked.trk.links.jsonl`
(full link logs), `linking.config_snapshot.yaml`, `run_bench.sh`, `HASHES.txt`.
