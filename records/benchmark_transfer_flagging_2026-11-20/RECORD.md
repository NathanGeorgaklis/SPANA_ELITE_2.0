# Transfer flagging fix in tools/benchmark.py: re-score of stored DEV runs (2026-11-20)

CODE TASK (Nathan / chat), QA tooling. Branch `qa/benchmark-transfer-flagging`. No tracker, linker or tracking-config
change; scoring only, no re-runs. EVAL not used.

## Rule change (spana/evaluation.py, one implementation for the DEV benchmark and the synthetic QA suite)
- **Old:** a transfer was flagged if the track had `possible_predecessors` or `recoveries > 0`, was
  IDENTITY_UNCERTAIN on the transfer frame, or had a new uncertainty episode since its previous attribution.
- **New:** flagged only if the track was IDENTITY_UNCERTAIN on some processed frame within
  `transfer_flag_tolerance_s` of the transfer frame (either side). Otherwise silent, recoveries and links included.
  Tolerance 0.5 s in `config/benchmark.yaml` 1.0.0 (= `tracking.identity_resolve_s`), set before re-scoring.
  Uncertainty is read from every processed frame of the tracks document (`TransferFlags`), not only the labelled
  frames (every 6th).
- New columns (summary and per event): `recovery_caused_*` (a tracker recovery after the previous attribution),
  `link_caused_*` (an offline link, from the decision log's `link` rows, after the previous attribution),
  `transfers_on_linked_tracks`, each event's `previous_frame`. `tools/dev/link_bench.py` no longer counts
  transfers: it pools these columns and the link log.

## Re-scored (same tracks, same labels; `rescore.sh`, `flag_rescore.json`)

| run | transfers | silent, old rule | silent, new rule | recovery-caused (silent) | link-caused (silent) | on linked tracks |
|---|---|---|---|---|---|---|
| ours (tracker-bench-v1) | 7 | 0 | **3** | 0 (0) | — | — |
| ByteTrack | 14 | 14 | 14 | 0 | — | — |
| BoT-SORT | 20 | 20 | 20 | 0 | — | — |
| linking, clearance on | 8 | 0 | **3** | 0 (0) | 1 (0) | 4 |
| linking, clearance off | 7 | 0 | **3** | 0 (0) | 0 (0) | 0 |

**Three earlier "flagged" transfers turn silent, the same three in every run on our tracker:**
1. FRESH_02 T000045 at frame 276, player 9 -> 10. The track was spawned with two `possible_predecessors`
   (ambiguous spawn, frame 265) and was never IDENTITY_UNCERTAIN until frame 294 (0.6 s after the transfer, just
   outside the window). The old rule flagged it only for `possible_predecessors`.
2. and 3. FRESH_07 T000018 at frames 66 and 72, player 14 -> 15 and back. The track had one earlier recovery
   (before the previous attribution, so not the cause) and was not uncertain until frame 239. The old rule flagged
   them only for `recoveries > 0`. This is the track behind the follower's one wrong frame on FRESH_07 player 14
   (no help and reviewer): the follower showed the wrong player there with no flag.

Expected checks from the task:
- Clearance on: FRESH_02 T000010 player 8 -> 10 at frame 186 is reported link-caused, on a linked track, flagged
  (IDENTITY_UNCERTAIN within the window). Correct.
- Clearance off: 0 link-caused, 0 on linked tracks, **but 3 silent** (the tracker's own three above, none caused by
  a link). The expectation "clearance-off still has 0 silent" does not hold under the new rule.

Synthetic QA suite: `tests/test_tracker.py` h1 same-look case at 60 fps now shows 1 silent transfer, on a track
spawned with `possible_predecessors` (the test's documented "known limit", previously counted as flagged for that
reason). The test now pins it (1 at 60 fps, 0 at 10 / 25) instead of asserting 0.

## What changes in past conclusions
- "Our tracker: 0 silent transfers" (ByteTrack compare 2026-11-18, tracker bench v1 2026-11-20, rival-eligible,
  extent link, linking v1 acceptance) reads **3 silent of 7** under the new rule (2 events on one FRESH_07 track,
  1 on a FRESH_02 ambiguous spawn). ByteTrack / BoT-SORT are unchanged (all silent before and after), so "not
  adopted" stands. Wrong-player time, the gating metric there, is unchanged (it never used the flag).
- Linking v1 clearance-off: 0 transfers on linked tracks and 0 link-caused still hold; "0 silent" holds only for
  transfers the linker caused.
- Reported to Nathan first, as the task asked; approved and merged to main the same day.

## Files
`flag_rescore.json` (per set, every flipped event), `link_summary_on.json` / `link_summary_off.json`
(`tools/dev/link_bench.py` on the new reports), `benchmark.config_snapshot.yaml`, `rescore.sh`, `HASHES.txt`.
