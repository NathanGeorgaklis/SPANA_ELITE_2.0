# Tracker bench v1: Roboflow `trackers` BoT-SORT vs ByteTrack vs the SPANA tracker on DEV (2026-11-20)

CODE TASK (Nathan): before any BoT-SORT adoption, run BoT-SORT with its published defaults on the same 4 DEV clips
and cached detections as `records/bytetrack_compare_2026-11-18`, score it with the same tools, and report fragments,
silent / flagged transfers and wrong-follower time side by side with our tracker and ByteTrack. The adoption task
was dropped; option B (an uncertainty layer on BoT-SORT) only if BoT-SORT closes the silent-transfer gap.
**The production tracker is unchanged. EVAL not used. Run on Nathan's Mac.**

Branch `tracking/tracker-bench-v1-botsort`. A separate, broader harness (`bench/`, candidates REF / A-D, TrackEval,
pre-registered rule "fewest silent wrong-player frames first") was pushed by another session to
`tracking/tracker-bench-v1` (`b07261c`, built, not run on DEV); this record does not use it. Its candidate B config
is the same published-defaults set, so its later run can cross-check these numbers. Provenance strings in the
artifacts read `d6511e2+tracker-bench-v1` (the local branch name at run time); code hashes are in `HASHES.txt`.

## Scope
- **Clips:** DEV FRESH_02 / 05 / 07 / 08 (3,449 labelled player-frames, 14.4 followed player-minutes).
- **Detections:** the cached local-detector person view `runs/person_view_compare/F0x_local.det.json` (sha256 as in
  the ByteTrack record), identical for all three trackers; only the tracker differs.
- **Scorers:** `tools/benchmark.py` NESTED_V2; `tools/target_score.py` with `config/default.yaml`, without and with
  `--simulate-reviewer`. Pooled by `tools/dev/tracker_bench.py` (same pooling as the ByteTrack record; it reproduces
  that record's numbers exactly from its archived reports).

## Method
- **BoT-SORT:** `trackers` 2.7.0 (Roboflow, Apache-2.0) `BoTSORTTracker`, every published `__init__` default
  (`config/botsort_compare.yaml`, checked against the installed signature by `tests/test_botsort_compare.py`):
  activation 0.7, high / low split 0.6, min consecutive frames 2, IoU thresholds 0.2 / 0.5 / 0.3, lost buffer 30
  frames at 30 fps (scaled by the clip fps), no re-ID model. **Camera motion compensation on** (sparseOptFlow,
  downscale 2): each decoded BGR frame is passed to `update()`.
  - Installed for the bench only: wheel + jsonargparse in `~/spana-botsort-run/trklib`, loaded via PYTHONPATH; the
    project venv is unchanged and nothing under `spana/` imports it.
  - Conversion `tools/dev/botsort_compare.py` → the shared `tools/dev/bytetrack_compare.to_tracks`: outputs with an id
    ≥ 0 become ACTIVE rows on the exact detection row (-1 = unconfirmed, not a track); gaps in BoT-SORT's lost pool
    become LOST; a track ends on the first frame BoT-SORT no longer holds it (read from its track list after each
    update, not recomputed); identity always CERTAIN (BoT-SORT has no uncertain state).
- **ByteTrack:** re-run (supervision 0.25.1 defaults) through the shared conversion: the 4 tracks artifacts are
  identical to the 2026-11-18 ones except the provenance commit.
- **Ours:** current main tracker (`d6511e2`, production `config/default.yaml`, `rival_eligible_only` ON) re-run on
  the same detections. Reproduces STATUS exactly (105 fragments, 107 ID switches, 7 flagged transfers, 0 silent,
  reviewer 0.783 / 2 wrong / 5.50 actions per min).

## Results

**Tracking (pooled over the 4 clips; same detections, NESTED_V2):**

| | ours | ByteTrack | BoT-SORT |
|---|---|---|---|
| fragments | 105 | 49 | **42** |
| ID switches | 107 | 52 | **45** |
| identity transfers | **7** | 14 | 20 |
| of which flagged (tracker marked identity uncertain) | **7** | 0 | 0 |
| of which silent | **0** | 14 | **20** |
| confirmed tracks, median life (F02 / F05 / F07 / F08) | — | 8.1 / 3.5 / 9.0 / 2.6 s | 9.8 / 4.1 / 13.0 / 3.0 s |

Per clip, silent transfers: ByteTrack 5 / 0 / 6 / 3, BoT-SORT 5 / 0 / 8 / 7, ours 0 / 0 / 0 / 0.

**Following one tapped player:**

| | correct time | wrong-player time | no answer | review actions per followed min |
|---|---|---|---|---|
| ours, no user help | 0.535 | **1 frame (0.03 %)** | 0.412 | 0 |
| ByteTrack, no user help | 0.665 | 217 frames (6.3 %) | 0.248 | 0 |
| BoT-SORT, no user help | **0.696** | **272 frames (7.9 %)** | 0.196 | 0 |
| ours + simulated after-match reviewer | 0.783 | **2 frames (0.06 %)** | 0.204 | 5.50 |
| ByteTrack + simulated reviewer | 0.823 | 282 frames (8.2 %) | 0.061 | 2.23 |
| BoT-SORT + simulated reviewer | **0.825** | **316 frames (9.2 %)** | 0.046 | **1.81** |

Wrong-player time per clip, BoT-SORT (no help / reviewer): FRESH_02 7.5 % / 9.1 %, FRESH_05 0 / 0, FRESH_07
11.2 % / 11.2 %, FRESH_08 7.9 % / 11.4 %.

Notes:
- BoT-SORT is scored on 3,447 frames, not 3,449: on 4 players the simulated tap lands one labelled frame later
  because BoT-SORT's first sighting is unconfirmed (-1), so there is no track to tap yet.
- ByteTrack's tracking and no-help numbers equal the 2026-11-18 record; its reviewer numbers moved (278 → 282 wrong,
  2.72 → 2.23 actions per min) because `config/default.yaml` target / offer settings changed since (same tracks).

## Findings
1. **BoT-SORT has the best continuity of the three:** 42 fragments (ours 105), the longest tracks, the highest
   correct time and the fewest review actions (1.8 per min vs ours 5.5).
2. **It does not close the silent-transfer gap; it widens it.** 20 identity transfers, all silent (ByteTrack 14,
   ours 0 of 7). A follower on BoT-SORT shows the wrong player 7.9 % of the time unaided and 9.2 % with the
   reviewer, worse than ByteTrack (6.3 % / 8.2 %) and far from ours (1 / 2 frames). FRESH_08 goes from 0 wrong
   (ByteTrack, no help) to 7.9 %.
3. **Camera motion compensation on does not stop the swaps:** FRESH_07 has 8 silent transfers and FRESH_08 7 (the
   clip with the fastest pans). A CMC-off run was not part of this task, so how much of BoT-SORT's difference from
   ByteTrack comes from CMC versus its other defaults is not measured.
4. **Verdict against the task's condition:** BoT-SORT is not ≥ our tracker on DEV under the fail-closed rule
   ("no identity switching on uncertainty"; wrong-player time is the gating metric). Not adopted. Option B (an
   uncertainty layer on BoT-SORT) is not triggered: BoT-SORT does not close the silent-transfer gap.
5. What it shows again (as ByteTrack did): how much continuity these detections allow. The open tracking work stays
   "ByteTrack-style continuity inside our tracker, keeping its flags" (STATUS next steps).

## Files
- `compare_summary.json`: all numbers above, per clip and pooled, for the three trackers.
- `botsort_compare.config_snapshot.yaml` (= `config/botsort_compare.yaml`).
- `run_bench.sh`: the full run (tracking, scoring, summary).
- `HASHES.txt`: code, wheel, inputs and every output on Nathan's Mac.
