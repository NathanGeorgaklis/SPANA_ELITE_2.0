# Tracker bench v1, A-D harness run on DEV (2026-11-21)

CODE TASK (Nathan, option 1): rebase the `bench/` harness (`b07261c`, built by another session, never run) onto main
and run it on DEV, mainly to learn whether D (GTA-Link offline linking) on BoT-SORT removes BoT-SORT's silent
identity transfers, the reason it was rejected (`records/tracker_bench_v1_2026-11-20`, DECISIONS 2026-11-20). C
(Deep-EIoU, no licence) was run for reference only. **Production tracker unchanged. EVAL not used. Nathan's Mac.**

Branch `tracking/tracker-bench-v1-gtalink` = main `b758c0b` + the harness (cherry-pick of `b07261c`).

## Scope and method
- **Clips:** DEV FRESH_02 / 05 / 07 / 08, frozen labels (`config/tracking/bench_v1.yaml`).
- **Detections:** the same local-detector files as the 2026-11-20 BoT-SORT record (`runs/person_view_compare/F0x_local.det.json`);
  the bench cache holds byte-identical copies (same sha256, `HASHES.txt`) plus one SPANA re-ID v2 embedding per detection.
- **Candidates** (published defaults, `config/tracking/*.yaml`, nothing tuned): REF ours (`config/default.yaml`), A ByteTrack and
  B BoT-SORT + CMC (Roboflow `trackers` 2.7.0), C Deep-EIoU @ `2fc7d27` (EVAL-ONLY), D GTA-Link @ `e4d5cc4` on B and on C.
  REF / A / B were run too: they are cheap and D needs B's output.
- **Scoring** (`bench/metrics.py`): TrackEval HOTA / IDF1 / IDSW; each labelled player tapped on its first unambiguous
  NESTED_V2 frame and the tapped track followed **as is** (no target layer, no reviewer). Wrong frames = the tapped track is
  on another player; unflagged = the tracker called that frame certain.
- **Rule** (fixed in `bench_v1.yaml` before any result): commercial-compatible only; fewest unflagged wrong frames, then
  coverage, then IDF1.
- Run: `python -m bench.run cache ... --embed` (x4), `track`, `score` (`bench/README.md`), bench venv `.venv-bench`.

## Results (pooled over the 4 clips, 73 tapped players)

| tracker | licence | HOTA | IDF1 | IDSW | target coverage | wrong frames | of which unflagged |
|---|---|---|---|---|---|---|---|
| **ours (REF)** | SPANA | 0.542 | 0.674 | 83 | 0.626 | 13 | **0** |
| A ByteTrack | Apache-2.0 | 0.589 | 0.775 | 37 | 0.786 | 109 | 109 |
| B BoT-SORT + CMC | Apache-2.0 | 0.595 | 0.776 | 42 | 0.714 | 273 | 273 |
| **D GTA-Link on B** | MIT + Apache-2.0 | 0.607 | 0.796 | 52 | 0.745 | 254 | **254** |
| C Deep-EIoU | none (EVAL-ONLY) | 0.607 | 0.812 | 28 | 0.810 | 152 | 152 |
| D GTA-Link on C | MIT + none | 0.610 | 0.815 | 30 | 0.814 | 152 | 152 |

Unflagged wrong frames per clip (F02 / F05 / F07 / F08): ours 0 / 0 / 0 / 0; B 76 / 0 / 135 / 62; D on B 57 / 0 / 135 / 62;
ByteTrack 57 / 0 / 32 / 20; Deep-EIoU 60 / 24 / 1 / 67. Full table: `tracker_bench_v1.csv` (per clip + pooled, fps).

**Recommended by the pre-registered rule: ours (REF).** It is the only candidate with 0 unflagged wrong frames.

## Findings
1. **GTA-Link does not fix BoT-SORT's silent swaps.** It removes 19 of 273 unflagged wrong frames (FRESH_02 76 -> 57);
   FRESH_07 (135) and FRESH_08 (62) are unchanged. It raises continuity a little (coverage 0.714 -> 0.745, IDF1 0.776 ->
   0.796), not identity safety. Every wrong frame from an external tracker is unflagged: none of them has an uncertain state.
2. **The 2026-11-20 BoT-SORT result is reproduced by an independent harness:** 273 unflagged wrong frames here vs 272 in
   that record (different scorer code, same detections). The "not adopted" decision stands.
3. Deep-EIoU has the best continuity and overall scores (IDF1 0.812, coverage 0.810) but still 152 silent wrong frames,
   and it has no licence, so it can never be the product tracker. GTA-Link on it changes nothing material.
4. Ours: lowest HOTA / IDF1 / coverage, 0 silent wrong frames (13 wrong frames, all while it marked identity uncertain).
   The gap is continuity, as in the earlier records; it stays the open tracking work inside our tracker.
5. Numbers differ from `tools/target_score.py` by design (no target layer here): e.g. ours 13 wrong frames here vs 1 there;
   this harness's ByteTrack is `trackers` 2.7.0, not the supervision 0.25.1 one in the 2026-11-18 record (109 vs 217).

## Files
- `tracker_bench_v1.csv`: the scored table (= `bench/results/tracker_bench_v1.csv`).
- `HASHES.txt`: inputs, caches, every run file, external commits, bench package versions.
