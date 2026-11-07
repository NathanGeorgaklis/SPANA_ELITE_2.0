# DEV validation of detection + multi-player tracking on real footage (2026-11-07)

Roles: Failure Investigator; Player Detection Engineer; Multi-Player Tracking Engineer; QA / Regression Engineer.

## Data (all DEV; no EVAL used or created)
- Ten fresh clips from Nathan: `~/Downloads/SPANA_fresh_clips`, three NPL NSW matches, 1080p, 30/60 fps, ~13 s each.
  Hashes are in `CLIP_SHA256SUMS.txt`, sources in the clips' README.
- Processed: **FRESH_01** (set piece, 30 fps), **FRESH_10** (crowded set piece, 30 fps), **FRESH_04** (counter, 60 fps
  analysed at 30). Not yet processed: 02, 03, 05–09. Detection costs ~2.2 s/frame on 2 CPUs.
- **No human labels exist.** These results are label-free diagnostics: automatic flags from the run's own evidence,
  checked by eye on rendered frames. They are **not a benchmark**: there is no recall, ID-switch rate or
  precision against frozen truth.

## Failures found → first failing layer → fix → regression

| # | Failure (real footage) | First failing layer | Fix | Regression |
|---|---|---|---|---|
| 1 | Clips 02/03/09 refused by the variable-frame-rate guard | Video input timing: the last packet sits 2–3 frame intervals late (trailing PTS from the stream copy) | Frame time = container PTS, which must strictly increase | `tests/test_video.py` (generated video with a dropped frame) |
| 2 | Tracks output rejected by the validator | Output serialisation: IoU 3e-5 rounded to 0 while `overlap_with` was still set | `overlap_with` follows the rounded value | `test_tiny_overlap_serialises_consistently` |
| 3 | Duplicate tracks; neighbours' detections stolen | Detection merge: two depth-aligned players form a **union box** that survives NMS (6–12 per frame after the full + tile merge) | Drop a box that contains two distinct, more confident detections | `UnionBoxes` tests. Visual check: 11/12 true unions, 1 doubtful |
| 4 | Lost tracks never re-associated | Motion model: **size velocity** from jittery real box sizes collapsed or inflated predicted boxes (height ×0.03–×6 within 1–2 s) | No size extrapolation (centre constant-velocity, size random walk) | `test_size_not_extrapolated_under_jitter`, `RealFootageRegressions` |
| 5 | True continuations vetoed in crowds | Appearance evidence: crops that overlap another player were used as evidence | A descriptor is UNKNOWN when the box overlaps another detection (> `clean_max_iou`) | `AppearanceEvidence` test |
| 6 | Fragmentation cascade in crowds | Hidden-rival rule: every stale lost track (up to 2 s) could refuse a live continuation | A rival must be within `occlusion_max_s` and reachable by its own motion | QA suites unchanged (front swap still safe) |

## Gate calibration on isolated real players (FRESH_01, n = 1317 continuations)
- Appearance similarity < 0.70: **0.4 %** of same-player continuations.
- Mahalanobis distance > 9.21: **0.1 %**.
- Gates are well set for isolated players. Different-player similarity reaches 0.92 at p95, so the colour descriptor
  cannot separate same-kit players. That is expected; it is used only as a veto.

## State after the fixes (label-free diagnostics)

| clip | confirmed tracklets | observed confirmed / frame (median) | track losses | no detection near | taken by new track | taken by duplicate | taken by neighbour |
|---|---|---|---|---|---|---|---|
| FRESH_01 | 189 → 179 | 21 | 160 → 154 | 62 | 39 | 23 | 30 |
| FRESH_10 | 151 → 134 | 23 | 138 → 117 | 27 | 47 | 22 | 21 |
| FRESH_04 | 49 → 50 | 5 | 45 → 46 | 28 | 9 | 7 | 2 |

Synthetic safety after all fixes:
- QA scenarios: 0 unflagged transfers, 0 wrong recoveries, 0 duplicates.
- Adversarial set: unchanged (168 runs: 15 unflagged transfers / 11 wrong recoveries, all in the documented
  classes).
- 74 tests pass.

## Remaining failure modes (not fixed; evidence-limited)
1. **Fragmentation in clusters remains high:** ~180 / ~134 confirmed tracklets for ~25 people in 13 s. The causes:
   - detector drop-outs or weak boxes under partial occlusion (~40 % of losses; detection layer);
   - competition from new or duplicate tracks after an ambiguity refusal (~45 %).

   Several plausible fixes trade fragmentation for identity risk. They need human-labelled truth to judge safely.
2. **Non-players are tracked** (spectators, people behind the fence, signage). This belongs to pitch context
   (a later stage) and inflates every fragmentation count above.
3. **The appearance descriptor cannot separate same-kit players.** No re-ID model was added, because the evidence
   does not show appearance as the first failing layer: the losses come from detection drop-outs and the
   ambiguity rules.
4. Runtime: ~2.2 s/frame with tiles on CPU.

## Readiness
**Not ready for target identity.** The tracker is safety-biased (it splits rather than swaps), but fragmentation in
real clusters is far beyond what an identity layer should have to re-link. Next: human-labelled DEV truth for 3–4
clips (boxes + IDs), frozen before re-running. Then measure recall, ID switches and fragmentation, and address the
two remaining causes in order: detection drop-outs first, then the competition after an ambiguity refusal.

## Reproduce
Scripts in `tools/dev/` read runs from `/tmp/dev/runs/` and clips from the staged folder (DEV diagnostics, not
product code). Commands: `python -m spana.cli detect … --stride N`, `python -m tools.refilter_detections`,
`python -m spana.cli track`, `python -m tools.track_review`.
