# Motion-gate (Mahalanobis) failures: diagnosis on DEV FRESH_02 and FRESH_05 (2026-11-08)

- Scope: DEV only. EVAL was not used. Tracker behaviour was not changed.
- Tool: `tools/dev/motion_gate_probe.py`. It replays `spana track` with read-only probes.
- Check: the replayed tracks hash-equal the reference artifacts (`F02M.trk.json` and `F05.trk.json`, see `records/multibody_flag_2026-11-08`).
- Probe output: `FRESH_02_gateprobe.json` and `FRESH_05_gateprobe.json`. Each failure case lists:
  - predicted and detection centre
  - innovation in body heights
  - Mahalanobis² and the threshold (`gate_chi2` 9.21)
  - √P, √R, √S
  - velocity
  - time since the last observation
  - streak length
  - competing candidates
  - the camera estimate
  - the common-mode innovation of the frame's other matches

## Hypothesis "uncertainty collapses during continuous tracking": rejected

| | FRESH_02 | FRESH_05 | consistent filter |
|---|---|---|---|
| accepted continuous matches | 18801 | 2777 | |
| mean NIS (Mahalanobis² of accepted matches) | 0.163 | 0.413 | 2.0 |
| share over gate | 0.07 % | 0.00 % | 1 % |
| √S median, streak 1–4 / 5–29 / ≥30 (h) | 0.081 / 0.072 / 0.072 | 0.081 / 0.072 / 0.072 | |

- √S levels off at about 0.072 h and does not keep shrinking. The measurement-noise floor √R is 0.05 h.
- For ordinary motion the filter is over-conservative: NIS is far below 2.
- All rejected matches had dt = 1/30 s. These are continuous-tracking rejections, not gaps.

## What the rejected innovations are

There are 17 cases: 8 on FRESH_02 and 9 on FRESH_05. The truth column gives:
- 8 confirmed true continuations: the successor track took over that same detection, or the frozen label confirms it on a labelled frame.
- 9 not verifiable: the detection went to another track, or the frame is unlabelled.

| class | FRESH_02 | FRESH_05 | signature |
|---|---|---|---|
| A. box extent change (partial ↔ full body) | 6 (189, 219, 221, 241, 280, 381) | 2 (4, 10) | One edge stays put (shift ≤ 0.15 h) while the other moves 0.3–1.5 h. Height ratio 1.2–2.55; `height_ok` also fails in 5 of the 8. |
| B. camera-motion estimate error | 0 | 5 (354 ×2, 374, 386, 630) | Lateral innovation 0.13–0.30 h. Camera status OK but only 32–35 inliers (`min_inliers` 25). The frame's other matches share a common-mode innovation of 0.03–0.10 h. |
| C. young track with spurious velocity | 2 (75, 79) | 0 | Streak 3, velocity 2.4–3.5 h/s. Both edges shifted. |
| other | 0 | 2 (144, 190) | 190: x innovation 1.32 h, a different player. 144: mixed shift, streak 72. |

Camera support against common-mode innovation, from all accepted matches per frame:
- FRESH_02, inliers ≥ 150 (387 frames): median 0.0017 h, p90 0.0043 h.
- FRESH_05, inliers 50–149 (340 frames): median 0.0101 h, p90 0.0275 h.
- FRESH_05, inliers < 50 (47 frames): median 0.0524 h, p90 0.1065 h.

## Interpretation

- **Class B (FRESH_05).** `BoxKalman.apply_camera` maps the state and covariance through the estimated transform but adds no noise for the transform's own uncertainty. A weakly supported estimate is therefore treated as exact. The failing layer is the camera estimate → Kalman interface.
- **Class A (FRESH_02).** The track had been following a partial-body box. The motion model has no term for a change of box extent, and the height gate rejects the change independently. A covariance change alone would not fix it. The first failing layer to inspect is detection (partial boxes), before any change to tracking.
