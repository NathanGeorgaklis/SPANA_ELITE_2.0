# Detector preprocessing speed-up (2026-11-20, DEV only)

Owner layer: performance. This follows `records/identity_speed_2026-11-18`, REMAINING item 2.
It uses the same host (M3 8 GB), the same DEV G1 2.5 min window (4,500 frames) and the same `run_pass.sh`.
Config: `profile_v2.yaml` is `config/default.yaml` (fp32 + compile) with `profiling.enabled: true`.

## Finding
`detect.preprocess` took 26.9 ms/frame on the CPU, 7.4 % of the pipeline after compile.
- A worker thread that overlaps it with the GPU network was tried first (`probe/gil_probe*.py`) and does not work:
  - The GPU wait releases the GIL, but the compiled MPS graph takes the GIL back for every kernel launch.
  - A busy Python thread slows the network from 331 to 1606 ms/frame.
  - Real overlap with preprocessing saved 7 of 38 ms (sequential 383 vs overlapped 376 ms).
  - Not adopted.
- Breakdown of preprocessing (`probe/pre_probe.py`): cvtColor + resize 0.9 ms, uint8→float32 / 255 4.1 ms, normalise `(x - MEAN) / STD` 17.1 ms, transpose + copy 2.5 ms.
  - The normalise step broadcasts over a last axis of length 3, and the steps allocate full-size temporaries.

## Change
`spana/detection/rfdetr_local.py` `preprocess` does the same float32 operations per element (u8 / 255f, − mean_c, / std_c), one channel plane at a time, in place into the (1, 3, R, R) output.
- The output is bit-identical to the old formula: 60 real DEV frames, random frames, 4 size / resolution pairs and uint8 extremes (`tests/test_detect_preprocess.py`).
- No new values, so nothing new in config. The CPU and ONNX paths use the same function.

## Result (2.5 min window)

| | fp32 + compile (2026-11-18) | + preprocess (this change) |
|---|---|---|
| pipeline ms/frame | 361.9 | **347.1** |
| detect.preprocess ms/frame | 26.9 | **5.7** |
| detect.model ms/frame | 324.1 | 329.4 (run-to-run variation; the network is unchanged) |
| real-time factor / 90 min projection | 10.86x / 16.3 h | **10.41x / 15.6 h** |
| detections, sorted rows | | identical |
| track frames | | identical |
| identity compare vs the 2026-11-18 baseline (218 probes) | 1.000 | **1.000**: 0 state / box diffs, 0 conflicts, LOST/UNKNOWN 0.908 unchanged, 3 reacquisitions, median 0.067 s |

Next stages: detect.model 329 ms (95 %), camera_motion 5.1 ms, tracking 3.8 ms. The 1x budget is 33 ms/frame.

Files:
- `profiles/`
- `compare_pre_v2.json`
- `probe/`: probe scripts, DEV measurement only
- `HASHES.txt`: sha256 of the artifacts in `runs/identity_speed/`
