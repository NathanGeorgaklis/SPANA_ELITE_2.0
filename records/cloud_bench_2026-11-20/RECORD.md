# Cloud bench, first run (2026-11-20, DEV only)

Owner layer: performance. This is the first run of `tools/cloud_bench.py` on Modal: tag `l4_fp32_r1`, suite `dev_standard`.
- Footage: the DEV G1 2.5 min window (frames 45000–49500), FRESH_02 (G1) and FRESH_05 (G2).
- Upload approved by Nathan on 2026-11-20 (DECISIONS; `records/dev_permissions/`). Game 3 (FRESH_07 / 08) is not licensed for third-party processing, so the tool refuses it (tested). The labelled check therefore covers 2 clips, not 4.
- Code: commit 73fedf9 (the `code_object_ids` in `run.json` equal the final commit's trees). Product config with profiling on, `--device cuda` (fp32, eager).
- Each unit ran on its own NVIDIA L4 (3 at once), with torch 2.15.0+cu132 and Modal's default CPU.
- Wall time 795 s, of which about 80 s was the image build, which is cached for later runs. 872 GPU-seconds.

## Result: not comparable to the Mac yet (`comparable_to_mac: false`)

| check | result |
|---|---|
| decode identical (first 30 frames of each unit) | **no, on all 3 units** |
| window identity vs Mac baseline (218 probes) | **fail**: agreement 0.931 (min 0.99), 3 conflicting answers (max 0), 8 probes not reproducible; LOST + ABSTAIN −0.55 pt (ok) |
| labelled identity, FRESH_02 + FRESH_05 (`performance.option_*` limits) | **pass**: ID switches 31 → 31, unflagged transfers 0 → 0, wrong reviewer frames 1 → 0, recall 0.984 → 0.986; info: identity transfers 3 → 4, reviewer correct-time 0.851 → 0.842 |

### Why decoding differs (`probe/decode_probe.py`, FRESH_02 frames 0–2)
- Both sides run opencv-python-headless 5.0.0, but the wheels bundle different FFmpeg builds:
  - Mac: avcodec 61.19, swscale 8.3;
  - Linux: avcodec 62.28, swscale 9.5.
- The decoded BGR frames differ by a mean of 1.46 levels and up to 23: 74 % of the values differ, and 23 % differ by more than 2.
  - That is a colour-conversion difference, not rounding.
  - The detector therefore sees different pixels, and the window identity failure cannot be attributed to the GPU alone.
- Before cloud results can stand in for the Mac, decoding must give the same pixels on both platforms. That is a video-layer change and is not done here.

## Speed (window, per processed frame; Mac = `records/identity_speed_2026-11-20`)

| stage | Mac M3 (MPS, compile) | Modal L4 (CUDA fp32, eager) |
|---|---|---|
| detect.model | 329.4 ms | **79.4 ms** |
| detect.preprocess | 5.7 ms | 10.0 ms |
| tracking | 3.8 ms | 15.7 ms |
| camera_motion | 5.1 ms | 8.0 ms |
| pipeline (per-frame, one-offs apart) | 347.1 ms | **120.0 ms** (detect 92.0 + track 28.0) |
| real-time factor (1 GPU) | 10.41x | **3.60x** |

- The L4 runs the network 4.1x faster than the M3.
- The CPU stages are 2–4x slower in the container (gVisor, Modal's default CPU share, no OpenCL), so tracking is now a large part of the time.

Full-match projection (90 min = 162,000 frames; detection split over N GPUs, tracking sequential; arithmetic, not measured):

| GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| minutes | 324 | 200 | 138 | 107 |

- At 8 GPUs, sequential tracking (28 ms × 162,000 ≈ 76 min) is the floor.
- "Minutes" therefore also needs:
  - faster tracking: more CPU per container, or tracking in chunks;
  - a faster detector runtime: TensorRT / fp16, gated by identity.

## Files
- `l4_fp32_r1/`: run.json (per-unit wall, GPU, frame hashes), summary.json (`spana.cloud_bench` 1.0.0), judge.json, target scores, window compare and profile summary, per-unit profiles and benchmarks.
- `probe/decode_probe.py`
- `HASHES.txt`: sha256 of the large det / trk artifacts in `runs/cloud_bench/l4_fp32_r1/` (gitignored).
