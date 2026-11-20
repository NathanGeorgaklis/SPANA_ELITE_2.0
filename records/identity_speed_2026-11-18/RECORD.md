# Identity tracking speed: profile + biggest stage (2026-11-18 .. 2026-11-20, DEV only)

Owner layer: performance (target identity + tracking path). DEV only, no EVAL, no full-match run.
Host: MacBook Air M3, 8 GB, macOS 15.7.3, Python 3.13, torch 2.15.0, rfdetr 1.12.1, OpenCV 5.0.0 (8 threads).

## Clip
DEV G1 (`Sydney_United_APIA.mp4`, DEV match), one 2.5 min window: 4,500 frames at 30 fps (frames 45000..49500).
`run_pass.sh <tag> <config>`: `detect --backend local --runtime torch --device mps` then `track`, with profiling on
(`profile.yaml` = `config/default.yaml` + `profiling.enabled: true`). `target`: `tools/identity_speed.py baseline`.

## STEP 1: profile (no behaviour change), fp32 eager (the old default)
Pipeline 419.7 ms/frame. That is **12.59x real time, so a 90 min match projects to 18.9 h** (`performance.projection_footage_s` 5400).
The budget for `performance.realtime_factor_target` 1.0 is 33.3 ms/frame.
One-off costs (67 s on this window) are kept apart from the per-frame figures. About 20 s of that is the decoder seeking to frame 45000, which does not happen at a match start.

| stage | pass | ms/frame | share | device | batch |
|---|---|---|---|---|---|
| detect.model (RF-DETR medium network, 1280x1280) | detect | 380.2 | 90.6 % | mps (torch, fp32) | 1 frame |
| detect.preprocess (resize + normalise) | detect | 27.6 | 6.6 % | cpu (numpy / OpenCV) | 1 frame |
| camera_motion (LK + RANSAC) | track | 5.2 | 1.25 % | cpu (OpenCV) | 1 frame |
| tracking (Kalman + association) | track | 3.8 | 0.9 % | cpu | 1 frame |
| decode | detect + track | 0.78 + 0.47 | 0.3 % | cpu (OpenCV) | 1 frame |
| io.validate_write | track | 0.35 | 0.1 % | cpu | |
| identity_scoring (follower) | target | 0.2 | n/a | cpu | all tracks |
| detect.decode / person_view / appearance | detect / track | 0.16 / 0.14 / 0.13 | < 0.1 % each | cpu | appearance: 1 crop per call (HSV torso histogram) |
| ball | none | | | not built | |

There is no deep re-ID embedding in the path: appearance is the HSV torso histogram (CPU, 0.13 ms/frame).
Files: `profiles/baseline.*.json`.

Baseline identity (`baseline_identity.compact.json.gz`; full `baseline.identity.json` hashed in `HASHES.txt`):
- 218 probes. A probe is a simulated tap on every ACTIVE, confirmed, CERTAIN track every `performance.identity_probe_every_s` 10 s. Each probe is followed to the end of the window.
- States per frame: TARGET 0.083, LOST 0.889, ABSTAIN 0.019, CANDIDATE 0.009. 209 gaps, 3 reacquisitions (median latency 0.067 s).
- Tracker: 847 tracks, IDENTITY_UNCERTAIN share 0.112.

## STEP 2: the biggest stage (detect.model, 90.6 %)
Model-only probe, 120 frames, same window (`probe/probe_model*.json`; `tools/detector_speed_probe.py`).
The agreement column is the share of fp32-eager boxes with score ≥ 0.5 that are reproduced at IoU ≥ 0.9 with the same class.

| variant | model ms/frame | agreement with fp32 eager |
|---|---|---|
| fp32, 1 frame (old default) | 373–388 | reference |
| fp32, 2 / 4 frames per batch | 389 / 435 | identical (batching is slower on this GPU) |
| bf16 / fp16, 1 frame | 302 / 311 | fp16 0.9968 |
| fp16, 4 frames | 305 | |
| **fp32 + torch.compile** | **338 (re-check on an idle machine 2026-11-20: 373 → 337)** | **1.000** (max logit diff 1.53, box 0.30 px of 1280) |
| fp16 + torch.compile | 260 | 0.9978 |

The network is compute-bound: the encoder alone takes 325 of 360 ms and already uses SDPA attention.

## STEP 3: identity check (`tools/identity_speed.py compare`, tolerances in config `performance`)

The first three columns are whole-pipeline runs on the 2.5 min window, re-tracked, with the 218 probes replayed.

| | fp32 eager (baseline) | **fp32 + compile (adopted)** | fp16 + compile (gated off) |
|---|---|---|---|
| pipeline ms/frame | 419.7 | **361.9** | 293.2 |
| detect.model ms/frame | 380.2 | 324.1 | 256.4 |
| real-time factor / 90 min projection | 12.59x / 18.9 h | **10.86x / 16.3 h** | 8.80x / 13.2 h |
| probe agreement per frame (min 0.99) | | **1.000** | 0.959 (FAIL) |
| probes not reproducible (no unique box with IoU ≥ 0.90) | | 0 | 8 |
| frames with a different state / box | | 0 / 0 | 4,271 / 9 |
| conflicting TARGET answers (max 0) | | 0 | 0 |
| LOST + ABSTAIN share rise (max 0.01) | | 0.000 | −0.0003 |
| reacquisitions / median latency | 3 / 0.067 s | 3 / 0.067 s | 3 / 0.067 s |
| tracks / IDENTITY_UNCERTAIN share | 847 / 0.112 | 847 / 0.112 | changed / 0.135 |

fp32 + compile, checked further:
- Same tracks and tracklets. Track boxes ≤ 0.4 px apart, detection boxes ≤ 0.2 px apart after sorting rows.
- The tracks sha differs only through provenance and the model description (`compiled: true`).

Labelled DEV clips (FRESH_02 / 05 / 07 / 08):
- `run_labelled.sh`: detections re-run under each config, then re-tracked.
- Scored with `tools/benchmark.py` (NESTED_V2) and with `tools/target_score.py`, both as-is and with `--simulate-reviewer`. Files: `labelled/`.

| clip | ID switches base / compile32 / fp16c | identity transfers (unflagged) base / compile32 / fp16c | follower correct-time / wrong frames, reviewer, base = compile32 | fp16c reviewer |
|---|---|---|---|---|
| F02 | 24 / 24 / 24 | 3 (0) / 3 (0) / **6 (2)** | 0.771 / 1 | 0.753 / 1 |
| F05 | 7 / 7 / 7 | 0 / 0 / 0 | 0.964 / 0 | 0.964 / 0 |
| F07 | 38 / 38 / 36 | 2 / 2 / 2 | 0.773 / 1 | 0.773 / 1 |
| F08 | 52 / 52 / 50 | 1 / 1 / 1 | 0.689 / 0 | 0.661 / 0 |

- fp32 + compile is identical to the baseline on every labelled metric: every benchmark field, and target_score with and without the reviewer.
- fp16 + compile adds two **unflagged** identity transfers on F02, i.e. silent swaps. It also lowers reviewer correct-time on F02 and F08.
- Decision: fp16 + compile fails, and it stays gated off (`inference.mps_precision: fp32`).
- Note: the detect profiles in `labelled/` for compile32 (~570 ms/frame) overlapped with a GPU unit-test run on the same machine, so they are not speed evidence. The fp16 runs that came after are not affected (~262 ms model). The speed evidence for compile32 is the 2.5 min run and the idle re-check (`probe/probe_model_compile_recheck.json`).

## Result
- `config/default.yaml` `inference.mps_compile: true` and `mps_precision: fp32` are the new default.
- The CPU path always runs fp32, eager. `tests/test_inference_speed.py` covers the CPU fallback and the GPU determinism (fp32 + fp16).
- **10.86x real time (16.3 h per 90 min match), down from 12.59x (18.9 h).** Identity output is unchanged.

**The 1x target is not reachable by an engineering change to this stage.** The budget is 33 ms/frame. RF-DETR medium at 1280 on the M3 GPU takes 324 ms compiled, and 260 ms even at fp16, which fails identity.
Reaching 1x needs a decision. Possible directions, none of them taken:
- a lower input resolution (conflicts with DECISIONS 2026-11-16 "input 1280");
- a smaller detector;
- detector cadence (restricted by the "no frame skipping that breaks temporal evidence" rule);
- other hardware (a discrete GPU / cloud).
That decision is Nathan's.

## Files
- `profiles/`: raw `spana.profile` 1.0.0 per pass + `*.profile_summary.json` (baseline / compile32 / fast).
- `compare/compare_{compile32,fast}.json`: the identity comparison (`spana.identity_speed_compare` 1.0.0).
- `probe/`: model-stage probes.
- `labelled/`: bench + target_score per clip and variant.
- `baseline_identity.compact.json.gz`: per probe, run-length state / reason / track runs, gaps, relinks, track ends and per-frame identity confidence.
- `HASHES.txt`: sha256 of the large artifacts kept in `runs/identity_speed/` (gitignored).
- `profile.yaml`, `compile32.yaml`, `fast.yaml`, `run_pass.sh`, `run_labelled.sh`: how the runs were made. The run scripts are DEV record scripts, not product code.
