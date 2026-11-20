# Detector speed options (2026-11-20, DEV only)

Owner layer: performance. This measures which route can bring the detector toward the intermediate target. Nothing in the product changed: the default stays medium at 1280.
The target and limits are in config `performance.option_*` (Nathan, 2026-11-20):
- Speed: 3x real time.
- Viability on the labelled DEV clips, against the current detector: no more wrong follower frames (with the simulated reviewer), no more unflagged identity transfers, ID switches up at most 5 %, detection recall down at most 1 point.
- Ball recall is reported, not gated.

Tool: `tools/detector_options.py`.
- `speed`: stage 1.
- `record`: DEV model records (`dev_records/`), with the same weights and sha.
- `judge`: stage 2.

Host: M3 MacBook Air 8 GB, torch 2.15 on MPS, fp32 + torch.compile (the product default).

## Stage 1 (`stage1.json`): 120 frames of the DEV G1 window (from 45000) + the frozen DEV bench (66 frames)
- The projected pipeline is the other stages from `records/identity_speed_2026-11-20` (12.0 ms/frame) + the variant's model + preprocess.
- Agreement = the 1280 boxes with score ≥ 0.50 that are reproduced at IoU ≥ 0.5 on the window.
- Bench persons found counts persons with class ignored, score ≥ 0.50.

| variant (W x H) | model ms | real-time factor | h / 90 min | bench mAP50 | bench persons found (drop) | bench ball AP50 / recall | agreement persons / ball |
|---|---|---|---|---|---|---|---|
| medium 1280x1280 (current) | 319.3 | 10.10 | 15.2 | 0.695 | 0.947 | 0.845 / 0.705 | reference |
| medium 1088x1088 | 209.4 | 6.77 | 10.2 | 0.695 | 0.946 (0.1 pt) | 0.828 / 0.705 | 0.993 / 0.989 |
| **medium 960x960** | **151.3** | **5.00** | **7.5** | **0.708** | **0.944 (0.4 pt)** | **0.845 / 0.682** | **0.991 / 0.957** |
| medium 768x768 | 87.5 | 3.06 | 4.6 | 0.693 | 0.935 (1.2 pt) | 0.853 / 0.614 | 0.989 / 0.914 |
| medium 1280x960 | 221.6 | 7.14 | 10.7 | 0.694 | 0.951 (−0.4 pt) | 0.862 / 0.659 | 0.998 / 0.957 |
| medium 1280x736 (native 16:9) | 155.9 | 5.15 | 7.7 | 0.708 | 0.924 (2.3 pt) | 0.821 / 0.591 | 0.989 / 0.903 |
| medium 1088x640 | 107.5 | 3.67 | 5.5 | 0.691 | 0.930 (1.7 pt) | 0.781 / 0.545 | 0.988 / 0.936 |
| medium 960x544 | 77.1 | 2.72 | 4.1 | 0.679 | 0.909 (3.8 pt) | 0.751 / 0.591 | 0.989 / 0.828 |
| small 1280x736 / 960x960 / 960x544 / 512x512 (COCO, speed only) | 155.1 / 149.7 / 76.2 / 37.2 | 5.13 / 4.93 / 2.71 / 1.58 | | | | | |
| nano 1280x736 / 960x544 / 384x384 (COCO, speed only) | 154.2 / 75.4 / 21.2 | 5.07 / 2.67 / 1.06 | | | | | |

Findings:
- **Network time follows the input pixel count; the model size barely matters.**
  - small and nano take the same time as medium at the same input (960x544: 77 / 76 / 75 ms).
  - Their published speed comes from their small native inputs (512 / 384), which are far below what our far-away players and the 9–13 px ball need.
  - A smaller RF-DETR is therefore not a speed route on its own, and retraining one gains nothing at a given input size.
- **No variant meets both 3x and the 1-point recall limit,** so the agreed rule gives no stage 2 candidates.
  - Under 3x: only 960x544 (2.72x), which loses 3.8 points of persons and a quarter of the balls.
  - 768x768 is just over the target (3.06x) and loses 1.2 points.
- The native 16:9 input (1280x736) loses persons (−2.3 pt). The model was trained on frames stretched to a square, so non-square inputs would need retraining to be judged fairly.
- The bench has only 44 balls, so one ball is 2.3 points of recall: the ball numbers are indicative only.

## Stage 2 (`judge_r*.json`, `labelled/`): the square options nearest the line, run as information for the decision
Full pipeline on FRESH_02 / 05 / 07 / 08: detect under a DEV record, track, `tools/benchmark.py` (NESTED_V2), `tools/target_score.py` with and without the simulated reviewer.
The base is the current detector, whose labelled outputs equal the 2026-11-18 baseline.

| | base 1280 | **960** | 1088 | 768 |
|---|---|---|---|---|
| viable (all limits) | | **yes** | no | no |
| wrong follower frames, reviewer (max +0) | 2 | **1** | 18 | 12 |
| unflagged identity transfers (max +0) | 0 | **0** | 2 | 3 |
| ID switches (max +5 %) | 121 | **102** | 112 | 108 |
| detection recall (max −1 pt) | 0.978 | **0.981** | 0.979 | 0.977 |
| fragments | 120 | 100 | 111 | 106 |
| follower correct-time / wrong frames, no reviewer | 0.496 / 1 | **0.549 / 0** | 0.507 / 1 | 0.530 / 3 |
| follower correct-time, reviewer / reviewer actions | 0.779 / 90 | 0.757 / 77 | 0.745 / 88 | 0.743 / 89 |

Reading:
- **960 passes every limit** and is better on most measures: fewer ID switches and fragments, no wrong frames without the reviewer, fewer reviewer actions.
- Its one loss is reviewer correct-time (−2.2 pt), with 13 fewer reviewer actions.
- **Caution: the result is not monotonic.** 1088 sits between 960 and 1280 in pixels, yet it fails: 2 silent transfers on F07 and 18 wrong reviewer frames. 768 fails too.
  - Small detection changes move tracking outcomes a lot on 4 clips. 960's pass is therefore weak evidence, not proof.
  - Before any adoption, 960 needs more DEV footage: other DEV windows or a DEV half if allowed, and the identity probes on the 2.5-min window.
- Adopting 960 overrides DECISIONS 2026-11-16 ("input 1280"). That decision is Nathan's.

## Files
- `stage1.json`, `judge_r{960,1088,768}.json`, `labelled/` (bench + target_score per option).
- `dev_records/`: DEV records at 1088 / 960 / 768, with the same weights and sha. Their file paths are relative to `runs/detector_options/records/`, where they were used.
- `run_option.sh`
- `HASHES.txt`: the large det / trk artifacts in `runs/detector_options/`, and the COCO small / nano checkpoints (public, `~/.roboflow/models`, speed only).
