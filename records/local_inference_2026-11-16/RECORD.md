# Local inference for detector v1 (2026-11-16): feasibility and speed on the MacBook (Apple M3, 8 GB)

Scope: DEV only. Nothing tuned; EVAL not used. Goal: run detector v1 (`spana-elite-2-rfdetr-medium-t1`) without
hosted calls (cost per frame, frames leave the machine).

## Weights and export
- The workspace plan allows downloading the trained checkpoint: `models/rfdetr_v1/weights.pt` (not committed),
  sha256 8e030d40db7e86ee801263578525901d32221e8c1f583e4d695fa2bd52f74987, 139 MB. It holds 509 tensors and the
  training args. It does not store the input resolution or class names.
- `tools/export_rfdetr_onnx.py` (rfdetr 1.12.1, Apache-2.0, run in a separate Python 3.11 environment because rfdetr
  needs Python >= 3.10; the engine's Mac venv is Python 3.9):
  - `rfdetr-medium-1280.onnx`, sha256 369b7611af20a1302e43db8b04e99021d0ce99a001c2eeaf9770534308588382
  - `rfdetr-medium-576.onnx`, sha256 edf83195a4c952292a5e0ec1315e9a239d04c99e81a5e883caf14855affb56e3
  - Input (1, 3, R, R) float32: RGB / 255, ImageNet mean / std, plain resize to R x R.
  - Outputs: `dets` (1, 300, 4; cx, cy, w, h normalised) and `labels` (1, 300, 5 logits).
  - Class index (from agreement with the hosted model): 1 ball, 2 goalkeeper, 3 player, 4 referee; 0 never used.

## Agreement with the hosted model (66 cached DEV benchmark frames; hosted predictions from the scoring run)
| input | hosted boxes >= 0.5 matched (IoU >= 0.7) | median IoU | median score difference |
|---|---|---|---|
| 1280 | 820 / 823 (0.996) | 0.953 | 0.014 |
| 576 | 785 / 823 (0.954) | 0.906 | 0.028 |
The hosted model runs at 1280, as the training record says.

## Accuracy on the frozen DEV benchmark (`data/labels/dev_bench/bench_v1`, same metric code as tools/dev_bench_score.py)
| | hosted | local 1280 | local 576 |
|---|---|---|---|
| mAP50 | 0.701 | 0.694 | 0.640 |
| player AP50 / recall | 0.892 / 0.804 | 0.897 / 0.823 | 0.906 / 0.822 |
| ball AP50 / recall | 0.873 / 0.705 | 0.845 / 0.705 | 0.690 / 0.568 |
| persons found at score_high by the person view, class ignored | 0.931 (759/815) | **0.945 (770/815)** | 0.903 (736/815) |
| person-view false boxes >= score_high | - | 26 | 48 |
Local 1280 is as accurate as hosted. 576 loses persons and the ball, so it is not an option.

## Speed at 1280 on the M3 (s per frame)
| route | s/frame | runs in the engine's Python 3.9 venv |
|---|---|---|
| ONNX Runtime 1.19.2, CPU | 2.5 | yes |
| ONNX Runtime 1.19.2, CoreML (NeuralNetwork or MLProgram) | fails (error after ~5 min compile) | - |
| ONNX Runtime 1.32, CoreML MLProgram | 8.4 | no (Python >= 3.10) |
| PyTorch 2.15 MPS (GPU) | **0.36** | no (Python >= 3.10 + PyTorch) |
At 576: ONNX Runtime CPU 0.26 s/frame, and CoreML 0.59 s/frame (only 654 of 923 nodes in 154 partitions).

## Options put to Nathan (decided below: option A)
- The fast local route (0.36 s/frame) needs the engine on Python >= 3.10 with PyTorch + rfdetr as the detection
  backend.
- ONNX CPU at 1280 works today but is about 7 times slower.
- Hosted inference is fast but costs a call per frame.

## Decision and implementation (Nathan chose option A, 2026-11-16)
- Mac project venv moved to Python 3.13.18 (uv-managed, user-local), installed from requirements.lock.txt with every
  pin kept, plus requirements-local-detector.txt (torch 2.15.0, torchvision 0.29.2, rfdetr 1.12.1). The resolved set
  is in requirements-local-detector.lock.txt. Full suite on the new venv: 285 passed, 4 skipped (same as Python 3.9),
  then 291 passed, 4 skipped with the new tests. The old venv is kept as `.venv-py39/` (gitignored) as a fallback.
  numpy 2.5.3 needs Python >= 3.12, so 3.11 does not resolve.
- `spana/detection/rfdetr_local.py` and `detect --backend local` (model record `models/rfdetr_v1.json`). PyTorch on CPU
  and on MPS gives the same raw outputs as the ONNX export: max |logit| difference 3e-4, |box| 2e-5.

## Tracking on DEV FRESH_02 / FRESH_05 with the local backend (torch, MPS)
Same frames, tracker, scorer (NESTED_V2) and config as `records/person_view_2026-11-16/RECORD.md`. No hosted call.

| metric | F02 YOLOX | F02 hosted | F02 local | F05 YOLOX | F05 hosted | F05 local |
|---|---|---|---|---|---|---|
| detection recall | 0.9725 | 0.9804 | 0.9774 | 0.9889 | 0.9978 | 0.9978 |
| missed players | 28 | 20 | 23 | 5 | 1 | 1 |
| recall, occluded players | 0.741 | 0.815 | 0.815 | 1.000 | 1.000 | 1.000 |
| duplicate detections | 126 | 17 | 17 | 84 | 3 | 1 |
| track coverage | 0.9597 | 0.9705 | 0.9686 | 0.9800 | 0.9978 | 0.9978 |
| ID switches | 28 | 25 | 24 | 22 | 8 | 7 |
| fragments | 28 | 24 | 24 | 21 | 8 | 7 |
| duplicate track frames | 16 | 5 | 7 | 3 | 0 | 0 |
| identity transfers (unflagged) | 3 (0) | 3 (0) | 3 (0) | 0 (0) | 0 (0) | 0 (0) |

- Speed, end to end (decode + preprocess + model + person view): FRESH_02 391 frames in 168 s (0.43 s/frame); FRESH_05
  781 frames in 347 s (0.44 s/frame).
- Local equals hosted within small differences (FRESH_02: 3 more missed boxes, 2 more duplicate track frames, 1 fewer ID
  switch) and stays far ahead of YOLOX.
- Artifacts in runs/person_view_compare/ (not committed); sha256 in `HASHES.txt`. Scorer reports:
  `F0[25]_local.bench.json`.
