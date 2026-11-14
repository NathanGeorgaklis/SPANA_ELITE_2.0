# DECISIONS

Seeded 2026-11-14 from code and commit history only. Append new rows; never rewrite old ones.

| date | decision | reason | owner layer |
|---|---|---|---|
| 2026-11-07 | Fresh build; no code from earlier SPANA builds (README) | clean start | repo |
| 2026-11-07 | Layers exchange versioned, validated, atomic, never-overwritten JSON artifacts (`spana/artifacts.py`) | layers depend on artifacts, not internals | repo |
| 2026-11-07 | Config fails closed: no code defaults, unknown/missing keys rejected; seconds and body-height units (`spana/config.py`) | no hidden thresholds | repo |
| 2026-11-07 | Every change committed and pushed to GitHub; repo is source of truth (commit 8dd2231, previous CLAUDE.md) | work never lives only in a session | repo |
| 2026-11-07 | DEV and EVAL strictly separate; EVAL never used without Nathan's explicit approval (previous CLAUDE.md, `spana/reid_labels.py` split_guard, `spana/dataset/config.py`) | unbiased evaluation | repo |
| 2026-11-08 | Human labels are frozen (`tools/freeze_labels.py`) before any inference is scored against them, and never edited (commits 863ba6b, ee9ecda, cb40ea9, afa93ee) | no label leakage | evaluation |
| 2026-11-07 | Detector: pinned YOLOX-s COCO ONNX, sha256-checked, provisional (`models/yolox_s.json`, commit 3d68382) | Apache-2.0, reproducible | player detection |
| 2026-11-07 | Box size is never extrapolated by a size velocity (`spana/tracking/kalman.py`, commit 4650492) | DEV FRESH_01 size jitter | player tracking |
| 2026-11-08 | Possible multi-body boxes are flagged; they cannot spawn a track or update appearance (commit a9f88f3) | DEV merged-player boxes | player detection |
| 2026-11-08 | Camera-motion uncertainty widens track position variance (commit b2cd82e) | DEV motion-gate failures | player tracking |
| 2026-11-08 | Low-score association round only for confirmed tracks (commit e85223a) | DEV partial-body duplicates (M1) | player tracking |
| 2026-11-08 | Lasting IDENTITY_UNCERTAIN state; resolved only by clean appearance evidence (commits 2755b34, 940568d) | no silent identity swaps | player tracking |
| 2026-11-08 | Default benchmark rule NESTED_V2, frozen before re-scoring; NESTED_V1 kept (commit 0ab4c5a) | foot-point attribution for overlapping players | evaluation |
| 2026-11-08 | Nested-box (M2) duplicate fix deferred until Nathan approves (`records/HANDOVER.md`) | a part box is sometimes a different real person | player tracking |
| 2026-11-08 | Youtu re-ID not used (not licence-safe); SPANA-owned re-ID model designed, not trained (commits ebd99f1, 0f4a211) | licensing | target identity |
| 2026-11-08 | TRAIN re-ID matches need explicit written rights in the manifest (commit 21e8abf) | footage rights | target identity |
| 2026-11-12 | Roboflow via REST API, not the SDK (commit 6c068c8) | SDK conflicts with Python 3.9 / OpenCV pins | player detection |
| 2026-11-12 | Roboflow uploads/hosted inference need allow-listed, licence-confirmed matches; EVAL never uploaded (commits c19988e, af4cbca) | footage licensing | player detection |
| 2026-11-12 | Roboflow split moves by delete + re-upload with restore file (commit 9380526) | re-upload alone does not move splits | player detection |
| 2026-11-14 | Chat does architecture/diagnosis/review; Claude Code implements; state lives in `docs/` (this commit) | shared state for both sides | repo |
| 2026-11-14 | Detector v1 trained on dataset version 2; its 2 test frames (Roboflow refuses to train without test images) come from a match outside training and are never scored (`version.platform_test_frames: 2`, `records/roboflow_version/version_2.json`) | platform requirement; held-out testing stays EVAL's job | player detection |
| 2026-11-14 | The licence gate (`match_of`) accepts a `split:valid` tag on a TRAIN-licensed match, because whole-match assignment to valid is allowed (`splits.assignable`); a DEV or EVAL match still accepts only its own split, and `test` is never assignable | M001 is a held-out TRAIN match used for model selection | player detection |
| 2026-11-14 | The version filter drops every excluded tag: `excluded:blur` as well as `excluded:near-duplicate` (`DatasetConfig.excluded_tags`) | blurred frames must not train | player detection |
| 2026-11-14 | Detector v1 (`spana-elite-2-rfdetr-medium-t1`) is the baseline. No threshold tuning (`detector.confidence` / `iou` stay 0.50) until the DEV benchmark exists and has been reviewed | M001 shares a broadcaster with the training matches, so valid scores do not show generalisation | player detection |
| 2026-11-14 | DEV benchmark frames live in their own Roboflow project `spana-elite-dev-bench` (own annotation group); config refuses `dev_bench.project == roboflow.project`, and the training version gate refuses any `bench:*`-tagged frame | benchmark frames can never enter a training version | evaluation |
