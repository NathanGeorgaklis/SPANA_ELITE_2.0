# Appearance cues for same-kit players: DEV comparison (2026-11-08)

- Scope: DEV only (FRESH_05, FRESH_07, plus FRESH_02). EVAL was not used.
- Tracker behaviour was not changed. The nested-box fix was not resumed.
- Tool: `tools/dev/appearance_cue_eval.py`. Full numbers in `cue_eval.json`.

## Data
- Crops of every labelled player box on labelled frames: FRESH_05 450, FRESH_07 1211, FRESH_02 1018.
- **Hard pairs:**
  - contact pairs (labelled boxes IoU ≥ 0.30 on some frame);
  - the player pairs of IDENTITY_UNCERTAIN tracks still unresolved (F05F, F07F, F02F), mapped to labels with NESTED_V2.
- **Rival test**, which mirrors identity resolution: a query crop of player A is compared with a reference crop of
  A and a reference crop of B, each the latest one ≥ 1 s earlier. The test is correct when the query is nearer A.
- **Verification:** AUC of same-player similarity (≥ 1 s apart) against different-player similarity, overall and
  on hard pairs, plus the true-positive rate at 1 % false accepts on hard pairs.

## Cues (models not committed)

| cue | source | licence | sha256 | CPU cost (2 cores) |
|---|---|---|---|---|
| hist | the tracker's torso HSV histogram | ours | — | negligible |
| osnet | `anriha/osnet_x0_25_msmt17` ONNX (OSNet x0.25, MSMT17) | repo MIT; MSMT17 data terms are research-only | e78604f4…d277 | 14 ms/crop |
| youtu | `opencv/person_reid_youtureid` `person_reid_youtu_2021nov.onnx` (OpenCV Zoo, Tencent Youtu) | Apache-2.0 (repo LICENSE); training data not stated | 05796833…580d | 70 ms/crop |
| dinov2 | `onnx-community/dinov2-small` (general self-supervised, CLS token) | Apache-2.0 | f22797ea…c20a | not timed |
| number | tesseract digits-only OCR on the 4× upscaled torso band | Apache-2.0 | — | — |

## Results

**Rival test accuracy**

| cue | FRESH_05 | FRESH_07 | FRESH_02 |
|---|---|---|---|
| hist | 0.992 | 0.898 | 0.885 |
| osnet | 0.976 | 0.922 | 0.911 |
| **youtu** | **1.000** | **0.941** | **0.950** |
| dinov2 | 0.848 | 0.787 | 0.804 |

**Hard-pair verification**

| cue | AUC F05 | AUC F07 | AUC F02 | TPR @ 1 % FAR F05 | F07 | F02 |
|---|---|---|---|---|---|---|
| hist | 1.000 | 0.839 | 0.805 | 0.996 | 0.055 | 0.050 |
| osnet | 0.983 | 0.864 | 0.813 | 0.732 | 0.101 | 0.155 |
| **youtu** | 0.997 | **0.878** | **0.890** | 0.927 | **0.152** | **0.284** |
| dinov2 | 0.752 | 0.690 | 0.651 | 0.189 | 0.099 | 0.070 |

**Same-colour pairs.** Hard pairs whose mean torso histograms are ≥ 0.70 similar: FRESH_07 5 of 16, FRESH_02 10 of
12, FRESH_05 0 of 2. Per-pair rival accuracy, histogram → youtu:

| pair | hist | youtu |
|---|---|---|
| FRESH_07 12\|13 | 0.68 | 0.77 |
| FRESH_07 5\|6 | 0.71 | 0.90 |
| FRESH_07 12\|9 | 0.80 | 0.78 |
| FRESH_02 3\|5 | 0.92 | 1.00 |
| FRESH_02 15\|16 | 0.92 | 0.98 |
| FRESH_02 10\|8 | 0.91 | 0.96 |

**Robustness (rival accuracy, youtu vs hist)**
- Occluded query: FRESH_07 0.851 vs 0.833; FRESH_02 0.800 vs 0.783.
- Player height < 90 px (FRESH_07): 0.957 vs 0.942.
- Every cue loses 5–15 points when the query is occluded.

**Shirt numbers**
- Confident reads (≥ 60): FRESH_05 4.7 %, FRESH_07 2.8 %, FRESH_02 2.3 % of crops.
- 0–5.6 % per height bin.
- Agreement with the player's majority read: 0.61, 0.65, 0.80.
- Hard pairs with two different reliable numbers: 1 of 30.

## Limits of this comparison
- Label boxes are used as crops, so some overlap a neighbour; this applies equally to every cue.
- Reference gaps are mostly 1–3 s. Longer gaps are barely sampled (FRESH_05 11 triplets, FRESH_02 4).
- There are no camera cuts inside these clips, so robustness across cuts is not measured.
- The youtu model's training data is not documented. Its suitability for commercial deployment must be confirmed
  before production use.
