# Multi-body detection flag — DEV validation (2026-11-08)

DEV only (FRESH_02, FRESH_05). EVAL was not used. Matching rule: NESTED_V1. Labels: FRESH_02 v1 and FRESH_05 v1,
both frozen before any scoring. FRESH_05 v1 was frozen in commit ee9ecda, before FRESH_05 was ever run.

## Change
- **Detection layer** (`postprocess.flag_multi_body`). A box is flagged, not deleted, when two conditions hold:
  - it encloses a confident detection: IoA ≥ 0.60 and score ≥ 0.50;
  - it is ≥ 1.20× that detection's width or height.
  The thresholds are in `detection.multibody_*`.
- **Calibration.** Done on FRESH_02 only (`tools/dev/multibody_calibration.py`). Among spawn-capable (score ≥ 0.50) boxes it flagged 23 of 46 two-person boxes and 8 of 694 single-person boxes.
- **Tracker.**
  - A flagged box cannot start a track (`spawn_refused: MULTI_BODY_DETECTION`).
  - Any match through a flagged box is contested (`contested_multi_body`), so it never updates appearance.
- **Detections schema 1.3.0.** Adds a 7th column, multi_body (0/1). `tools/refilter_detections.py` migrates 1.2.0 caches.

## Results

FRESH_02: 1018 GT boxes, 18 people. FRESH_05: 450 GT boxes, 12 people.

| metric | FRESH_02 before | FRESH_02 after | FRESH_05 no flag | FRESH_05 flag |
|---|---|---|---|---|
| detection recall | 0.9705 | 0.9705 | 0.9867 | 0.9867 |
| ID switches | 67 | 37 | 34 | 28 |
| fragments | 59 | 37 | 29 | 26 |
| identity transfers (unflagged) | 8 (3) | 8 (3) | 0 (0) | 0 (0) |
| duplicate track frames | 26 | 15 | 10 | 7 |
| track coverage | 0.9627 | 0.9607 | 0.9756 | 0.9756 |

- FRESH_02: refiltering the cached 1.2.0 detections also removed 1 union box. This comes from rounding in the stored rows.
- FRESH_05: detected at stride 2 (60 fps, so 30 Hz analysis) over frames [0, 782), which covers every labelled frame.

## Remaining track changes, by first cause

`tools/dev/fragmentation_causes.py` gives:
- **FRESH_02 (37):**
  - 8 ended while observed
  - 8 taken by another track
  - 7 hidden-rival refusals
  - 8 Mahalanobis gate failures with a high-score detection ≤ 0.35 h away
- **FRESH_05 (28):**
  - 9 Mahalanobis gate failures (7 of them ≤ 0.35 h)
  - 6 with no detection nearby
  - 4 low-score IoU failures
  - 3 taken by another track

## Artifacts

These are not committed. The commands are in each artifact's provenance.

```
a5cf1d09…4083  F02.det.json   (1.2.0 cache)
ffc63c60…eb8e  F02M.det.json  (refiltered 1.3.0)
53848ce4…d9ed  F02M.trk.json
2aee6176…0112  F05.det.json
e7338871…e341  F05B.det.json  (flag disabled: multibody_min_extent 1000)
510761df…06d9  F05.trk.json
2d5b5931…0cfa  F05B.trk.json
```
