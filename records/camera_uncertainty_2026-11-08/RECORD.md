# Camera-motion uncertainty: DEV validation (2026-11-08)

- Scope: DEV only (FRESH_05 and FRESH_02). EVAL was not used.
- Detections are unchanged from `records/multibody_flag_2026-11-08`.
- Partial-body handling is not changed.

## Change
- `CameraObservation.motion_std_px` measures how uncertain the camera estimate is. It is the per-axis RMS
  disagreement, in source pixels, between *all* tracked background features (inliers and outliers) and the fitted
  transform, multiplied by `camera.disagreement_noise_scale` (1.0).
- `BoxKalman.apply_camera` adds that variance to the position covariance. Every per-frame `camera` decision now
  logs the value.

## Calibration
`tools/dev/camera_disagreement_calibration.py` compares the camera uncertainty with the shared camera error: how
far all matched players in a frame are off their predictions together. On FRESH_05, by camera inlier count:

| inliers | frames | shared error, median px | all-feature RMS, median px | error ÷ RMS, p90 |
|---|---|---|---|---|
| 25–49 | 47 | 9.5 | 18.6 | 1.15 |
| 50–99 | 334 | 1.5 | 3.6 | 1.54 |

The other candidate measures underestimate the error at low inliers:
- the RMS of the inliers alone (error ÷ RMS, p90 14.9);
- the standard error of the translation (0.49 px against 9.5 px).

With few inliers the fit follows the wrong motion even though its inliers agree with each other. The disagreeing
features are what measure that ambiguity.

On FRESH_02, all 387 frames have ≥ 200 inliers. There the RMS disagreement is 0.31 px and the shared error 0.11 px.

## Results (NESTED_V1)

| | FRESH_05 before | FRESH_05 after | FRESH_02 before | FRESH_02 after |
|---|---|---|---|---|
| Mahalanobis rejections of continuing tracks (all frames) | 16 | 6 | 19 | 19 |
| track changes whose first cause includes the Mahalanobis gate | 9 | 3 | 8 | 8 |
| ID switches | 28 | 23 | 37 | 37 |
| fragments | 26 | 22 | 37 | 37 |
| identity transfers (unflagged) | 0 (0) | 0 (0) | 8 (3) | 8 (3) |
| detection recall | 0.9867 | 0.9867 | 0.9705 | 0.9705 |
| track coverage | 0.9756 | 0.9711 | 0.9607 | 0.9607 |

- **FRESH_02.** All 20,030 association decisions are identical to before. The camera uncertainty there has a median
  of 0.22 px and a maximum of 5.9 px. Output boxes still differ slightly, through the Kalman gain.
- **FRESH_05 coverage −2 player-frames.** Player 5, frames 348 and 372. In that stretch the player had only
  low-score detections (< 0.5), which are matched by overlap (IoU) alone with a 0.40 threshold. The changed filter
  state lowered the overlap at frame 348 to 0.34. The same track was rejected by that same overlap check one frame
  later in the run before the fix. This is the existing low-score failure mode, not a new one.

Tracks: F05C 3c531f17…bcb4, F02C 88c5a894…3b26 (not committed).
