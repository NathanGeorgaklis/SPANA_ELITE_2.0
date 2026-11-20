# Pan-aware appearance: blurred crops no longer split a player (tracking, DEV, 2026-11-20)

Owner layer: player tracking. Branch `tracking/pan-aware-appearance` (from main `d6511e2`). Mechanism `a61c2a2`
(off); this commit turns it on. DEV FRESH_02 / 05 / 07 / 08 re-tracked from the existing local-detector detections,
and the 10-minute G2 window. No EVAL, no full match.

## Re-baseline first (CODE TASK step 1)
Main `1a161ca` (re-ID resolution merge, off by default) reproduces main `24ccd58` on DEV:
- tracks identical (apart from the new `resolved_by` field);
- 105 fragments, 7 flagged transfers, 0 silent;
- re-ID offers 16 right / 0 wrong;
- unaided 0.536 / 1 wrong frame;
- reviewer 0.783 / 2 / 5.50.

The 10-minute window with this branch's switch off is identical to the 2026-11-20 main window run.

## Why
`records/veto_root_cause_2026-11-20`: FRESH_08 player 10 was split by the colour veto because his stored appearance
had been learned only from motion-blurred crops during a fast pan (torso band about half background), then compared
with a sharp crop. On DEV, 9 of 22 wrong own-box vetoes had pan-learned models. Lowering `min_similarity` was
unsafe (2 right vetoes at 0.62 / 0.67).

## Built (`appearance.pan_aware`, `pan_blur_px` 5.0, `weak_min_similarity` 0.50)
- Camera speed per frame = the translation of that frame's camera transform (0 unless camera status OK).
- A crop taken above `pan_blur_px` is "blurred". It never overwrites a stored appearance that is mostly sharp.
  `appearance_blur` is the share of the stored appearance learned from blurred crops, averaged like the
  appearance itself.
- While the crop or the stored appearance is blurred, the colour veto applies only below `weak_min_similarity`.
  This covers every veto site: S1–S3 gating, S3b, S3c, S4 APPEARANCE_MISMATCH and the gate report.
- Positive appearance evidence is unchanged: identity resolution, S4 long-gap recovery and the hidden-rival margins.
- Priors from the root-cause analysis, not fitted: wrong vetoes' models were learned at >= 9 px / frame and right
  vetoes' at <= 2.4, except one right veto at similarity 0.07. Config 2.13.0.
- **Tests** (`tests/test_pan_aware_appearance.py`, 7):
  - a pan-learned appearance no longer splits its own player (it does today);
  - another player is still vetoed after a pan;
  - a pan starting neither splits a sharp track nor overwrites its appearance;
  - another player during a pan is still vetoed;
  - on by default.

  Full suite 544 passed, 6 skipped.

## Measured on DEV (switch off = main, byte-identical tracks)
| | main | **pan-aware on** |
|---|---|---|
| fragments (F02 / F05 / F07 / F08) | 105 (23 / 7 / 28 / 47) | **99** (23 / 7 / 28 / **41**) |
| ID switches | 107 | **101** |
| identity transfers | 7, all flagged | the same 7 events (frame, players); none new, none gone |
| silent transfers / wrong recoveries | 0 / 0 | 0 / 0 |
| follower unaided correct / wrong frames | 0.536 / 1 | **0.561** / 1 |
| reviewer correct / wrong frames / actions per min | 0.783 / 2 / 5.50 | **0.785 / 2 / 5.08** (−7.6 %) |
| re-ID offers right / wrong / undecided | 16 / 0 / 0 | 14 / 0 / 0 |
| predecessor offers right / wrong / undecided | 10 / 0 / 2 | 8 / 0 / 2 |

- Only FRESH_08 (the pan-heavy clip) changes. FRESH_02 / 05 / 07 benchmarks are byte-identical.
- **FRESH_08 player 10:** followed on one track, TARGET 3–313, with one-frame hidden abstains on 94 / 103. Main split
  him on frames 45, 99, 104 and 148.
- **Fewer offers** because there are fewer losses to recover from. **FRESH_08 player 6 lost his right re-ID offer**
  (main offered it at 269): the one regression seen.
- **Pass bar (agreed 2026-11-18): all met.**

## 10-minute DEV window (G2 frames 170,250–188,249; unlabelled)
| | off | on |
|---|---|---|
| confirmed tracks started | 1,703 | 1,687 (−0.9 %) |
| ends LOST_EXPIRED / LEFT_FRAME | 911 / 774 | 890 / 779 |
| vetoes that alone stopped a match | 1,359 | 1,228 (−10 %) |
| identity-uncertainty episodes | 615 | 610 |
| new uncertainty within 1 s, per confirmed observed track-frame | 7.5 % | 7.4 % |
| observed confirmed track time IDENTITY_UNCERTAIN | 18.2 % | 18.3 % |

No sign of harm. The gain is small on this window: its camera is slower than FRESH_08's.

## Decision
ON by default (`appearance.pan_aware: true`). It met the agreed tracking bar on DEV and the window check showed no
harm signal.
