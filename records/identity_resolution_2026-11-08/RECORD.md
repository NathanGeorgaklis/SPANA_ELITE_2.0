# Why IDENTITY_UNCERTAIN rarely resolves: DEV FRESH_02 / FRESH_05 / FRESH_07 (2026-11-08)

- Scope: DEV only. EVAL was not used.
- No engine code changed. The nested-box fix was not resumed.
- Runs: F02E, F05E, F07E (`records/identity_uncertain_2026-11-08`).
- Tools:
  - `tools/dev/identity_resolution_probe.py`: hash-checked replay. For every frame a track is uncertain, it records
    why resolution evidence did or did not advance.
  - `tools/dev/torso_clean_check.py`: whether a looser crop-cleanliness rule would have produced appearance.

## Per-track blocker
Rule: an observed uncertain span shorter than `identity_resolve_s` (15 frames) means the track ended first.
Otherwise the blocker is the first that applies, in this order: own appearance never stored; every rival's
appearance never stored; appearance margin; no clean crop.

| class | tracks | observed uncertain track-frames |
|---|---|---|
| resolved | 4 | 152 (8 %) |
| track ends first (lost within < 0.5 s of observation) | 41 | 158 (8 %) |
| **own appearance never stored** | **10** | **1012 (54 %)** |
| no clean crop (crop overlaps another detection) | 3 | 235 (13 %) |
| appearance margin too small (same-looking players) | 2 | 151 (8 %) |
| rival appearance never stored | 3 | 144 (8 %) |
| other (favouring evidence not sustained) | 1 | 25 (1 %) |

Per-track classes are in `track_classes.json`. By clip:
- FRESH_02: 23 tracks; 3 resolved, 12 ended first.
- FRESH_05: 8 tracks; 1 resolved, 6 ended first.
- FRESH_07: 33 tracks; 0 resolved, 23 ended first.

## The largest cause: a track born in contact never gets an identity reference
- All 10 "own appearance never stored" tracks overlapped another track on every observed frame from birth to the
  trigger. The minimum IoU with the nearest track was 0.08–0.65. They became uncertain 5–21 frames after birth.
- Their crops were never clean, so the appearance model was never initialised. Uncertainty then freezes the
  appearance by design. They stay uncertain for the rest of their life: 1012 observed frames, about 100 per track.
  Examples:
  - FRESH_07 T000191: frames 247–390. This is the correctly tracked player 18 from the earlier diagnosis.
  - FRESH_07 T000175 (194–390) and T000145 (142–364).
  - FRESH_02 T000279 (282–390) and T000282 (304–390).
- **The rival side is usually gone.** For these frames the rival track is UNOBSERVED or ENDED in 534 of 656 frame
  checks (FRESH_02 264 of 260+, FRESH_05 94, FRESH_07 375). There is never a frame with both crops clean at once,
  so a simultaneous pairwise comparison is impossible.
- **The rival's appearance is usually known.** 9 of these 10 tracks have at least one rival with a stored
  appearance. The current rule needs both stored appearances, so that one-sided knowledge is never used.
- **A looser crop rule would not help.** If "clean" were judged on the torso band alone, it would have given an
  appearance to only 4 of these 10 tracks (0 on FRESH_02, 0 on FRESH_05, 4 on FRESH_07). It would also have made
  15–56 % of NO_CLEAN_CROP frames usable.

## Estimate of a one-sided exclusion test (offline, from probe rows)
On "own appearance unknown" frames with a clean crop, the crop is clearly unlike every rival's stored appearance
(similarity < `min_similarity` 0.70) in:
- FRESH_02: 77 of 130 frames;
- FRESH_05: 0 of 47;
- FRESH_07: 25 of 349.

The first such frame comes for FRESH_02 T000263 (@267), T000279 (@302), T000282 (@318) and FRESH_07 T000175 (@244),
T000191 (@270). The 0.5 s persistence requirement was not checked here.

In most of the remaining frames the crop looks like a rival (similarity ≥ 0.70): FRESH_02 53, FRESH_05 47,
FRESH_07 172. Players in the same kit cannot be separated by a torso colour histogram. Leaving those uncertain is
correct, not a defect.
