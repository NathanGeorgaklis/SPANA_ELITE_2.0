# Identity-resolution margin, calibrated on DEV (2026-11-18)

Question:
- When the tracker cannot tell two players apart (IDENTITY_UNCERTAIN), how large must the appearance margin be before
  it may resolve the case?
- The target is ≤ 0.5 % wrong resolutions (design, `records/design/REID_MODEL_DESIGN.md`).

Scope:
- **Model:** re-ID model v2 (`c269450e…`), the colour histogram, and both combined.
- **Data:** DEV FRESH_02 / 05 / 07 / 08, the rival test of `tools/dev/appearance_cue_eval.py` (4,488 tests).
- **Product unchanged:** nothing is wired into the tracker. EVAL not used.

## Method (`tools/dev/reid_calibrate.py`, `config/reid_calibration.yaml`)
- **Margin:** score(query, A) − score(query, B).
  - Resolve to A if margin ≥ m, to B if ≤ −m, otherwise unknown.
  - Resolving to B is wrong.
- **Choosing m** (fail closed: the bound, not the observed rate):
  - margins are scanned from the largest down;
  - m is the lowest margin reached while the exact one-sided 95 % upper bound of wrong / resolved stays ≤ 0.5 %,
    counting from the first margin that meets it.
  - With 0 wrong, meeting the bound needs about 600 resolutions.
- **Combined option:** per-cue z-normalisation, with statistics from the calibration clips; the colour weight is the
  grid value that resolves the most.
- **Honesty check (leave one clip out):** statistics, weight and m are fitted on three clips and applied to the
  fourth; the held-out results are pooled.
- **Final values:** fitted on all four clips.

## Results

**Held out (fitted on the other three clips), pooled over 4,488 tests:**

| option | resolved | wrong | wrong rate | 95 % upper bound | meets ≤ 0.5 %? |
|---|---|---|---|---|---|
| colour | 1,983 (44 %) | 20 | 1.01 % | 1.46 % | **no** |
| re-ID v2 | 1,036 (23 %) | 1 | 0.10 % | **0.46 %** | **yes** |
| colour + re-ID | 1,979 (44 %) | 7 | 0.35 % | 0.66 % | **no** |

**Per held-out clip** (margin fitted on the other three; resolved / wrong / tests):

| clip | colour | re-ID v2 | colour + re-ID (colour weight) |
|---|---|---|---|
| FRESH_05 | 34 / 0 / 53 | 39 / 0 / 53 | 47 / 0 / 53 (0.7) |
| FRESH_07 | 815 / 0 / 1,496 | 456 / 1 / 1,496 | 723 / 0 / 1,496 (0.3) |
| FRESH_02 | 158 / 0 / 1,329 | 296 / 0 / 1,329 | 317 / 1 / 1,329 (0.7) |
| FRESH_08 | 976 / **20** / 1,610 | 245 / 0 / 1,610 | 892 / **6** / 1,610 (0.5) |

**Fitted on all four clips** (the proposal for integration):

| option | margin | resolved | wrong | 95 % upper bound |
|---|---|---|---|---|
| **re-ID v2** | **0.481** (difference of cosine similarities) | 1,154 (26 %) | 1 | 0.41 % |
| colour + re-ID | 1.349 (z units, colour weight 0.6) | 2,191 (49 %) | 5 | 0.48 % |
| colour | 0.518 | 1,806 (40 %) | 3 | 0.43 % |

- **Re-ID v2 by query height** (all four clips): it resolves 37 % / 20 % / 41 % of the queries below 90 px /
  90–130 px / above 130 px, with 0 / 1 / 0 wrong.
- **Leave-one-clip-out margins for re-ID:** 0.481–0.517. The re-ID margin transfers between clips; the colour margin
  does not (0.32–0.64).

## Findings
1. **Only re-ID alone resolves safely on a clip it was not calibrated on.** It had 1 wrong in 1,036 resolutions held
   out (bound 0.46 %). It clears about a quarter of the uncertain cases.
2. **Colour's scale shifts between matches** (kit, light), so a margin fitted on some clips over-resolves on another:
   - FRESH_08 held out: 20 wrong with colour, and 6 wrong with colour + re-ID.
   - The combination resolves almost twice as much, but its held-out bound (0.66 %) misses the target.
   - The combination's in-sample result (0.48 %) looks fine only because it was fitted on the same clips.
3. **This refines the fusion record.** The combination ranks pairs better on average
   (`records/reid_retrain_m001_m002_2026-11-18`), but a fixed resolution margin that transfers to unseen matches
   comes from re-ID alone.
4. **Not adopted.** These values are a proposal for the integration step.

## Proposal for the integration step (needs Nathan's go-ahead)
- **Resolution rule:** in IDENTITY_UNCERTAIN, compare the query with each rival's reference using re-ID v2 only.
  Resolve only if the margin is ≥ 0.481 (all-clip fit) or ≥ 0.517 (the most conservative leave-one-out margin).
  Otherwise stay unknown.
- **Pinning:** the rule is pinned to model sha `c269450e…`. A new model needs a new calibration.
- **Behind a config flag, off by default.** Judged end to end on the DEV clips:
  - silent swaps stay 0;
  - wrong resolutions stay 0;
  - uncertain track-frames and review actions go down.
- **Colour:** keep it for tracking association only. Retry the combination after more TRAIN matches, or with
  per-match colour normalisation.

## Files
- `calibration_summary.json`
- `reid_calibration.config_snapshot.yaml`
- `HASHES.txt`
