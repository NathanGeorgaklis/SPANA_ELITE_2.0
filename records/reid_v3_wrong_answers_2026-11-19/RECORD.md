# Re-ID v3: the confident wrong answers on DEV are crops of the wrong person (2026-11-19)

Scope:
- Analysis only, on Nathan's Mac. No labels, models, configs or product code changed. EVAL not used.
- Question (Nathan): why does v3 fail the held-out margin calibration (`records/reid_retrain_m001_m003_2026-11-19`)?
  The failure is decided by the FRESH_08 wrong answer at margin −0.556.
- **Method:** `find_test.py` (in this folder) maps a rival-test index of `tables_p3.json` back to its crops.
  - It re-runs the same crop and triplet construction as `tools/dev/appearance_cue_eval.py`.
  - It checks that the re-ID similarities reproduce the table to 1e-4.
  - It writes the query / reference crops and the query frame.
  - Outputs: `~/spana-f08-wrong/out*/` on the Mac.

## The rival tests v3 answered wrong with |margin| ≥ 0.476 (the fitted margin)

| clip | test | pair | v3 margin | what the crops show |
|---|---|---|---|---|
| FRESH_08 | 556 | 13 vs 17 | −0.556 | **Reference of 13** (frame 252) is labelled *occluded*: the crop is mostly a light-blue opponent standing in front of him. The query (frame 282) is a clean red #13, which looks like red #17 more than like a blue player. Colour also chose 17 (−0.612). Test 557 (−0.407) uses the same occluded reference. |
| FRESH_07 | 906 | 4 vs 5 | −0.492 | **Query of 4** (frame 342) is labelled *occluded* and overlaps 5 at IoU 0.64: the crop is mostly blue #5. |
| FRESH_02 | 134 | 8 vs 10 | −0.476 | **Query of 8** (frame 180) overlaps 10 at IoU 0.70 (not labelled occluded): an opponent covers most of the crop. |

## Findings
1. **No confident wrong answer comes from a clean picture.** In each one, the query or the reference shows mostly
   another player. The model judged what it saw; the test asked about someone else.
2. **The DEV rival test is not the tracker's situation.**
   - The test takes the last crop of each player at least 1 s before the query, whatever its overlap.
   - The tracker only uses clean, uncontested crops for both its reference and its query (`appearance.clean_max_iou`
     0.05; `records/reid_resolve_dev_2026-11-18`).
   - So the calibration counts errors the tracker could not make, and the held-out pass/fail of v2 and v3 turns on
     them.
3. **This also argues against letting re-ID use heavily overlapped crops** (the earlier proposal in
   `records/reid_resolve_dev_2026-11-18`): these are exactly where its confident errors come from.

## Proposed next step (not done)
- **Calibrate on the tracker's own condition:** keep a rival test only if the query and both references are clean,
  i.e. not labelled occluded and no other labelled player overlaps by more than `appearance.clean_max_iou`, read
  from config.
- Re-run the v2 and v3 calibrations on that subset. If v3 passes held out, switch the tracker to v3 with its
  calibrated margin, still off by default, and re-run the end-to-end DEV check.
- Keep the all-crops calibration as a reported number.
