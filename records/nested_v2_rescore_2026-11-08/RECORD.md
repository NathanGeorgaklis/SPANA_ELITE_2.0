# Re-score under NESTED_V2: FRESH_02, FRESH_07 (and FRESH_05) (2026-11-08)

- Same runs as `records/unconfirmed_low_score_2026-11-08`: F02D, F07, F05D.
- Tracker unchanged. DEV only. EVAL was not used.
- NESTED_V1 re-scores are identical to the earlier reports.

| | FRESH_02 V1 → V2 | FRESH_07 V1 → V2 | FRESH_05 V1 → V2 |
|---|---|---|---|
| identity transfers (unflagged) | 10 (3) → 5 (1) | 5 (3) → 0 (0) | 0 → 0 |
| ID switches | 33 → 28 | 48 → 44 | 20 → 20 |
| fragments | 31 → 26 | 47 → 44 | 19 → 19 |
| ambiguous attributions | — → 40 | — → 24 | — → 1 |
| coverage / recall / duplicate track frames | unchanged | unchanged | unchanged |

**Removed as false (V1 transfers that V2 does not count):**
- **FRESH_07, T000191 ×3 (unflagged).** The track is on player 18 throughout (foot ≤ 0.09 h). The frames where it
  overlaps occluded player 17 are now ambiguous.
- **FRESH_07, T000219 and T000241 (flagged).** Both tracks were born while players 7 and 5 overlapped, with 7
  labelled occluded. From the first clear frame they stay on player 5.
- **FRESH_02, T000010 ×2 (unflagged).** The track is on player 26. The one-frame flip to player 1 at 318 is gone.
- **FRESH_02, T000214 and T000274 ×2 (flagged).** Both tracks were born inside an overlap (player IoU 0.70 / 0.90).
  They stay on player 8 from their first clear frame.

**Still counted (real):**
- **FRESH_02, T000014 10→8 @174 (unflagged).** The crossing diagnosed in `records/unflagged_transfers_2026-11-08`.
- **FRESH_02, T000229 9→8 @210 and 8→9 @216 (flagged).**
- **FRESH_02, T000277 9→10 (flagged).** Now dated 300, its first clear frame on player 10 (it was 276 under V1).
- **FRESH_02, T000305 3→5 @390 (flagged).**
