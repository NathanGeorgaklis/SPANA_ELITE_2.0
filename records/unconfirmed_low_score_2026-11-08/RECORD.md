# Low-score round restricted to confirmed tracks: DEV validation (2026-11-08)

- Scope: DEV only. EVAL was not used.
- FRESH_07 v1 labels were frozen (commit cb40ea9) before FRESH_07 was ever run.
- The FRESH_07 baseline was tracked with the code before this change (HEAD~1 of the fix commit) on the same detections.

## Change
- S2 (low-score, IoU-only) now runs only for confirmed tracks.
- Unconfirmed tracks match only high-score detections (S3).

## Results (NESTED_V1)

| | FRESH_02 before → after | FRESH_05 before → after | FRESH_07 before → after |
|---|---|---|---|
| ID switches | 37 → 33 | 23 → 20 | 59 → 48 |
| fragments | 37 → 31 | 22 → 19 | 57 → 47 |
| duplicate track frames | 15 → 13 | 7 → 3 | 15 → 11 |
| identity transfers (unflagged) | 8 (3) → 10 (3) | 0 (0) → 0 (0) | 5 (3) → 5 (3) |
| track coverage | 0.9607 → 0.9607 | 0.9711 → 0.9689 | 0.9637 → 0.9595 |
| detection recall | 0.9705 | 0.9867 | 0.9744 |

- **FRESH_02: +2 flagged transfers.** Both are on one track, T000229, which moves between players 9 and 8 at
  frames 210 and 216. This is in the players 8/9/10 contact cluster. Both transfers are flagged, so the tracker
  marked the association as ambiguous. Unflagged transfers stay at 3.
- **Unconfirmed churn.** Tracks that previously survived on low-score boxes now terminate as UNCONFIRMED_MISSED
  and respawn:
  - FRESH_02: 41 → 198 (spawns 213 → 316). 172 of the 198 are inside labelled ignore regions (crowd); on labelled players it is 19 → 15.
  - FRESH_07: 57 → 179.
  - Confirmed tracklets drop on FRESH_02 (171 → 115): fewer weak-evidence tracks become confirmed.
- **FRESH_07 unflagged transfers (3)** are unchanged by this fix. They are all track T000191 swapping between
  players 17 and 18 (frames 294, 342, 348).

## Artifacts (not committed)
    0b0c4ab7…53b854a0  F07.det.json
    0d241f15…dc47ffe0  F02D.trk.json
    8bcbd6f0…730df42d  F05D.trk.json
    b147ed89…922318ec  F07B.trk.json
    17f11645…b0297478  F07.trk.json
