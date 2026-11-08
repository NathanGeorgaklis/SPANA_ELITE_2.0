# Unflagged identity transfers on DEV FRESH_07 and FRESH_02: diagnosis (2026-11-08)

- Scope: DEV only. EVAL was not used.
- No code changed. The nested-box fix was not resumed.
- Runs: `F07.trk.json` and `F02D.trk.json` (see `records/unconfirmed_low_score_2026-11-08`).
- Tools:
  - `tools/dev/transfer_trace.py`: hash-checked replay with a read-only probe. It records gate reports (centre /
    Mahalanobis / height / appearance / motion) of every track near each detection, and `--span` traces an explicit
    frame range.
  - `tools/dev/transfer_attribution.py`: per labelled frame, the track box against both players' labels (IoU and
    foot-point offset in body heights) and the players' mutual IoU.

## Summary of the 6 unflagged transfers

| clip | track | benchmark transfer | verdict |
|---|---|---|---|
| FRESH_07 | T000191 | 17→18 @294, 18→17 @342, 17→18 @348 | **not a tracker transfer**: attribution ambiguity |
| FRESH_02 | T000010 | 26→1 @318, 1→26 @324 | **not a tracker transfer**: attribution ambiguity |
| FRESH_02 | T000014 | 10→8 @174 | **real transfer** during a crossing |

## FRESH_07 T000191 (players 17 and 18)

- T000191 is the front player 18 at every labelled frame from 270 to 360. Its foot point (bottom-centre) is within
  0.09 h of player 18's label, while player 17 (labelled occluded, behind) is 0.12–0.30 h away
  (`FRESH_07_attribution.txt`).
- The two players' labels overlap heavily: mutual IoU 0.37–0.53 at 270–294 and 0.33–0.52 at 342–360. The
  track's IoU with the two labels is close (e.g. 0.58 against 0.56 at 342).
- The NESTED_V1 assignment ranks by IoU only, so it alternately gives the track to the occluded player. It then
  reports 18 unmatched while visible (a coverage loss) and a "transfer".
- The tracker side was consistent:
  - 270–336 are all S1 matches.
  - Player 17's own track T000179 went OCCLUDED at 267 (detection taken by T000191) and expired at 327.
  - Player 17 was later picked up by new tracks T000245 and T000249: fragmentation, not a transfer.

## FRESH_02 T000010 (players 26 and 1)

- The track stays on player 26: foot offset ≤ 0.09 h, IoU 0.53–0.61 at every labelled frame from 294 to 348.
- At 318 the players' mutual IoU is 0.65, and the track's IoU with player 1 (0.58) exceeds its IoU with 26 (0.56)
  for that one labelled frame. That one frame produces the 26→1→26 pair.

## FRESH_02 T000014 (players 10 and 8): real transfer

The players cross between frames 166 and 174.

| frame | what happened |
|---|---|
| 169–170 | The only high-score box on the pair is a multi-body box (d10, flagged). T000014 matched it (contested). |
| 171 | One box for both players (d10, IoU 0.65 with 10, 0.66 with 8). T000014 (cost 0.13) and T000007 (player 8) both pass every gate for it. T000014 wins, contested with rival T000007. T000007 took a low-score box. |
| 172 | T000014 matched d12 (multi-body, IoU 0.77 with player 8), contested. |
| 173–174 | T000014 is on player 8 (IoU 0.91 / 0.81, foot −0.03 h). From 180 it is PREDICTED, then lost. |

- **Appearance evidence: none.** The descriptor is None in every frame from 166 to 174, because every crop
  overlaps another detection (by design, contaminated crops are not described).
- **Motion evidence: ambiguous.** Both tracks pass centre, Mahalanobis and motion-consistency for the same boxes.
- **Ambiguity flags: raised, then dropped.** `contested` was raised at 167, 169, 170, 171 and 172, but it is a
  per-frame output only. The track carries no persistent ambiguity state: `possible_predecessors` and
  `recoveries` are empty. So the identity stays "clean" after a crossing it could not resolve, and the benchmark
  counts it as unflagged.
