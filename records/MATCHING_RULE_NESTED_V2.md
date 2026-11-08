# Frozen matching rule NESTED_V2 (frozen 2026-11-08, requested by Nathan)

- Implemented in `spana/evaluation.py` as `NESTED_V2` (`_match`, `_foot_distance`).
- Selected with `tools/benchmark.py --rule NESTED_V2`.
- NESTED_V1 is unchanged and stays the benchmark default for comparison. Its re-scored results are identical to
  the earlier reports.

## Rule

- **Eligibility.** Exactly NESTED_V1: IoU ≥ 0.50, or the same body drawn in another box style. See
  `MATCHING_RULE_NESTED_V1.md`.
- **Overlap group.** For one track or detection box, this is the set of eligible labelled players whose label boxes
  overlap another eligible player's box with **IoU ≥ 0.30**.
- If the group has two or more players:
  1. **Any of them labelled occluded (visibility 0.5): AMBIGUOUS.** An occluded player's label is drawn to
     estimated feet that are hidden, so its foot point is not evidence. The one-to-one assignment keeps the
     NESTED_V1 IoU preference.
  2. **Otherwise, by foot point.** The pair score uses the foot-point distance (bottom-centre, in units of each
     label's height) instead of IoU: score = 1 + max(0, 1 − distance). If the two nearest players' distances
     differ by **< 0.10 label heights**, the box is AMBIGUOUS.
- **Effect of an ambiguous track match:** it counts as coverage but makes no identity update. It does not count as
  a transfer, an ID switch or a fragment, and it does not set the track's or the player's last identity. The next
  clear attribution is compared with the last clear one, so a real swap through an ambiguous stretch is still
  counted. It is reported as an `ambiguous_attributions` event: frame, track, assigned player, last clear player.
- **Detections** use the same assignment. Their ambiguity only matters for which player is counted missed.

## Why these values

- **Mutual IoU 0.30** was given in the request.
- **Margin 0.10 label heights** is about twice the tracker's box-measurement noise (`kalman.pos_std_h` 0.05 h).
  Smaller foot differences cannot separate two players.
- **The occluded-label exception** comes from an observed case: FRESH_02 frames 378–390, track T000008; see
  `records/nested_v2_rescore_2026-11-08/FRESH_02_T000008_f378-390.png`. There, deciding by the occluded
  label's estimated foot attributed a correctly tracked player to the player behind him.

## Honesty note

The rule was written after the FRESH_02 and FRESH_07 tracking results had been seen: the unflagged-transfer
diagnosis in `records/unflagged_transfers_2026-11-08`. The occluded-label exception was added after a first
version scored FRESH_02. The thresholds come from geometry and measurement noise, not from fitting transfer
counts. Applied unchanged from this commit on. Any change is NESTED_V3, with earlier results kept.

Tests: `tests/test_matching_rule_v2.py`.
