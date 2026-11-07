# Frozen matching rule NESTED_V1 (frozen 2026-11-08, approved by Nathan: "Option A")

Decides when a detection or observed-track box counts as the same body as a human ground-truth box.
Implemented in `spana/evaluation.py` (`NESTED_V1`, `eligible_matrix`); the default of `tools/benchmark.py`.

A pair (label G, detection D) is eligible if **IoU(G, D) ≥ 0.50**, OR all of:
1. the centre of D lies inside G **and** the centre of G lies inside D;
2. one box lies inside the other by **≥ 0.70** of its own area;
3. height ratio (smaller / larger) **≥ 0.60**;
4. width ratio (smaller / larger) **≥ 0.40** (widths vary most between box styles: arms, stride, margins).

Eligible pairs are assigned one-to-one (Hungarian), preferring the larger IoU.

Why it does not weaken correctness:
- It is a strict superset of the standard IoU 0.5 rule, so no standard match is lost.
- The extra pairs must be the same body drawn tighter or looser. A neighbour's box fails condition 1; a partial
  body (half or legs) fails condition 3; a label left behind by a moving player fails condition 1; a two-player
  union box fails conditions 1–2 for each player. Tests: `tests/test_matching_rule.py`.
- Duplicate detections and duplicate tracks keep their definitions (eligible under the rule, or nested by IoA
  ≥ 0.6) and are still reported.

Honesty note: the rule was written after the FRESH_02 v1 results at IoU 0.5 / 0.3 had been seen (box-style
mismatch diagnosis). Its thresholds come from box-geometry reasoning, not from fitting FRESH_02 numbers. It is
frozen from this commit and applied unchanged to every later clip, including the unseen DEV clips
(FRESH_05/07/08). Any change = NESTED_V2 with a reason; earlier results are kept.
