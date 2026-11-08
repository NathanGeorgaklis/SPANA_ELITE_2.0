# Partial ↔ full body failures on DEV FRESH_02: trace (2026-11-08)

- Scope: DEV only. EVAL was not used.
- No code changed.
- Runs traced: `F02M.det.json` and `F02C.trk.json` (see `records/camera_uncertainty_2026-11-08`).
- Tools:
  - `tools/dev/partial_body_trace.py` gives a per-frame trace of the old track's last 12 frames: every detection on the player with its score, size and multi-body flag, and its owner (match stage / spawn / refusal).
  - `tools/dev/nested_box_pairs.py` gives part/whole pair statistics against the labels.

## Per case (FRESH_02 Mahalanobis-gate track changes with a box-extent jump)

| case | track state | what happened | mechanism |
|---|---|---|---|
| f189 P10 T000134 | unconfirmed (born f186) | f188: S2 gave it the low-score partial d32 (s0.36, h60). Its own full box d17 (s0.60, h90) came later in S3 and was refused as INSIDE_OBSERVED_BOX. f190: the full box spawned T000136. | M1 |
| f221 P17 T000150 | unconfirmed (born f216) | f217: S2 gave it the low-score partial d30 (s0.45, h69). The full box d18 (s0.64, h112) then spawned duplicate T000151. Same pattern at f219, spawning T000153. | M1 |
| f381 P11 T000207 | unconfirmed (born f378) | f379: S2 gave it the low-score partial d140 (s0.12, h52). The full box d17 (s0.73, h84) spawned duplicate T000208. | M1 |
| f280 P12 T000177 | confirmed | It followed a confident upper-body box (s0.60–0.75, h66–72) for ≥ 12 frames. The full body (h108–114) existed only at low score (0.15–0.48), so S1 could not use it. f277: the full box reached s0.65 and spawned T000184/T000185 because it *encloses* the track box (IoA 1.00); the duplicate rule only tests the opposite direction. f280: the partial disappeared and the track was lost. | M2 |
| f219 P9 T000145 | confirmed | Born from an upper-body box (h76). The full boxes nearby were merged with a neighbour and correctly flagged multi-body. f218: a clean full box d17 (s0.72, h91) enclosed the track box (IoA 1.00) and spawned T000152. f219: the partial disappeared. | M2 |
| f241 P15 T000009 | confirmed | It followed upper-body boxes (h45–72, mostly low score, via S2) while T000130, a different track (not the successor), held a full box overlapping the same column. Two players are overlapping in depth here. | ambiguous occlusion, not a pure part/whole error |
| f75 / f79 P4 | unconfirmed | A head box of a player behind; spurious initial velocity (3.5 h/s). | not partial-body |

- **M1 — association order.** The low-score round (S2, IoU only) runs for every continuing track *including
  unconfirmed ones*, before the unconfirmed tracks' high-score round (S3). A new track therefore takes a low-score
  partial box ahead of its own high-score full box, and the full box then spawns a duplicate. In ByteTrack-style
  association the low-score round is for confirmed tracks only.
- **M2 — one-directional duplicate refusal.** `_duplicate_of` refuses a new track whose box lies inside an observed
  box (IoA(new in observed) ≥ `duplicate_ioa` 0.85). It does not test the reverse: a new box that encloses an
  observed track box. A full-body box therefore starts a second track on a player already tracked through a
  partial box.

## Frequency (all frames)

| | FRESH_02 | FRESH_05 |
|---|---|---|
| S2 matches made by unconfirmed tracks | 225 | 36 |
| … where a spawn in the same frame enclosed the matched box (≥ 0.85) | 11 | 10 |
| spawns whose box encloses an observed track box (≥ 0.85) | 25 of 213 | 10 of 63 |
| … same column (width ratio ≥ 0.8, ≥ 1.2× taller) | 16 | 3 |

**Nested part/whole detection pairs** (IoA ≥ 0.85, width ratio ≥ 0.8, height ratio ≥ 1.2):
- On labelled players, the part is the same person in FRESH_02 25 of 28 cases and in FRESH_05 25 of 26 cases.
- In the other 3 and 1 cases it is a different, real person.
- So the pair is mostly a duplicate but not always one. Deleting part boxes at detection would sometimes remove a real player.
- The part outscores the whole in about 24 % of pairs on FRESH_02 and 22 % on FRESH_05.
- Most FRESH_02 pairs (504 of 544) are on unlabelled people (crowd / ignore regions).
