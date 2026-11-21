# Diagnosis: contact losses (a labelled player's track changes), DEV (2026-11-22)

CODE TASK (Nathan), player tracking. Diagnose first, no code change. Branch `tracking/contact-continuity-diag`, base
main `e6a6236`. DEV FRESH_02 / 05 / 07 / 08, the cached local-detector detections. Tracks: the default tracker
(`~/spana-player-confirmation-run/base`; current main reproduces FRESH_05 exactly). No EVAL, no full match.
`config/default.yaml` unchanged.

**Method (`diagnose.py`).** A track change is track A on one labelled frame and track B on the next labelled
frame that attributes the player unambiguously (NESTED_V2).
- **Count:** 103 on DEV. The reviewer's 56 are the subset it had to act on.
- **For each change, taken from the tracker's decision log:**
  - what happened to A between the two frames;
  - for the first frame A went unmatched: the reason, and the first gate that failed for the nearest detection,
    with that frame's evidence (IoU with A's prediction, motion distance, height ratio, colour similarity, score,
    whether another track took the box);
  - B's origin and relation to A.
- **`veto_details.py`** adds the appearance-veto detail.

## 1. Causes (103 track changes)

| what happened to A | first failing gate (nearest detection) | changes | evidence at that frame |
|---|---|---|---|
| unmatched: **appearance veto** | colour similarity < `appearance.min_similarity` 0.70 | **36** | motion passed in all 36 (IoU with prediction median 0.76, 34 >= 0.40); similarity median 0.64 (0.09-0.70); box free in 30; the vetoed box overlaps no other tracked box (35 / 35 <= 0.05) |
| unmatched: TAKEN_BY_OTHER | none (the box passed every gate) | 15 | another track took it (the rival in 10, B in 3, contested in 2); A already IDENTITY_UNCERTAIN in 14 |
| unmatched: REFUSED | match refused: AMBIGUOUS_WITH_HIDDEN_TRACK 9, HIDDEN_TRACK_EXPLAINS_BETTER 1 | 10 | box IoU 0.69-0.89 with A's prediction, but a hidden track explains it equally well: the silent-swap guard |
| unmatched: motion / extent gates | mahalanobis 8, centre 7, height 4 | 19 | fast motion or the box changing extent beyond `max_height_ratio` / extent link; IoU mostly < 0.6 |
| unmatched: low-score box | low_score_iou_ok | 7 | only a low-score box (0.10-0.49) with IoU < `low_score_min_iou` 0.40 |
| unmatched: no detection | NO_DETECTION_NEARBY | 4 | detector miss |
| unmatched: other | PASSING_CANDIDATE_UNUSED | 1 | |
| A kept, but on another player's box | none (A observed) | 11 | A IDENTITY_UNCERTAIN in 10: the flagged transfers / two-player boxes |

- **Relation of B to A:**
  - B is a spawn naming A as a possible predecessor, and also A's rival: 44;
  - B names A as its only predecessor: 14;
  - B is A's rival: 6;
  - B took A's box: 5;
  - **no recorded relation: 34.**
- **State:** A was IDENTITY_UNCERTAIN at the miss in 45 of 92 cases; B was uncertain at the next labelled frame in
  62 of 103.

## 2. The appearance veto (the largest cause)
- **What happens:** in each of the 36 the player's own next box passes every motion gate. Colour vetoes it, and the
  box goes unclaimed in 30. A new track is born on it: on the same frame in 19, and in 29 that track names A as a
  predecessor.
- **The crop is not the problem.** It is clean: no other tracked box overlaps it.
- **The reference is.** A's colour model is stale:
  - frozen since its uncertainty episode began (A uncertain: 17);
  - or not updated during the contact, when no clean crops exist (A CERTAIN: 18).
- **This matches `records/veto_root_cause_2026-11-20`.** There, models were learned from blurred crops in a pan and
  frozen in a crowd. That record also showed that lowering the threshold is unsafe: right vetoes sit at 0.62 and
  0.67, inside this range.
- **The doubt link already targets exactly this** (`tracking.appearance_doubt_link`, built 2026-11-18, off). The
  continuation is flagged IDENTITY_UNCERTAIN (identity_doubt), never certain. Measured here with that one switch on
  (`run_doubt_link.sh`, `doubt_link_measurement.json`; `config/default.yaml` unchanged):

| | default | doubt link on |
|---|---|---|
| track changes (labelled) | 103 | 100 |
| ... caused by the appearance veto | 36 | 28 |
| identity transfers / silent / on linked tracks | 7 / 0 / 0 | 8 / 0 / 0 (new: FRESH_08 T000047, 13 -> 14, on a doubt-linked track, flagged) |
| fragments / ID switches | 101 / 103 | 98 / 100 |
| unaided: correct / wrong / ABSTAIN frames | 0.508 / 0 / 513 | 0.506 / 0 / 584 |
| reviewer (policy baseline): correct / wrong / actions per min | 0.879 / 0 / 6.48 | 0.885 / 0 / 6.55 |

- **Why the doubt link takes only 8 of the 36.** It links only when no other track gates the box, and in contact a
  neighbour's gate covers it.
- **A narrower rule** ("no other unmatched track gates it") could take at most 5 of the remaining 28 (upper bound
  from the logs). The others:
  - 11: another unmatched track also gates the box, so it is genuinely ambiguous and fail-closed refuses;
  - 5: another track took the box;
  - 4: two boxes in the motion gate;
  - 3: IoU < 0.40.

## 3. First failing layer
- **Player tracking: association in contact.** Almost every cause is a fail-closed refusal where the tracker lacks
  decisive evidence:
  - a hidden track explains the box (10);
  - the rival takes it (15);
  - the colour reference is stale (36);
  - motion or extent breaks (19);
  - a low-score box (7).
- Detection misses are 4.
- The 34 changes with no recorded relation are mostly in this set too: the new track is born elsewhere, or later,
  without naming A.

## 4. Smallest generic fix (NOT built; awaiting Nathan)
- **Proposal: turn on the existing doubt link** (`tracking.appearance_doubt_link: true`), measured as above:
  - track changes 103 -> 100, 0 silent, 0 wrong frames, 0 transfers on linked tracks;
  - +1 flagged transfer (on a doubted track);
  - reviewer 0.879 -> 0.885 correct at 6.48 -> 6.55 actions per min;
  - unaided ABSTAIN frames 513 -> 584 (doubted tracks abstain).
- **Possible follow-up:** narrow its exclusivity to unmatched tracks, at most 5 more changes. That would be a code
  change in `_doubt_links` with its own measurement.
- **Recommendation:** the gain is small. The measured success criterion ("fewer track changes") moves by 3 of 103,
  and reviewer effort does not fall.
  - Nothing generic in association removes the contact losses without the evidence the tracker lacks: a reference
    that stays current through contact, which same-kit re-ID does not give (records/reid_v5_dev_eval_2026-11-21).
  - My recommendation is to NOT spend a tracking task on this now. Turn the doubt link on only if Nathan values the
    +0.006 reviewer correctness over the extra flagged transfer.
- **Not proposed:**
  - lowering `min_similarity`: unsafe, see `records/veto_root_cause_2026-11-20`;
  - relaxing the hidden-track refusal: that is the silent-swap guard;
  - low-score recovery: measured 2026-11-18, off;
  - wider motion gates.
