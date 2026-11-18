# TRAIN_M002 cluster-review audit: why it disagrees (2026-11-18), analysis only

Scope:
- Nathan's review state (saved from the page) for the TRAIN_M002 cluster review: 224 clusters, 18 audit tracklets.
- Labels, clusters and the audit limit are unchanged.
- Tool: `tools/dev/reid_audit_diagnose.py`.
- M002 is a night match (floodlit), so every tracklet below is night.
- Contact sheets stay on Nathan's Mac (`HASHES.txt`).

## The 18 audit answers
- 11 reject, 2 unsure, 5 players.
- 9 audit tracklets are alone in their cluster, so their cluster gives no label to compare: 7 rejects and 2
  players (B7, A15).

## Every audit tracklet where the answer differs from the cluster

| tracklet | Nathan's answer | cluster label as counted | cluster after edits (live cluster-mates now) | cluster before edits (original cluster-mates) | category | length | median box height | appearance change inside (percentile) |
|---|---|---|---|---|---|---|---|---|
| T004652-1 | A24 | A22 | A22 | tie A22 / A24 | wrong cluster (same team) | 12.9 s | 101 px | 54 |
| T000018-1 | B14 | unsure | unsure | unsure | other: cluster marked unsure | 20.7 s | 76 px | 56 |
| T001545-1 | A42 | unsure | unsure | unsure | other: cluster marked unsure | 9.5 s | 135 px | 85 |
| T008476-1 | unsure | A42 | A42 | A42 | my answer was unsure | 10.6 s | 115 px | 64 |

- **T004652-1, a real error:**
  - The proposed cluster C026 mixed two red team-mates: one #24 tracklet, one #22 tracklet and this audit tracklet,
    which shows #24.
  - Nathan split the #24 tracklet out and labelled the rest A22.
  - The audit tracklet was hidden from the sheet, so it stayed behind with A22.
  - The clustering was wrong; Nathan's answer is right.
- **T000018-1 / T001545-1, data loss rather than wrong labels:**
  - Nathan marked these clusters unsure, from their 3-thumbnail sheets.
  - In the audit (8 thumbnails) the numbers were readable (#14, #42), and the cluster-mates look like the same
    players.
- **T008476-1:** the cluster says A42, Nathan was unsure. The tool does not count it.
- **No tracklet** shows an appearance change typical of a player switch (all below the 95th percentile of
  within-tracklet change).
- **No name mismatch.**
- **No out-of-date cluster label in this data:** the label counted equals the after-edits label for every audit
  tracklet.

## Rates

| | disagreements / counted | rate | Wilson 95 % CI |
|---|---|---|---|
| (a) real conflicts: player vs player, against the after-edits label | 1 / 1 | 100 % | 21–100 % |
| (b) as the tool counts | 3 / 7 | 43 % | 16–75 % |

The limit is 5 %.

## Bugs in the audit, by the agreed rule (not fixed yet)
1. **(b) counts unsure / reject as disagreements.** 2 of its 3 disagreements are clusters Nathan marked unsure against
   a player answer. Only unsure answers on the audit side are excluded.
2. **(b) can count an out-of-date cluster decision.**
   - A split or move leaves hidden audit tracklets in the original cluster under the decision it had before the
     edit.
   - That holds even when no visible member remains: the cluster disappears from the page, but its old label is still
     compared.
   - This did not occur in this state, but the code allows it (shown in `tests/test_reid_audit_diagnose.py`).
3. **The audit cannot measure anything at this size and design.**
   - Only 1 of 18 audit tracklets yields a player-vs-player comparison: 9 are singletons, 11 are rejects.
   - With n = 1, one error is 100 % and the 5 % limit can never be passed with confidence.
   - Even with zero errors, about 60 comparable items would be needed before the upper 95 % bound falls under 5 %.
4. **Hidden audit tracklets cannot follow a split.** When a cluster mixes two players, the reviewer's correct split
   leaves the audit tracklet on whichever side keeps the cluster id (T004652-1).

## Proposed fix (for approval)
1. **What counts:**
   - A disagreement is a player-vs-player mismatch only, compared with the label of the tracklet's live cluster-mates
     now, not with a stored cluster decision.
   - Reject / unsure mismatches are reported separately, as lost data rather than wrong labels.
   - A cluster with no live non-audit members gives no opinion.
2. **Stale decisions:** a split or move clears the stored decision of the clusters it changes; the page asks to
   confirm them again.
3. **When the audit is drawn:**
   - Draw it after the review, from tracklets whose reviewed cluster says a player and that have at least one live
     cluster-mate, shown blind.
   - This measures the error of the labels that actually get frozen.
   - Nothing is hidden during the review, so splits can place every tracklet.
   - Trade-off: the reviewer may remember a tracklet from the cluster sheets.
4. **Sample size and gate:**
   - Size the sample for the limit: draw until at least `audit.min_comparable` comparable answers exist.
   - Gate on the Wilson upper bound, not the point rate.
   - Both are new config values; they need Nathan's choice.
5. **Same logic in the freeze check** (`tools/reid_cluster.py`), with tests for each case above.

## Other observation
11 of the 18 randomly sampled long tracklets were rejected by Nathan, so a large share of the 8 s pack is not a
single on-pitch player. Not investigated further here.

## Fix applied (2026-11-18, Nathan: "apply the audit fix, minimum 60 comparable answers"), commit 4791913
1. **What counts:** disagreement = player vs player only. Reject / unsure answers are reported as lost data and are
   not counted.
2. **When the audit is drawn:** after the clusters are locked.
   - It draws from tracklets of player clusters that still have another live member, in a fixed order (FNV-1a of
     `<seed>:<tracklet>`, identical in the page and the freeze check).
   - It stops at the 60th answer that names a player. The freeze requires exactly that shortest prefix.
   - Nothing is hidden during the review any more.
3. **No stale decisions:** a split or move clears the decision of the clusters it changes.
4. **Gate:** the exact one-sided (Clopper-Pearson) 95 % upper bound must be ≤ 0.05.
   - With 60 comparable answers, 0 errors gives 4.87 % (passes) and 1 error gives 7.66 % (fails).
   - The two-sided Wilson bound used in the previous display gives 6.02 % even at 0 errors in 60 and needs 73. It
     could never pass at 60, which is why the gate uses the exact bound.
5. **The 1.0.0 audit format is refused at freeze.**
6. **TRAIN_M002 review folder upgraded in place** (`tools.reid_cluster upgrade`).
   - Clusters, tracklets and exclusions are byte-identical; the old files are kept as `*.1.0.0.bak`.
   - New review.json sha `a1ed4f75072ae92e0b242a04b00c33cb27c993873e24e961526c6448458d5290`.
   - The page carries Nathan's saved review over (decisions kept, old audit dropped).
   - The 18 tracklets hidden before are marked "not seen", with the earlier audit answer.
7. **Feasibility:** in Nathan's saved state, 79 tracklets are eligible (player clusters with a live mate) for 60
   comparable answers. Feasible, but tight if many answers are reject / unsure.

## Post-fix audit on TRAIN_M002: 4 of 60 disagree (freeze refused, correctly)
- Inputs (Nathan's Mac):
  - state `Downloads/TRAIN_M002_cluster_review_state_v2.json`, sha `0f24700d…2fb3`;
  - diagnosis `spana-audit-diagnose3/out/diagnosis.json`, sha `0e146eac…8044`;
  - code 5d22b7a.
- **Audit:**
  - 86 tracklets could be drawn; 66 were drawn: 60 comparable plus 6 unsure (lost data, 4 of them in A6 clusters).
  - 4 disagreements: 6.7 %, Wilson 2.6–15.9 %, exact one-sided 95 % bound about 14.7 % (limit 5 %).
- **The 4** (all red, team A, night; boxes 108–140 px; no tracklet switches player partway):

| tracklet | Nathan | cluster | what the sheets show |
|---|---|---|---|
| T000696-1 | A42 | C003 = A5 | #42 readable in the audit; the 4 cluster-mates show no number (side / back views) |
| T006553-1 | A5 | C017 = A14 | #5 readable; its 3 cluster-mates clearly show #14: a mixed cluster |
| T004400-1 | A22 | C052 = B22 | #22 in the red kit, so team A; the cluster was given the team-B key B22 |
| T005798-1 | A14 | C052 = B22 | #14 in the red kit: C052 mixed two team-A players and was labelled with a team-B key |

- **Pattern:**
  1. The clustering merges same-kit team-mates (3 of 4).
  2. The reviewer saw only 3 thumbnails per tracklet in the cluster sheet (`review.sheet_thumbs_per_tracklet`), where
     the number was not readable. The 8 audit thumbnails showed it.
  3. One player-key slip: B22 for a red #22, while A22 exists.
- **Remaining risk:** 20 drawable tracklets are unaudited. These are player labels propagated by a cluster and never
  seen alone. 21 player clusters are singletons, decided one tracklet at a time.
