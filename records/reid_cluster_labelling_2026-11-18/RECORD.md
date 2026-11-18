# Group-then-confirm re-ID labelling for TRAIN matches (2026-11-18)

Scope:
- Replaces labelling the TRAIN_M002 8 s pack one tracklet at a time (414 tracklets) with:
  - clustering;
  - a review of whole clusters;
  - a blind audit of a random sample.
- TRAIN only: DEV and EVAL packs are refused, and no DEV / EVAL data is used.
- Labelling tooling only; the product is unchanged.
- Nothing is match-specific: every parameter is in `config/reid_cluster.yaml`.

## Flow
1. **Build:** `tools/reid_cluster.py build` (needs only the pack folder and the re-ID ONNX).
   - **Features per tracklet**, computed from the pack's own thumbnails, so no video is needed:
     - the mean spana re-ID embedding (pilot v1, `682d0a09…`);
     - the mean torso colour histogram.
   - **Distance:** 0.7 × (1 − cosine) + 0.3 × (1 − Bhattacharyya).
   - **Clustering:** average linkage, stopping above `clustering.max_distance`.
   - **Same-frame rule:** two tracklets observed on a common frame can never share a cluster (cannot-link).
   - **Excluded:** tracklets the tracker flagged (identity contest) get the status `excluded_tracker_uncertain`.
     They are decided as unsure, not trained on, not clustered and not audited.
   - **Odd ones out first:** within a cluster, tracklets are ordered by their mean distance to the other members.
   - **Audit sample:** a seeded random 5 % of the clustered tracklets (at least 10).
2. **Review page:** `tools/reid_cluster_review.html`.
   - One contact sheet per cluster.
   - Cluster actions: assign the whole cluster to a player, reject, unsure, split ticked tracklets.
   - Tracklet actions: reject, unsure, move to another cluster.
   - **CONFLICT:** a cluster holding two tracklets on the same frame, or a player in two places, is shown as CONFLICT
     and blocks export.
   - **Audit tab:** the sampled tracklets alone, without their cluster; they are also hidden from the cluster sheets.
     - Disagreement = the audit answer differs from what the reviewed clusters imply.
     - Not counted in the rate: unsure answers, and tracklets whose cluster gives no opinion (every other member is
       also audited, e.g. a singleton).
     - For audited tracklets, the human answer is what gets frozen.
     - The page refuses to export above the limit.
3. **Export:**
   - The standard `spana.reid_label_export` 1.0.0, named `<match>_reid_v1.json`, with one part per tracklet.
   - Plus `labelling_flow: cluster_confirm`, a `cluster_review` block, and `status` on excluded parts.
4. **Freeze:** `tools/freeze_reid_labels.py`.
   - All existing checks still apply.
   - For `cluster_confirm` exports, `check_cluster_export` recomputes from the pack and the export:
     - the config hash;
     - exclusions = exactly the tracker-flagged tracklets;
     - each tracklet in exactly one cluster, and no same-frame conflict;
     - audit tracklets = exactly the seeded sample;
     - every exported decision follows from the review;
     - the disagreement rate with its Wilson 95 % CI.
   - It refuses above `audit.max_disagreement_rate` (0.05).
   - The audit statistics are written into FROZEN.json.
   - Frozen files have the same layout as M001's, so crops, training and the DEV comparisons need no change.

## Choosing max_distance (TRAIN only)

Sweep on long tracklets (≥ 120 observed frames), excluding tracker-flagged ones:

| weight colour | max distance | M001 clusters (156 tracklets) | M001 errors / tracklets with a clean label | M002 clusters (355) | M002 singletons |
|---|---|---|---|---|---|
| 0.3 | 0.10 | 127 | 0 / 37 | 279 | 224 |
| 0.3 | 0.15 | 109 | 1 / 37 | 224 | 155 |
| 0.3 | 0.20 | 98 | 1 / 37 | 183 | 106 |
| 0.3 | 0.30 | 84 | 1 / 37 | 130 | 51 |
| 0.3 | 0.35 | 76 | 1 / 37 | 105 | 27 |

Colour weights 0 and 0.6 give the same M001 error.

- **M001 cannot choose the threshold.** Only 37 of its long tracklets carry a single whole-tracklet player label.
  The pilot model was also trained on M001.
- **Contact sheets of the 8 largest M002 clusters** (judged by eye, labels not used):
  - **0.35:** clusters group by kit, not by person. Dark kit #6 / #2 / #14 shared one cluster; red #14 / #5 another.
  - **0.20:** mostly one person per cluster, but one red cluster mixes #5 and #14.
  - **0.15:** the largest clusters are each one person (#2, #3, a referee…).
- **Chosen: 0.15** (`config/reid_cluster.yaml`).
  - An impure cluster fails the strict 5 % audit and costs the labeller a rework.
  - The price is many small clusters.
- **Same finding as the DEV tests:** this pilot model separates kits and teams, not same-kit team-mates, on a match it
  has not seen (`records/reid_pilot_v1_2026-11-18`, `records/reid_fusion_dev_2026-11-18`).

## TRAIN_M002 proposal (to review)
- 414 tracklets: 59 excluded (tracker-flagged), 355 clustered into **224 clusters**.
  - Cluster sizes: 155 of 1, 39 of 2, 12 of 3, 7 of 4, 9 of 5, 1 of 6, 1 of 7.
- 18 audit tracklets.
- Decisions needed: about 206 clusters plus 18 audit answers, instead of 414 tracklets.
  - Fewer cluster decisions if some clusters hold only audit tracklets: those give no cluster opinion and need no
    decision.
  - **About half the work**, not the one-cluster-per-player ideal. That needs a re-ID model that separates
    team-mates, which more labelled matches should bring.
- **Audit result: pending.** Nathan's audit answers come with the export. The disagreement rate and its CI will be
  added here at freeze time (they are also in FROZEN.json).

## Tests
- `tests/test_reid_cluster.py` covers:
  - the same-frame rule beats identical features;
  - the threshold and odd-ones-out order;
  - a deterministic audit sample and the Wilson CI;
  - thumbnail → box mapping;
  - build (exclusions, no same-frame cluster, TRAIN only, never overwrites);
  - the freeze gate: accepts a correct review; refuses above the limit, an incomplete audit, a CONFLICT, a wrong
    status, another config, and decisions that don't follow the review;
  - manual exports unchanged;
  - evaluate against frozen labels;
  - the review page end to end (Node Playwright): assign, CONFLICT shown and blocking export, audit, export →
    freeze.

## Files
- `reid_cluster.config_snapshot.yaml`: the config used.
- `HASHES.txt`.
