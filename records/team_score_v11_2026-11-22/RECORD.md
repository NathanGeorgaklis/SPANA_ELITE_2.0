# Team classification v1.1 (roles off + cluster-overlap abstention) rescored on DEV + TRAIN, 2026-11-22

**What changed (Nathan's decision 2026-11-22 on `records/team_score_dev_2026-11-22`: option 3 + option 2):**
- `team.roles_enabled: false` (new, default off). Colour outliers (referees, keepers, anything far from both kits)
  answer UNKNOWN / COLOUR_OUTLIER and are never put in the nearest team.
- `team.min_separation_radius_ratio: 1.0` (new, a stated prior, not fitted). The match model abstains with
  CLUSTERS_OVERLAP unless the two kit centres are farther apart than the larger cluster radius, i.e. unless the
  other team's centre lies outside each cluster's radius (max member distance).
  - DISCLOSED: the M001 and DEV numbers were known when 1.0 was stated. 1.0 is the geometric "centre outside the
    radius" point, not a value searched for.
  - TRAIN_M003 passes at 1.19, closest to the line.
- Option 1 rejected (7 role ids are too few to fit `role_min_distance`).
- Versions: config 2.18.0, `spana.teams` 1.1.0. Nothing else changed: same tracks and labels as
  `records/team_score_dev_2026-11-22`, same scorer, same runs (`../team_score_dev_2026-11-22/run_dev.sh`,
  `run_train.sh` with `OUT=~/spana-team-score-run-v11`).

## DEV (held out: every value a prior, never fitted on a DEV clip)

| match | answered | wrong | bound | role answers on team players |
|---|---|---|---|---|
| G1 FRESH_02 | 626 | 0 | 0.48 % | 0 |
| G2 FRESH_05 | 376 | 0 | 0.79 % | 0 |
| G3 FRESH_07 + 08 | 1,050 | 0 | 0.29 % | 0 |
| **pooled** | **2,052** | **0** | **0.146 % — PASS (<= 0.5 %)** | **0 — PASS** |

- **Overlap rule (option 2) changes no DEV answer.** All three models are FITTED with unchanged separations
  (0.80 / 0.86 / 0.98, ratio 1.83 / 2.28 / 2.20).
- **The only DEV changes are option 3.** 22 segments changed:
  - every former REFEREE / GOALKEEPER segment, and every COLOUR_OUTLIER_NO_ROLE segment, became UNKNOWN /
    COLOUR_OUTLIER;
  - 0 TEAM_A / TEAM_B answers changed.
- **Roles (counts, 7 ids): no role id is answered now.**
  - FRESH_02 26 REF: UNKNOWN 61
  - FRESH_05 11 GK_A: UNKNOWN 24
  - FRESH_07 23 / 24 REF: UNKNOWN 66 / 48
  - FRESH_08 20 GK_B: UNKNOWN 48
  - FRESH_08 25 REF: UNKNOWN 36
  - FRESH_08 26: no attributed frame
  - Before: 3 right (126 frames), 0 wrong.

| clip | coverage (was) | team-player coverage (was) |
|---|---|---|
| FRESH_02 | 0.665 (0.725) | 0.711 (0.711) |
| FRESH_05 | 0.841 (0.895) | 0.889 (0.889) |
| FRESH_07 | 0.687 (0.687) | 0.763 (0.763) |
| FRESH_08 | 0.372 (0.461) | 0.423 (0.449: the 16 frames of T000049, formerly a wrong GOALKEEPER, are now UNKNOWN) |

## TRAIN sanity check (labelled tracklets of M001-M003, team sides only)

| match | model | separation / larger radius | answered | wrong |
|---|---|---|---|---|
| TRAIN_M001 | **ABSTAIN CLUSTERS_OVERLAP** | 0.4385 / 0.4497 (0.98) | 0 (was 11,861) | 0 (was 686) |
| TRAIN_M002 | FITTED | 0.9257 / 0.4497 (2.06) | 19,683 | 0 |
| TRAIN_M003 | FITTED | 0.534 / 0.4485 (1.19) | 10,195 | 0 |

Acceptance met and M001 abstains. Per Nathan's instruction the three branches are merged to main (DECISIONS
2026-11-22).
