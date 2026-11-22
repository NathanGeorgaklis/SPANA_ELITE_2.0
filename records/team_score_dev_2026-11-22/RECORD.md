# Team classification v1 scored on DEV (+ TRAIN sanity check), 2026-11-22

CODE TASK (Nathan), evaluation. Branch `evaluation/team-score-v1` (on `evaluation/team-labels-dev`, on
`team/classification-v1`; none merged to main).
- **What:** `spana.teams` 1.0.0 with the `config/default.yaml` `team` priors (2.17.0), unchanged.
- **DEV:** the frozen side labels `data/labels/<clip>/teams_v1`.
- **TRAIN check:** the labelled tracklets of TRAIN_M001-M003.
- No EVAL. Nothing adopted.

**Result: v1 FAILS the acceptance rule on DEV.**
- Pooled: 16 wrong of 2,194 answered frames, bound 1.11 % > 0.5 %.
- There are 16 GOALKEEPER answers on a team player (one track, FRESH_08 T000049).
- TEAM_A / TEAM_B were never swapped on any DEV frame.

## Method
- **Run** (`run_dev.sh`):
  - `spana.cli teams` per match: G1 = FRESH_02, G2 = FRESH_05, G3 = FRESH_07 + FRESH_08 in one model.
  - Tracks: the default tracker run (`~/spana-player-confirmation-run/base`; HASHES.txt).
- **Scored** by `tools/team_score.py dev`:
  - **Unit:** a labelled frame x an observed track that NESTED_V2 attributes to one GT id unambiguously. Two-player
    boxes are excluded and counted.
  - **Truth:** A / B map to TEAM_A / TEAM_B by the per-match permutation with the most agreement. It is a naming
    choice, reported with both counts, and it was never close (G3: 1,050 vs 0). GK_A / GK_B -> GOALKEEPER, REF ->
    REFEREE.
  - **Answered** = any label but UNKNOWN. **Wrong** = answered and not the truth.
- **Leave one DEV match out:**
  - Every `team` value is a prior, fixed before the first score and never fitted on any DEV clip.
  - So each match's result is already held out, and the pooled number is the leave-one-match-out result.
  - Nothing was tuned in this task.
- **Deterministic:** rescoring the stored G3 artifact gives a byte-identical score file.

## 1. DEV: errors and acceptance

| match | answered | wrong | exact one-sided 95 % bound | role answers on team players |
|---|---|---|---|---|
| G1 (FRESH_02) | 682 | 0 | 0.44 % | 0 |
| G2 (FRESH_05) | 400 | 0 | 0.75 % (n too small for <= 0.5 % even at 0 wrong) | 0 |
| G3 (FRESH_07 + 08) | 1,112 | 16 | 2.18 % | 16 frames, 1 id (FRESH_08 id 18) |
| **pooled (held out)** | **2,194** | **16** | **1.11 % — FAIL** | **16 — FAIL** |
| pooled without that one track (not a result, for scale only) | 2,178 | 0 | 0.14 % | 0 |

- **The single error source** is FRESH_08 T000049 = GT id 18, a red (B) player. The segment is labelled GOALKEEPER
  on 16 labelled frames.
  - Its colour is 0.483 from TEAM_B, just past `team.outlier_distance` 0.45, so it counts as a colour outlier.
  - The detector called it "goalkeeper" on 70 % of its rows (`role_min_share` 0.60).
  - The rule did what it was designed to do. The design let an off-colour team player become a role when the
    detector agrees.
- **Margin of the true roles.** The true role segments on DEV sit at 0.64-0.93 from the nearest team centre, with
  hint shares of 0.87-1.00:
  - FRESH_02 referee: 0.92 / 0.93;
  - FRESH_05 keeper: 0.64;
  - FRESH_08 keeper: 0.90 / 0.92.
  - The wrong one sits at 0.48 / 0.70.
- **No team swap anywhere.** Every team answer on DEV was the right team (A->TEAM_B 0, B->TEAM_A 0).
- **TEAM_CHANGE flags:** none on DEV.

## 2. DEV coverage per clip (for Nathan's review; not gated)

| clip | units | answered | coverage | of GT boxes | team players | answers on tracks without a GT id | why UNKNOWN (frames) |
|---|---|---|---|---|---|---|---|
| FRESH_02 | 941 | 682 | 0.725 | 0.670 | 0.711 | 5 | outside segment 207, colour outlier no role 34, too few clean 18 |
| FRESH_05 | 447 | 400 | 0.895 | 0.889 | 0.889 | 1 | colour outlier no role 29, outside segment 13, too few clean 5 |
| FRESH_07 | 1,151 | 791 | 0.687 | 0.653 | 0.763 | 16 | outside segment 184, colour outlier no role 169 (110 of them the 2 referees), too few clean 7 |
| FRESH_08 | 697 | 321 | 0.461 | 0.409 | 0.449 | 2 | outside segment 280, too few clean 88, colour outlier no role 8 |

- **"Outside segment"** is the largest cause: the frame is in an uncertainty episode, or the track is not yet
  confirmed. It follows the tracker's contact losses (`records/contact_continuity_diag_2026-11-22`). FRESH_08 is
  the crowded attack into the box.
- **Answers on tracks with no GT id** (spectators, bench, unlabelled people) are not scorable: 24 frames in all.

## 3. Roles (counts, 7 ids)

| clip | id | label | answered (labelled frames) | why not answered |
|---|---|---|---|---|
| FRESH_02 | 26 | REF | REFEREE 56, UNKNOWN 5 | outside segment 5 |
| FRESH_05 | 11 | GK_A | GOALKEEPER 24 | |
| FRESH_07 | 23 | REF | UNKNOWN 66 | colour outlier, no role hint 65 |
| FRESH_07 | 24 | REF | UNKNOWN 48 | colour outlier, no role hint 45 |
| FRESH_08 | 20 | GK_B | GOALKEEPER 46, UNKNOWN 2 | outside segment 2 |
| FRESH_08 | 25 | REF | UNKNOWN 36 | outside segment 30, too few clean 6 |
| FRESH_08 | 26 | REF | (no attributed frame) | |

- **3 of 7 role ids answered right, 0 role ids answered wrong.**
- **The wrong role answer is on a team player** (section 1).
- **In G3 the referees' black kit is a clean colour outlier,** but the detector rarely says "referee" there (2-4 %
  of person boxes), so they stay UNKNOWN. That is fail closed, a coverage loss, not an error.

## 4. TRAIN sanity check (labelled tracklets of TRAIN_M001-M003; `run_train.sh`)
- The tracks the re-ID labels were made on, sha256-checked against each label FROZEN.json.
- 10-minute windows, not full matches.
- **Team sides only:** the TRAIN detections are COCO person-only (YOLOX, `spana.detections` 1.3.0), so there is no
  role hint and REFEREE / GOALKEEPER cannot be answered.
- Same `team` priors.

| match | separation | units | answered | wrong | bound | wrong frames by player |
|---|---|---|---|---|---|---|
| TRAIN_M001 | **0.44** | 40,076 | 11,861 | **686** | 6.15 % | A5 307, A18 112, A6 86, A11 62, A8 55, A3 45, A17 18, B43 1 |
| TRAIN_M002 | 0.93 | 24,006 | 19,683 | 0 | 0.015 % | |
| TRAIN_M003 | 0.53 | 15,396 | 10,195 | 0 | 0.029 % | |

- **M001 is a weak-separation match.**
  - The two kit centres are 0.44 apart, no farther than each cluster's own radius (0.44-0.45), so the clusters
    overlap.
  - Some A segments sit nearer the B centre, and their frames vote B decisively.
  - The model-level guard (`team.min_cluster_separation` 0.30) let it through.
- **DEV cannot show this failure.** Every DEV match separates at 0.80-0.98.
- **TRAIN can show it but cannot tune it.** TRAIN is for training, and thresholds are fitted on DEV only.

## 5. Options for Nathan (nothing built, nothing adopted)
1. **Keep the role rule out of team players** (the DEV failure).
   - (a) A role needs to be farther from both teams than a new `team.role_min_distance` (> `outlier_distance`). An
     off-colour team player then stays UNKNOWN.
   - (b) Or a role needs a stronger detector majority (`role_min_share` up).
   - Either is a threshold chosen after seeing DEV. To be judged leave one match out, it must be fitted on two
     matches and scored on the third. G3 is the only match with the failure, so fitting on G1 + G2 must already
     put the value above 0.48.
     - The G1 + G2 true roles are at 0.64 and 0.92. Any value between 0.48 and 0.64 removes the error and keeps
       every DEV role.
     - That is still only 7 role ids.
2. **Make the match model abstain when the clusters overlap** (the TRAIN M001 failure): for example, require
   `separation` to exceed the larger cluster radius by a margin, as a ratio rather than the absolute 0.30.
   - DEV has no case to measure it on (it would change no DEV answer).
   - TRAIN shows the need but cannot set the value.
   - The choice of rule and value is Nathan's.
3. **Roles off in v1:** answer only TEAM_A / TEAM_B / UNKNOWN until pitch context exists.
   - On DEV that removes all 16 wrong frames: pooled 0 / 2,052, bound 0.15 %.
   - It loses the 3 correct role ids (126 labelled frames).
4. **Coverage** (FRESH_08 0.46) is driven by tracking: uncertainty episodes and unconfirmed tracks. No team-layer
   change is proposed for it.

## Files
- `run_dev.sh`, `run_train.sh`: the runs. Outputs go to `~/spana-team-score-run/`, outside the repo.
- Scores: `DEV_G1.score.json`, `DEV_G2.score.json`, `DEV_G3.score.json`, `TRAIN_M001.score.json`,
  `TRAIN_M002.score.json`, `TRAIN_M003.score.json`.
- `HASHES.txt`: the spana.teams documents and tracks scored.
