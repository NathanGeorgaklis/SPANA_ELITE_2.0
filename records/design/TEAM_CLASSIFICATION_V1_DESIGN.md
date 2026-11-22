# Team classification v1: design (2026-11-22, NOT built; awaiting Nathan's go-ahead)

CODE TASK (Nathan), team classification. Roles: Lead -> Team Classification -> QA. Design only, no code. Branch
`team/classification-v1-design`, base main `757b318`. DEV only.

**Goal.** Each confirmed track gets one of `TEAM_A`, `TEAM_B`, `GOALKEEPER`, `REFEREE`, `UNKNOWN`, decided per match.
- No hard-coded kits or colours.
- The label comes from evidence over many frames of the track, never from one frame.
- When the evidence is unclear, the answer is `UNKNOWN`.

## 1. What exists today (checked)

| input | where | usable for v1 |
|---|---|---|
| tracks | `spana.tracks` 2.3.0 + `*.decisions.jsonl` | yes: per-frame state (OBSERVED / HELD / LOST), `confirmed`, identity state and `uncertain_episodes`, `detection_index` per row, camera state |
| torso colour | `spana/appearance.py` `torso_descriptor` (HSV hue-saturation histogram of the torso band, `appearance.*` crop geometry, `clean_max_iou` 0.05); `TorsoColourEmbedder` in `spana/tracking/linking.py` | yes: the v1 feature. It is weak between same-kit teammates, but that is the strength wanted here: same kit = same team |
| detector class | `spana.detections` 1.4.0 `class_id`, `class_names` [player, goalkeeper, referee, ball], a role hint only (decision 2026-11-16) | as a vote only. It is unreliable: DEV bench 91 of 708 players boxed as referee. On the DEV clips the referee class is 7 % (F02), **31 % (F05)**, 2 % (F07), 4 % (F08) of person boxes, and F08 has 517 goalkeeper boxes. It cannot decide a role alone |
| re-ID v2 embeddings | `models/spana_reid_v2.json` | measured as an alternative feature only (section 6). It is trained to separate players, not to group teams |
| pitch position | none (pitch context NOT DEFINED) | no. Keeper-in-the-box is not used in v1 |
| linking team input | `spana/tracking/linking.py` already accepts `teams: {track_id: label}` (`weight_team`, `TEAM_CONFLICT`) | a later consumer. Not wired in v1 (that would need its own decision) |

**Labels.**

| labels | split | team / role content | usable for scoring v1 |
|---|---|---|---|
| `data/reid/TRAIN/TRAIN_M00{1,2,3}/v1` (`players.json`, `identities.jsonl`) | TRAIN | a side per player: A / B / GK_A / GK_B / REF / OTHER (`spana.reid_labels.SIDES`). M001 10 / 10 / 1 / 1 / 3 REF, M002 8 / 9 / 1 / 1 / 2, M003 6 / 9 / 1 / 1 / 2; per tracklet on TRAIN tracker runs of the full broadcast | not for scoring or tuning (TRAIN, purpose `train`). A method sanity check on TRAIN tracklets is allowed, but it is long footage, so it needs Nathan's OK (no full-match runs) |
| `data/labels/FRESH_0{2,5,7,8}/v1` (gt.txt, NESTED_V2) | DEV | **none**: one id per player (18 / 12 / 21 / 22 = 73 ids, 3,463 player boxes on 264 labelled frames); the class column is 1 for every row | the frames and ids are what v1 is scored on, but a side per id is missing |
| `data/labels/dev_bench/bench_v1` | DEV | per box player / goalkeeper / referee (708 / 34 / 73) on 66 full-match frames, no A / B | only for the detector role hint per frame, not for tracks |

**So the DEV side labels the task refers to do not exist for DEV.** They exist for the three TRAIN matches only. Scoring
v1 on DEV needs one new small label set (section 5.1).

## 2. Method (v1)

Per match: the tracks documents of all DEV clips of one match go in together (G3 = FRESH_07 + FRESH_08), so TEAM_A
means the same kit in both clips.

1. **Evidence frames.** A track-frame gives evidence only when:
   - the track is confirmed and the row is OBSERVED;
   - the identity is CERTAIN, i.e. not inside an uncertainty episode (a contact frame may hold the rival);
   - the crop is clean (IoU <= `appearance.clean_max_iou` with every other observed box) and at least
     `appearance.min_crop_height_px` tall;
   - the camera is not CUT / UNKNOWN.
   - Frames are sampled every `team.frame_stride` frames (cost).
2. **Segments.** A track is split at every uncertainty episode. Each part between episodes is a segment and is
   labelled on its own. Frames inside an episode answer `UNKNOWN`. A teammate swap keeps the team, but a rival swap
   does not, and the tracker cannot tell which one happened.
3. **Segment descriptor.** The per-bin median of the segment's clean torso histograms. A segment needs at least
   `team.min_segment_frames` evidence frames over at least `team.min_segment_s`; otherwise `UNKNOWN`
   (TOO_FEW_CLEAN_FRAMES).
4. **Team model, per match.** Two clusters of the segment descriptors, weighted by evidence frames:
   - k-means with k = 2, Bhattacharyya distance, deterministic initialisation (the farthest pair), so reruns are
     byte-identical.
   - Trimmed: segments farther than `team.outlier_distance` from both centres are removed and the fit is repeated,
     `team.trim_rounds` times. Referees and keepers wear their own kits, so they fall out here and do not pull the
     team centres.
   - The model abstains for the whole match (`UNKNOWN` everywhere, MATCH_MODEL_ABSTAIN) when the centres are closer
     than `team.min_cluster_separation`, or either cluster has fewer than `team.min_cluster_segments` segments.
   - Naming: TEAM_A is the cluster that holds the earliest-starting segment. The names are arbitrary per match, with
     no home / away meaning and nothing carried across matches.
5. **Team label (temporal vote).** Each evidence frame of a segment votes for A when
   `d_A + team.vote_margin < d_B` (and the reverse for B); otherwise it abstains.
   - The segment gets TEAM_X only when all three hold:
     - its median descriptor is within `team.outlier_distance` of X;
     - X has at least `team.min_votes` votes;
     - X holds at least `team.min_vote_share` of all frames (abstaining frames count against it).
   - Otherwise `UNKNOWN` (LOW_VOTE_SHARE).
6. **Roles.** These apply only to a segment that is a colour outlier from both teams. Over its observed rows:
   - REFEREE when detector `referee` is at least `team.role_min_share` of its rows, with at least
     `team.role_min_frames` rows;
   - GOALKEEPER the same way with `goalkeeper`;
   - an outlier with neither, or with both classes above `team.role_conflict_share`, is `UNKNOWN`
     (COLOUR_OUTLIER_NO_ROLE / ROLE_HINT_CONFLICT).
   - A segment inside a team cluster is never REFEREE / GOALKEEPER, even if the detector says so. That is the
     F05-type confusion.
   - v1 gives GOALKEEPER without a team. A keeper's side needs pitch position, which comes later.
7. **QA flag, not a decision.** A track whose segments get different decided labels (for example TEAM_A then TEAM_B)
   is flagged `TEAM_CHANGE`. That is evidence of an identity transfer. v1 changes nothing in tracking or identity.
8. **Fail closed throughout.** No label is inferred from neighbours, from position, or from one frame. A segment never
   inherits another segment's label.

## 3. Output artifact (INTERFACES style; proposed)

- **Tracks -> team classification** (`spana/teams/`, CLI `python -m spana.cli teams --match-id --config
  [--video V --tracks T --decisions D --detections DET]... --out`). Repeat the bracket once per clip of the match.
- **Out: `spana.teams` 1.0.0**, JSON:
  - header `{schema: "spana.teams", schema_version: "1.0.0", match_id, inputs: [{clip, video_sha256, tracks_sha256,
    decisions_sha256, detections_sha256}], team_config (snapshot + sha256), feature: "torso_colour_hs_hist_v1",
    provenance}`;
  - `model: {status: FITTED | ABSTAIN, reason, clusters: [{label: TEAM_A | TEAM_B, segments, evidence_frames,
    radius}], separation, trimmed_segments}`;
  - `tracks: [{clip, track_id, segments: [{frame, frame_end, label, reasons[], evidence_frames, votes {TEAM_A,
    TEAM_B, abstain}, dist {TEAM_A, TEAM_B}, role_hint_share {player, goalkeeper, referee}}], flags: [TEAM_CHANGE]}]`.
  - Frames inside uncertainty episodes are not in any segment and read as `UNKNOWN`.
- **State enum** (`spana/records.py`): `TEAM_A, TEAM_B, GOALKEEPER, REFEREE, UNKNOWN`.
- **Reason enum:** `NOT_CONFIRMED, IN_UNCERTAIN_EPISODE, TOO_FEW_CLEAN_FRAMES, LOW_VOTE_SHARE,
  COLOUR_OUTLIER_NO_ROLE, ROLE_HINT_CONFLICT, MATCH_MODEL_ABSTAIN`.
- **Validation** in `spana/artifacts.py`:
  - segments inside the track's live frames and outside its episodes;
  - labels and reasons from the enums;
  - a decided label has no blocking reason;
  - the input sha256s match.
- **Consumers:** none in v1. Statistics and events read it only through validated events, later. Linking's `teams`
  input stays unwired.

## 4. Config keys (proposed; `config/default.yaml` `team`, schema 2.17.0, the next free version)

All values are priors until measured on DEV. Crop geometry is reused from `appearance.*`, not duplicated.

```yaml
team:
  feature: torso_hs            # torso_hs (v1) | reid (measurement only)
  frame_stride: 3              # sample every n-th frame for evidence
  min_segment_frames: 8        # clean evidence frames for a segment to be labelled
  min_segment_s: 1.0
  min_cluster_segments: 3      # per team cluster, else the match abstains
  min_cluster_separation: 0.30 # Bhattacharyya distance between the two team centres
  outlier_distance: 0.45       # from both centres -> outlier (role candidate)
  trim_rounds: 2
  vote_margin: 0.05
  min_votes: 5
  min_vote_share: 0.80
  role_min_share: 0.60         # detector role-hint share among an outlier segment's rows
  role_min_frames: 5
  role_conflict_share: 0.30
```

## 5. Scoring on DEV (QA)

### 5.1 Labels needed (one small human step)
- **New frozen set** `data/labels/<clip>/teams_v1/`: one side per frozen GT id, from the same enum `A / B / GK_A /
  GK_B / REF / OTHER`, plus `match_id`, so FRESH_07 and FRESH_08 share A / B.
  - 73 decisions, about 15 minutes.
  - `labels_frozen.json` v1 is not edited; this is a new file with its own sha256.
- **Tool:** `tools/team_label_sheet.py` writes one contact sheet per clip (a few clean crops per GT id). Nathan
  labels; `tools/freeze_team_labels.py` freezes the result with the existing split guard. DEV only.
- Labels are never inferred by the system.

### 5.2 Metric (`tools/team_score.py`)
- **Unit:** a labelled frame (every 6th frame) where the NESTED_V2 attribution (`tools/target_score.attributions`)
  gives a track one GT id unambiguously. Two-player boxes are excluded.
- **Mapping:**
  - TEAM_A / TEAM_B to label A / B: the one permutation per match with the most agreement, fixed for all clips of
    the match and reported;
  - GK_A / GK_B to GOALKEEPER, REF to REFEREE;
  - OTHER expects UNKNOWN.
- **Reported:**
  - answered share (coverage) per class;
  - **wrong answered frames** (the fail-closed number) with the exact one-sided 95 % bound;
  - the confusion matrix;
  - wrong segments / tracks;
  - answers on tracks with no GT id (spectators, bench), counted separately;
  - `TEAM_CHANGE` flags against the tracks' true transfers;
  - per clip and per match.
- **Proposed acceptance (Nathan to confirm):**
  - wrong answered frames bound <= 0.5 % (the same exact-bound rule as re-ID);
  - 0 REFEREE / GOALKEEPER answers on a team player;
  - coverage is reported, not gated, in v1.
- **Held out:** the values above are priors fixed before the first score. If any is tuned, it is tuned
  leave-one-match-out over G1 / G2 / G3, and the held-out numbers decide.
- **Adjacent checks:**
  - tracks, target and reviewer outputs byte-identical (v1 writes only its own artifact);
  - full test suite;
  - the run is deterministic (two runs, identical bytes).

## 6. Measured alternatives in the same run (no decision implied)
- `team.feature: reid`: the same method on re-ID v2 embeddings (cosine distance). It is reported next to colour,
  never adopted without its own decision.
- **Detector role hint only:** the majority class per segment, as a baseline. It shows what the colour model adds.

## 7. Build plan after go-ahead
- **One commit per layer:**
  1. `team/classification-v1`: `spana/teams/`, `spana.teams` 1.0.0 artifact plus validation, config 2.17.0, CLI, unit
     tests (synthetic two-kit tracks, outliers, abstention, segments at episodes, determinism).
  2. `evaluation/team-labels-dev`: the sheet tool and the freeze tool. Then Nathan labels.
  3. `evaluation/team-score-v1`: the scorer and the DEV run record.
- No EVAL, no full match. The TRAIN sanity check only with Nathan's OK.

## 8. Open questions for Nathan
1. Is it OK to label the 73 DEV GT ids with a side (section 5.1)? Nothing can be scored on DEV without it.
2. Acceptance as proposed (bound <= 0.5 % wrong answered frames, 0 role answers on team players)?
3. Is GOALKEEPER without a side acceptable for v1?
4. Should the TRAIN side labels be allowed as a method sanity check? These are long footage, so it is a
   non-full-match run limited to the labelled tracklets.
