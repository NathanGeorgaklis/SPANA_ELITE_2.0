# Re-ID resolution of same-look IDENTITY_UNCERTAIN episodes, DEV (2026-11-20)

CODE TASK (Nathan / chat), target identity. Branch `identity/reid-episode-resolution`, base main `67df7ce`.
DEV FRESH_02 / 05 / 07 / 08, the same cached local-detector detections (`runs/person_view_compare/F0x_local.det.json`).
Scored with `config/benchmark.yaml` unchanged. No EVAL, no full match. Run on Nathan's Mac.

**Outcome: STOP condition met.** Re-ID v2 does not separate same-kit teammates on two of the three DEV clips that
have them (FRESH_07, FRESH_08: 0 of 66 clean same-look rival tests reach the calibrated margin). The margin was not
lowered. The rule is built, logged and tested, and stays OFF by default. Safety held: 0 silent transfers, 0 / 0 wrong
frames, 0 transfers on linked tracks, 2 of 2 re-ID resolutions right. The gain is small: unaided 0.508 -> 0.513,
reviewer 0.680 -> 0.680.

## 1. What existed (STATUS Next task 1(c), config `reid_resolve`, off)
- **Embedder:** SPANA re-ID v2 (`models/spana_reid_v2.json`, ONNX sha256 `c269450e…`, 256-d, cosine).
  - Backbone: DINOv2-S/14 (Apache-2.0, `models/dinov2_small.json`).
  - Fine-tuned with `tools/reid_train.py` / `config/reid_train.yaml` (12 epochs, 2026-11-18,
    `records/reid_retrain_m001_m002_2026-11-18`).
  - Training data: frozen TRAIN labels of TRAIN_M001 (Moreton City Excelsior) and TRAIN_M002 (Eastern Suburbs).
    That is 46 identities and 13,109 train crops.
- **Provenance:** both manifests (`data/reid/TRAIN/*/v1/manifest.json`) record written permission (3 / 4 November
  2026) for commercial AI / ML use, including re-ID training. They also record `training_permitted`, player consent,
  minors covered, and "model survives footage removal". No DEV or EVAL footage is in the training set. The licence
  check (`records/reid_licence_2026-11-08`) rejected the public re-ID models.
- **Margin:** 0.517, the most conservative leave-one-clip-out value for <= 0.5 % wrong
  (`records/reid_calibration_dev_2026-11-18`, floored `records/reid_margin_floor_dev_2026-11-19`).
- **Rule (config 2.12.0):** colour and re-ID were combined per rival pair: either against resets the run, either
  clears. Predecessor rivals of an ambiguous spawn got no re-ID verdict at all. On current main it resolves 2
  episodes (FRESH_02), and the reviewer goes 0.680 -> 0.685.

## 2. Why episodes stayed uncertain: two rule defects
1. **Colour "cannot tell" blocked re-ID.** Colour TWO_SIDED returns False whenever the colour margin is under
   0.10, and PREDECESSOR returns False whenever two predecessors look alike. Under "either against resets", colour
   vetoed every re-ID clear on same-look rivals, which are exactly this task's case.
2. **Ambiguous spawns could not be cleared by re-ID** (the predecessors were skipped).

## 3. Change (config 2.14.0; `spana.tracks` / `spana.target` contracts unchanged)
- **Separate re-ID run** (`spana/tracking/tracker.py` `_reid_evidence`, `_advance_reid_run`):
  - It uses clean (IoU <= 0.05), uncontested crops only.
  - **Pair rule:** margin = sim(crop, own reference) − sim(crop, rival reference at the contest). The pair clears at
    >= 0.517 and is against at <= −0.517.
  - **Ambiguous spawn:** one predecessor must beat every other by >= 0.517, and it must be the same pick on every
    frame of the run. A changed pick resets the run.
  - **Missing data** (a missing embedding or reference, the contest frame itself, an appearance doubt) is no
    evidence.
  - **Colour** can only veto, and only when it clearly favours the rival, or is clearly unlike the picked
    predecessor.
  - **Resolution:** resolved_by REID only when the run spans `tracking.identity_resolve_s` (0.5 s) AND holds
    `reid_resolve.min_evidence_frames` (new, 3, a prior) clearing frames. The config refuses values < 2: never one
    frame.
  - **Unchanged:** the colour path is as with re-ID off. Re-ID evidence against still resets it. A colour run no
    longer takes re-ID clears.
- **Logging:** every evidence frame of an uncertain track gets a `reid_evidence` row with its status, per rival the
  scores / margin / verdict, or the refusal reason. Statuses: CONTESTED, NOT_CLEAN, NO_EMBEDDING, NO_OWN_REFERENCE,
  APPEARANCE_DOUBT, EVALUATED. Refusal reasons: BELOW_MARGIN, NO_RIVAL_REFERENCE, NO_PREDECESSOR_REFERENCE,
  AT_CONTEST. Further rows:
  - `reid_run_reset` (AGAINST, COLOUR_CONTRARY, PICK_CHANGED);
  - `identity_resolved` with `reid` (margins), `reid_margin`, `run_frames`, `run_s`, `predecessor`.
- **Follower guard** (`spana/identity/follower.py`): `resume_after_reid` resumes only on a track that was CERTAIN
  when it became the answered track.
  - Re-ID re-confirms the identity the track had *before* its episode (or the predecessor it was born from). A tap
    made during the episode may be on a different player.
  - The synthetic h1 same-look spawn at 60 fps shows the case: the new track first held B (merged box), then
    re-ID resolved it to A, rightly.
- **Unchanged as required:** the spawn / crossing rules, the linker, `config/benchmark.yaml`, and the margin.
  `reid_resolve.enabled` and `target.resume_after_reid` stay false.

## 4. DEV results (`compare_summary.json`; 3,449 labelled player-frames, 14.36 followed min)

| run | silent | wrong unaided / reviewer | transfers on linked tracks | correct unaided | correct reviewer | reviewer unanswered (ABSTAIN) frames | actions / min | uncertain F02 / F05 / F07 / F08 |
|---|---|---|---|---|---|---|---|---|
| main 67df7ce (re-ID off) | 0 | 0 / 0 | 0 | 0.508 | 0.680 | 735 | 4.11 | 21.7 / 0 / 15.5 / 38.6 % |
| this branch, re-ID off | 0 | 0 / 0 | 0 | 0.508 | 0.680 | 735 | 4.11 | = main (tracks identical on all 4 clips) |
| old rule (2.12.0) on | 0 | 0 / 0 | 0 | 0.513 | 0.685 | 718 | 4.11 | 21.7 / 0 / 15.5 / 38.6 % |
| **new rule on, follower guard (as committed)** | **0** | **0 / 0** | **0** | **0.513** | **0.680** | **735** | 4.11 | 21.7 / 0 / 15.5 / 38.6 % (F08 +2 frames) |
| new rule on, without the guard | 0 | 0 / 0 | 0 | 0.513 | 0.685 | 718 | 4.11 | same |
| linked (clearance off), new rule + guard | 0 | 0 / 0 | 0 | 0.548 (main 0.543) | 0.681 (main 0.681) | 733 | 3.62 | |

**Per clip, new rule + guard (main):**

| clip | unaided | reviewer | reviewer ABSTAIN frames |
|---|---|---|---|
| F02 | 0.494 (0.477) | 0.696 (0.696) | 213 (213) |
| F05 | 0.716 (0.716) | 0.964 (0.964) | 14 (14) |
| F07 | 0.628 (0.628) | 0.686 (0.686) | 244 (244) |
| F08 | 0.242 (0.242) | 0.487 (0.487) | 264 (264) |

- **Tracks:** boxes, ids and states are identical to main with re-ID on; only identity fields can change.
  - Transfers stay 7 flagged, 0 silent. Fragments 101 / ID switches 103 (linked: 92 / 94).
  - FRESH_08 T000056: its colour resolution comes 2 frames later (221 -> 223), because a colour run no longer takes
    re-ID clears.
- **Follower:** the guard costs the reviewer's +17 frames on FRESH_02. That gain came from a tap made during an
  episode that re-ID later resolved (right on DEV, but unanchored). Unaided +17 frames stay (selection while
  CERTAIN).

## 5. Episodes (`episodes_ours.json`, `tools/dev/reid_episode_report.py`, labels window 2 s)
- **126 episodes. Resolved 14:**
  - REID 2: both right vs labels. FRESH_02 T000003 (player 13, rival T000013, margin 0.546) and T000007 (player 26,
    rival T000008, margin 0.589). Both runs were 16 frames / 0.5 s. Colour cleared the same two episodes on the
    same frames.
  - APPEARANCE 12: 2 right, 10 unknown, 0 wrong.
- **Not resolved (refused): 112** (72 track ended, 40 clip end). Reasons:

| reason | episodes | what it means |
|---|---|---|
| NO_REFERENCE | 49 | 35 ambiguous spawns: a predecessor has no re-ID reference; 8 no own reference; 4 both; 2 rival |
| NO_CLEAN_CROP | 29 | only contested / overlapping crops while uncertain |
| NOT_OBSERVED | 24 | no matched frame inside the episode |
| BELOW_MARGIN | 7 | same-look rivals evaluated, never a clear margin |
| RUN_TOO_SHORT | 2 | cleared on some frames, never 0.5 s / 3 frames |
| AGAINST | 1 | evidence against |

- **The reference-less predecessors** (96 references over 39 spawn episodes) are all tracks that were themselves
  uncertain and never had a clean CERTAIN observation. They come from chains of ambiguous spawns or crossings.
  There is no trustworthy reference of who they were, so refusing is correct (fail closed).
- **Linked tracks:** the same counts (`episodes_linked.json`).

## 6. STOP check: can re-ID v2 tell same-kit teammates apart on DEV? (`similarity_distributions_v2.json`)
- **Source:** the cached DEV rival tables of re-ID v2 (`tools/dev/appearance_cue_eval.py --dump-tables`, same
  model hash).
- **Teammate proxy:** the DEV labels carry no team field, so a pair counts as "same look" when the mean torso-colour
  similarity of its crop pairs is >= 0.70 (the tracker's own same-look test). This gives 155 of 634 player pairs.

**Cosine similarity of single crop pairs (pooled):**

| | n | p05 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| same player | 24,000 | 0.258 | 0.462 | **0.595** | 0.709 | 0.831 |
| same-look different players (teammate proxy) | 6,185 | 0.050 | 0.269 | **0.407** | 0.534 | 0.682 |
| different-look players | 17,815 | −0.153 | 0.029 | 0.167 | 0.315 | 0.545 |

- Same player vs same-look teammate: AUC 0.70. 39 % of teammate pairs are above the same-player p25.

**Clean rival tests at margin 0.517** (query of A vs references of A and B; all three crops clean):

| clip | same-look tests | resolved right | wrong | median margin | different-look: tests / right / wrong |
|---|---|---|---|---|---|
| FRESH_02 (G1) | 129 | 50 | 0 | 0.458 | 357 / 102 / 0 |
| FRESH_05 (G2) | 0 | – | – | – | 38 / 34 / 0 |
| FRESH_07 (G3) | 52 | **0** | 0 | 0.191 | 457 / 165 / 0 |
| FRESH_08 (G3) | 14 | **0** | 0 | 0.121 | 370 / 73 / 0 |

**Verdict:**
- The margin is safe (0 wrong anywhere).
- On G3, re-ID v2 separates no same-look pair at it. It separates some only on G1.
- Lowering the margin is not the answer: on G3 the same-look medians (0.12–0.19) are far below 0.517.
- End to end, BELOW_MARGIN is only 7 of 112 refusals: the binding limits on DEV are the missing references (49) and
  the missing clean crops (29 + 24 not observed).

## 7. Next (for Nathan)
1. **Re-ID v2 on more permissioned TRAIN footage** (the task's stated next step):
   - TRAIN_M003 already exists (v3 failed the floored held-out check, `records/reid_margin_floor_dev_2026-11-19`);
   - more matches with same-kit teammates.
   - Re-calibrate the margin per DECISIONS 2026-11-18 / 2026-11-19.
2. **References for spawned tracks.** 39 spawn episodes are blocked because their predecessors were themselves
   unresolved, so better re-ID alone will not reach them. A design question for Nathan; not attempted.
3. **Keep `reid_resolve` off.** Turning it on now is safe on DEV but worth +0.005 unaided and 0 for the reviewer.

## Tests
`tests/test_reid_episode_resolution.py` (13):
- a same-kit crossing that colour cannot separate resolves by re-ID (>= 3 frames, >= 0.5 s, margins logged);
- no clear margin stays uncertain and logs BELOW_MARGIN / NOT_CLEAN;
- against resets and below-margin holds; never one frame, and the config refuses min_evidence_frames < 2;
- colour clearly favouring the rival vetoes;
- missing references are refusals;
- predecessor pick: it resolves, a changed pick resets, same-look predecessors never resolve, a missing reference is
  refused;
- the h1 same-look spawn end to end at 10 / 25 / 60 fps resolves to the predecessor that carried the same player;
- follower anchor: a selection while CERTAIN resumes, a selection during the episode never does.

Full suite: 596 passed, 6 skipped.

## Files
- **Scripts:** `run_bench.sh` (CFG / OUT select config / output), `summarize.py`, `distributions.py`.
- **Results:** `compare_summary.json` (main, existing_on, new_off, new_on, new_on_guard),
  `episodes_ours.json` / `episodes_linked.json`, `similarity_distributions_v2.json`.
- **Configs:** `default.config_snapshot.yaml`, `reid_on.config.yaml`. `HASHES.txt`.
- **Outputs:** `~/spana-reid-episode-run/{main,existing_on,new_off,new_on,new_on_guard}`.
