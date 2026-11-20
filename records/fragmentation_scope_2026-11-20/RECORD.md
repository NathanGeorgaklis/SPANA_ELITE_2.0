# What fragments players now, and housekeeping state (DEV, 2026-11-20)

Analysis only (CODE TASK steps 3 and 4). Tracks: pan-aware tracker (`089ee08`, records/pan_aware_appearance_2026-11-20)
on DEV FRESH_02 / 05 / 07 / 08. No EVAL, no full match, nothing changed.

## Remaining fragments (`probes/frag_causes_now.py`)
A fragment is consecutive labelled frames where a player's unambiguous NESTED_V2 track changes: 101 here (the
benchmark counts 99). Cause = the old track's first logged non-match after its last labelled frame.

| cause | count | player labelled in picture throughout | camera > 5 px / frame |
|---|---|---|---|
| **colour veto alone** (GATE_FAILED:appearance_veto) | **33** | 28 | 0 |
| motion / size gate | 18 | 11 | 5 |
| the box was taken by another track | 14 | 14 | 1 |
| recovery refused (rival / ambiguity) | 11 | 11 | 0 |
| old track still observed (a parallel duplicate took over) | 11 | 11 | 0 |
| colour veto plus motion / size | 9 | 6 | 3 |
| no detection nearby / unused candidate | 5 | 3 | 2 |

By clip: FRESH_02 23, FRESH_05 7, FRESH_07 29, FRESH_08 42.

1. **The colour veto is still the largest single cause (33, plus 9 combined).** These are not pan cases: the camera
   was below 5 px / frame in every one. In records/veto_root_cause_2026-11-20, the slow-camera wrong vetoes compared
   a freshly updated appearance at 0.56–0.70. Lowering the threshold is unsafe (right vetoes at 0.62 / 0.67).
2. **Almost every fragment happens with the player in the picture.** On these labelled clips, exits are not the
   problem. On the unlabelled 10-minute window, track ends are mostly exits (779 LEFT_FRAME) and crowds
   (890 LOST_EXPIRED), so longer footage has a different mix.

## Proposed next CODE TASK (not started)
**A re-ID second opinion on a colour veto of an exclusive own-box match.**
- When the colour veto alone refuses a track's only, exclusive, geometrically near-perfect next box, ask the re-ID
  model whether it is the same player. The calibrated machinery already exists: `reid_resolve`, merged in #13, off.
- Re-ID may only lift the veto with a calibrated positive verdict; otherwise the veto stands (fail closed).
- **Target:** the 33 veto-only fragments.
- **Bar:** the agreed tracking bar, plus the 10-minute window.
- **Risk:** re-ID's known confident errors (records/reid_v3_wrong_answers_2026-11-19) must be measured on these exact
  cases first.

The other buckets each need their own diagnosis, and none is as large: motion / size 18, taken 14, refused 11,
duplicates 11.

## Housekeeping found (step 4; no changes made)
- **52 remote branches are fully merged into main**, so they can be deleted safely. They include every branch of
  this work, from `evaluation/offer-verdict-offered-track` to `tracking/veto-root-cause`.
- **12 branches are NOT merged.** They are other work in progress and are left alone:
  - `ball/ball-track-v1`
  - `claude/docs-shared-state`
  - `data/reid-train-m003-v1` (TRAIN_M003 re-ID labels)
  - `evaluation/dev-label-windows-v1`
  - `identity/reid-direct-label-v1`
  - `identity/reid-fusion-dev-eval`
  - `identity/reid-retrain-v3`
  - `infra/modal-reid-train`
  - `performance/detect-preprocess-overlap`
  - `performance/detector-speed-options`
  - `performance/identity-tracking-speed`
  - `tracking/pan-aware-appearance` (this work, awaiting merge)
- **Nathan's Mac working copy** is on `performance/cloud-bench` (`73fedf9`), with uncommitted edits to
  `docs/DECISIONS.md` and `records/dev_permissions/DEV_PERMISSIONS_2026-11-14.md`, and untracked files. Its
  `config/default.yaml` is schema 2.4.0, so main's code (2.13.0) cannot load it. Runs here use clean `git archive`
  copies instead; the working copy was not touched.
