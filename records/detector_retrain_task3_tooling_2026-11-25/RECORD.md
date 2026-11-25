# Detector retraining, task 3 tooling: dataset version 3 builder (2026-11-25)

Owner layer: player detection (dataset tooling). Branch `detection/dataset-v3-builder`, from
`detection/ball-retraining-plan` d230607. No labelling, no footage, no upload, no training. Tested on synthetic review
files only.

## What was built
- `spana/dataset/detector_review.py`: review file schema `spana.detector_review` 1.0.0 (docs/INTERFACES.md).
- `tools/dataset_v3.py`, `config/dataset_v3.yaml`:
  - `check --review FILE`: preflight only, writes nothing.
  - `build --review FILE`: freeze -> preflight -> assemble -> stage. Exit 3 with every reason on a refusal.
- Freeze: `data/detector_review/v<N>/{review_frozen.json, FROZEN.json}`, as `tools.freeze_ball_labels`. Never
  overwritten; the same bytes again are the same freeze. A malformed review is not frozen.
- Stage: `runs/dataset_v3/v3_review<N>/` (`stage.json` + `images/<split>/`), `uploaded: false`. Never overwritten.

## Preflight refusals (each tested)
| refusal | code |
|---|---|
| match not a TRAIN match on the allow-list (third-party confirmed), or without a split | NOT_ALLOW_LISTED |
| candidate not declared TRAIN; frame name or image sha256 in a frozen DEV label set | DEV_OR_EVAL |
| ball box on a spare ball / on a look-alike | BOXED_SPARE_BALL / BOXED_LOOKALIKE |
| two ball boxes on one frame (new frames; base labels after removals) | TWO_BALL_BOXES |
| frame file missing or unreadable; base frame missing from the image cache | MISSING_FRAME |
| no sha256; sha256 does not match the file | MISSING_HASH / HASH_MISMATCH |
| look-alike category under `min_per_category` (shortfall reported, nothing padded) | CATEGORY_SHORTFALL |
| MATCH_BALL frame without a box; negative frame with the match ball unaccounted for | MATCH_BALL_NO_BOX / BALL_UNACCOUNTED |
| other: unknown category, audited box not in the base labels, other base labels, name clash, box outside frame | CATEGORY, EXISTING_LABEL, BASE_LABELS, FRAME_NAME, BOX |

EVAL has no footage or labels in the repository to compare against: an EVAL frame is refused by the allow-list (its
match is not a TRAIN match) and by the declared split, not by content.

## Checked against the real frozen labels of version 2 (no pixels)
- `config/dataset_v3.yaml` names labels `7c3846b4...`, the labels of `version_2.json`.
- Without a verdict on it, the preflight refuses version 2's one two-ball frame; with the second box marked
  SPARE_BALL by image id + box, the plan is exactly `remove_box` + `add_tag has-spare-ball` on that image.

## Tests
- `tests/test_dataset_v3.py`, 20: happy path (freeze, staging, tags, boxes, spare-ball removal, scale copies),
  every refusal, never-overwrite rules, determinism (two independent builds, byte-identical), `check` writes nothing.
- Adjacent: `test_detector_mine.py` and the dataset / Roboflow tests, 159 passed, 1 skipped.
- Full suite: 827 passed, 15 skipped, 2 failed (`test_botsort_compare.py`: `supervision` not installed here; same
  as before this change).

## Not built / open (see docs/STATUS.md)
1. From Nathan's review CSV to the review file: frame export from the TRAIN clips (full-size JPEG + sha256).
2. Ball boxes for the missed-ball candidates (hint ring only) and the match-ball answer on negative frames.
3. Player / goalkeeper / referee labels on the mined frames (pre-label + review in the project).
4. The uploader that applies `stage.json` (uploads, `remove_box`, tags). Nothing is uploaded by this tool.
5. `min_per_category` 40 against 15 tiled-only objects: expect a CATEGORY_SHORTFALL on the real review.
6. Mined frame indices are clip-relative (10-minute clips). Frame names must not reuse the version-2 naming; a name
   already in the base labels is refused.
