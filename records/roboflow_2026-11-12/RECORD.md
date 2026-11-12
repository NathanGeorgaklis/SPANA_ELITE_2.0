# Roboflow connection for the player / ball detection dataset (2026-11-12)

Roles: Lead Engineer; Player Detection Engineer; Data / Licensing Steward; QA / Regression Engineer.
Built and tested only with mocked Roboflow clients. **Nothing uploaded, no frame sent to Roboflow, no footage
read.** DEV and EVAL not involved.

## Target
- Workspace `nathans-workspace-9brdc`, project `spana-elite`.
  - Object detection, private, 0 images, splits train / valid / test.
  - Checked read-only via the Roboflow connector on 2026-11-12.
- Classes: player, goalkeeper, referee, ball.
- Split mapping (`config/roboflow.yaml`):
  - SPANA TRAIN goes to Roboflow `train`.
  - SPANA DEV goes to Roboflow `valid`.
  - SPANA EVAL has no mapping and is never uploaded.

## What was added
| piece | file |
|---|---|
| Config (strict loader, same rules as default.yaml; tuples now supported generically) | `config/roboflow.yaml`, `spana/dataset/config.py`, `spana/config.py` |
| Licensing / split policy (`authorize`) | `spana/dataset/config.py` |
| API key from env / gitignored `.env`, redaction | `spana/secrets.py`, `.env.example`, `.gitignore` |
| Deterministic sampling (interval + heavy motion + cut, cap with even thinning) | `spana/dataset/sampling.py` |
| Upload manifest (schema, validation, prior-split guard) | `spana/dataset/manifest.py` |
| Ingestion CLI | `tools/roboflow_ingest.py` |
| Generic `Detector` interface + Roboflow backend | `spana/detection/base.py`, `spana/detection/roboflow_backend.py` |
| DEV-only baseline measurement | `tools/dev/roboflow_baseline.py` |
| Tests (no network) | `tests/test_roboflow_dataset.py` |

## Policy enforced in code (fail closed)
- **EVAL:** `split=eval` is refused before the video is opened (CLI exit 3). The loader also rejects any
  `split` other than `dev` for inference tools.
- **Allow-list:** the match must be in `licensed_matches`, and each match belongs to exactly one split.
  - A registered match is refused in any other split.
  - The ingestion tool also refuses a match that an earlier manifest recorded under another split.
- **Third-party processing:** a Roboflow upload, or a hosted inference call (any `api_url` not on
  `local_api_hosts`), needs `third_party_processing_confirmed: true`.
  - It is `false` for every match until Nathan confirms the licence covers it.
  - `--dry-run` (local sampling and manifest only) does not need this flag.
- **Secret:** the API key travels in the HTTP JSON body, never in a URL. Errors are redacted before display or
  storage. Tests check that the key never appears in manifests or exceptions.

## Allow-list contents (to confirm with Nathan)
- `TRAIN_M001_MORETON_CITY`, split train.
  - Written permission dated 2026-11-03 (records/reid_train/TRAIN_M001_STATUS.md).
  - `third_party_processing_confirmed: false`.
- **Discrepancy to resolve.** The task says one licensed TRAIN match (letter of 3 Nov 2026). The re-ID records
  also list TRAIN_M002 and TRAIN_M003 with written permission dated 4 Nov 2026. These are not added to the
  allow-list until Nathan says whether that permission covers this dataset.
- No DEV match listed yet. Nathan will name the DEV clip and its match_id.

## Baseline model (test use only)
- `football-players-detection-3zvbc/11`, Roboflow Universe `roboflow-jvuqo/football-players-detection-3zvbc`.
  - YOLOv8x, 1280 px input; classes ball / goalkeeper / player / referee; dataset licence CC BY 4.0.
  - Universe-reported mAP 0.929 / precision 0.911 / recall 0.896 on its own validation set, which is not ours.
- Choice: Roboflow's own, most-used football detector (331 stars, ~26k downloads), with the exact class list.
  Version 11 has the best reported mAP and recall of the 1280 px versions.
- **Caveat.** The source footage of that dataset is not documented; this is the same provenance question as
  records/reid_licence_2026-11-08. Use it for the DEV baseline measurement only. Do not ship it and do not train
  on its outputs without a licence review.
- `confidence 0.25` and `iou 0.50` are untuned priors. They are not to be tuned until the baseline is reviewed.

## Dependencies
- Added: `roboflow 1.7.3`, `python-dotenv 1.2.3`, `requests 2.34.2`.
- **`inference-sdk` not added.** Version 1.8.4 requires `numpy<2.4` and the non-headless `opencv-python<=4.13`.
  Installing it downgraded numpy 2.5.3 to 2.2.6 and replaced cv2 5.0 with 4.12.
  - This was reverted: numpy 2.5.3 and opencv-python-headless 5.0.0.93 restored; `pip check` clean.
  - The adapter instead calls the inference-server API (`POST /infer/object_detection`) with `requests`. Roboflow's
    hosted endpoint and a local inference server both serve this API.

## Tests
- `python -m pytest tests/test_roboflow_dataset.py -q`: 22 passed.
- `python -m pytest tests -q`: 148 passed, 3 skipped (before this change: 126 passed, 3 skipped).

## Next (blocked on Nathan)
1. Confirm the licence covers third-party processing, i.e. Roboflow storage and hosted inference. Then set
   `third_party_processing_confirmed: true` for the match.
2. Give the TRAIN video path and match_id; resolve the TRAIN_M002 / M003 question.
3. Give a DEV clip path and match_id; it is added to `licensed_matches` with split dev.
4. Then run, in order:
   - `python -m tools.roboflow_ingest --video … --match-id … --split train --dry-run` (review the frames);
   - the same command without `--dry-run`;
   - `python -m tools.dev.roboflow_baseline --video DEV --match-id …`.

## Update 2026-11-12: licence confirmed
- Nathan confirmed the licences cover third-party processing (Roboflow upload and hosted inference).
- Nathan confirmed TRAIN_M002_EASTERN_SUBURBS and TRAIN_M003_SYDNEY_OLYMPIC are covered by written permission
  (dated 2026-11-04, per their status records).
- Allow-list: M001, M002 and M003, all split train, `third_party_processing_confirmed: true`.
- Still no DEV match: the baseline cannot run until a DEV clip + match_id is given.
- Videos to ingest: the H.264 transcodes in `Downloads/SPANA_reid/TRAIN_M00x/work/`. These are the 10-minute
  windows. Each transcode's sha256 must equal its status record before upload:
  - M001 `2475f26c…4c58`
  - M002 `62d91f7e…707e`
  - M003 `3db10eb1…da50`

## Update 2026-11-12: Roboflow SDK replaced by a direct REST upload client
- **Finding: the SDK cannot run beside the engine on Nathan's Mac.** The Mac `.venv` is Python 3.9.6 and
  roboflow >= 1.3 requires Python >= 3.10. The last release for 3.9, 1.2.16, pins
  `opencv-python-headless==4.10.0.84`, which would downgrade the engine's OpenCV 5.0.
- **Change:** new `spana/dataset/upload.py` calls the REST endpoint the SDK itself uses:
  - `POST https://api.roboflow.com/dataset/<project>/upload`, multipart `file` / `name` / `split` / `metadata`,
    with `batch` and `tag` as query parameters;
  - the API requires the key in the query string, so every error text is redacted;
  - retries apply only on network errors and 5xx;
  - before uploading, `GET /<workspace>/<project>` checks the key can read the project and that its type is
    `object-detection`; otherwise it fails closed.
- `roboflow` removed from requirements. New config keys: `roboflow.api_url`, `roboflow.project_type` and
  `roboflow.upload_timeout_s`.
- Tests: 25 dataset tests passed; full suite 151 passed, 3 skipped, run with the SDK uninstalled.

## Update 2026-11-12: TRAIN frames uploaded (M001, M002, M003)
- **Approval:** Nathan approved the upload after the dry runs.
- **Where it ran:** on Nathan's Mac, repo at commit `6c068c8`, `.venv` with Python 3.9.6, numpy 2.0.2,
  OpenCV 5.0.0, requests 2.32.5 and python-dotenv 1.2.1. Tests there: 150 passed, 4 skipped.
- **Inputs:** H.264 transcodes of the 10-minute windows (50:00–60:00). Each sha256 was checked against its
  status record before use:
  - M001 `2475f26c…4c58`;
  - M002 `62d91f7e…707e`;
  - M003 `3db10eb1…da50`.
- **Result:** 300 frames per match, 900 in total.
  - All 900 uploaded: 0 failed, 0 duplicates.
  - All in Roboflow split `train`.
  - Tagged `match:<id>` and `split:train`, with metadata for match, split, frame index, time, video sha256
    and image sha256.
- **Checks:**
  - Each upload selected exactly the same frames, with identical image bytes, as its dry run, so sampling is
    deterministic on real footage.
  - Each manifest validates, has 300 distinct upload ids, and contains neither the key nor local folder paths.
  - Roboflow image search: 300 images per match tag, 900 with `split:train`.
- **Manifests** (`records/roboflow_uploads/`):
  - `TRAIN_M001_MORETON_CITY__train__20261112T120617Z.json` sha256 `21cd380242c5b5e42411d1bf98a1ce46150eda2c4245aa5c75d2cf4f3233b19a`
  - `TRAIN_M002_EASTERN_SUBURBS__train__20261112T120732Z.json` sha256 `2f58ccebf03540c06fad1308304506c79208330df442dfbba041f0ab8631eece`
  - `TRAIN_M003_SYDNEY_OLYMPIC__train__20261112T120726Z.json` sha256 `641bba6083d253d552d64cbf6897826a3d49e4da0964f1d16b8fe1ab6c3ccc0b`
- **Sampling mix:**
  - M001: 155 interval and 163 motion frames.
  - M002: 283 interval, 17 motion and 2 cut frames.
  - M003: 131 interval and 181 motion frames.
  - The motion threshold (0.06) is still an untuned prior; calibrate it on DEV only.
- **Not labelled yet.** No DEV or EVAL frames were uploaded.

## 2026-11-13: dataset preparation for detector v1 (code; sample pending)
- **Pre-label model:** `coco/36`, i.e. RF-DETR base COCO-pretrained, Apache-2.0.
  - It is the id Roboflow's own inference server (`inference` 1.8.4, `inference/models/aliases.py`) resolves the
    alias `rfdetr-base` to.
  - Not a Universe dataset or model.
  - The hosted endpoint answers `POST https://serverless.roboflow.com/coco/36` with the base64 image as the body;
    confidence and overlap are fractions.
  - The JSON route `/infer/object_detection` returns 404 on the hosted endpoint, so the DEV-baseline adapter was
    switched to the per-model route. The baseline had never run, so no result changes.
- **Image bytes:** on Nathan's Mac, each image's `url` (`original.jpg`) returned exactly the bytes uploaded
  (sha256 equal to `user_metadata.image_sha256`). Every tool verifies this before using pixels.
  - The cloud container cannot reach source.roboflow.com, so networked steps run on the Mac.
- **Splits: Roboflow's API cannot move single images between splits.**
  - The only mechanism is a random, ratio-based rebalance (`datasets_rebalance_splits`), which would split
    matches across splits.
  - `tools/roboflow_split.py` therefore:
    - plans the match-level assignment and writes the before/after manifest;
    - applies near-duplicate exclusion tags;
    - refuses `--apply` whenever a move is needed (`splits.move_method: none`). Nothing is changed in that case.
- **Licence gate (`match_of`):** applied to every frame by pre-labelling, split assignment, progress and version
  generation. It checks:
  - exactly one match tag;
  - the match is on the allow-list and not EVAL;
  - split tags agree with the allow-list;
  - `metadata.match_id` agrees with the tag;
  - third-party processing is confirmed.
- **Version gate:** refuses to generate a version if:
  - any pre-labelled frame is unreviewed;
  - any frame in the Dataset is unreviewed;
  - a frame fails the gate;
  - a match is in two splits;
  - there is a test split;
  - train or valid is empty.
- **Version settings:** colour-changing augmentations are refused by the config loader.
- **Tests:** `tests/test_roboflow_prep.py` (19) and `tests/test_roboflow_dataset.py` (25) pass. Full suite:
  170 passed, 3 skipped.

## 2026-11-13: split-move probe support (Nathan chose M001 as valid, option (a): re-upload probe)
- **Why this is a committed tool:** an ad-hoc probe script on the Mac was blocked by its safety classifier. That
  is correct under the project rules: every change to Roboflow goes through committed, tested code.
- **New move method:** `splits.move_method: reupload` re-uploads each frame's identical, sha-verified bytes with the
  new split, keeping the same name, tags and metadata.
- **Probe:** `tools/roboflow_split.py --probe N`, with `N <= splits.probe_max` (1), moves only the first planned
  frame and verifies it:
  - the upload must return the same image id;
  - a fresh listing must show the new split.
  - Otherwise the run stops, and the manifest records what happened.
- **Full moves:** `--apply` needs `move_method: reupload`. That is only set after a successful probe.
- **Tests:** 48 dataset tests pass (4 new for moves and probes); full suite 174 passed, 3 skipped.

## 2026-11-13: audit and split probe results (run on Nathan's Mac, commit `ae36133`)
- **Audit** `records/roboflow_audit/audit_20261112T185229Z.json` (sha256 `6caa4a1e…92ba2e`):
  - 900 images, 300 per match, all in `train`. None in the Dataset, none annotated.
  - 0 untagged images, 0 licence-gate rejections, 0 unverified images.
  - Near-duplicates at Hamming ≤ 4: 34 groups, 60 frames to exclude (M001 15, M002 41, M003 4); no group spans
    two matches.
  - Pair distances: 6 at 0 bits, 30 at 2 bits, 63 at 4 bits. Group sizes: 26 × 2, 3 × 3, 2 × 4, 1 × 5, 1 × 9,
    1 × 11.
  - Some linked frames are minutes apart (e.g. M001 f360 and f28680). A whole-frame pHash is dominated by the
    pitch and stands, so the 4-bit prior may also link same-camera wide shots with different player positions.
  - **Exclusions not applied**; they need Nathan's visual check, or a stricter threshold.
- **Split probe** `records/roboflow_splits/split_20261112T192514Z_probe1.json` (sha256 `0ef687a9…46ff`):
  - The identical bytes of M001 frame 0 were re-uploaded with `split=valid`.
  - Roboflow returned the same id (`duplicate: true`), but the split stayed `train`. This was confirmed again by a
    separate search.
  - **Result: re-upload does not move splits.** Nothing else changed and no second copy exists.
  - `move_method` stays `none`.
- **Pre-label sample: not run.** The Mac's auto-mode classifier refuses Roboflow-writing commands started from this
  session, even with Nathan's request. Nathan has to run it on the Mac, or allow it there.

## 2026-11-13: Nathan's decisions: allow the tools on the Mac, option (b) delete + re-upload, stricter audit
- **Allow rule:** Nathan wants `.venv/bin/python -m tools.roboflow_*` allowed on the Mac. The Mac's Claude Code
  refused to let this session read or edit its permission settings, which is correct: an agent may not widen its
  own permissions. Nathan adds the rule himself.
- **Move method `delete_reupload`** (replaces the disproven `reupload`):
  1. Verify every frame's bytes.
  2. Write a restore file (no local paths).
  3. Remove the frames with their upload batch, permanently. Use the match's whole batch when it holds exactly
     those frames; otherwise move just those frames into a new batch first.
  4. Check the frames are gone.
  5. Upload them again with the new split, same name, tags and metadata.
  6. Check each name exists exactly once, in the new split, with the same sha256.
  - Any failed check stops the run.
  - `--restore FILE` re-uploads any frame from a restore file that is missing.
  - `--probe 1` tests the method on one frame first.
- **Endpoints:** batch calls follow Roboflow SDK 1.7.3: `GET /{ws}/{project}/batches`,
  `POST .../annotation-batches`, `DELETE .../annotation-batches/{id}?permanent=true`. Per Roboflow's description,
  permanent deletion removes the image sources, not only the project link.
- **Audit threshold:** `audit.phash_hamming_max` 4 → 2, at Nathan's request after the 4-bit audit linked frames
  minutes apart. The audit is re-run with the new value.
- **Tests:** dataset tests (25 + 25) pass; full suite 176 passed, 3 skipped.
