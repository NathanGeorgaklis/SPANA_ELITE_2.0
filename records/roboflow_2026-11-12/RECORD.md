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

## 2026-11-13: results on Nathan's Mac (commit `9380526`)
- **Audit (Hamming ≤ 2)** `records/roboflow_audit/audit_20261112T195344Z.json` (sha256 `9c02a8ce…c7f3`):
  - 13 groups, 24 frames to exclude: M001 3, M002 20, M003 1.
  - Pair distances: 6 at 0 bits, 30 at 2 bits.
  - Mostly consecutive regular frames 2 s apart. The largest group is 10 frames of M002 within 18 s, a static
    shot.
  - Exclusions not applied yet.
- **delete_reupload probe (M001 f0000000) worked:**
  - Old id `PhuHQ4Zj…` was removed with its own one-frame batch.
  - Re-uploaded as new id `dGpLC8iE…` in split `valid`, with the same sha256, tags and metadata. Exactly one copy.
  - Project total stays 900 (899 train, 1 valid). Batches: M001 train 299, M001 valid 1.
  - Restore file `records/roboflow_splits/restore_20261112T195427Z.json`.
  - The tool's final check raised "search stopped after 899 of 900": Roboflow's search index lags writes. The state
    was verified separately afterwards.
  - Because that error was not a PolicyError, no split manifest was written for this run.
  - Fixes:
    - listings and post-move checks retry (`audit.search_attempts` / `search_wait_s`, `splits.verify_attempts` /
      `verify_wait_s`);
    - any error during a split run is now written to its manifest.
  - `move_method` is set to `delete_reupload`.
- **Pre-label sample (coco/36, RF-DETR base COCO):** 10 frames each from M002 and M003, evenly spaced.
  - Logs: `records/roboflow_prelabel/prelabel_20261112T195536Z.json` (sha256 `8cf5ada6…`) and
    `prelabel_20261112T195607Z.json` (sha256 `495a796c…`).
  - All 20 labelled: 232 player boxes (M002 115, M003 117), 0 ball boxes.
  - Ball detections below the 0.85 threshold: 25 (M002 7, M003 18).
  - Dropped COCO labels: bench 28, chair 26, car 12, truck 2, couch 1.
  - Checked via the API: image `buQN0GpB…` holds 16 `player` boxes, as logged, and carries `prelabel:v1`.
- **Search omits annotations** of images that are not in the Dataset. Progress therefore reads each pre-labelled or
  reviewed frame's annotation from `GET /{ws}/{project}/images/{id}`; checked live, it returns `annotation.boxes`
  with a label per box.
- **Tests:** dataset tests 52 pass; full suite 178 passed, 3 skipped.

## 2026-11-13: Nathan approved: sample looks very good, tag exclusions, full pre-label, move M001
- **M001 move blocked:** `tools.roboflow_split --assign valid=TRAIN_M001_MORETON_CITY --apply` (permanently
  deletes and re-uploads 299 frames) was refused by the Mac's auto-mode safety check. It was not retried. Nathan
  runs it himself.
- **Exclusions applied:** 24 frames tagged `excluded:near-duplicate` (manifest
  `records/roboflow_splits/split_20261112T202247Z_apply_exclusions_only.json`, sha256 `caf4cdc6…`).
  - Tagging before the move is safe: the move re-uploads each frame with its current tags.
- **Full pre-label runs (M002 and M003, in parallel) hit Roboflow's rate limit:**
  - M002: 16 labelled, 254 failed. M003: 10 labelled, 279 failed. All failures were HTTP 429 on the platform API.
  - Because each frame is tagged before its labels are saved, failed frames got no tag and no annotation.
    Roboflow shows exactly 46 `prelabel:v1` frames (20 sample + 26), so no frame is half done and a re-run picks the
    rest up.
  - Logs `prelabel_20261112T202308Z.json` (sha256 `118cd406…`) and `prelabel_20261112T202318Z.json`
    (sha256 `cb1186fe…`).
  - Their `boxes_per_class` totals also count frames that were not saved.
- **Fixes:**
  - Every Roboflow platform call and upload retries HTTP 429, honouring Retry-After, else
    `rate_limit_wait_s` × attempt (`roboflow.rate_limit_retries` 6, `rate_limit_wait_s` 10).
  - Pre-labelling pauses `prelabel.min_interval_s` (1 s) between frames. Run one pre-label process at a time.
  - The summary counts boxes only on frames whose labels were saved.
- **Tests:** dataset tests 54 pass; full suite 180 passed, 3 skipped.

## 2026-11-13: full pre-label attempt at commit `cc17620` (Nathan: "go ahead")
- **Mac:** checked out `cc17620`; tests 179 passed, 4 skipped.
- **M002 full run started 08:42 and stalled before its first frame.**
  - No frame was tagged in 8 minutes (Roboflow still showed 46 `prelabel:v1`), so it was stopped at about 08:50.
  - A read-only search at 08:49 returned 200 with no rate-limit headers. The stall is therefore most likely in the
    429 back-off of the per-image tag call: Retry-After is capped at 300 s, so 6 retries can wait up to 30 min. This
    is not confirmed: the process prints nothing until it ends.
  - Afterwards Roboflow still showed exactly 46 `prelabel:v1` frames, so nothing was half-done.
- **One-frame diagnostic run** (`--limit 1`) was refused by the Mac's auto-mode safety check. Not retried.
- **Still to run on the Mac (by Nathan, or once the safety check allows it):**
  - full pre-label of M002 and M003, one at a time;
  - the M001 move;
  - then pre-label M001.

## 2026-11-13: pre-label v2 (Nathan: re-label v2 replacing v1)
Roles: Data / Licensing Steward; Player Detection Engineer; Lead Engineer; QA / Regression Engineer.
Branch `claude/prelabel-v2` (from `claude/nifty-carson-j51h2l`; `main` has none of the Roboflow tooling).

### Licence gate and the 20-frame stop (diagnosed before any change)
- **ROOT CAUSE (M002/M003 labelled, M001 not):** the gate worked as configured. `licensed_matches` lists M001, M002
  and M003, all `third_party_processing_confirmed: true` (commit `af4cbca`, Nathan confirmed 2026-11-12). Nathan
  re-confirmed on 2026-11-13 that the config is right. M001 was never selected because every run passed
  `--match TRAIN_M002…` or `--match TRAIN_M003…`: M001 was held back for its move to `valid`, which is still blocked.
- **ROOT CAUSE (did not stop at 20):** `prelabel.sample_size` only capped `--limit`; a bare `--confirm-full-run`
  bypassed it with no link to a reviewed sample. Full runs with that flag: 2026-11-12 20:23Z (M002, M003) and
  2026-11-13 04:57Z (PID 12236, `--all --confirm-full-run --match TRAIN_M002_EASTERN_SUBURBS`, still running when
  v2 work started; Nathan chose to let it finish). It explains 46 -> 196 `prelabel:v1` frames.
- **FAILING LAYER:** run control in `tools/roboflow_prelabel.py` `select()`; licence data, not gate code. Gap: the
  allow-list was checked at selection only, not again at each write.
- **FIX:** `spana/dataset/guard.py` `GuardedProject`: every write re-reads the image and re-checks `match_of` and the
  reviewed tag immediately before writing. Beyond `sample_size`: `--confirm-full-run` AND `--approved-sample <log>`
  of a written (not dry-run) v2 sample whose `prelabel_config_sha256` equals the current one.
- **Regression found while switching the tag to v2:** `roboflow_version` / `roboflow_progress` only knew
  `prelabel.tag`, so unreviewed `prelabel:v1` frames would have passed the version gate. Both now use
  `prelabel_tags()` (current + superseded) and `excluded_tags()` (near-duplicate + blur).

### Ball recall (20 v1 sample frames: M002 10, M003 10)
- `--diagnose-ball` (read-only), raw ball scores before any threshold, top 3 per frame and pass:
  `records/roboflow_prelabel/ball_diagnostic_20261113T052348Z.json` (sha256 `c73df0b7…`).
  - Full frame: best score median 0.35, 0 frames >= 0.85, 7 >= 0.5.
  - 640 px tiles: median 0.48, 0 >= 0.85, 9 >= 0.5. 320 px tiles: median 0.56, 2 >= 0.85, 13 >= 0.5.
- Visual check of the top candidates: the full-frame top candidate is the real ball in ~11/20 frames (scores
  0.32-0.79; mostly M003 daylight). M002 night frames: ball 5-8 px, rarely the top candidate at any scale. Tiles add
  ~0.05-0.15 to true balls and also raise false positives (broadcast logo overlay, 44 px, 0.46-0.67).
- **Verdict:** mainly a threshold problem (0.85 above every true-ball score), plus a resolution limit on small balls.
- **Nathan's choice:** ball threshold 0.50, ball-only tiling 640 px / 25 % overlap / NMS merge (IoU 0.30 or centre
  containment), ball boxes larger than 4 % of frame height dropped and logged. Player threshold unchanged (0.50).
  Dry-run preview (`prelabel_20261113T053035Z_dryrun.json`, sha256 `eb94a545…`, before the size cap): 11 balls in
  9 frames = 8 real, 2 logo (removed by the cap), 1 doubtful.

### Blur gate and grass filter
- Variance of Laplacian on a 640 px-wide grey copy. 900 cached TRAIN frames: p1 340, p5 448, p50 1098. Sharp
  low-texture frames score 220-260, so the prior is 200 (Nathan): catches strong blur only (1/20 samples, M002
  f17160); some panning blur (~417) passes.
- Grass mask (HSV range, closing, components >= 2 % of area, dilation 1 % of height). First dry run
  (`prelabel_20261113T052832Z_dryrun.json`, sha256 `cd7e45c0…`): 10 player boxes dropped; 6 spectators in the stand
  (right) and 4 occluded on-pitch players whose box bottom lay on another player's kit (wrong).
  - Fix: `grass.convex_hull: true` (each grass component filled to its hull). Offline re-check on the same boxes:
    the 4 occluded players kept, the 6 spectators still dropped. Regression test
    `GrassFilterTests.test_without_hull_the_occluded_player_would_be_lost`.
  - Limitation: bench / sideline staff standing on grass inside the hull are not removed by this rule.

### Tests
- `tests/test_roboflow_prelabel_v2.py` (27, mocked client): allow-list on every write path, reviewed frames,
  mid-run changes, stop at sample size and approval hash, dry run writes nothing, blur gate, grass filter on a
  synthetic frame, tile coverage and merge without duplicate balls, ball / player thresholds independent, size cap,
  backup before any change, failed read-back keeps `prelabel:v1`.
- Code review of the first v2 commit (5 findings, all fixed, each with a regression test in `ReviewFixTests`):
  the guard's refresh now also reads the annotation state (search omits it outside the Dataset), so a frame a human
  already labelled is never tagged; any unexpected per-frame error fails that frame only; a render error cannot fail
  a frame that was already written; the image is re-read immediately before its first write; migration requires
  `saved` and compares box positions on read-back; the ball size cap runs before the tile merge.
  Full suite: 212 passed, 4 skipped.

### 2026-11-13: v2 sample migration (20 frames) result
- Server state after the run (read back per image): 19 frames `prelabel:v2`, 1 frame `excluded:blur`
  (M002 f0017160, blur 151.68 < 200, 0 boxes), 0 `prelabel:v1`, 0 `reviewed`. 226 boxes = 217 player + 9 ball.
- Per match: M002 10 frames: 9 labelled, 1 excluded (blur), 106 players kept, 0 dropped off-pitch, 1 ball in 1 frame.
  M003 10 frames: 10 labelled, 111 players kept, 6 dropped off-pitch (spectators, f0012120), 8 balls in 8 frames,
  2 ball boxes dropped by the size cap (44-45 px broadcast logo).
- Logs: `prelabel_20261113T072316Z.json` (sha256 cdddc729255c55b6..., 17 labelled, 3 failed closed) and
  `prelabel_20261113T075828Z.json` (33dedd19f2e80ef6..., the 3 retried: 2 labelled, 1 excluded blur).
  Renders: `runs/roboflow_prelabel/20261113T072316Z/`, `.../20261113T075828Z/` (not committed).
- Backups: `backups/backup_20261113T054224Z.json` (a6a3440525e31fb5...) is the true v1 backup of all 20 frames
  (232 boxes). `072130Z`, `072316Z` and `075828Z` were written by later attempts; the first sample frame
  (M002 f0000000) already held v2 boxes under a v1 tag in those.
- Failures on the way, each failed closed (v1 tag kept) and has a regression test:
  1. ROOT CAUSE: tag writes returned HTTP 429 with Retry-After 1956 s; the wait was capped at a fixed 300 s, so
     every retry landed inside the window. LAYER: `spana/dataset/ratelimit.py`. FIX: cap from config
     `roboflow.rate_limit_max_wait_s` (3600); waits and per-frame progress are printed to stderr.
  2. ROOT CAUSE: pooled HTTP connections dropped by the server during the 31 min wait; the next request on each
     (project API, inference API) timed out. LAYER: HTTP clients. FIX: one re-send on ConnectionError.
  3. ROOT CAUSE: the annotate API rejects an empty Darknet TXT (HTTP 400 InvalidAnnotationFormat). LAYER:
     `_write_labels` migrate path. FIX: boxes are cleared with an object-less Pascal VOC file; verified on the
     server (0 boxes).
- Open: the tag-write quota that produced the 1956 s Retry-After is unknown; a larger run may hit it again.
  Not run: v2 on any frame outside the 20; the remaining 183 `prelabel:v1` frames are untouched.
  Full suite: 212 passed, 4 skipped.

### 2026-11-13: v2 migration of the remaining 183 `prelabel:v1` frames (approved by Nathan: "Migrate the 183")
- Command: `python -m tools.roboflow_prelabel --migrate --all --confirm-full-run --approved-sample
  records/roboflow_prelabel/prelabel_20261113T075828Z.json`. Selected 183 (only frames tagged `prelabel:v1`),
  183 labelled, 0 failed, 366 writes, no HTTP 429.
- Per match: M002 173 frames, 1991 players kept, 0 dropped off-pitch, 95 balls in 86 frames, 8 ball boxes dropped.
  M003 10 frames, 96 players kept, 20 dropped off-pitch, 7 balls in 7 frames, 1 ball box dropped.
- Read back per image (GET images/{id}) after the run: 183 / 183 tagged `prelabel:v2`, none `prelabel:v1`,
  2189 boxes, box count equal to the log for every frame. The search index still listed 4 M003 frames as
  `prelabel:v1` minutes after the run (index lag; the per-image read is authoritative).
- Project totals by direct read + listing: 202 `prelabel:v2` (M002 182, M003 20), 1 `excluded:blur`, 0 `reviewed`.
- Log `prelabel_20261113T082430Z.json` (sha256 9fca429118abbb9b...), backup
  `backups/backup_20261113T082430Z.json` (b41c5a001f19dd0a..., 183 frames, v1 annotations).
  Renders `runs/roboflow_prelabel/20261113T082430Z/` (not committed).
- Not checked visually: the 95 M002 ball boxes of this run (the sample found 1 ball in 9 M002 frames).

### 2026-11-13: visual check of the v2 ball boxes (contact sheets)
- `python -m tools.roboflow_ball_sheet` on the three v2 logs (offline; reads logs and the image cache only).
  Sheets in `runs/roboflow_prelabel/ball_sheets/` (not committed): M002 96 boxes, M003 15 boxes.
- M002, judged from the crops: 2 clear false positives (f0005880 drink-bottle crate 0.54; f0009960 a boot next to
  the real ball 0.53), about 10 doubtful (4-8 px, score 0.52-0.64, e.g. f0008880, f0010800 x2, f0011040, f0013440,
  f0019320, f0021720, f0021960), the remaining ~84 look like the match ball, including balls in the air against
  the dark background. M003: 14 of 15 look right; f0016140 (12x7 px white streak, 0.58) is doubtful.
- No threshold change: real balls also score 0.53-0.60, so raising the ball threshold would remove as many real
  balls as doubtful ones. The doubtful boxes are left for the human review.

### 2026-11-14: v2 sample on never-labelled frames (10 per match, approved by Nathan: "run the sample")
- `python -m tools.roboflow_prelabel --limit 10 --match <M002 | M003>` (evenly spaced over the eligible frames).
- M002 (`prelabel_20261113T175902Z.json`, sha256 ae2b8cc4cadb1a50...): 9 labelled, 1 no_boxes, 88 players kept,
  0 dropped off-pitch, 9 balls in 7 frames. f0034728 is a broadcast transition graphic (no play visible): it got
  no boxes and the `prelabel:v2` tag; there is no rule that excludes non-play frames.
- M003 (`prelabel_20261113T180008Z.json`, sha256 c4a412c29eaca50d...): 10 labelled, 106 players kept, 1 dropped off-pitch (a spectator,
  f0000552), 4 balls in 4 frames, 2 ball boxes dropped by the size cap (broadcast logo).
- Ball crops checked on contact sheets (`runs/roboflow_prelabel/ball_sheets/sample2/`): the 13 boxes look like
  footballs; M002 f0032280 has two (at least one is a spare ball beside the pitch).
- Seen on the M003 f0000552 render: some players are not boxed (below the 0.50 threshold or cut by the frame
  edge) and the referee is boxed as `player`; both are left for the human review.
- Still eligible (never pre-labelled, not excluded): M002 87, M003 269, M001 297 (M001 held until its move to
  `valid`). Not run beyond this sample.
