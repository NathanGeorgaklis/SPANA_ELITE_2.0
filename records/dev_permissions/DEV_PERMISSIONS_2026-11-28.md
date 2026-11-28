# DEV matches: permission as stated by Nathan (2026-11-28)

Nathan gave these statements in the session of 2026-11-28, after saving two more full match videos while collecting
footage for detector retraining (`records/detector_retrain_task2_2026-11-25`). The written permissions themselves are
held by Nathan, not in this repository.

| match_id | Match | Rights holder | Permission |
|---|---|---|---|
| DEV_G4_MACARTHUR_RAMS_WESTERN_RANGERS | Macarthur Rams v Western Rangers. A NEW match: not Game 1, 2 or 3 (Nathan) | Macarthur Rams / Western Rangers | written permission dated 22 November 2026 |
| DEV_G2_NWS_SPIRIT_SD_RAIDERS | NWS Spirit v SD Raiders (Game 2, already on the allow-list) | NWS Spirit / SD Raiders | written permission dated 22 November 2026 (a second statement; the first is dated 31 October 2026) |

For both, as stated by Nathan:
- Consent basis for players filmed: authorised participation in the football match and written permission covering
  use of the footage and player imagery / data for SPANA's AI/ML development and evaluation.
- SPANA Sports Pty Ltd may use the match footage as DEV data for computer-vision development, testing, validation and
  benchmarking, including player detection, tracking and re-identification research.
- Frames and derived images may be uploaded to authorised third-party processing services where required.
- No minors appear in the footage.
- **DEV-only: must not be used as TRAIN or EVAL data unless separately authorised.**

Consequences:
- Neither match is mined, labelled or uploaded for detector retraining. Nathan had saved them as "M004" and "M005"
  next to the TRAIN matches; they are not TRAIN matches and get no TRAIN_ id.
- DEV_G4 is added to `licensed_matches` in `config/roboflow.yaml` with split `dev`. No clip of it has been used for
  anything, so it can supply unseen clips for the retraining plan's final check (task 6).
- For Game 2 the statement of 31 October 2026 (`DEV_PERMISSIONS_2026-11-14.md`) says ML training is permitted and
  this one says DEV-only. The project treats Game 2 as DEV either way; if training on it were ever proposed, the
  stricter statement applies until Nathan says otherwise.

## Correction, later on 2026-11-28: training IS permitted for both (replaces "DEV-only" above)

Nathan then gave a fuller statement for each match, and, asked directly whether the signed permission dated
22 November 2026 allows training and the "DEV-only" wording was a mistake, answered "yes". As stated by Nathan, for
both Macarthur Rams v Western Rangers and NWS Spirit v SD Raiders:
- Consent basis: authorised participation in the match and written permission covering use of the footage and player
  imagery / data for SPANA's AI/ML development, training, testing and validation.
- The footage may be used for computer-vision development, including TRAIN and DEV purposes.
- Permitted uses include extracting and storing frames and player crops; annotations and labels; training,
  retraining and fine-tuning detection, tracking and re-identification models; benchmarking and validation; derived
  datasets, embeddings and training examples; and use of the resulting models in SPANA's commercial products.
- Frames and derived images may be uploaded to authorised third-party processing, annotation, cloud-compute and
  machine-learning services where required.
- No minors appear in the footage. ML training and retraining are expressly permitted.
- The footage may be assigned to TRAIN or DEV as SPANA's dataset protocol requires. It must not be used as EVAL data
  unless designated as such before any model development, training or tuning is performed against it.

What the project does with that (permission allows a use; the dataset protocol decides it):
- **Game 2 (NWS Spirit v SD Raiders) stays DEV.** FRESH_03 to FRESH_06 are scored DEV clips; no Game 2 frame enters
  any training or validation split.
- **Macarthur Rams v Western Rangers is not yet assigned.** It stays on the allow-list as `dev` (id DEV_G4_...), and
  nothing is run on it, until Nathan decides after the full-broadcast mining of TRAIN_M001 to M003: either it stays
  an unseen DEV match (task 6), or it becomes a TRAIN match under a new TRAIN_ id, which is one-way.
