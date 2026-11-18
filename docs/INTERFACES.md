# INTERFACES

Taken from the code as of 2026-11-14. Records: `spana/records.py`. Artifacts: `spana/artifacts.py`
(canonical JSON, validated, atomic, never overwritten; any change to keys/types/units/meaning bumps the version).

## Video input — `spana/video.py`, `spana/provenance.py`
- In: video path (optional `--video-sha256` guard). Out: `VideoMeta` (incl. sha256, PTS timing) + BGR frames.
- Provenance block in every artifact: git state, package versions, config sha256, input hashes, argv.

## Player detection — `spana/detection/`
- Interfaces (`base.py`): `PlayerDetector.detect(frame_bgr) -> List[Detection]` (persons only);
  `Detector.detect(frame_bgr) -> List[Detection]` + `class_names` (generic multi-class, e.g. Roboflow backend).
- `Detection`: `box` (x1,y1,x2,y2 source px), `score`, `class_id`, `multi_body`.
- Artifact out: **`spana.detections` 1.4.0** — top level `schema, schema_version, provenance, video, model,
  class_names, detection_config, frame_stride, frames`; frame `{frame, t, detections}`; row
  `[x1, y1, x2, y2, score, class_id, multi_body(0/1)]`; `class_id` indexes `class_names` (1.4.0 added it; 1.2.0 /
  1.3.0 person-only artifacts migrate with `tools/refilter_detections.py`, names rebuilt from `model.person_class_id`).
- CLI: `python -m spana.cli detect --video --config --model-record --out [--start --stop --stride]` (onnx, person
  only); `--backend roboflow --dataset-config --match-id` (football detector as a person view; hosted inference, so
  the match must be an allow-listed DEV match with third-party processing confirmed; EVAL refused); `--backend local
  --model-record models/rfdetr_v1.json --dataset-config --runtime {torch,onnx} --device {mps,cpu}` (the same football
  detector run on this machine, `spana/detection/rfdetr_local.py`; weights / ONNX sha256-checked; frames stay local).

## Detection -> tracking (defined 2026-11-14; wired 2026-11-16)
What the tracking layer receives from any detector, the football detector included:
- One record per processed frame: `frame` (int, source video frame index, strictly increasing), `t` (seconds,
  PTS), `detections`. This is the `spana.detections` 1.4.0 frame built by `spana.artifacts.detection_frame`.
- Persons only (`PlayerDetector`). One row per detection: `[x1, y1, x2, y2, score, class_id, multi_body]`. Box in
  source pixels, inside the frame, `x2 > x1`, `y2 > y1`. Score in [0, 1]. `class_id` names the box's class via the
  artifact's `class_names`; for the football detector it is a role hint only (player / goalkeeper / referee), which
  the tracker does not use. Roles come from a later layer (decision 2026-11-16).
- Football detector -> persons: `spana.detection.person_view.PersonView` wraps `RoboflowDetector`:
  1. keeps `detector.person_classes` (`config/roboflow.yaml`) and drops the ball;
  2. applies `detection.score_low` and `min_box_height_frac`;
  3. merges duplicates across classes with class-agnostic NMS at `detection.nms_iou`, keeping the higher-scoring box;
  4. sets `multi_body` from `detection.multibody_*`.
  The artifact records `model.person_view`.
- Thresholds: one source, default.yaml `detection`:
  - The football detector is asked for boxes down to `detection.score_low` (0.10). PersonView refuses a detector
    whose request confidence is higher.
  - The tracker uses `score_low` / `score_high` as for YOLOX.
  - `detector.confidence` (0.50, roboflow.yaml) stays the operating point for detector evaluation only.
- `cmd_track` checks the artifact's `detection_config` against default.yaml `detection` for both backends.
  The roboflow backend does not use the tiling and union keys in that section.
- Checked by `tests/test_detection_interface.py` and `tests/test_person_view.py` (including detect -> track on a
  synthetic clip and the licence gate).

## Player tracking — `spana/tracking/`
- In: `spana.detections` (must match video sha256 and current `detection_config`) + video frames.
- Artifact out: **`spana.tracks` 2.2.0** — top level `schema, schema_version, provenance, video,
  detections_sha256, frames, tracklets`; frame `{frame, t, camera{status, affine, inliers, hist_distance},
  tracks[], ended[[track_id, EndReason]]}`; track keys: `track_id, state, confirmed, source, box,
  detection_index, detection_score, confidence, contested, age_frames, missed_frames, time_since_observed_s,
  observed_frames, epoch, occluder, max_overlap_iou, overlap_with, recoveries, last_gap_s,
  possible_predecessors, identity, identity_rivals, uncertain_episodes, identity_doubt`; tracklet keys: `track_id, epoch,
  first_frame, last_frame, last_observed_frame, observed_frames, confirmed, end_reason, recoveries,
  possible_predecessors`.
  - 2.2.0: `identity_doubt` (bool) = the track continued through an appearance veto on an exclusive motion link
    (`tracking.appearance_doubt_link`, decision `doubt_link`). `identity` is CERTAIN only with no
    `identity_rivals` and no `identity_doubt`, so IDENTITY_UNCERTAIN may have no rivals. 2.1.0 documents still
    validate with their 2.1.0 meaning (no `identity_doubt` key, uncertain <=> rivals).
- Side output: decision log `*.decisions.jsonl` (every association, refusal, spawn, end).
- CLI: `python -m spana.cli track --video --config --detections --out`.

### State enums (`spana/records.py`)
| enum | values |
|---|---|
| `TrackState` | ACTIVE, OCCLUDED, LOST, TERMINATED (only via `ended`) |
| `IdentityState` (per track, orthogonal to TrackState) | CERTAIN, IDENTITY_UNCERTAIN |
| `PositionSource` | OBSERVED (ACTIVE only), PREDICTED, HELD (OCCLUDED/LOST only) |
| `CameraStatus` | OK, UNKNOWN, CUT, FIRST |
| `EndReason` | UNCONFIRMED_MISSED, UNCONFIRMED_TIMEOUT, LOST_EXPIRED, CAMERA_CUT, CAMERA_MOTION_UNKNOWN, LEFT_FRAME |
| `TargetState` (target identity) | TARGET, CANDIDATE (2.0.0: proposed relink awaiting confirmation), LOST, ABSTAIN |
| `TargetReason` (target identity) | TRACK_OBSERVED, NOT_CONFIRMED, IDENTITY_UNCERTAIN, AFTER_UNCERTAIN, TRACK_OCCLUDED, TRACK_LOST, TRACK_ENDED, RELINK_PREDECESSOR (2.0.0), USER_VOUCHED (2.2.0) |

## Target identity — `spana/identity/` (defined 2026-11-16)
Follow the user's chosen player through a clip, and say so when unsure.
- In: a validated `spana.tracks` artifact + the user's selection.
- Selection (versioned; new kinds are added, never changed): `{kind, version, frame, track_id}`.
  - `track_at_frame` 1.0.0 = a tap on a box the app showed: the track must be live and ACTIVE (observed) on that
    frame. Otherwise `SelectionError` (refused, never guessed).
  - Tap is the chosen way to pick (decision 2026-11-16). Shirt number / photo may come later as hints.
- Out: **`spana.target` 2.3.0** (`spana/artifacts.py` `validate_target`; canonical, atomic, never overwritten).
  2.2.0, 2.1.0, 2.0.0 and 1.0.0 documents still validate with their own meaning.
  - 2.2.0: no `target_config.resume_after_doubt_resolved`.
  - 2.1.0: no `target_config.tap_vouches_doubt`, no USER_VOUCHED.
  - 2.0.0: `confirmations` instead of `user_actions`, no `gaps`, no `rejection` in relinks.
  - 1.0.0: TARGET only on the selected track, `track_end`, `track_id` null without an answer, CANDIDATE refused.
  - Top level: `schema, schema_version, provenance, video, tracks_sha256, selection, user_actions, target_config,
    track_ends, relinks, frames, gaps`.
  - `target_config`: `{resume_after_uncertain, relink_max_gap_s, relink_auto_confirm, tap_vouches_doubt,
    resume_after_doubt_resolved}` from default.yaml `target`.
  - `user_actions` (2.1.0; decisions 2026-11-16: non-blocking offer, tap to re-select, after-match review).
    Versioned records, each applied on its own frame and refused (`UserActionError`) if it does not apply. At most
    one track-changing action (`confirm_relink` / `track_at_frame`) per frame.

| action | fields | effect |
|---|---|---|
| `confirm_candidate` 1.0.0 | `frame, track_id` | accept the CANDIDATE shown on that frame |
| `confirm_relink` 1.0.0 | `frame, from_track, to_track, proposed_frame` | accept an offer by its relink entry, also after it was dropped (the offered track must be live on that frame); TARGET on it from that frame on |
| `reject_candidate` 1.0.0 | `frame, from_track, to_track, proposed_frame` | decline an offer: withdrawn (`REJECTED_BY_USER`), never re-proposed for that loss |
| `track_at_frame` 1.0.0 | `frame, track_id` | a tap on an observed box mid-clip: re-selects the player from that frame on; earlier answers kept |

  - Review list (not part of the document): `spana.identity.follower.review_list(gaps, cfg)` returns the gaps
    lasting at least `target.review_min_gap_s` (`gap_duration_s` = end_t - start_t); 0 lists every gap.
  - `gaps` (2.1.0): the review timeline. Maximal runs of frames without a TARGET answer:
    `{start_frame, end_frame, start_t, end_t, frames, states, offers: [[from_track, to_track, proposed_frame]]}`.
    The validator recomputes them from the frames.
  - `track_ends`: `{track_id, frame, reason: EndReason}` for each answered track that ended.
  - `relinks`: one entry per proposal: `from_track, to_track, lost_frame` (from_track's last observation),
    `proposed_frame, gap_s, evidence {possible_predecessors, candidates}, confirmation` (null, the user's record,
    or `{kind: auto_confirm, frame}`), `rejection` (2.1.0: null or the user's record), `outcome`
    (PENDING / CONFIRMED / DROPPED) and `dropped_reason` (kept when an offer is confirmed after it was dropped).
  - Frame `{frame, t, state, reason, track_id, box}`, one per tracks frame from the selection frame to the end.
    `track_id` always names the track the answer refers to; `box` is set exactly for TARGET and CANDIDATE.
  - Validated against the tracks artifact:
    - same frames;
    - TARGET / CANDIDATE boxes are those of an ACTIVE, confirmed, CERTAIN track, except TARGET USER_VOUCHED (2.2.0):
      an ACTIVE, confirmed track uncertain only by an appearance doubt (tracks `identity_doubt`, no rivals), whose
      latest tap (selection / re-selection) is on it, made during the doubt, with no rival and the same
      `uncertain_episodes` since, and `tap_vouches_doubt` true;
    - TARGET only on the selection, a re-selection or a relink confirmed at or before that frame;
    - every re-selection is an observed box on its frame;
    - every relink supported by the tracks: the new track names the lost one, it is the only such track born in
      the window, and the gap is within `relink_max_gap_s`.
- Follower (`spana/identity/follower.py`), per frame, on the ANSWERED track (the selection, or a confirmed relink):

| state | reason | when |
|---|---|---|
| TARGET | TRACK_OBSERVED | answered track ACTIVE, confirmed, CERTAIN |
| TARGET | USER_VOUCHED | ... ACTIVE, confirmed, uncertain only by an appearance doubt that the user tapped it during (`target.tap_vouches_doubt`; ends on a rival or a new episode) |
| ABSTAIN | NOT_CONFIRMED | ... not yet confirmed |
| ABSTAIN | IDENTITY_UNCERTAIN | ... IDENTITY_UNCERTAIN now |
| ABSTAIN | AFTER_UNCERTAIN | ... was uncertain since it became the answered track and `target.resume_after_uncertain` is false (2.3.0: an episode that was only ever a no-rival doubt does not count when `target.resume_after_doubt_resolved` is true) |
| ABSTAIN | TRACK_OCCLUDED | ... OCCLUDED |
| LOST | TRACK_LOST | ... LOST |
| LOST | TRACK_ENDED | ... ended (any EndReason incl. CAMERA_CUT) |
| CANDIDATE | RELINK_PREDECESSOR | proposed relink, awaiting the user's confirmation (below) |

  - Never answers before the selection frame.
  - A gap the tracker bridged on the same track id is followed again (the tracker's decision, not this layer's).
  - **Relink (decision 2026-11-16).** Only the tracker's unique predecessor link (DEV 44 / 44). It applies while
    the answered track A is not observed, and only if A was confirmed, CERTAIN and untainted at its last
    observation.
    - B is proposed as CANDIDATE while all hold:
      - B is the only track born within `target.relink_max_gap_s` after A's last observation whose
        `possible_predecessors` names A;
      - B is ACTIVE, confirmed and CERTAIN;
      - no CUT / UNKNOWN camera frame since A's last observation.
    - The candidate is dropped (no further proposal for that loss) when A is observed again, a second such track
      appears, a cut follows, or B stops qualifying.
    - The user's confirmation makes B the answered track (TARGET from that frame). Losses can repeat (chains).
    - `target.relink_auto_confirm` (default false) promotes immediately; for measurement only.
- CLI: `python -m spana.cli target --tracks --config --select-track --select-frame [--actions FILE] --out`
  (`--confirm` kept as an alias).
- DEV scorer: `tools/target_score.py`. Per labelled player: selected on the first labelled frame with an
  unambiguous NESTED_V2 track match.
  - Reports correct / wrong / unverified / candidate (correct, wrong, unverified) / abstain / lost time, per player
    and pooled per clip.
  - Relinks are counted as proposed / right / wrong / undecided / confirmed / dropped.
  - `--simulate-user` confirms every candidate judged right.
  - `--simulate-reviewer` works through the gap list. Per gap: confirm a right offer or reject a wrong one,
    otherwise tap the player on the gap's first labelled frame. Reports the actions taken (effort).

- Re-ID headroom (analysis tool, not product code): `tools/reid_headroom.py`.
  - Per labelled player, handovers between their tracks: LOSS (old track gone before the new one was born) /
    OVERLAP.
  - Ranks and picks of three clues among the tracks born meanwhile: tracker `possible_predecessors`,
    camera-compensated foot position, torso colour similarity.
  - Simple relink rules (picks / correct / wrong) and the follower's ceiling with perfect relinking.

- Overlap-guard cost (analysis tool): `tools/guard_cost.py`.
  - What a TARGET → ABSTAIN overlap guard would cost (correct answers lost) and gain (wrong answers avoided) per
    (iou, min_s, hold_s) setting.
  - Sliced into all, contested and near-ball moments.
  - Ball positions from `tools/ball_pass.py` (local detector, one class, schema `spana.class_pass` 1.0.0; DEV
    analysis only).

## Evaluation (supporting) — `spana/evaluation.py`, `spana/labels.py`, `tools/benchmark.py`
- In: tracks + detections artifacts, frozen MOT labels (`data/labels/<clip>/v1`). Out: `Report` metrics JSON.
- Match rules: IoU 0.5, NESTED_V1, NESTED_V2 (default).

## Re-ID labels (supporting) — `spana/reid_labels.py`
- Schemas `spana.reid_match` 1.1.0, `spana.reid_identities` 1.0.0. Splits TRAIN / DEV / EVAL (EVAL needs approval).
- Side labels: A, B, GK_A, GK_B, REF, OTHER (labels only; no classifier consumes them).

## Detection dataset (supporting) — `spana/dataset/`
- Roboflow splits train/valid/test; SPANA splits train/dev/eval (eval never uploaded).
- Upload manifest frame statuses: uploaded, duplicate, failed, not_uploaded.
- DEV benchmark manifest **`spana.dev_bench` 1.0.0** (`tools/dev_bench.py`, `records/dev_bench/`): match, project,
  bench / prelabel tags, detector model id + thresholds, rules, config sha256s, clips (file name, sha256), per-frame
  clip index, frame index, time, image sha256, blur, pHash, upload id, status, pre-label class counts, error.
  Benchmark frames live in `dev_bench.project` only and carry `match:<id>`, `split:dev`, `dev_bench.bench_tag`.

## Not yet defined
| layer | inputs / outputs | state enum |
|---|---|---|
| ball tracking | NOT DEFINED | NOT DEFINED |
| team classification | NOT DEFINED | NOT DEFINED |
| pitch context | NOT DEFINED | NOT DEFINED |
| possession/touches | NOT DEFINED | NOT DEFINED |
| event segmentation | NOT DEFINED | NOT DEFINED |
| event classification | NOT DEFINED | NOT DEFINED |
| outcomes | NOT DEFINED | NOT DEFINED |
| statistics | NOT DEFINED | NOT DEFINED |
| clips | NOT DEFINED | NOT DEFINED |
