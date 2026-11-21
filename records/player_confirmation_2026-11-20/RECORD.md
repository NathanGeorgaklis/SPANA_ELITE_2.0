# Player confirmation resolves target identity episodes, DEV (2026-11-20 / 21)

CODE TASK (Nathan / chat), owner layer target identity. Branch `identity/player-confirmation`, base main `b758c0b`.
DEV FRESH_02 / 05 / 07 / 08, the same cached local-detector detections (`runs/person_view_compare/F0x_local.det.json`).
Tracker, linker, spawn / crossing rules, re-ID margin and `config/benchmark.yaml` unchanged. No EVAL, no full match.
Run on Nathan's Mac (`run_bench.sh`, outputs in `~/spana-player-confirmation-run`, summary `compare_summary.json`).

**Outcome.** Built as specified (event, IS_ME / NOT_ME, conflict rule, gallery, undo, logging, simulated reviewer),
with ONE deviation that needs Nathan's decision: an IS_ME answers only the frames the player confirmed
(`target.confirmation_scope: range`), not the whole episode. The episode-wide design fails the acceptance on DEV
(51 wrong frames with always-correct taps, section 2). With range scope and correct taps: silent 0, wrong 0 / 0,
0 transfers on linked tracks, reviewer correct 0.680 -> **0.870**, unanswered (ABSTAIN) frames 735 -> **295**, at
4.11 -> 7.52 actions per followed minute. The human-confirmed gallery let re-ID (on, separate run) clear **no**
episode, NO_REFERENCE ones included (section 5).

## 1. What was built (config 2.15.0, spana.target 2.10.0; `spana.tracks` contract unchanged)
- Event (`spana/identity/confirmation.py`, INTERFACES): `player_confirmation` 1.0.0 `{confirmation_id, track_id,
  frame, frame_end, verdict IS_ME|NOT_ME, source player}`, `revoke_confirmation` 1.0.0. A confirmation must name a
  track observed on its first frame, live throughout, in one uncertainty episode, else it is refused.
- IS_ME: the track becomes the answered track (like a tap), TARGET PLAYER_CONFIRMED on the confirmed frames.
  Only the target's document changes: tracks, rivals and other identities are untouched (by construction, and
  measured below).
- NOT_ME: that episode is removed from target candidates (ABSTAIN PLAYER_REJECTED, never offered, no gallery crop).
- Conflict (not applied, logged with evidence, `reconfirmation_list`): OTHER_IDENTITY_CERTAIN (another followed
  identity's TRACK_OBSERVED answer is on the track), TARGET_CERTAIN_ELSEWHERE, PLAYER_CONTRADICTION, REID_CONTRARY
  (every clean crop nearer another identity's reference than the target's by 0.517 = the calibrated re-ID margin),
  COLOUR_CONTRARY (0.30, a prior, not fitted). NOT_ME against the target's own CERTAIN track: CONFLICT, the target
  abstains there (CONFIRMATION_CONFLICT).
- Gallery: clean crops (`spana.cli crops`, CropFeatures: observed, uncontested, no other detection above
  reid.clean_max_iou, >= 40 px) of an applied IS_ME's frames, tagged with the confirmation id.
- Gallery re-ID (`target.gallery_reid_resolve`, needs `reid_resolve.enabled`; both OFF by default): later episodes of
  the answered track, per frame vs the latest gallery crop before the episode and each rival's latest clean CERTAIN
  crop; run rules of reid_resolve (margin 0.517, identity_resolve_s, min_evidence_frames 3); after the run qualifies,
  each clearing frame is answered (GALLERY_REID); nothing carries past it.
- Undo: revoked confirmations are ignored; every output recomputes. Logged (status REVOKED, with the revocation).
- Simulated reviewer (`tools/target_score.py --confirmations`): a tap on an IDENTITY_UNCERTAIN track issues IS_ME over
  the following labelled frames of the gap where the track is still the player (truth on labelled frames: the
  best-case reviewer convention, as before); wrong taps from `config/review_sim.yaml` (seeded, the tapped track's
  rivals first, else the nearest other labelled player); claims = the other players' reviewer runs without
  confirmations; a CONFLICT is retried once.
- Bug found and fixed while running: `CropFeatures.load` re-read the npz array per crop and kept views alive (1.7 GB,
  54 s for FRESH_08 -> 178 MB, 4 s, identical results); the first run was killed by memory (exit 137).

## 2. Why range scope: the episode-wide design shows the wrong player (DEV, always-correct taps)

| scope | reviewer correct | wrong frames | unverified | ABSTAIN frames | actions / min |
|---|---|---|---|---|---|
| episode (as specified, `confirmation_scope: episode`) | 0.865 | **51** | 70 | 232 | 7.10 |
| range (default) | **0.870** | **0** | 32 | 295 | 7.52 |

Every confirmation was right on its frame; the wrong frames come later in the same episode:
- A track changes player INSIDE one uncertain episode without any new event in the tracks document: FRESH_02
  T000048 (ambiguous spawn, rivals T000014 / T000036) carries player 9, then player 10 from frame 276 (17 wrong);
  T000033 (player 10 -> 8, frame 276); FRESH_07 T000034 (player 18 -> 17 after repeated contests).
- The tracker's later resolution re-confirms the PRE-episode player: FRESH_07 T000018 was player 14, crossed with
  15, the player confirmed 15 correctly during the episode, colour resolved the track back to 14 (frame 82) and the
  episode-wide answer followed it (28 wrong frames, player 15).
Range scope: after the confirmed frames the track is uncertain for the target again; if it was still uncertain at the
range's end the episode taints the track as any episode does (resume_after_uncertain false). A range ending on a
CERTAIN frame continues like a tap on a certain track.

## 3. DEV results (3,449 labelled player-frames, 14.36 followed min)

| run | silent | wrong | transfers on linked tracks | correct | ABSTAIN frames | actions / min |
|---|---|---|---|---|---|---|
| unaided (no actions) | 0 | 0 | 0 | 0.508 | 513 | 0 |
| reviewer, taps only (main) | 0 | 0 | 0 | 0.680 | 735 | 4.11 |
| **reviewer + confirmations, always correct** | **0** | **0** | **0** | **0.870** | **295** | 7.52 |
| linked tracks: reviewer / + confirmations | 0 | 0 / 0 | 0 | 0.681 / 0.870 | 733 / 293 | 3.62 / 6.90 |
| + confirmations, 5 % wrong taps | 0 | 1 | 0 | 0.870 | 294 | 7.59 |
| + confirmations, 30 % wrong taps (stress) | 0 | 27 | 0 | 0.820 | 345 | 7.80 |

Tracking unchanged: 7 transfers, all flagged, 0 silent; fragments 101 / ID switches 103 (linked 92 / 94; 0 on linked
tracks). Per clip (reviewer -> + confirmations): F02 0.695 -> 0.854, F05 0.964 -> 0.964, F07 0.685 -> 0.920,
F08 0.487 -> 0.758. 70 confirmations, all applied, 0 conflicts, 2,309 gallery crops.
Effort rises (4.11 -> 7.52 actions / min): a confirmed range ends where the reviewer can still vouch, and the episode
then opens a new gap that the reviewer works through. Answers on unlabelled frames inside a range are unscored.

## 4. Wrong taps: blocked vs through, containment
- 5 % (`config/review_sim.yaml`): only 1 of 71 taps drawn wrong (a wrong tap needs another labelled player's
  observed track on that frame). FRESH_07 player 9, frame 78, track T000023 (player 8): **through** - the track was
  itself uncertain (rivals T000005 / T000013 / T000016), no identity held it CERTAIN, no clean crop at frame 78, so
  no evidence existed. 1 wrong frame (the range of a wrong tap is its frame only).
- 30 % stress (`review_sim_stress.yaml`): 22 wrong taps, **8 blocked** (OTHER_IDENTITY_CERTAIN 5 confirmations,
  COLOUR_CONTRARY 3, REID_CONTRARY 1 alongside an OTHER_IDENTITY_CERTAIN), **14 through** (27 wrong frames). 3 correct
  taps blocked (FRESH_02 player 16, COLOUR_CONTRARY): that player's earlier wrong tap (frame 246, track of player
  17) went through and became the target's own appearance reference.
- **Containment:** in both runs every player whose answers differ from the always-correct run made a wrong tap
  of their own (5 %: 1 / 1; 30 %: 13 changed / 14 with wrong taps; 0 changed without one). The tracks are never written;
  no other identity changed.

## 5. Does the human-confirmed gallery let re-ID clear episodes? (separate run, reid_resolve + gallery re-ID on)
- Tracker with reid_resolve on: same 126 episodes as `records/reid_episode_resolution_2026-11-20` (resolved REID 2,
  APPEARANCE 12; refused NO_REFERENCE 49, NO_CLEAN_CROP 29, NOT_OBSERVED 24, BELOW_MARGIN 7, RUN_TOO_SHORT 2,
  AGAINST 1); tracks 7 / 0 silent; reviewer 0.680, + confirmations 0.870 / 0 wrong (identical to off).
- Gallery re-ID evaluated 30 later episodes of answered tracks (with a gallery): **0 cleared, 0 frames answered**,
  0 against. Main hold per episode: NO_CLEAN_CROP 15, no evaluable frame 6, NO_GALLERY_REFERENCE 5 (gallery crops
  only from inside the episode), NO_RIVAL 2, NO_RIVAL_REFERENCE 1, BELOW_MARGIN 1. **No NO_REFERENCE episode was
  cleared**: the gallery supplies the target's own reference, but the episodes still lack a clean crop or a
  reference for the rivals / other predecessors, which a human gallery does not provide.

## 6. Undo (confirm then revoke == no confirmation, every output: frames, gaps, relinks, track ends, re-entry log,
gallery, gallery re-ID log)
- Confirmations alone + their revocations vs no action: **30 / 30 players equal** (every run).
- The full reviewer action list + revocations vs the list without the confirmations: 29 / 30 equal; 1 not
  comparable (a later action applies only because of a confirmation, so the list without it is refused), 0 unequal.
- Unit tests: `tests/test_player_confirmation.py` (24, incl. undo with taps around it).

## 7. Remaining
- Nathan: accept range scope (the deviation) or not. Episode scope stays available for measurement only.
- Effort (7.5 actions / min) is now the cost: a range-length UI (confirm a stretch, not a frame) is the lever.
- Gallery re-ID gives nothing on DEV; stays off. It needs clean crops and rival references, as `reid_resolve`.
- COLOUR_CONTRARY 0.30 is a prior; with a polluted target reference it blocks correct taps (section 4).
