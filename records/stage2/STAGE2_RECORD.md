# Stage 2: player detection + multi-player tracking foundation (2026-11-07)

Spec: Nathan 2026-11-07 20:20. Roles: Lead Engineer; Player Detection Engineer; Multi-Player Tracking Engineer;
Failure Investigator; QA / Regression Engineer (independent reviewer agent; its attack scripts are in
`qa_attacks/`).

## Architecture
`video → detector (full frame + tiles, NMS) → detections artifact → camera motion + appearance descriptor →
multi-player tracker → tracks artifact (2.0.0) + decision log + tracklet summaries`. Detection and tracking
share no code and are joined only by the versioned artifact.

## Experiments (negative results included)

| # | Change | Observed | Decision |
|---|---|---|---|
| 1 | Occlusion state not sticky | A re-emerging player is missed for a frame or two, flips to LOST, then splits | Sticky OCCLUDED |
| 2 | Sticky OCCLUDED alone | **Transfer**: a same-looking third player emerging 1.5 h away is recovered as the hidden player | Rejected |
| 3 | Re-emergence must be next to the occluder (0.5 h) | Too blunt: at 10 fps, or after detector misses, the true player is already 1.1 h away | Rejected |
| 4 | Recovery must fit the hidden player's own motion (acceleration bound) | Exp. 2 fixed, exp. 3 recovered | Kept |
| 5 | QA review of `3d68382` | H1: continuing matches give a track to a neighbour when the player's own detection drops (130 crowd transfers). H2: unknown appearance passes short-gap recovery. H4: a shifted partial box becomes a 2nd confirmed track. H5: validator gaps. M1: confirmation depends on fps | **Rejected** |
| 6 | Appearance veto in every stage; hidden-rival refusal; motion bound on all recoveries; nesting blocks confirmation; time-based unconfirmed gaps | Crowd transfers 130 → 26, fps dependence gone | Kept |
| 7 | Hidden-rival refusal when costs tie and appearance does not favour the continuing track | front_swap fixed; **fragmentation explodes at 60/120 fps** (refusal repeats every frame) | Refusal kept |
| 8 | Skip a track's own recorded possible predecessors in the hidden-rival check | Churn gone; fragmentation stable across fps | Kept |
| 9 | Kalman (Mahalanobis) gate on top of the max-speed gate | Crowd transfers 26 → 6 | Kept |
| 10 | Every unobserved confirmed track counts as a hidden rival (not only recoverable ones) | Front-swap transfer at a 0.24 s gap fixed | Kept |
| 11 | Refuse every contested match of an unconfirmed track | **21 fragments** for two players walking side by side | Rejected |
| 12 | Synthetic harness applies the detector's NMS | Removed impossible inputs (two boxes at IoU 0.75); a few more realistic merged-box cases at 10/25 fps | Kept (test realism) |

## Results (synthetic + one real-model decoding check; NO real football footage)
- 61 tests pass (`TEST_OUTPUT.txt`).
- `QA_SCENARIOS.txt` (13 scenarios × 10/25/60 fps):
  - **0 unflagged transfers, 0 wrong recoveries, 0 duplicate frames**;
  - 1 flagged transfer (15-player crowd at 60 fps);
  - fragments only where continuity is unverifiable (same-look occlusion/crossing: 1–2; crowds: 4–20).
- `QA_ATTACKS_UNFLAGGED.txt`, the reviewer's adversarial set, 168 runs:
  - 29 transfers, 15 of them unflagged;
  - 11 wrong recoveries;
  - 0 duplicates.
- YOLOX-s decode reproduces YOLOX's published demo result (bicycle 0.955, dog 0.913, truck 0.612).
- Speed on 2 CPUs at 1080p: 0.26 s/frame without tiles, 2.16 s/frame with tiles.
- Config sha256 `46230fe3…195c45`.

## Known failures (measured)
1. **Birth-time crossover at low fps.** Two players first detected within ~0.5 h of each other at 10 fps can be
   swapped on their second frame, before any velocity is known. 13 of the 15 unflagged attack transfers are this
   class (crowd seeds at 10/25 fps; `margin_gone`); there are none at 60/120 fps. Fix path: an appearance (ReID)
   embedding that discriminates same-kit players, or the identity layer.
2. **Never-tracked look-alike on a hidden player's path** (`third_on_path_*`): it is recovered as the hidden
   player (11 wrong recoveries). This is indistinguishable with local evidence. These recoveries are always
   flagged (`recoveries > 0`), so the identity layer must verify them.
3. **Merged box of two same-looking players** right after an ambiguous spawn: a transfer can follow, but only on
   a track already flagged with `possible_predecessors`.
4. **Fragmentation by design:** same-look occlusions and crossings, unknown camera motion (new epoch) and cuts
   all split tracks; the identity layer re-links them.
5. **A real player mostly behind another** (≥ 85 % of his box inside it) is never spawned, so his coverage is lost.
6. Appearance descriptor: a torso hue-saturation histogram. It cannot separate same-kit players, and players
   under 24 px have none.
7. Runtime: 2.2 s/frame with tiling on CPU, so ~80 h for a 90-minute match. A GPU and/or frame striding is
   needed before production use.
8. All thresholds are uncalibrated priors. They need DEV footage with human labels; EVAL stays untouched.

## Persistence
Git: github.com/NathanGeorgaklis/SPANA_ELITE_2.0 `main`. Weights are not in Git: `tools/fetch_model.py` downloads
them and verifies the sha256. Not configured: Mac snapshot archive, second independent backup.
