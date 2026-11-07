# Stage 1: foundation, player detection, multi-player tracking (2026-11-07)

Roles: Lead Engineer; Player Detection Engineer; Multi-Player Tracking Engineer; Failure Investigator;
QA / Regression Engineer (independent reviewer agent).

## Scope
Config, video input, provenance, the detector interface plus a pinned ONNX backend, the generic multi-player
tracker, and the artifacts `spana.detections` / `spana.tracks` 1.1.0. Nothing downstream of tracking yet.

## Experiments and decisions (negative results included)

| # | Hypothesis / change | Observed | Decision |
|---|---|---|---|
| 1 | "Contested" should fire at a two-player crossing | It did not fire: the motion model separates them (cost 0 vs 0.24). The test had conflated association ambiguity with physical overlap | Keep `contested` for association ambiguity; add separate `max_overlap_iou` / `overlap_with` |
| 2 | With UNKNOWN camera motion, keep associating without compensation | **False continuity**: under a 0.75 h/frame pan a neighbour lands on the old prediction | UNKNOWN motion ends continuity (new epoch), same as a cut |
| 3 | v1 (`1e6bc49`): LOST tracks re-match in stage 1 with a gate growing to 2 h | QA H1–H3: **false continuity** for a new player near a LOST track, a player crossing an occluded player's spot, and a LOST track taking a continuing player's detection | **Rejected.** Continuing tracks are matched first. Re-matching after a gap is only allowed within `rematch_max_s` and only for exclusive pairs. The distance cost is in absolute body heights |
| 4 | Fix 3 | Every QA scenario: no false continuity at 10/25/120 fps. The new regressions fail on v1 (4 of 5) | Accepted |

## Results (synthetic only; no real footage, no model pinned)
- 47 tests pass (`TEST_OUTPUT.txt`): unit, synthetic integration (crossing, occlusion, pass-through, pan, zoom,
  cut, unknown motion, score flicker, fps 10–120, scale 0.5–2x, determinism), validator mutations, and an
  end-to-end CLI run on a generated video.
- Config sha256 `dbe93176…146b`.

## Known limits
1. **Residual motion-only risk inside `rematch_max_s` (0.2 s).** If a player vanishes and a different player
   appears within about 0.15 s and about 1 body height of that spot, the gap is bridged (measured grid: fps
   10–120, offset 0.1–1.0 h). Motion cannot tell this apart from a detector miss. It is recorded on the track
   (`gaps_bridged`, `last_gap_s`). **The target-identity layer must treat every bridged gap as unverified.**
2. UNKNOWN camera motion fragments every track. How often it happens on real broadcast footage is UNKNOWN; it
   must be measured on DEV.
3. Stage S2 (low-score continuation) applies only to tracks observed on the previous frame. This is
   frame-based on purpose (direct continuity). Gate growth is time-based.
4. All config values are uncalibrated priors.
5. VFR guard: only the container timestamps OpenCV exposes are checked. A VFR stream reported as CFR is not
   caught.
6. No person-detection model is pinned yet (model, licence and version are Nathan's decision).

## Persistence
Git: `main` on github.com/NathanGeorgaklis/SPANA_ELITE_2.0, verified by fresh clone and per-file sha256.
**Not configured:** Mac source-of-truth copy, full snapshot archive, second independent backup.
