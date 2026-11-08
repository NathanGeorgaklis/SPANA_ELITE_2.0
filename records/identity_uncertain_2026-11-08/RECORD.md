# IDENTITY_UNCERTAIN + NESTED_V2 default: DEV validation (2026-11-08)

DEV only (FRESH_02, FRESH_05, FRESH_07). EVAL was not used.

## Changes
- `tools/benchmark.py` default rule is now **NESTED_V2**. NESTED_V1 is still available with `--rule NESTED_V1`.
- **Tracker: `IdentityState` (CERTAIN / IDENTITY_UNCERTAIN)**, orthogonal to the lifecycle state.
  - **Trigger:** a contested match of a confirmed track, where a rival confirmed track (observed, or unobserved for
    at most `occlusion_max_s`) passes every gate and its own motion bound for the same detection, and appearance
    cannot separate the two by `appearance.min_margin`. **Both tracks** become uncertain, with each other as rivals.
  - **While uncertain:** the appearance model is frozen, so the track cannot adapt to whoever it is now following.
  - **Resolve:** clean evidence (an uncontested match with a clean crop) must favour the track's own appearance
    (≥ `min_similarity`) over every rival's recorded appearance by ≥ `min_margin`, continuously for
    `tracking.identity_resolve_s` (0.5 s). A rival without a known appearance, or evidence favouring a rival,
    never resolves it.
  - Every trigger, reset and resolution is logged in the decision log.
- **Tracks schema 2.1.0:** per-track `identity`, `identity_rivals`, `uncertain_episodes`.
- **Evaluation:** a transfer is flagged if the track is IDENTITY_UNCERTAIN at the transfer frame, or entered that
  state since its previous attribution. The existing flags (possible predecessors, recoveries) still count.

## Results (NESTED_V2; "before" = the previous run, `records/nested_v2_rescore_2026-11-08`)

| | FRESH_02 | FRESH_05 | FRESH_07 |
|---|---|---|---|
| identity transfers (unflagged) | 5 (1) → 5 (**0**) | 0 → 0 | 0 → 0 |
| ID switches | 28 → 29 | 20 → 20 | 44 → 39 |
| fragments | 26 → 27 | 19 → 19 | 44 → 39 |
| coverage | 0.9607 → 0.9607 | 0.9689 → 0.9689 | 0.9595 → 0.9595 |
| detection recall / duplicate track frames / ambiguous attributions | unchanged | unchanged | unchanged |
| confirmed observed track-frames that are uncertain | 835 / 15103 (5.5 %) | 126 / 2819 (4.5 %) | 912 / 11195 (8.1 %) |
| tracks ever uncertain / resolutions | 23 / 3 | 6 / 1 | 24 / 0 |

- **FRESH_02, T000014 (the silent crossing swap):** it is now flagged. Its identity state was IDENTITY_UNCERTAIN at
  the transfer.
- **The other 4 FRESH_02 transfers** were already flagged. 3 of them are also identity-uncertain.
- **Association decisions are unchanged on FRESH_05.** On FRESH_02 and FRESH_07 they changed, because frozen
  appearance changes later appearance vetoes:
  - FRESH_02: player 8 has one extra fragment (4 → 5).
  - FRESH_07: players 4, 5 and 7 have 1, 3 and 1 fewer fragments.
- **Resolutions checked:** none of the 4 resolved tracks has any later transfer.
- **NESTED_V1, for comparison:** FRESH_02 10 transfers / 0 unflagged; FRESH_07 6 / 0 (was 5 / 3).

## Tests
- `tests/test_identity_uncertain.py` contains:
  - a real-crossing fixture: the exact tracker inputs of FRESH_02 frames 120–204 in the ROI, plus the frozen
    labels of players 8 and 10. It reproduces the silent swap when the identity flag is ignored, and shows the
    swap flagged with it.
  - synthetic tests: a crossing without appearance ends uncertain; distinct kits never trigger; separated
    players never become uncertain; clean own appearance resolves; contradicting appearance never resolves.
- The fixture was built by `tools/dev/make_tracking_fixture.py`.
- Full suite 114 passed. Synthetic QA scenarios are identical to before.

Tracks (not committed): F02E 0b6e6226…b9c9, F05E 0f045dc5…ae2a, F07E 34a27d86…c92f.
