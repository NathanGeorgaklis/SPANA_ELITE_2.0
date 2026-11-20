# Re-ID branches merged onto main: DEV check, re-ID resolution off and on (2026-11-19)

Scope:
- Nathan's CODE TASK "Merge re-ID branches onto main".
- **Branch:** `identity/reid-merge-v1` = main 5c7dac5 + `tracking/margin-floor-guard` 68bf1a9 (which contains
  `tracking/reid-resolve-v1`); merge commit 5890b14.
- **DEV run:** FRESH_02 / 05 / 07 / 08, the same local detections and labels as every DEV record. Re-ID signatures
  for the re-entry offers (`spana.cli embed`, re-ID v2) were made per variant.
- Run on Nathan's Mac. EVAL not used. No full-match runs.

## Merge
**Schema clashes**, each resolved with the next free version on main (no number reused):

| | main | re-ID branches | merged |
|---|---|---|---|
| `spana.tracks` | 2.2.0 `identity_doubt` | 2.2.0 `resolved_by` | **2.3.0** = 2.2.0 + `resolved_by` (2.1.0 / 2.2.0 validate) |
| `spana.target` | 2.5.0 | 2.2.0 `target_config.resume_after_reid` | **2.6.0** = 2.5.0 + `resume_after_reid` (2.0.0–2.5.0 validate) |
| config | 2.7.0, `reid` = signatures | 2.4.0, `reid` = tracker evidence | **2.8.0**: main's `reid` unchanged + new **`reid_resolve`** |

**Tracker:**
- main's appearance doubt (no rival) and the re-ID pair verdict are combined.
- A doubt has no rival reference, so only appearance clears it.
- `resolved_by` is REID only if re-ID cleared every rival on every frame of the run and there was no doubt.
- A doubted track's re-ID reference stays frozen, like its colour model.

**Model registry:**
- Uses main's pinned `ReidModelRecord`. `load_pinned` refuses a record whose sha256 is not
  `reid_resolve.model_sha256`, and an ONNX file whose hash is not the record's.
- `models/spana_reid_v3.json` added; config does not reference it.
- Weights stay out of git (`*.onnx` ignored), as for every other model here. Git LFS was not used.

**Defaults unchanged:** `reid_resolve.enabled: false`, v2, margin 0.517; `target.resume_after_reid: false`.

**Tests:** full suite 503 passed, 5 skipped. New: pinned-model refusals, the registry pins, tracks 2.3.0 / 2.2.0 /
2.1.0 compatibility, target 2.6.0 / 2.5.0.

## DEV results (4 clips; 3,449 labelled player-frames; 14.36 followed player-minutes)

| | main | merge, re-ID off | merge, re-ID on |
|---|---|---|---|
| tracks (frames + tracklets, without `resolved_by`) | — | identical to main | identical to main |
| silent swaps (unflagged transfers) | 0 | 0 | 0 |
| flagged transfers | 7 | 7 | 7 |
| ID switches / fragments | 112 / 110 | 112 / 110 | 112 / 110 |
| doubt episodes cleared (by re-ID) | 4 (0) | 4 (0) | 4 (**2**, FRESH_02) |
| **unaided:** correct time | 0.532 | 0.532 | **0.537** |
| unaided: correct / wrong / abstain / LOST frames | 1,835 / 1 / 291 / 879 | 1,835 / 1 / 291 / 879 | 1,852 / 1 / 274 / 879 |
| **reviewer:** correct time / wrong frames | 0.777 / 2 | 0.777 / 2 | **0.782** / 2 |
| reviewer: actions (per followed min) | 82 (5.71) | 82 (5.71) | 82 (5.71) |
| reviewer: abstain / LOST frames | 428 / 274 | 428 / 274 | 411 / 274 |
| re-entry and relink offers right / wrong / undecided (unaided; reviewer) | 25 / 0 / 3; 35 / 0 / 5 | same | same |

- **Wrong frames:** the same players in all variants (unaided: FRESH_07 player 14 on one frame; reviewer: + FRESH_02
  player 9). Corrected 2026-11-20: first written as "8 and 13", which were list positions in the report, not player
  ids.
- **Main is reproduced:** these equal main's own latest record (`records/reentry_position_score_2026-11-18`:
  0.532 / 1; reviewer 0.777 / 2 / 5.71).

## Findings
1. **The merge changes nothing when re-ID resolution is off:** tracks, follower and reviewer equal main exactly.
2. **Turning it on adds no wrong frame and no silent swap.**
   - Tracks are unchanged; only `resolved_by` is new.
   - The 2 FRESH_02 crossing episodes it clears were already cleared by colour (both judged right in
     `records/reid_resolve_dev_2026-11-18`).
   - With `resume_after_reid` the follower answers after them: +17 correct frames (unaided 0.532 → 0.537, reviewer
     0.777 → 0.782), and 17 fewer abstentions.
3. **Actions per minute are unchanged (5.71).** The re-ID gain is in frames the reviewer was not fixing anyway.
4. **The LOST share is unchanged** (unaided 879 / 3,449 = 25.5 %; reviewer 274 = 7.9 %). Re-ID resolution acts only
   on crossings, not on lost tracks.
5. **Still off by default.**
   - The gain is small: 99 / 1,299 uncertain frames have a clean crop (`records/reid_resolve_dev_2026-11-18`).
   - v3 has not passed the held-out check (`records/reid_margin_floor_dev_2026-11-19`).
   - Turning it on is Nathan's decision.

## Addendum 2026-11-20: re-checked on the newer main (24ccd58)
- **Why:** main moved after PR #13 opened.
  - It took `spana.target` 2.6.0–2.8.0 (crop-set signatures, exit side, offer hold) and config 2.8.0–2.11.0.
  - It turned on `tracking.rival_eligible_only` and `target.offer_hold_s` 0.2.
- **Merge commit 80fc877:** the re-ID additions moved to the next free numbers.
  - `spana.target` **2.9.0** = 2.8.0 + `resume_after_reid`.
  - Config **2.12.0** = 2.11.0 + `reid_resolve` + `target.resume_after_reid`.
  - `spana.tracks` stays **2.3.0** (main is still at 2.2.0).
  - The tracker merged cleanly. Full suite: 536 passed, 5 skipped.
- **Same DEV run**, against main 24ccd58:

| | main 24ccd58 | merge, re-ID off | merge, re-ID on |
|---|---|---|---|
| tracks (frames + tracklets, without `resolved_by`) | — | identical to main | identical to main |
| silent swaps / flagged transfers | 0 / 7 | 0 / 7 | 0 / 7 |
| ID switches / fragments | 107 / 105 | 107 / 105 | 107 / 105 |
| doubt episodes cleared (by re-ID) | 4 (0) | 4 (0) | 4 (**2**, FRESH_02) |
| unaided: correct time / wrong frames | 0.536 / 1 | 0.536 / 1 | **0.540** / 1 |
| unaided: abstain / LOST frames | 310 / 805 | 310 / 805 | 293 / 805 |
| reviewer: correct time / wrong frames / actions per min | 0.783 / 2 / 5.50 | same | **0.787** / 2 / 5.50 |
| offers right / wrong / undecided (unaided; reviewer) | 26 / 0 / 2; 34 / 2 / 3 | same | same |

- **Wrong frames:** the same players in all variants (unaided FRESH_07 player 14 at frame 66, shown on player 15;
  reviewer + FRESH_02 player 9 at frame 276, shown on player 10). Corrected 2026-11-20 (first written as list
  positions 13 / 8).
- **Wrong offers:** main's 2 wrong offers with the reviewer are main's own; re-ID resolution changes no offer.
- **Conclusion unchanged:**
  - Off = main exactly.
  - On = 0 new wrong frames, 0 silent swaps, +17 correct frames.
  - Actions unchanged.
  - Stays off by default.
