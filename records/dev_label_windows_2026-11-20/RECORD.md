# DEV labelling windows chosen from track logs (2026-11-20)

Nathan's go-ahead for step 3 (more labelled DEV footage, `records/reid_margin_floor_dev_2026-11-19` option (c)).
- **Rule:** windows chosen by generic criteria from the track logs (target occlusions, leaving / re-entering the
  picture, flagged transfers, similar-looking rivals nearby), never hand-picked.
- **Plus:** dense windows behind the current wrong frames.
- **Scope:** DEV only; EVAL not touched; the long-play branch not touched. Everything ran on Nathan's Mac.
- **Choices (Nathan):**
  - the search covers the G2 10-minute window plus the never-labelled DEV clips;
  - 3 generic windows of ~13 s at 0.2 s;
  - dense windows ±1 s around each wrong frame, every frame;
  - each window is a frame range of its source video (no re-encoded cut).

## Built (branch `evaluation/dev-label-windows-v1`)
- **`tools/dev/label_windows.py select / build`** + `config/label_windows.yaml`.
  - Event types, for confirmed tracks only:
    - occlusion: a track becomes OCCLUDED;
    - exit: a track ends LEFT_FRAME;
    - entry: a track's first box is within 2 % of a border;
    - flagged: an `identity_uncertain` episode;
    - similar_rival: the same, when the rival's stored colour matched the crop, `rival_similarity` ≥ the tracker's
      `appearance.min_similarity` 0.70.
  - **Scoring:**
    - Score = Σ weight · min(count, cap 3).
    - A window needs ≥ 2 event types and must not overlap footage that is already labelled.
    - At most 1 window per source.
    - Ties go to more event types, then more episodes.
  - **Dense windows:** every wrong TARGET answer in `target_score` reports gets a window.
- **`tools/label_pack.py` range packs (pack 1.1.0):** a named frame range [start, end) of the source video.
  - Frame numbers and `clip.sha256` stay those of the source video, so the labels line up with tracks of that
    video.
  - 1.0.0 packs are unchanged.
- **Labelling page:** keeps a range pack's saved work under video + pack name, so packs of one video don't
  overwrite each other. 1.0.0 packs keep their key.
- **`tools/freeze_labels.py`:** records the range in FROZEN.json and refuses a label record of another pack.
- **Bugs found and fixed on the way:**
  - The frame-file variable in `make_pack` shadowed the new name (caught by a test).
  - `target_score`'s `per_player` is a list, not a dict. The same slip had put list positions instead of player ids
    into `records/reid_merge_dev_2026-11-19` (corrected).
  - Flags are logged on every contested frame, so they are now counted per episode.
- **Tests:** `tests/test_label_windows.py` (8). The page was checked in headless Chromium: two range packs of one
  video keep separate saved work, a 1.0.0 pack keeps its key, and there are no page errors. Full suite 547 passed,
  5 skipped (at 81a62ce).

## Inputs
**Tracks:** main 1a161ca default config.
- FRESH_01 / 03 / 04 / 06 detected with the same local detector command as the existing DEV clips.
  - The detector covers each clip's first 13.0 s, so each clip offers exactly one window.
- G2 10-minute window (match frames 170,250–188,249) re-tracked from its existing detections.
  - FRESH_05's footage is excluded: match frames 171,150–171,681, from the start in `records/window_f05_2026-11-18`
    and FRESH_05's length of 17.68 s.

**Event totals:**

| source | occlusion | exit | entry | flagged | similar_rival |
|---|---|---|---|---|---|
| FRESH_01 (G1) | 63 | 15 | 13 | 69 | 0 |
| FRESH_03 (G2) | 42 | 23 | 15 | 20 | 0 |
| FRESH_04 (G2) | 3 | 10 | 6 | 0 | 0 |
| FRESH_06 (G2) | 19 | 10 | 9 | 29 | 0 |
| G2 window (10 min) | 1,540 | 774 | 712 | 1,448 | 7 |

562 candidate windows were scored.

## Chosen (`windows.json`)

| pack | frames | score | occlusion / exit / entry / flagged / similar rival |
|---|---|---|---|---|
| `DEV_G2_NWS_SPIRIT_SD_RAIDERS_f187440_187830` (match 104:08–104:21) | 65 at 0.2 s | 14 | 30 / 20 / 20 / 22 / **2** |
| `FRESH_01_G1_2697_setpiece_delivery_keeper_claim_throw_f0_390` | 65 at 0.2 s | 12 | 63 / 15 / 13 / 69 / 0 |
| `FRESH_03_G2_2712_midfield_possession_switch_of_play_f0_780` | 65 at 0.2 s | 12 | 42 / 23 / 15 / 20 / 0 |
| `FRESH_02_G1_3000_midfield_buildup_passing_dense_f246_307` | 61, every frame | — | wrong frame 276: player 9 shown on player 10 (T000045) |
| `FRESH_07_G3_1205_midfield_passing_switch_dense_f36_97` | 61, every frame | — | wrong frame 66: player 14 shown on player 15 (T000018) |

- FRESH_06 (score 12, fewer episodes) and FRESH_04 (score 9) were not chosen.
- The G2 window is the only footage with similar-looking rivals: 7 episodes in 10 minutes, 2 of them in the chosen
  window.
- **Packs:** `Downloads/SPANA_dev_label_packs_2026-11-20/` on Nathan's Mac (92 MB, 317 frames). Open each pack's
  `index.html`.
- **The dense packs** share their video with the v1 labels (same sha256). They are separate label sets
  (`<clip>_dense_f…`) and never a new version of FRESH_02 / FRESH_07 v1.

## Notes
- **What the labels are for:**
  - The 3 generic windows (~195 labelled frames) are the extra DEV footage the held-out re-ID check needs (about 3
    clips' worth).
  - The dense windows measure how long each known wrong answer lasts. Today only every 6th frame is labelled there.
- **Scoring the new windows needs tracks over the same frames:**
  - FRESH_01 / 03 have them: `spana-label-windows/out/`.
  - For the G2 window: track the same 10-minute detections.
- **Not done here:** labelling (Nathan), freezing, scoring.
