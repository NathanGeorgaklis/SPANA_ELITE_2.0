# Persistent label for the selected player in the review video (2026-11-21)

Owner layer: target identity, review video only (`tools/target_render.py`). Branch `identity/persistent-target-label`.
No tracker / follower / config change; `spana.target` documents are byte-identical. No EVAL.

## Diagnosis (confirmed before changing)
The changing "T-number" over the selected player is the tracker's track id. It changes correctly whenever the
player's answer moves to another track (an accepted re-entry offer / relink, a re-selection, a camera cut).
`tools/target_render.py` labelled the TARGET box `"<track_id>  following"`, so the review video looked like identity
was lost while `spana.target` still followed the same player.

Reproduced on the synthetic demo (`tools/target_demo.py`: scripted identities, real follower). Every TARGET answer is
player #7, but the label changed `T000001 following` (f12) -> `T000005 following` (f80, after the accepted offer) ->
`T000006 following` (f195, re-tap after the cut).

## Change (display only)
- TARGET: the label is `YOU` on every TARGET frame, whichever track carries it. The track id is a small secondary
  tag next to it (leading `T0` stripped, the same as the grey per-track ids).
- CANDIDATE: unchanged, `Is this them?  accept / decline`.
- ABSTAIN / LOST: no box and no label on any player. `answer_label()` returns nothing for these states even if an
  answer carried a box (fail closed). The grey per-track ids of every observed track stay: they are track ids, not
  identity claims.
- Automatic re-entry linking stays OFF (DECISIONS 2026-11-18): offers only, user confirms.

## Evidence
- `demo_before_after.png`: demo frames 50 / 100 / 200. Top = main (`T000001` / `T000005` / `T000006 following`),
  bottom = this branch (`YOU` with tag 1 / 5 / 6). Same-kit #9 is beside #7 in frame 100.
- Tests (`tests/test_target_render.py`, class `PersistentLabelTest`): `YOU` drawn on every TARGET frame across
  >= 3 tracks and no `<track> following` text; CANDIDATE keeps the offer text and never `YOU`; ABSTAIN and LOST
  (with a forged box) draw neither. The new tests cannot import against the old renderer. Targeted: 5 passed. Full:
  664 passed, 14 skipped, 2 failed. The 2 failures are `tests/test_botsort_compare.py`: the bench-only `supervision`
  package is missing in the cloud container, and they fail identically on main.

## DEV QA: NOT RUN (blocked in the cloud session)
The cloud container has the frozen DEV labels only: no DEV footage, no tracks / embeddings artifacts. That left two
gaps:
- No before / after DEV review videos were rendered.
- No re-entry verdicts were re-scored.

The change cannot move any verdict: re-entry verdicts are computed from `spana.target`, and this change does not touch
it. The last DEV numbers on main are re-ID offers 16 right / 0 wrong (`records/offer_hold_2026-11-19`).

To run on Nathan's Mac (FRESH_08: player 4 leaves at f49 and comes back, same-kit G3 teammates; FRESH_07: same-kit
crossings). Paths follow `records/offer_hold_2026-11-19`; set `VIDEO_*`.

1. Per-player target documents and re-entry verdicts (unchanged by this task; `relinks[].verdict` = right / wrong /
   undecided = correct / wrong / unresolved reconnection):

   python -m tools.target_score --config config/default.yaml \
     --clip data/labels/FRESH_07_G3_1205_midfield_passing_switch/v1 out/F07_mdefault.trk.json out/F07_sv2.emb.json \
     --clip data/labels/FRESH_08_G3_2103_attack_into_box_clearance/v1 out/F08_mdefault.trk.json out/F08_sv2.emb.json \
     --simulate-user --artifacts-dir out/plabel --out out/plabel_score.json

2. Before / after videos for each re-entry player (before = a main checkout, after = this branch):

   python -m tools.target_render --video $VIDEO_F08 --tracks out/F08_mdefault.trk.json \
     --target out/plabel/FRESH_08_G3_2103_attack_into_box_clearance.player4.target.json --out out/plabel_F08_p4_after.mp4

   (the same from the main checkout with `--out out/plabel_F08_p4_before.mp4`)

Expected result: 0 wrong reconnections. On every TARGET frame the after video shows `YOU` on the same player, while
the small tag changes at each accepted offer.
