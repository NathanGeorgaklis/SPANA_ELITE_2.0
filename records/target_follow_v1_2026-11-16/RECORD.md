# Target follow v1 (2026-11-16): baseline for following a tapped player on DEV

Scope: DEV only (FRESH_02, FRESH_05; frozen labels v1). Nothing tuned. EVAL not used. No full-match run.

Goal: a first number for the product's core promise: follow the player the user tapped, and say "not sure" rather
than show the wrong player. This is the baseline before re-identification.

## What was run
- Code: commit 67b817a (branch `identity/target-follow-v1`): `spana/identity/follower.py`, `spana.target` 1.0.0,
  `tools/target_score.py`.
- Run on Nathan's Mac (Python 3.13 `.venv`) from a scratch extract of that commit (`git archive`, spana/ tools/
  config/ tests/ only, in `~/spana-target-v1-run`).
  - Nathan's project copy was not changed.
  - So the `provenance.git` of the target documents is null; the commit is the one above.
  - `tests/test_target_follow.py` passed there (14).
- Inputs: the `--backend local` tracks from `records/local_inference_2026-11-16` (same sha256 as its HASHES.txt):
  - `runs/person_view_compare/F02_local.trk.json`
  - `runs/person_view_compare/F05_local.trk.json`
- Config: `config/default.yaml` 2.1.0, `target.resume_after_uncertain: false` (config sha256 56d84f7c…).

```
python -m tools.target_score --config config/default.yaml --out out/target_score.json --artifacts-dir art \
  --clip data/labels/FRESH_02_G1_3000_midfield_buildup_passing/v1 runs/person_view_compare/F02_local.trk.json \
  --clip data/labels/FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper/v1 runs/person_view_compare/F05_local.trk.json
```

## Method (tools/target_score.py)
- Simulated tap, per labelled player: on the player's first labelled frame, select the observed track matched to
  the player's label box.
  - The match uses NESTED_V2 (spana.evaluation matching; ignore regions as in tools/benchmark.py) and must be
    unambiguous.
  - All 30 players were selectable on their first labelled frame (0 frames skipped).
- Follow: `spana.identity.follower` on that track (one `spana.target` document per player; 30 documents, hashes
  below).
- Score: over the player's labelled frames from the selection on. Labels are every 0.2 s, so a share of frames is
  a share of time. Each frame falls in exactly one category:

| category | meaning |
|---|---|
| correct | TARGET box attributed to this player |
| wrong | TARGET box attributed to another labelled player |
| unverified | TARGET box with no clear attribution (NESTED_V2 ambiguous, matched to nobody, or ignored): not credited, not counted wrong |
| abstain / lost | no answer |

  Also counted: TARGET answers on labelled frames where the player is not labelled.

## Results (pooled per clip; per player in target_score.json)
| clip | players | player-frames | correct | **wrong** | unverified | abstain | lost | answers while absent |
|---|---|---|---|---|---|---|---|---|
| FRESH_02 | 18 | 1018 | 0.482 | **0.000** | 0.015 | 0.154 | 0.349 | 0 |
| FRESH_05 | 12 | 450 | 0.716 | **0.000** | 0.002 | 0.024 | 0.258 | 1 (on nobody) |

- **Wrong-time is 0 on both clips:** 0 of 1468 player-frames went to another player, and 0 answers landed on
  another player while the target was absent.
- The single F05 answer while absent: player 10, frame 480, a box attributed to no labelled player, 8 frames before
  the track ended LEFT_FRAME.

Where the time without an answer goes (scored player-frames, by reason):

| clip | TRACK_ENDED | IDENTITY_UNCERTAIN | AFTER_UNCERTAIN | TRACK_OCCLUDED | NOT_CONFIRMED | TRACK_LOST |
|---|---|---|---|---|---|---|
| FRESH_02 | 340 | 100 | 17 | 23 | 17 | 15 |
| FRESH_05 | 116 | 0 | 0 | 1 | 10 | 0 |

Track ends of the followed tracks:
- F02: LOST_EXPIRED 10, LEFT_FRAME 2, UNCONFIRMED_MISSED 1; 5 tracks never ended.
- F05: LEFT_FRAME 11; 1 never ended.

Per player:
- F02: correct-time 0.98 (players 3, 7) down to 0.00 (player 2; its track ended LEFT_FRAME on frame 4 at the left
  edge while the player stayed labelled for 53 frames).
- F05: correct-time 0.56 to 1.00.

## Findings
1. Fail closed holds on DEV: the follower never answered with the wrong player.
   - The tracker's 3 known F02 identity transfers are all flagged (0 unflagged; local-inference record). None of
     them produced a wrong answer here.
   - Not checked frame by frame: whether each transfer happened inside an IDENTITY_UNCERTAIN episode (where the
     follower abstains; 100 player-frames) or on a track the follower had already stopped answering for.
2. Most of the time without an answer is a track ending while the player is still in the picture: 340 / 1018
   (F02) and 116 / 450 (F05) player-frames.
   - F02: mostly LOST_EXPIRED, from tracker fragmentation (24 fragments).
   - F05: every followed track ended LEFT_FRAME while most of those players were still labelled later: the track's
     prediction left the image, and the player came back on a new track.
   - Recovering those is re-identification, the next step. The tracker's fragmentation sets the ceiling of this
     follower.
3. IDENTITY_UNCERTAIN is the second cost on F02 (100 player-frames, 9.8 %). Same-kit contact clusters.
4. A new track starts unconfirmed, so a selection on a track's first frames is ABSTAIN (NOT_CONFIRMED) for about
   0.1 s.
   - 24 of the 30 selections were on the clip's first frame, where every track is new.
   - NOT_CONFIRMED: 27 player-frames in all.

## Sensitivity: `target.resume_after_uncertain: true` (same tracks, scratch config copy; not adopted)
| clip | correct | wrong | abstain |
|---|---|---|---|
| FRESH_02 | 0.482 → 0.499 (+17 frames, players 13 and 26) | 0 → 0 | 0.154 → 0.138 |
| FRESH_05 | unchanged | 0 | unchanged |

Kept at false (fail closed): +1.7 points on 2 players is too little evidence that the tracker's resolution is safe
for same-kit rivals. Revisit on more DEV clips (see DECISIONS).

## Files
- `target_score.json`: scorer output for the default config (per player, pooled, events).
- `HASHES.txt`: sha256 of the inputs, outputs and the 30 target documents.
  - The target documents and the sensitivity report stay on the Mac in `~/spana-target-v1-run/` (not committed).
