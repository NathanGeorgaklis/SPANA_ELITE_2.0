# FRESH_07 / FRESH_08 (2026-11-16): follower v1 and re-ID headroom on two more DEV clips

Scope: DEV only (frozen labels v1). Short DEV clips, not a full match. Nothing tuned. EVAL not used.

Why: the re-ID headroom (`records/reid_headroom_2026-11-16`) rested on 25 losses from 2 clips. That is too few to
choose which clues a relinker may trust, or the gap limit for the tracker's own link.

## What was run (Nathan's Mac, Python 3.13 `.venv`)
1. **Detection and tracking:** Nathan's project copy (branch `detection/local-onnx`, 25a2d6b), exactly as for
   FRESH_02 / FRESH_05 in `records/local_inference_2026-11-16`.
   - `detect --backend local --runtime torch --device mps`, then `track`, `config/default.yaml`.
   - Artifacts in `runs/person_view_compare/F07_local.*`, `F08_local.*` (not committed; sha256 in HASHES.txt).
   - Detection time: FRESH_07 161 s.
2. **Tracking benchmark:** `tools/benchmark.py` (NESTED_V2).
3. **Scoring:** from the scratch extract `~/spana-reid-headroom-run` (code of branch `identity/reid-headroom-v1`,
   b903759; the tools' code is unchanged from that commit):
   - `tools/target_score.py` with the default config, and again with `resume_after_uncertain: true`;
   - `tools/reid_headroom.py` with the clips' videos.
4. **Fix during the run:** FRESH_08 player 26 (5 labelled frames) is never matched to a track unambiguously. The
   first run of `reid_headroom` failed on that; fixed in b903759 (reported as not selectable).

## Tracking (NESTED_V2)
| clip | detection recall | coverage | ID switches | fragments | identity transfers (unflagged) | players |
|---|---|---|---|---|---|---|
| FRESH_07 | 0.982 | 0.962 | 38 | 38 | 2 (0) | 21 |
| FRESH_08 | 0.963 | 0.932 | 52 | 51 | 1 (0) | 21 |

## Follower v1 (`tools/target_score.py`, default config)
| clip | players selectable | player-frames | correct | **wrong** | unverified | abstain | lost |
|---|---|---|---|---|---|---|---|
| FRESH_07 | 21 / 21 | 1205 | 0.586 | **0.0008 (1 frame)** | 0.007 | 0.085 | 0.322 |
| FRESH_08 | 21 / 22 | 776 | 0.305 | **0.000** | 0.004 | 0.099 | 0.592 |

All four DEV clips together: correct 1756 of 3449 player-frames (0.509), **wrong 1 (0.0003)**.

`resume_after_uncertain: true` changes nothing on FRESH_07 / FRESH_08. Over all 4 clips it changes only FRESH_02
(+17 correct), with 0 wrong anywhere.

### The one wrong player-frame (FRESH_07, player 14, frame 66)
- Track T000018 follows player 14 cleanly before and after frame 66.
- At frame 66, player 14 crosses in front of player 15 (labelled occluded). The detector returns one box about
  54 px wide covering both, with no multi-body flag.
- Under NESTED_V2 that box goes to player 15: its centre lies 0.8 px outside player 14's label box, while it fits
  player 15's.
- So it is not an identity swap, but on that frame the answer box held two people.
- The track gave no warning: ACTIVE, confirmed, CERTAIN, not contested. Its only overlap was IoU 0.45 with player
  15's track.
- Possible guard (not built; a policy and threshold for Nathan): abstain while the target box overlaps another
  track above a configured IoU.

### Answers while the player was not labelled (7, none on another player)
- FRESH_07 player 17, 6 frames: the player leaving at the right edge, unlabelled once partly out of frame, still
  followed correctly.
- FRESH_07 player 1, 1 frame.

## Re-ID headroom (`tools/reid_headroom.py`)
| clip | LOSS | OVERLAP | LOSS gap, median | v1 correct | ceiling (perfect relinks) |
|---|---|---|---|---|---|
| FRESH_07 | 35 | 3 | 0.07 s | 0.583 | 0.889 |
| FRESH_08 | 48 | 4 | 0.20 s | 0.302 | 0.747 |
| all 4 clips | 108 | 13 | | 0.507 (1756 / 3463) | **0.853 (2955 / 3463)** |

19 of the 108 losses have gaps over 1 s, mostly LEFT_FRAME re-entries (FRESH_05 6, FRESH_08 9, FRESH_02 2,
FRESH_07 2).

### Relink rules over all 108 losses (picks / correct / wrong)
| rule | F02 | F05 | F07 | F08 | **all** |
|---|---|---|---|---|---|
| tracker's unique predecessor, any gap | 12/12/0 | 2/0/2 | 15/14/1 | 25/22/3 | 54/48/6 |
| ... gap ≤ 0.5 s | 12/12/0 | 0/0/0 | 14/14/0 | 23/22/1 | 49/48/1 |
| **... gap ≤ 0.5 s, candidate confirmed** | 11/11/0 | 0/0/0 | 13/13/0 | 20/20/0 | **44/44/0** |
| ... gap ≤ 0.25 s, candidate confirmed | 10/10/0 | 0/0/0 | 13/13/0 | 18/18/0 | 41/41/0 |
| position and appearance agree | 7/6/1 | 4/4/0 | 17/13/4 | 20/19/1 | 48/42/6 |
| ... candidate confirmed | 7/6/1 | 4/4/0 | 15/13/2 | 19/18/1 | 45/41/4 |
| all three agree | 5/5/0 | 0/0/0 | 10/10/0 | 16/16/0 | 31/31/0 |
| best position alone | 18/12/6 | 7/5/2 | 35/24/11 | 48/26/22 | 108/67/41 |
| best appearance alone | 12/9/3 | 7/6/1 | 22/18/4 | 40/27/13 | 81/60/21 |

"Candidate confirmed" means the picked track became confirmed. A relinker would wait for that (`confirm_s`, about
0.1 s), as the follower already does.

### What the "wrong" picks were
Checked by who each picked track is ever attributed to.
- **Predecessor rule:**
  - the wrong picks on long gaps (FRESH_05 0.9 s and 4.5 s, FRESH_07 3.2 s, FRESH_08 10.2 s and 4.4 s) are
    other or unconfirmed tracks;
  - the only short-gap wrong pick (FRESH_08 player 19, 0.23 s) is a 2-frame unconfirmed fragment, never
    attributed to anyone.
  - The confirmed-candidate requirement removes it.
- **Position and appearance agreement:** 4 picks are really another player's track, all in same-kit contact
  clusters:
  - FRESH_02 player 8 → player 10's track;
  - FRESH_07 player 5 → player 7's track, and player 7 → player 5's track (a mutual swap);
  - FRESH_08 player 17 → player 16's track.
  - The other 2 are unconfirmed fragments.

## Findings
1. **Fail closed holds on 4 DEV clips:** 1 wrong player-frame in 3449, caused by a two-person box at a crossing,
   not by following the wrong track. The follower itself never switched players.
2. **Relinking is the big lever:** with this tracker, perfect relinking would raise correct-time from 0.51 to
   0.85 over the 4 clips.
3. **One clue is safe enough to automate on DEV:** the tracker's own unique predecessor link, within 0.5 s, on a
   candidate that confirms. It was correct 44 of 44 times, about 41 % of all losses, on all 3 clips with short
   gaps.
4. **Position plus colour agreement is not safe on its own:** 4 real wrong players in 45 picks, all same-kit
   clusters. At best it can propose a CANDIDATE for the user to confirm, and even then it needs care.
5. **Long re-entries are unsolved** (19 losses over 1 s, up to 61 candidates). Existing clues do not single out
   the player. This needs a stronger appearance model (the SPANA re-ID model) and/or team classification to prune
   the other team.
6. FRESH_08 is the hardest clip: correct-time 0.31, 51 fragments, many LEFT_FRAME ends, a 0.75 ceiling.

## Files
- `f0708_summary.json`: compact summary (benchmark, follower per player and events, every handover with each
  clue's pick, rules, ceiling).
- Full reports stay on the Mac (sha256 in HASHES.txt).
- `HASHES.txt`.
