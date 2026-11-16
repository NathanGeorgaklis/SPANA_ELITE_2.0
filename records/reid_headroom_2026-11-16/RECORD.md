# Re-ID headroom (2026-11-16): how much relinking could recover, and how far existing clues go

Scope: DEV only (FRESH_02, FRESH_05; frozen labels v1). Analysis, no new model, nothing tuned. EVAL not used. No
full-match run.

Question: when the followed player's track ends, could the clues we already have (the tracker's
`possible_predecessors`, camera-compensated position, torso colour histogram) pick the player's next track? And is
relinking worth building at all?

## What was run
- Code: `tools/reid_headroom.py` at commit be6195a (branch `identity/reid-headroom-v1`).
- Run on Nathan's Mac (Python 3.13 `.venv`) from a scratch extract of that commit in `~/spana-reid-headroom-run`.
  Nathan's project copy was not changed.
- Inputs:
  - The `--backend local` tracks from `records/local_inference_2026-11-16` (same sha256).
  - The DEV clips themselves, sha256-checked, for the appearance clue.
- Config: `config/default.yaml` 2.1.0 (appearance section unchanged: torso band, 16x8 HS histogram, `tau_s` 1.0).

```
python -m tools.reid_headroom --config config/default.yaml --out out/reid_headroom.json \
  --clip data/labels/FRESH_02_G1_3000_midfield_buildup_passing/v1 runs/person_view_compare/F02_local.trk.json \
  --video .../FRESH_02_G1_3000_midfield_buildup_passing.mp4 \
  --clip data/labels/FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper/v1 runs/person_view_compare/F05_local.trk.json \
  --video .../FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper.mp4
```

## Method
- **Chain:** per labelled player, the tracks attributed to them on labelled frames (NESTED_V2, unambiguous).
- **Handover:** every A → B change in that chain.
  - **LOSS** = A was seen before B was born and not again until B's labelled frame, so a relinker had to act.
  - **OVERLAP** = A and B coexisted: parallel tracks or an identity switch. That is the tracker's job, not
    relinking.
- **Candidates:** the tracks born after A's last observation, up to and including B's birth.
- **Clues:** each one picks the strictly best candidate, or abstains on a tie or unknown.

| clue | what it measures |
|---|---|
| predecessor | the only candidate whose tracklet lists A in `possible_predecessors` |
| position | nearest first foot point to A's last foot point, carried through the camera transforms, in A's box heights |
| appearance | highest Bhattacharyya similarity between A's torso histogram (its last 1 s) and the candidate's (its first 1 s) |

- **Correctness:** a pick is correct if it is B, otherwise counted as WRONG. That is pessimistic: the pick may be
  an unlabelled fragment of the same player.
- **Ceiling:** labelled frames on which some track attributed to the player is ACTIVE, confirmed and CERTAIN. That
  is what the v1 follower would score with perfect relinking.

## Results
### Ceiling: relinking is worth building
| clip | player-frames | v1 correct | ceiling (perfect relinks) | gain |
|---|---|---|---|---|
| FRESH_02 | 1018 | 491 (0.482) | 859 (0.844) | +368 |
| FRESH_05 | 450 | 322 (0.716) | 434 (0.964) | +112 |

The rest of FRESH_02's ceiling gap is frames whose track is unconfirmed or IDENTITY_UNCERTAIN: 941 frames are
attributed to the player, 859 of them on a track in a state the follower answers on.

### Handovers
| clip | handovers | LOSS | OVERLAP | across a cut | LOSS gap, median | candidates, median | A's end reason (LOSS) |
|---|---|---|---|---|---|---|---|
| FRESH_02 | 24 | 18 | 6 | 0 | 0.15 s | 2.5 | mostly LOST_EXPIRED |
| FRESH_05 | 7 | 7 | 0 | 0 | 2.12 s | 8 | all LEFT_FRAME |

The two clips show two different kinds of loss:
- **FRESH_02:** short gaps inside contact clusters (fragmentation).
- **FRESH_05:** long re-entries (0.9 to 6.3 s) after the camera pans away and back.

### Simple relink rules (LOSS handovers: picks / correct / wrong)
| rule | FRESH_02 (18) | FRESH_05 (7) | both (25) |
|---|---|---|---|
| tracker's unique predecessor | 12 / 12 / **0** | 2 / 0 / **2** | 14 / 12 / 2 |
| best position | 18 / 12 / 6 | 7 / 5 / 2 | 25 / 17 / 8 |
| best appearance | 12 / 9 / 3 | 7 / 6 / 1 | 19 / 15 / 4 |
| position and appearance agree | 7 / 6 / 1 | 4 / 4 / **0** | 11 / 10 / 1 |
| all three agree | 5 / 5 / 0 | 0 / 0 / 0 | 5 / 5 / 0 |

## Findings
1. **The headroom is large.** Perfect relinking would raise correct-time from 0.48 to 0.84 on FRESH_02 and from
   0.72 to 0.96 on FRESH_05. Most of v1's missing time is recoverable without improving the tracker itself.
2. **Short gaps: the tracker's own predecessor link is reliable.** On FRESH_02 (all its picks within 0.47 s) it picked
   correctly 12 of 12 times, with no wrong pick.
   - It is wrong on long gaps: both FRESH_05 picks were wrong (players 6 and 9, gaps 0.9 and 4.5 s).
   - So it can only be trusted within a short gap. Where that boundary lies is a config value, still to be set on
     more DEV clips.
3. **Long gaps (re-entries): position and appearance together** picked correctly 4 of 4 times on FRESH_05.
   - Each alone makes mistakes: position 2 wrong, appearance 1 wrong.
   - Appearance margins can be small (player 9: 0.977 against 0.969 for the runner-up), so colour alone separates
     teams more than players.
4. **The only wrong agreement** (FRESH_02, player 8, track 010 → 035): position and appearance both picked track
   033 in the contact cluster around frame 175.
   - Track 033 is later attributed to player 10 and then to player 8, so this is a real same-place, same-kit
     ambiguity. A relinker must abstain there, not guess.
5. **Appearance is often unknown in clusters:** no clean crop for B in 8 of 18 FRESH_02 losses (overlapping
   boxes), as designed. In crowds, position and the tracker's link carry the decision.
6. **6 OVERLAP handovers** (FRESH_02) are tracker identity switches or parallel tracks. Relinking cannot fix them;
   tracking work can.

## Implication for step 2 (relinker, not built yet)
- Offer a CANDIDATE (likely match; TARGET only after the user confirms in the app) when either holds:
  - (a) short gap: the tracker's unique predecessor;
  - (b) any gap: position and appearance agree on the same candidate.
- Abstain otherwise.
- On these 25 losses, (a) or (b) would have proposed 18 relinks: 17 correct and 1 wrong.
  - Correct: 12 from (a) on FRESH_02, 1 more from (b) on FRESH_02, and 4 from (b) on FRESH_05.
  - The wrong one is the finding-4 case. It would have reached the user as a CANDIDATE, not as an answer.
  - 7 losses would get no proposal.
- This needs a gap limit for (a) between 0.47 s (longest correct predecessor pick) and 0.88 s (shortest wrong
  one).
  - That window is narrow and based on a tiny sample.
  - The limit and any margin must go in config and be checked on FRESH_07 / FRESH_08 before they are trusted.
  - 25 events from 2 clips is a small sample.

## Files
- `reid_headroom.json`: full report (per handover: kind, gap, end reason, candidates, each clue's value, rank and
  pick; per-clip summary and rules; per-player ceiling).
- `HASHES.txt`.
