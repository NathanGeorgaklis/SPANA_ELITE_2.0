# Target relink v1 (2026-11-16): CANDIDATE from the tracker's unique predecessor link, on 4 DEV clips

Scope: DEV only (FRESH_02, 05, 07, 08; frozen labels v1; local-detector tracks). Short clips, not a full match.
Nothing tuned (`relink_max_gap_s` 0.5 is the prior from `records/reid_headroom_F07_F08_2026-11-16`). EVAL not
used.

## What was built (commit c2e1d11, branch `identity/target-relink-v1`)
- `spana/identity/follower.py`, `spana.target` 2.0.0. Rules in docs/INTERFACES.md. In short:
  - A candidate is proposed while the answered track is unobserved, and only if that track was confirmed,
    CERTAIN and untainted at its last observation.
  - Candidate = the only track born within `relink_max_gap_s` that names the lost track in
    `possible_predecessors`, once it is ACTIVE, confirmed and CERTAIN, with no camera cut since.
  - It is dropped when the original returns, a second such track appears, a cut follows, or the candidate stops
    qualifying.
  - TARGET only after the user's `confirm_candidate` 1.0.0 on a CANDIDATE frame, or with `relink_auto_confirm`
    (off by default). Chains are allowed.
- `tools/target_score.py`:
  - CANDIDATE frames are scored apart from answers, never as correct time.
  - Each relink is judged right / wrong / undecided on the first labelled frame (from the proposal) where the
    answer is on the new track and clearly attributed.
  - `--simulate-user` confirms, on its first frame, every candidate judged right.
- Run on Nathan's Mac from a scratch extract of c2e1d11 (`~/spana-relink-run`); his project copy was not
  changed. 42 target / re-ID / guard tests passed there; full suite in the container: 333 passed, 4 skipped.

## Results (labelled player-frames, 4 clips pooled: 3449)
| mode | correct | **wrong TARGET** | candidate correct | candidate wrong | no answer | relinks proposed (right / wrong / undecided) | confirmed |
|---|---|---|---|---|---|---|---|
| v1 (no relink) | 1756 (0.509) | 1 | - | - | 0.483 | - | - |
| 1. default (CANDIDATE, nobody confirms) | 1756 (0.509) | **1** | 169 (0.049) | 1 | 0.433 (+ 0.050 candidate) | 13 (12 / 0 / 1) | 0 |
| 2. auto-confirm | 2059 (0.597) | **3** | 0 | 0 | 0.393 | 18 (16 / 1 / 1) | 18 |
| 3. simulated user | 2059 (0.597) | **2** | 0 | 1 | 0.393 | 18 (16 / 1 / 1) | 16 |

Ceiling with perfect relinking: 0.853 (`records/reid_headroom_F07_F08_2026-11-16`).

Per clip, correct-time:

| clip | v1 = default | simulated user | auto |
|---|---|---|---|
| FRESH_02 | 0.482 | 0.519 | 0.519 |
| FRESH_05 | 0.716 | 0.716 | 0.716 (no relink: its losses are long LEFT_FRAME re-entries) |
| FRESH_07 | 0.586 | 0.688 | 0.688 |
| FRESH_08 | 0.305 | 0.490 | 0.490 |

## The acceptance check: wrong TARGET time must stay at the v1 level (1 frame)
- **Default (the shipping configuration): holds**, 1 wrong TARGET frame (the known v1 frame, FRESH_07 player 14,
  frame 66).
- **Simulated user: fails by 1 frame.** FRESH_02 player 9: T000014 → T000045 was proposed on frame 270 and
  judged right there (player 9), so the user confirms. On frame 276 the box of T000045 belongs to player 10.
  - That is the tracker's own track changing player after a correct relink, not a wrong relink.
  - T000045 names two predecessors: T000014 (player 9's lost track) and T000036 (still live). The re-ID headroom
    shows T000045 also taking over from player 10's track.
  - A user who confirmed correctly at 270 would see the wrong player at 276.
- **Auto-confirm: fails by 2 frames.** The frame above, plus FRESH_02 player 15's second hop T000032 → T000041
  (frame 245, gap 0.27 s).
  - T000041 is player 16's track; it names three predecessors, two of them still live.
  - The simulated user rejects this one (judged wrong), so in mode 3 it is only one wrong CANDIDATE frame.
- **Wrong candidates:** default 1 frame (player 9, frame 276, same cause); simulated user 1 frame (player 15,
  frame 246, never confirmed).

### A guard that was checked and NOT adopted
- Rule: refuse a candidate whose `possible_predecessors` also names another track still live at the proposal.
- Over the 18 auto-mode relinks it would refuse 11 right ones to avoid the 1 wrong one. It would also not have
  prevented the player 9 frame, which happened after a right proposal.
- Requiring the candidate to name only the lost track is stricter still: 12 of 18 relinks name 2 or 3
  predecessors, and 11 of those are right.
- Fitting a rule to 18 relinks is not justified; left for Nathan.

## Other findings
1. **Relinking with confirmation is worth it:** correct-time 0.509 → 0.597 pooled (+303 player-frames).
   - FRESH_08: 0.305 → 0.490. FRESH_07: 0.586 → 0.688.
   - 16 of 18 proposals right, 1 wrong, 1 undecided (FRESH_07 player 19: a 7-frame track never clearly
     attributed).
2. **Without confirmation, candidates are short-lived:** 12 of 13 default-mode proposals were dropped.
   - 9 CANDIDATE_NOT_USABLE: the new track briefly occluded, lost or uncertain.
   - 3 SEVERAL_CANDIDATES: a second track naming the lost one appeared later.
   - Of the 12 dropped proposals, 11 were right and 1 undecided.
   - In the app, a candidate needs a quick confirmation, or the drop rules for an unconfirmed candidate need
     revisiting. Candidates were shown for 4.9 % of player-time, almost all on the right player (169 / 173).
   - With confirmation (modes 2 and 3) the same relinks are kept, because a confirmed track follows the v1 rules.
3. **FRESH_05 gets no relink:** all its losses are LEFT_FRAME re-entries over 0.88 s, outside this rule by design.
4. **The remaining gap to the ceiling (0.597 vs 0.853)** is:
   - long re-entries (re-ID model / team classification);
   - losses where the tracker named no unique successor;
   - abstentions (uncertain / unconfirmed frames).

## Decision needed (Nathan)
- Accept relink v1 as built: safe in the shipping configuration, +8.8 points with confirmation, at +1 wrong frame
  from a tracker identity change after a correct confirmation.
- Or first make a confirmed track subject to an extra check: for example, abstain while a track that named
  several predecessors overlaps one of the others. That needs a measurement like `tools/guard_cost.py` before
  any threshold is chosen.

## Files
- `relink_summary.json`: per mode, per clip, counts, relink list (with verdicts and drop reasons) and wrong events.
- `HASHES.txt`: full reports (on the Mac), config copy for auto-confirm, input tracks.
