# FRESH_08 review follow-up: players 14 and 10, and the doubt link re-measured (DEV, 2026-11-20)

Diagnosis and measurement only. **No code or default changed.** Main `24ccd58` (code29/) on main's re-tracked,
re-embedded DEV clips. No EVAL, no full match.

## Player 14: a right offer on the old tracks, none on main
On main, his real returning track T000068 (born frame 258, labels: player 14) is ranked first:
- appearance 0.828, position 0.357, combined 1.007.

It is refused NOT_MUTUAL by **T000059 (player 13)**, which was last observed on frame 257, next to the birth:
- appearance 0.655, position 0.911, combined 1.11.

This is a genuine crossing:
- On frame 252, T000059's box was attributed to player 14.
- T000068 lists T000059 as its possible predecessor.
- Player 13 continues as T000072.

The two-way check is doing its job (fail closed). The old run's right offer at 263 came from a different signature
mode, which scored the rival lower. **No change.** Loosening the two-way check would remove exactly this protection
and needs Nathan's decision.

## Player 10: lost after frame 147
On frame 148, T000029's own next box failed only the torso-colour veto:
- distance 0.04 h, IoU 0.80, height ratio 1.06, not taken by any other track;
- similarity 0.66 against a minimum of 0.70.

The box spawned T000056, which is player 10 to the end of the clip. The follower then searches from his original
track T000012, because the T000029 offer was never accepted. At 154 it refused T000056 on the margin rule. An
unconfirmed offer is never chained (by design): a reviewer who accepts T000029 gets a new search from T000029's loss.

This is the colour-veto class of `records/continuity_attribution_2026-11-18`. Its fix (the appearance doubt link)
is OFF by decision (2026-11-18). It is re-measured below on today's tracker.

## Doubt link re-measured on main's tracker (rival_eligible_only + extent link on)
Re-tracked FRESH_02/05/07/08 from the same detections. Follower without re-ID offers.

| | main | doubt link | + self-consistent resolution (a) | + (a) + tap vouch + resume (ab) |
|---|---|---|---|---|
| fragments | 105 | 98 | 98 | 98 |
| identity transfers (silent) | 7 (0) | 8 (0) | 8 (0) | 8 (0) |
| unaided correct / wrong frames | 0.536 / 1 | 0.533 / 1 | 0.533 / 1 | 0.556 / 1 |
| predecessor offers right / wrong / undecided | 10 / 0 / 2 | 7 / 0 / 1 | 7 / 0 / 1 | 8 / **1** / 1 |
| reviewer correct / wrong / actions per min | 0.783 / 2 / 5.50 | 0.694 / 2 / 5.02 | 0.694 / 2 / 5.02 | 0.782 / 2 / 5.36 |

**(ab) now meets four of the five bar conditions** (reviewer >= 0.778, fewer actions, wrong frames <= 2, silent 0;
on 18 Nov it failed on actions). It fails the fifth: the extra transfer (FRESH_08 frame 258, player 13 → 14,
flagged, uncertain) is on a doubt-linked track (T000050). It is the same 13 / 14 crossing as above. It also adds a
wrong predecessor offer (FRESH_02 player 16 offered player 17).

**Stays OFF** (decision of 2026-11-18 unchanged). Player 10's frame-148 split stays open.

## 10-minute DEV window check of `tracking.rival_eligible_only` (records/rival_eligible_2026-11-19)
G2 frames 170,250–188,249, existing detections, main with the flag off and on. The off run reproduces the 18 Nov
window numbers exactly (1,747 tracks, ends 952 / 777).

| measure | off | on |
|---|---|---|
| confirmed tracks started in 10 min | 1,747 | **1,703** (−2.5 %) |
| starts per visible player-minute | 21.8 | 21.3 |
| ends LOST_EXPIRED / LEFT_FRAME | 952 / 777 | 911 / 774 |
| observed confirmed track-frames IDENTITY_UNCERTAIN | 17.8 % | 18.2 % |
| S3c links (continuing / across a gap) | 124 / 47 | 169 / 65 |

**Safety proxy** (unlabelled window), within 1 s after a link that exists only with the rule:

| | new uncertainty episode | track ends |
|---|---|---|
| links only with the rule (n = 324) | 8.0 % | 17.9 % |
| any confirmed observed track-frame, off / on | 7.6 % / 7.5 % | 9.8 % / 9.8 % |

- The 0.5-point difference in new uncertainty is within noise (about ±1.5 points at n = 324).
- Ends after links are high, as for S3c links on 18 Nov (19.9 %): links happen in hard moments.
- The link count is approximate: S4 recoveries do not log their detection, so renumbered tracks inflate "only
  with".
- **No sign of harm; the rule stays on.** The tracking bar's missing window check is now done.
