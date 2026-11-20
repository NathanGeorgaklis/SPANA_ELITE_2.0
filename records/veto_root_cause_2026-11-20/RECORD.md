# Why the colour veto rejects a player's own box (FRESH_08 player 10, frame 148), and two-way refusals on DEV (2026-11-20)

CODE TASK (Nathan): analysis only, DEV only. Main `74eb1a0` (code31/), replaying the runs that made
`out/F0x_main.trk.json` (same detections and config). **No code or default changed.**

## 1. Player 10, frame 148
- T000029's own next box (distance 0.04 h, IoU 0.80, height ratio 1.06, no other claimant) was vetoed. Its torso
  crop's similarity to the stored appearance was 0.6645, below `appearance.min_similarity` 0.70.
- **The stored appearance was learned only on frames 45–73** (21 updates from full-body boxes about 111 px tall;
  each crop agreed with the model at 0.84–0.97). The camera was panning fast then: median 27 px / frame over the
  last 10 updates.
- **From frame 74 to 147 there was no clean crop.** On 140–147 his box overlapped a neighbour's detection by IoU
  0.056–0.094, above `appearance.clean_max_iou` 0.05, so the descriptor was UNKNOWN and the model froze for 2.5 s.
- On frame 148 the overlap dropped to 0.007 and the first clean crop was taken, with the camera nearly still
  (3.5 px / frame). By then the player had crossed from mid-picture (x ≈ 900) to the left edge (x ≈ 90).
- Crops (`out/veto148/`):
  - frame 50 (the model): heavy horizontal motion blur, and the torso band is about half background;
  - frame 148: sharp, the band almost entirely red shirt with the white number.

  The histograms describe different things: blurred red plus background, against crisp red.
- **Hypothesis rejected:** the partial-body boxes of frames 95–98 never updated the appearance; every update came
  from a full-body box.

**Root cause:** an appearance model learned from motion-blurred crops during a fast pan, compared after a 2.5 s
freeze against a sharp crop.

## 2. Is it generic? Every veto that alone stopped a match on DEV
`probes/probe_veto_all.py`, 4 clips: 115 veto rows, 72 first vetoes per track. Verdict from the labels:
- **own** = the vetoed box spawned a track owned by the vetoed track's player (a wrong veto);
- **other** = another player (a right veto);
- **unknown** = no labelled owner.

| | own (wrong veto) | other (right veto) | unknown |
|---|---|---|---|
| first vetoes | 22 | 10 | 40 |
| similarity, median (q25–q75) | **0.67 (0.64–0.69)** | 0.27 (0.07–0.57) | 0.60 |
| stored appearance age at the veto, median (q75) | **0.03 s** (1.03) | 1.43 s (2.73) | 0.03 s |
| camera speed while the appearance was learned, median (q75), px / frame | 1.4 (15.3) | 0.14 (0.85) | 1.3 |

1. **A stale model is not the general cause.** Most wrong vetoes compare against an appearance updated on the
   previous frame. Player 10's 2.5 s freeze is the exception.
2. **Pan blur is a recurring cause.**
   - 9 of the 22 wrong vetoes (all on FRESH_08) had their appearance learned at ≥ 9 px / frame.
   - Of the 10 right vetoes, 1 was (FRESH_08 frame 47), at similarity 0.07, far from the line.
   - FRESH_02 / 05 / 07 have slow cameras: their wrong vetoes happen with fresh, sharp models.
3. **The threshold sits on the boundary.**
   - 21 of 22 wrong vetoes are at 0.53–0.70; 18 are at ≥ 0.60.
   - 2 right vetoes are in the same band (FRESH_02 frame 178 at 0.616, FRESH_07 frame 282 at 0.668).
   - Lowering `min_similarity` to 0.60 would let 18 own boxes through, and both of those right vetoes: two identity
     swaps. **Not safe.**
4. **What a fix would target** (not built; for a future CODE TASK): appearance learned under heavy camera motion.
   - Options:
     - do not update the stored appearance from crops taken above a camera-speed or blur bound;
     - or treat such a model as weak evidence (a veto from it abstains instead of vetoing);
     - or keep the sharpest crops instead of an average.
   - On DEV this would address up to 9 of 22 wrong vetoes, with 1 right veto (similarity 0.07) in that regime.
   - 40 vetoes have no labelled owner and must be watched when it is built.
   - The cleanliness test (detection IoU <= 0.05) froze player 10's model for 2.5 s in a crowd. That is safe (no
     evidence rather than mixed evidence), but it leaves fast-moving players with stale models.

## 3. Two-way (mutual) check: refused genuine crossings across DEV
`probes/count_not_mutual.py`, main's follower (unaided), 72 players. Each distinct NOT_MUTUAL decision, judged by
the refused track's labelled owner:

| refused track was | count |
|---|---|
| another player (a wrong offer prevented) | **6** |
| the followed player (a right offer withheld) | 1 (FRESH_08 player 14) |
| unknown | 0 |

- For scale: 16 re-ID offers were made.
- In 3 of the 6 prevented cases the right player was offered later anyway.
- **The two-way check prevents six wrong offers for each right one it withholds. Keep it; no crossing
  discriminator** (Nathan, 2026-11-20).
