# Re-entry offers by re-ID (DEV, 2026-11-18)

Option 1 of the re-ID choice (Nathan, 2026-11-18): build offers so Nathan can test them in the app.
- When the followed player's track is lost and the player comes back on a new track, the app offers that track
  ("looks like your player coming back: confirm?").
- Nothing is shown as the player without the user's confirmation.
- Known before building: an accepted offer counts as one review action, like a tap. So the review measure can't
  show its main benefit, which is one click instead of finding the player.

## Built (commit `45261f8`, branch `identity/reentry-reid-offers`)
- **Re-ID signatures per track** (`spana/identity/reid.py`, `python -m spana.cli embed`, **`spana.track_embeddings`
  1.0.0**).
  - The SPANA re-ID model v2 is pinned by sha256 (`models/spana_reid_v2.json`; weights stay on the Mac, not in git).
  - It embeds up to `reid.samples` 5 crops at each end of a track, at least 0.2 s apart. Crops must be clean:
    confirmed, CERTAIN, uncontested, >= 40 px tall, overlapping no other box.
  - The signatures are written as unit-length `first` / `last` vectors. A track without a clean crop has none and
    is never offered.
- **Re-entry offer in the follower** (`target.reentry_offer`, **`spana.target` 2.4.0**, CANDIDATE reason
  `REENTRY_REID`).
  - When: the followed track A is lost or has ended, and the predecessor relink window (0.5 s) has passed without
    an offer.
  - Which track: B, born within `reentry_max_gap_s` 10 s of A's last observation and usable now, whose first
    signature is closest to A's last one.
  - Conditions: similarity >= `reentry_min_similarity`, and above every other returning track by
    `reentry_min_margin`.
  - Accepted / declined like any offer. Never auto-confirmed, whatever `relink_auto_confirm` says. A declined offer
    is not repeated for that loss.
  - Evidence: the relink entry records kind, similarity, runner-up and candidates. `validate_target(doc, tracks,
    embeddings)` recomputes them.
- `spana.cli target --embeddings`, `tools/target_score.py` (embeddings per clip, offers reported by kind and
  verdict), and the review-video text in `tools/target_render.py`.
- Config schema 2.5.0 (+ `reid` section).
- Tests: `tests/test_reentry_offer.py` (14) and `tests/test_reid_embed.py` (3, a tiny ONNX model on a synthetic
  clip). Full suite 465 passed, 6 skipped.

## Measured
- **Scope:** DEV FRESH_02 / 05 / 07 / 08, today's default tracks. Signatures computed on Nathan's Mac (every
  confirmed track got one, e.g. FRESH_07 43 / 43).
- **Offer verdict:** judged against the labels on the first labelled frame with a clear attribution. "Undecided" =
  no such frame.

**Offers without review (all 4 clips):**

| min similarity / margin | offers right | wrong | undecided |
|---|---|---|---|
| 0.60 / 0.10 (prior) | 19 | 6 | 8 |
| 0.70 / 0.10 | 17 | 4 | 3 |
| **0.75 / 0.10 (chosen)** | **16** | **1** | 2 |
| 0.75 / 0.20 | 11 | 1 | 0 |
| 0.80 / 0.10 | 9 | 1 | 1 |

At 0.60 the wrong offers had similarity 0.63–0.81; the right ones 0.62–0.89.

**With the simulated after-match reviewer** (accepts right offers, declines wrong ones, else taps):

| | correct | wrong frames | actions / followed min | offer accepts / taps |
|---|---|---|---|---|
| no re-ID offers (today) | 0.778 | 2 | 5.64 | 17 / 62 |
| 0.60 / 0.10 | 0.757 | 2 | 5.50 | 38 / 35 |
| **0.75 / 0.10** | **0.776** | 2 | **5.57** | more accepts, fewer taps |
| 0.80 / 0.10 | 0.777 | 2 | 5.50 | |

## Findings
1. **Most offers are right at 0.75 / 0.10:** 16 of the 17 that could be judged. They replace taps with
   accept-clicks, so actions fall slightly (5.64 → 5.57) and wrong frames stay at 2.
2. **Correct time is −0.002 at 0.75 (−0.021 at the 0.60 prior).** The simulated reviewer accepts an offer on the
   frame it is made, which can be later than the first frame the player is visible. A tap would have gone on the
   gap's first labelled frame. Lower thresholds make more such offers, and more wrong ones.
   - A real user can tap earlier or accept later. The app test will show which matters.
3. **The thresholds are fitted on the same 4 DEV clips** (17 judged offers at the chosen setting). Treat them as a
   starting point for the app test, not a validated value.
4. **No change to answers without the user:** unaided correct time stays 0.532, 1 wrong frame. The follower never
   shows an offered track as the player on its own.

## How to try it
- Signatures for the 4 DEV clips' default tracks are on Nathan's Mac:
  `~/spana-continuity-run/out/F0x_mdefault.emb.json`.
- `spana.cli target ... --embeddings FILE` gives a target document with offers, and `tools/target_render.py` draws
  them as dashed amber boxes ("looks like your player coming back: confirm?").
- Wrong offers to look for: same-kit team-mates re-entering in the same pan.
