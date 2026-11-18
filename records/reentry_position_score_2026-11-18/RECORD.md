# Re-entry offer: position as a score alongside appearance (DEV, 2026-11-18)

CODE TASK "option 1" (Nathan). Before this change a returning track was chosen on appearance alone, and position was
only a pass/fail gate that rejected almost nothing.

## Built (branch `identity/reentry-position-score`; final code `b489373`)
- **`spana.identity.reid.position_fit`.**
  - Space: the image space of the existing gate. The lost player's last box centre is carried through the per-frame
    camera transforms, and distances are measured in body heights.
  - Formula: fit = max(0, 1 − d / (base_h + speed_h_per_s · gap)) · exp(−gap / time_scale_s), in [0, 1].
  - **Silenced (fit 0):** a CUT / UNKNOWN camera frame in the gap, the track ended LEFT_FRAME, the carried point
    left the picture, the camera moved the scene by more than `max_pan_frac` of the frame width, or no frame size.
    Any of these silences position for the WHOLE decision, including the two-way check: the ranking is then exactly
    appearance-only, so a pan return is never out-ranked by position.
  - A gap above `max_gap_s` silences only that candidate.
  - Camera / exit reasons are checked before the gap limit.
- **`reentry_choice`.**
  - Ranks by combined = appearance + weight · fit.
  - Unchanged: the appearance floor (`reentry_min_similarity`), the margin rule and the two-way (mutual) rule, now
    on combined scores. The existing gate is untouched.
  - Always returns a decision: an offer, or an abstention reason (NO_SIGNATURE, NO_CANDIDATE, BELOW_MIN_SIMILARITY,
    MARGIN, NOT_MUTUAL).
- **Logging** (`spana.target` 2.5.0):
  - `reentry_log`: each change of decision during a loss, with appearance, position, combined, informative and
    reason, runner-up, margin, reverse runner-up, and every candidate's scores.
  - Re-ID relink evidence carries similarity (appearance), position, combined, position_informative and
    second_combined.
  - The validator checks combined = appearance + weight · position and the margin; with the embeddings it also
    recomputes appearance.
- **Config schema 2.7.0, all values in `config/default.yaml`:** `reentry_position_weight`, `_base_h`,
  `_speed_h_per_s`, `_time_scale_s`, `_max_gap_s`, `_max_pan_frac`.
- **Tests** (`tests/test_reentry_offer.py`, +7):
  - a close and early candidate outranks a slightly more similar far and late one;
  - a pan or exit ranks exactly as appearance-only;
  - an exit silences every candidate;
  - a silenced decision is appearance-only in the two-way check too;
  - weight 0 is appearance-only whatever the scales;
  - the margin rule still abstains when the top two are close;
  - offers and abstentions are logged with their scores.

  Full suite 475 passed, 6 skipped.

Fitting fixed three issues found on the way, before choosing values:
1. A pan return could be out-ranked by a candidate that position favoured. Now a camera / exit reason silences the
   whole decision.
2. The gap limit was checked before the camera path. Now camera / exit is checked first.
3. The two-way check gave position credit to other lost players on silenced decisions. Now that check is
   appearance-only when the decision is silenced.

After these, pan returns are identical to appearance-only at every grid point.

## Fit (DEV only; 79 scorable labelled re-entries; `tools/dev/reentry_position_fit.py`)
**Grid:**
- weight {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0};
- speed {1, 2, 5} body heights / s;
- time scale {0.5, 1, 2} s;
- max gap {1, 3} s;
- `base_h` = 0.35 (= `tracking.gate_base_h`) and `max_pan_frac` 0.25 not fitted.

**Lenses:**
- "rank1": no floor, no margin, no two-way check;
- "shipped": floor 0.75, margin 0, two-way check (the offer rule).

**Chosen:** weight 0.5, speed 2.0, time scale 1.0 s, max gap 3.0 s.
- Rule: shipped wrong must not rise; then most shipped right; then fewest rank-1 wrong.

| setting | lens | right | wrong | missed | short (< 1 s) r / w / m | long r / w / m | no pan r / w / m | pan r / w / m |
|---|---|---|---|---|---|---|---|---|
| appearance only (before) | rank1 | 60 | 17 | 2 | 49 / 12 / 2 | 11 / 5 / 0 | 48 / 13 / 2 | 12 / 4 / 0 |
| **chosen** | rank1 | **64** | **13** | 2 | 53 / 8 / 2 | 11 / 5 / 0 | 52 / 9 / 2 | 12 / 4 / 0 |
| appearance only (before) | shipped | 33 | 2 | 44 | 26 / 1 / 36 | 7 / 1 / 8 | 27 / 2 / 34 | 6 / 0 / 10 |
| **chosen** | shipped | **36** | **2** | 41 | 29 / 1 / 33 | 7 / 1 / 8 | 30 / 2 / 31 | 6 / 0 / 10 |

(16 events are "pan", meaning position cannot speak for the true track; 63 have gaps under 1 s.)

**Sensitivity:**
- Rank-1: the gain holds from weight 0.2 to 0.5 (64 right / 13 wrong at the chosen scales; 62–66 / 11–15 across
  scales). It shrinks at 0.75 and is mostly gone at 1.0.
- Shipped: noisier.
  - Weight 0.1–0.3 adds one wrong offer (3 instead of 2).
  - Weight 0.5–0.75 keeps 2 wrong, with 33–36 right across scales.
  - So the shipped gain is +1 to +3 right offers at weight 0.5–0.75, and not present at 0.1–0.3.
- **Not a single exact value, but small and noisy:** a few events decide each row.

**The 17 rank-1 wrong picks** (the 14 examined in `records/reentry_position_mutual_2026-11-18` are among them):
- 6 fixed (now right), all short gaps: FRESH_02 player 15, FRESH_08 players 9, 13 (one of two), 8 (one of two), 18
  and 19 (one of two).
- 11 still wrong: the 5 long-gap / pan ones (position silenced or not decisive) and 6 short-gap ones in the FRESH_08
  crowd.
- **2 new wrong picks:** FRESH_02 player 8 (T000036 → T000033) and FRESH_07 player 5 (T000056 → T000046). A nearer,
  earlier track out-scored the right one.

**Shipped rule:**
- 1 wrong fixed (now an abstention), 1 new wrong (FRESH_07 player 7, previously an abstention).
- 3 right offers became abstentions (FRESH_02 player 10, FRESH_07 player 5, FRESH_08 player 18), and 6 abstentions
  became right offers.

## In the follower (4 DEV clips; offers are only for tracks usable at the time)
| | unaided correct / wrong | re-ID offers right / wrong / undecided | reviewer correct / wrong / actions per min |
|---|---|---|---|
| weight 0 (= previous offers, verified identical) | 0.532 / 1 | 16 / 0 / 1 | 0.776 / 2 / 5.64 |
| **weight 0.5 (new default)** | 0.532 / 1 | 15 / 0 / 1 | 0.777 / 2 / 5.71 |

- One right offer gained: FRESH_08 player 19.
- Two right offers lost to abstention: FRESH_07 player 15, FRESH_08 player 18.
- Still no wrong offer.

## Findings
1. **Offline, position as a score fixes 6 of the 17 wrong rank-1 picks, all in short gaps,** and adds 2 new ones.
   Net wrong 17 → 13, right 60 → 64. At the offer rule it adds 3 right offers with wrong unchanged.
2. **Pan / exit returns are untouched by design,** and so are long gaps: position cannot help them. That is option 2's
   job (better appearance signatures).
3. **In the follower the effect is neutral on DEV:** 15 vs 16 right offers, 0 wrong, reviewer +0.001, actions +0.07
   per minute. The follower offers only tracks usable at the time, one at a time, so the offline gain mostly
   doesn't reach it on these 4 clips.
4. **The small sample decides single events at every setting.** Treat the chosen values as a starting point, the
   same as the other re-ID thresholds.
5. **Unchanged:** the target state machine (uncertain / after-uncertain / relink / re-selection). Weight 0 reproduces
   the previous target documents' scores exactly; all target-identity tests pass.
