# Could re-ID re-link a returning player automatically? Offline check on DEV (2026-11-18)

**Question:** with re-ID v2, is there a setting at which a player coming back on a new track could be re-linked
WITHOUT the user, with essentially no wrong links? Only that kind of re-link would cut review actions
(`records/reentry_reid_offers_2026-11-18`).

- **Tool:** `tools/dev/reid_reentry_eval.py` (commit `b1e414c`). No product change. EVAL not used.
- **Ground truth:** frozen DEV labels of FRESH_02 / 05 / 07 / 08, NESTED_V2 attribution.
- **Re-entry:** a player's consecutive attributed tracks A → B where B is born after A's last observation.
- **Candidates:** every track born within 10 s after it.
- **Rule:** as the offer rule, i.e. the best cosine(A.last, C.first), accepted when it reaches the minimum
  similarity and beats the runner-up by the margin.
- **Right:** the chosen track is the player's.

## Population
| | count |
|---|---|
| labelled re-entries | 94 |
| within 10 s | 93 |
| lost track has a re-ID signature (scorable) | 79 |
| ... of which the true returning track has a signature | 67 |
| median candidates per re-entry | 17 |
| best match is the right player (no threshold) | 60 / 77 (78 %) |

Gaps: 63 of the 79 are under 1 s (tracker breaks), 6 are 1–3 s, and 10 are over 3 s (pans / long re-entries).

## Accuracy at each setting (all 79 scorable re-entries)
"Upper 95 %" = exact one-sided 95 % upper bound on the wrong rate among accepted links.

| min similarity | margin | accepted (coverage) | right | wrong | upper 95 % |
|---|---|---|---|---|---|
| 0.60 | 0.10 | 39 (49 %) | 33 | 6 | 28 % |
| 0.75 | 0.10 (offer default) | 31 (39 %) | 28 | 3 | 23 % |
| 0.85 | 0.10 | 14 (18 %) | 13 | 1 | 30 % |
| 0.60 | 0.20 | 15 (19 %) | 15 | 0 | 18 % |
| 0.75 | 0.20 | 11 (14 %) | 11 | 0 | 24 % |
| 0.90 | 0.00 | 8 (10 %) | 8 | 0 | 31 % |

**Wrong confident picks:**
- 0.873 (margin 0.18), FRESH_08 player 13 → an unlabelled track;
- 0.838 (margin 0.16), FRESH_08 player 19 → player 18;
- 0.833 (margin 0.006), FRESH_08 player 17 → player 16.

13 of the 17 wrong rank-1 picks are in FRESH_08 (the set-piece crowd, up to 30 candidates).

## Findings
1. **Automatic re-linking is not safe with re-ID v2.** At useful coverage (~40–50 %) about 1 pick in 10 is the
   wrong player.
   - The settings with 0 wrong on DEV cover under 20 % of re-entries, and their wrong-rate bound is still 18 % or
     more.
   - Two wrong picks sit at margins 0.16–0.18, just under the strictest margin tried.
   - Showing a wrong player without warning breaks the fail-closed rule.
   - To show a wrong rate of 1 % or less you would need about 300 accepted links with no mistakes, far more than
     the 4 DEV clips hold.
   - **The 16 Nov rule (relinks need the user) stays.**
2. **Offers remain the right use.** The shipped offer setting (0.75 / 0.10) picks wrong 3 times in 31 offline.
   That is higher than the 1 in 17 the follower showed, because offline every track in the window competes.
   - A wrong offer costs the user a decline, not a wrong answer.
   - If the app test shows too many wrong offers, margin 0.20 gave 0 wrong of 11–15 offline at half the coverage.
3. **Crowds are where re-ID fails** (FRESH_08): many same-kit candidates within 10 s.
   - Better re-ID needs more training identities (TRAIN_M003 is ready) and/or team classification to prune
     candidates.
4. **Identity at crossings (the next option) is ruled out for now on the same evidence.** Resolving
   IDENTITY_UNCERTAIN automatically is a silent swap when wrong. A cue that is wrong about 1 time in 10 at practical
   coverage cannot carry it. The fusion test's strict operating point (TAR 0.21 at 0.5 % FAR on hard pairs,
   `records/reid_retrain_m001_m002_2026-11-18`) needs an end-to-end check before any build.
