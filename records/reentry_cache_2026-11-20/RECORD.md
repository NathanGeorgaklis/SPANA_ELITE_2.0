# Re-entry rule memo: DEV scoring from minutes to seconds, exactly the same results (2026-11-20)

Owner layer: target identity. Branch `identity/reentry-cache` (from main `24ccd58`), code `16df6d8`.

## Finding (profile)
`tools/target_score --simulate-reviewer` on FRESH_08 (main's tracks and embeddings): 178 s.
- 173 s was in `follower._reentry_offer` → `reid.reentry_choice` (14,147 calls).
- The time inside it:

  | function | time |
  |---|---|
  | `exit_side` | 108 s |
  | `position_fit` | 31 s |
  | crop-set `appearance` | 30 s |
- Why: the follower re-runs the re-entry rule on every frame of a loss. Each run carried the lost box through every
  camera transform since the loss, for every candidate, giving the same answer as on the previous frame.

## Change
- `ReentryContext.memo` (excluded from equality): `exit_side`, `position_fit` and `appearance` are memoised per
  context.
- Each is a pure function of the document, the signatures and the config values in its key:
  - `exit_side`: (lost, last index, candidate);
  - `position_fit`: + the six `reentry_position_*` values;
  - `appearance`: + signature mode, wait samples, and the frame cutoff only while waiting.
- No rule changed and no value is approximated.
- Test: `tests/test_reentry_cache.py`. Across 4 scenes × 5 rule variants × every frame, a shared context decides
  exactly like a fresh context per call, and the memo does not change context equality. Full suite 510 passed,
  6 skipped.

## Verified on DEV (4 clips, main's tracks / embeddings, Nathan's Mac)
- Unaided scoring 11 s and reviewer scoring 12 s for all 4 clips. Before, the same runs took minutes (178 s for
  FRESH_08's reviewer alone, profiled).
- Both reports are identical to main's (`target_final_*`), apart from target-document hashes (provenance time).
- All 72 target documents are identical apart from provenance.

## Note
Context7 (library docs) was not needed. The cost was in our own rule, not in OpenCV / ONNX Runtime.
