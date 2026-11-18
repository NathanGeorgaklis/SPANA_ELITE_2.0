# Roboflow ByteTrack vs the SPANA tracker on DEV (2026-11-18)

Question (Nathan): why not let a Roboflow tracker follow the player once the user has tapped them?

Scope:
- **Clips:** DEV FRESH_02 / 05 / 07 / 08.
- **Detections:** the same local-detector detections fed to both trackers; only the tracker differs.
- **Scorers:** the same for both.
  - Tracking: `tools/benchmark.py`, NESTED_V2.
  - Following: `tools/target_score.py`.
- No tuning. EVAL not used. Run on Nathan's Mac.
- The product is unchanged.

## Method
- **ByteTrack:**
  - supervision 0.25.1 `sv.ByteTrack` with its defaults (`config/bytetrack_compare.yaml`): activation 0.25, lost buffer
    30 frames at 30 fps (1 s), matching 0.8, min consecutive frames 1.
  - This is the classic Roboflow tracker that the Workflows "Byte Tracker" block wraps.
  - Newer supervision releases moved it out, so 0.25.1 was loaded from its own folder via PYTHONPATH; the project's
    venv is unchanged.
- **Input:** every person-view detection, class-agnostic, as our tracker uses them.
- **Conversion:** `tools/dev/bytetrack_compare.py` writes ByteTrack's output as a validated `spana.tracks` artifact.
  - Outputs become ACTIVE rows on the exact detection rows.
  - Gaps inside ByteTrack's lost pool become LOST rows.
  - A track ends when ByteTrack removes it (`max_time_lost`).
  - Identity is always CERTAIN: ByteTrack has no notion of an uncertain identity.
  - Tests: `tests/test_bytetrack_compare.py`, including the real ByteTrack end to end.
- **Our tracker:** the existing local-detector tracks. Its numbers reproduce `records/target_recovery_v1_2026-11-17`
  exactly (1,756 / 1 wrong unaided; 2,685 / 2 wrong / 90 actions with the reviewer).

## Results

**Tracking (pooled over the 4 clips; same detections):**

| | ours | ByteTrack |
|---|---|---|
| broken tracks (fragments) | 120 | **49** |
| ID switches | 121 | **52** |
| identity transfers (a track moves to another player) | **6** | 14 |
| of which unflagged (silent) | **0** | **14** |
| confirmed tracks, median life (F02 / F05 / F07 / F08) | 4.4 / 3.6 / 2.1 / 1.5 s | 8.1 / 3.5 / 9.0 / 2.6 s |

**Following one tapped player** (3,449 labelled player-frames, 14.4 followed player-minutes):

| | correct time | wrong-player time | no answer | review actions per followed min |
|---|---|---|---|---|
| ours, no user help | 0.509 | **1 frame (0.03 %)** | 0.433 | 0 |
| ByteTrack, no user help | 0.665 | **217 frames (6.3 %)** | 0.248 | 0 |
| ours + simulated after-match reviewer | 0.778 | **2 frames (0.06 %)** | 0.211 | 6.27 |
| ByteTrack + simulated after-match reviewer | 0.826 | **278 frames (8.1 %)** | 0.058 | 2.72 |

**Wrong-player time per clip, ByteTrack:**

| clip | no help | with reviewer |
|---|---|---|
| FRESH_02 | 5.7 % | 9.1 % |
| FRESH_05 | 0 | 0 |
| FRESH_07 | 13.2 % | 13.1 % |
| FRESH_08 | 0 | 3.5 % |

## Findings
1. **ByteTrack keeps tracks alive much longer.** Fragments and ID switches drop by more than half. It also bridges
   short gaps that our tracker ends, so followed time is higher and the review needs fewer than half the actions
   (2.7 vs 6.3 per minute).
2. **It pays for that with silent identity swaps.** All 14 of its identity transfers are unflagged. A follower on
   ByteTrack shows the wrong player 6–8 % of the time overall, and 13 % on FRESH_07: about 1 second in 8, with no
   warning.
3. **After-match review cannot catch these swaps.**
   - The review works through the gaps (lost / unsure / offers). A silent swap is not a gap: the app keeps
     confidently following someone else.
   - Review taps can even increase wrong time (6.3 % → 8.1 %), because a re-selected ByteTrack track can swap again
     later.
   - With our tracker, wrong time stays at 1–2 frames because uncertainty is flagged and the follower abstains.
4. **Not adopted as is.** It breaks the hard rule "no identity switching on uncertainty", and any stats or clips built
   on those frames would credit the wrong player.
5. **What is worth taking from ByteTrack:** its continuity. That comes mainly from:
   - a second association pass on low-score detections;
   - a lost pool that re-matches by motion alone.
   Our tracker is stricter: it ends tracks that it cannot re-associate safely.

## Suggested next step (not built)
Combine the two strengths and measure them on these same clips:
- **Option 1:** ByteTrack-style continuity inside our tracker, keeping its identity-uncertainty flags.
  Concretely: a second low-score association stage, and re-matching LOST tracks by motion only when no same-kit
  rival is near.
- **Option 2:** ByteTrack's tracks with our overlap / contested detection layered on top, so the follower abstains
  around crossings.

Success is judged on the same scorers:
- wrong-player time must stay at about 0 (≤ ours + 1 frame);
- correct time and review actions per minute should move toward ByteTrack's.

## Files
- `compare_summary.json`: all numbers above, per clip and pooled.
- `bytetrack_compare.config_snapshot.yaml`.
- `HASHES.txt`.
