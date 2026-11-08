# IDENTITY_UNCERTAIN one-sided exclusion test: DEV validation (2026-11-08)

- Scope: DEV only. EVAL was not used.
- Runs on the same detections as `records/identity_uncertain_2026-11-08` (F02E, F05E, F07E → F02F, F05F, F07F).

## Change (`MultiPlayerTracker._identity_resolution`)

The track resolves when every rival pair is cleared on the same clean frame, and no pair has clean evidence
against it for `identity_resolve_s` (0.5 s). A frame without evidence for some pair holds the timer; it neither
advances nor resets it. Each rival pair is decided by one of these rules:

| rule | when it applies | clears the pair when |
|---|---|---|
| TWO_SIDED (unchanged) | both appearances stored | the crop matches own (≥ 0.70) and beats the rival's by ≥ 0.10 |
| RIVAL_ONLY | only the rival's appearance stored | the crop is clearly unlike the rival: ≤ 0.70 − 0.10 = 0.60 |
| OWN_ONLY | only own appearance stored | the crop matches own (≥ 0.70) and the rival's latest clean crop, taken after the latest contest, is ≤ 0.60 against own; no such crop means no evidence |
| none stored | neither appearance stored | never |

Clean, uncontested crops are now recorded per track as evidence only. They never change the appearance model,
which stays frozen while the track is uncertain.

## Results (NESTED_V2)

| | FRESH_02 | FRESH_05 | FRESH_07 |
|---|---|---|---|
| uncertain confirmed observed track-frames | 835 → **688** (5.5 % → 4.6 %) | 126 → 126 (4.5 %) | 912 → 912 (8.1 %) |
| tracks resolved | 3 → **5** | 1 → 1 | 0 → 0 |
| silent swaps | 0 → 0 | 0 → 0 | 0 → 0 |
| flagged swaps | 5 → 5 | 0 → 0 | 0 → 0 |
| coverage | 0.9607 → 0.9607 | 0.9689 → 0.9689 | 0.9595 → 0.9595 |
| ID switches / fragments / recall | unchanged | unchanged | unchanged |

**New resolutions, both on FRESH_02:**
- **T000279**, crowd-born, at frame 317 by RIVAL_ONLY against T000002.
- **T000002**, at frame 318 by TWO_SIDED against T000005 and OWN_ONLY against T000279.

Both are correct against the frozen labels. T000002 is player 13 and T000279 is player 12 at every labelled frame
from 282 to 390, before and after the resolution, with no transfer.

**FRESH_07: nothing new resolved.**
- T000175 had 3 resets: its crop looks like its rivals (similarity > 0.60). These are same-looking players.
- T000191 still has a rival, T000247, where neither appearance is stored, so that pair can never clear.
- T000252 and T000216 also had resets.

## Tests
- `tests/test_identity_uncertain.py::OneSidedResolution` covers:
  - safe crowd-born resolution, with the rule used recorded;
  - the full 0.5 s window is required;
  - a same-kit pair stays uncertain;
  - a true takeover never resolves;
  - neither appearance stored stays uncertain.
- The existing two-sided tests are unchanged and pass.
- Full suite 119 passed. Synthetic QA output is identical to before.

## Tracks (not committed)
- b10d6025…b2a9c67c  F02F.trk.json
- 23229cce…13a696b8  F05F.trk.json
- be52a9fd…c568dae8  F07F.trk.json
