# CODE REPORT: Football detector person view vs YOLOX: tracking on DEV FRESH_02 and FRESH_05

AGENTS USED: none. All work ran in one session; no subagents were spawned.

**COMMIT:** d979cf0 on `tracking/person-detections`, pushed. The remote showed no ahead/behind.

On both clips, tracking with the football detector (player, goalkeeper and referee merged into one "person" input) beats the YOLOX baseline on every metric. The biggest gains are far fewer duplicate boxes and, on FRESH_05, far fewer ID switches.

## FINDING

Scored against each clip's frozen labels, NESTED_V2 rule:

| metric | FRESH_02 YOLOX | FRESH_02 football | FRESH_05 YOLOX | FRESH_05 football |
|---|---|---|---|---|
| Detection recall | 0.973 | **0.980** | 0.989 | **0.998** |
| Missed players | 28 | **20** | 5 | **1** |
| Duplicate detections | 126 | **17** | 84 | **3** |
| Track coverage | 0.960 | **0.971** | 0.980 | **0.998** |
| ID switches | 28 | **25** | 22 | **8** |
| Fragments | 28 | **24** | 21 | **8** |
| Duplicate track frames | 16 | **5** | 3 | **0** |
| Unflagged identity transfers | 0 | 0 | 0 | 0 |

- **The role mix-up doesn't affect tracking.** On FRESH_05, 1,902 of 5,886 person boxes are tagged "referee" (the yellow-kit team), but the tracker ignores the class.
- **Fair comparison:** both detectors saw the same frames (391 and 781), with the same video hash, detection config, tracker and scorer. YOLOX was re-run with today's code; its numbers are close to the 8 November records.
- **Cost:** hosted inference was about 1,170 calls, fewer than the ~1,530 estimated, because the video reader decodes fewer frames than the file headers report.

## CHANGE

- `records/person_view_2026-11-16/`: RECORD.md, the four scorer reports, and `HASHES.txt` with exact sha256 values for all run artifacts. The artifacts themselves stay uncommitted in `runs/`.
- STATUS.md updated.
- No code, threshold or model changes in this step. Earlier in this task:
  - PR #3 merged into main (a88ad5f).
  - The person-view code is 9b3ac67.

## TESTS

- Full suite at 9b3ac67: 285 passed, 4 skipped.
- No code changed since then.

## EVIDENCE

The scorer reports in this folder, plus the hashes in `HASHES.txt`.

## REMAINING

- **Default tracking input.** Should the football detector's person view replace YOLOX as the default? Nathan's call. Each analysed frame is one hosted inference call; a local inference server would avoid that and keep frames on the machine.
- **Limits of this evidence:** two short clips (about 16 s and 18 s, 30 players). The tracker's settings were tuned on YOLOX output and weren't re-tuned here.
- **Next layers:** roles and teams belong to the team-classification layer. The FRESH_02 contact cluster still has 25 ID switches.

**STATUS.md UPDATED:** yes
