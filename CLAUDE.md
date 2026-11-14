# SPANA ELITE 2.0 — Claude Code rules

Read docs/STATUS.md, docs/DECISIONS.md and docs/INTERFACES.md at the start of every session.

## Your job
Implement CODE TASKs. Each task: one owner layer, one commit, pushed.
If a task conflicts with DECISIONS.md or these rules, stop and say so. Don't improvise.

## Workflow
reproduce → confirm first failing layer → smallest generic fix → targeted test
→ adjacent-layer check → commit + push → update STATUS.md (+ DECISIONS.md if a decision was made)
→ output CODE REPORT (format below).

## Hard rules
- No hard-coded players/teams/numbers/kits/venues/timestamps/frames/clips/angles.
- Thresholds in config/ only.
- Fail closed (UNKNOWN/LOST/ABSTAIN). No identity switching on uncertainty.
- Stats from validated events only. Clips from validated events only.
- Never touch EVAL data without explicit approval.
- No full-match runs unless the task says so.

## CODE REPORT
CODE REPORT: <title>
COMMIT: <hash> on <branch>
FINDING:
CHANGE:
TESTS: targeted | adjacent | full
EVIDENCE:
REMAINING:
STATUS.md UPDATED: yes/no
