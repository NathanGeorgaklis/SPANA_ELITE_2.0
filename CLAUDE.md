# SPANA Elite 2.0 — instructions for Claude sessions

## Persistence (mandatory)
- The source of truth is GitHub: https://github.com/NathanGeorgaklis/SPANA_ELITE_2.0 (branch `main`).
- Every code, config, test, tool or record change is committed AND pushed to this repo before a task is reported
  done — and before any long-running job, so work is never only in a session workspace.
- After pushing, verify the remote: `git fetch origin && git status -sb` must show no ahead/behind, or compare a
  fresh clone's per-file sha256 with the working tree.
- Never leave code only in chat, /tmp, a notebook or terminal history. Footage, model weights and run outputs are
  not committed (see .gitignore); their hashes and the commands that produce them are.
- Commit messages end with the session attribution lines provided by the environment.

## Project rules (summary; Nathan's full rules apply)
- Generic only: no player, team, kit, clip, frame, timestamp, venue or match-specific logic. Seconds and
  body-height units; thresholds in config/default.yaml.
- Diagnose the first failing layer before changing code; smallest generic fix; one regression test per failure.
- Fail closed: UNKNOWN / LOST / new uncertain track over a confident wrong answer.
- DEV and EVAL strictly separate: never inspect, tune on or score EVAL without Nathan's explicit approval.
- Human labels are frozen (tools/freeze_labels.py) before any inference is scored against them; never edited.
- Route work through the specialist roles and start replies with "AGENTS USED:".

## Layout
`spana/` engine · `config/` · `tests/` (python -m pytest tests -q) · `tools/` · `records/` stage and validation
records · `data/labels/` frozen DEV labels · `models/` pinned model records (weights via tools/fetch_model.py).
