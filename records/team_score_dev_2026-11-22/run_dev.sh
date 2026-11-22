#!/usr/bin/env bash
# Team classification v1 on DEV: spana.teams per match with config/default.yaml (priors, never fitted), then scored
# against the frozen side labels (teams_v1). Tracks: the default tracker run used since 2026-11-20
# (~/spana-player-confirmation-run/base, HASHES.txt there). Output: $OUT (outside the repo) + copies of the scores.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=~/SPANA_ELITE_2.0/.venv/bin/python
B=~/spana-player-confirmation-run/base
D=~/SPANA_ELITE_2.0/runs/person_view_compare
V=~/Downloads/SPANA_fresh_clips
L=data/labels
OUT=${OUT:-~/spana-team-score-run}
mkdir -p "$OUT"
C02=FRESH_02_G1_3000_midfield_buildup_passing
C05=FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
C07=FRESH_07_G3_1205_midfield_passing_switch
C08=FRESH_08_G3_2103_attack_into_box_clearance
$P -m spana.cli teams --match-id DEV_G1 --config config/default.yaml \
  --clip $V/$C02.mp4 $B/F02_ours.trk.json $D/F02_local.det.json --out "$OUT/DEV_G1.teams.json"
$P -m spana.cli teams --match-id DEV_G2 --config config/default.yaml \
  --clip $V/$C05.mp4 $B/F05_ours.trk.json $D/F05_local.det.json --out "$OUT/DEV_G2.teams.json"
$P -m spana.cli teams --match-id DEV_G3 --config config/default.yaml \
  --clip $V/$C07.mp4 $B/F07_ours.trk.json $D/F07_local.det.json \
  --clip $V/$C08.mp4 $B/F08_ours.trk.json $D/F08_local.det.json --out "$OUT/DEV_G3.teams.json"
$P -m tools.team_score dev --teams "$OUT/DEV_G1.teams.json" --clip $L/$C02/v1 $B/F02_ours.trk.json --out "$OUT/DEV_G1.score.json"
$P -m tools.team_score dev --teams "$OUT/DEV_G2.teams.json" --clip $L/$C05/v1 $B/F05_ours.trk.json --out "$OUT/DEV_G2.score.json"
$P -m tools.team_score dev --teams "$OUT/DEV_G3.teams.json" --clip $L/$C07/v1 $B/F07_ours.trk.json \
  --clip $L/$C08/v1 $B/F08_ours.trk.json --out "$OUT/DEV_G3.score.json"
