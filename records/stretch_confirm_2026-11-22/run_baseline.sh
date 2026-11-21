#!/bin/zsh
# Reviewer policy baseline (decision 2026-11-22, config/reviewer_policy.yaml) on the DEV tracks / crops of
# records/player_confirmation_2026-11-20 (~/spana-player-confirmation-run/base). DEV only.
set -euo pipefail
R=${R:-$HOME/SPANA_main}; B=${B:-$HOME/spana-player-confirmation-run/base}; O=${O:-$HOME/spana-reviewer-policy}
P=${P:-$HOME/SPANA_ELITE_2.0/.venv/bin/python}
typeset -A CLIP=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
                 F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
cd $R
a=(); for c in F02 F05 F07 F08; do a+=(--clip data/labels/${CLIP[$c]}/v1 $B/${c}_ours.trk.json); done
T=(env PYTHONPATH=. $P -m tools.target_score --config config/default.yaml $a --simulate-reviewer --confirmations --crops)
$T --out $O/policy_correct.json > $O/policy_correct.log 2>&1 &
$T --reviewer-policy earlier --out $O/earlier_correct.json > $O/earlier_correct.log 2>&1 &
$T --review-sim-config config/review_sim.yaml --out $O/policy_err5.json > $O/policy_err5.log 2>&1 &
wait
echo DONE
