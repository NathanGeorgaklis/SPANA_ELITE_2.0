#!/bin/zsh
# Re-ID v4 vs v2 on DEV (CODE TASK identity/reid-v4-dev-eval). DEV FRESH_02/05/07/08 frozen labels only; no EVAL.
# Same inputs for both models: labels, videos, and the main tracker's tracks / decision logs (uncertain-pair notes)
# from ~/spana-player-confirmation-run/base (tracker unchanged since main b758c0b).
# 1. tools/dev/appearance_cue_eval.py --dump-tables (DEV rival tests, all cues present in MODEL_DIR: hist + spana)
# 2. tools/dev/reid_calibrate.py (config/reid_calibration.yaml 1.2.0: margin floor all_tests, leave-one-clip-out)
# 3. records/reid_episode_resolution_2026-11-20/distributions.py (same-kit teammate proxy; clean same-look rival tests
#    at the adopted margin 0.517 and at each model's own floored final margin)
set -euo pipefail
R=${R:-$HOME/SPANA_v4eval}; E=${E:-$HOME/spana-reid-v4-eval}; P=${P:-$HOME/SPANA_ELITE_2.0/.venv/bin/python}
B=${B:-$HOME/spana-player-confirmation-run/base}
typeset -A CLIP=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
                 F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
cd $R
specs=(); for c in F05 F07 F02 F08; do n=${CLIP[$c]}
  specs+=(data/labels/$n/v1:$HOME/Downloads/SPANA_fresh_clips/$n.mp4:$B/${c}_ours.trk.json:$B/${c}_ours.trk.decisions.jsonl); done
for m in v2 v4; do
  ( PYTHONPATH=. $P -m tools.dev.appearance_cue_eval $E/out/cue_eval_$m.json $E/models_$m --dump-tables=$E/out/tables_$m.json $specs \
      > $E/out/cue_eval_$m.log 2>&1
    PYTHONPATH=. $P -m tools.dev.reid_calibrate --tables $E/out/tables_$m.json --out $E/out/calibration_$m.json \
      > $E/out/calibration_$m.log 2>&1 ) &
done
wait
for m in v2 v4; do
  $P records/reid_episode_resolution_2026-11-20/distributions.py $E/out/tables_$m.json 0.517 > $E/out/samekit_${m}_0517.json
done
echo DONE
