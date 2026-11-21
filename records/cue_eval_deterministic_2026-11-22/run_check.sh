#!/bin/zsh
# evaluation/cue-eval-deterministic DEV check: v2 dump twice under different hash seeds (byte compare), v5 once;
# reid_calibrate on each, compared with records/reid_v5_dev_eval_2026-11-21 outputs (~/spana-reid-v5-eval/out).
set -euo pipefail
R=$HOME/SPANA_main; E=$HOME/spana-cue-det; V=$HOME/spana-reid-v5-eval; P=$HOME/SPANA_ELITE_2.0/.venv/bin/python
B=$HOME/spana-player-confirmation-run/base
typeset -A CLIP=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
                 F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
cd $R
specs=(); for c in F05 F07 F02 F08; do n=${CLIP[$c]}
  specs+=(data/labels/$n/v1:$HOME/Downloads/SPANA_fresh_clips/$n.mp4:$B/${c}_ours.trk.json:$B/${c}_ours.trk.decisions.jsonl); done
for run in v2:1 v2:2 v5:1; do m=${run%:*}; s=${run#*:}
  ( PYTHONHASHSEED=$s PYTHONPATH=. $P -m tools.dev.appearance_cue_eval $E/out/cue_eval_${m}_s$s.json $V/models_$m \
      --dump-tables=$E/out/tables_${m}_s$s.json $specs > $E/out/cue_eval_${m}_s$s.log 2>&1
    PYTHONPATH=. $P -m tools.dev.reid_calibrate --tables $E/out/tables_${m}_s$s.json --out $E/out/calibration_${m}_s$s.json \
      > $E/out/calibration_${m}_s$s.log 2>&1 ) &
done
wait
echo DONE
