#!/bin/bash
# usage: dev_tables.sh NAME MODEL_DIR   (run from the repo; DEV report-only tables on the same tests as v2)
set -e
R=/Users/nathangeorgaklis/SPANA_ELITE_2.0; V=~/Downloads/SPANA_fresh_clips; P=~/spana-clean-calib/out/cue_eval_p2.json; O=~/spana-reid-v4/out
spec() { echo "$R/data/labels/$1/v1:$V/$1.mp4:pairs@$P:-"; }
cd $R
.venv/bin/python -m tools.dev.appearance_cue_eval $O/cue_eval_$1.json $2 \
  $(spec FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper) $(spec FRESH_07_G3_1205_midfield_passing_switch) \
  $(spec FRESH_02_G1_3000_midfield_buildup_passing) $(spec FRESH_08_G3_2103_attack_into_box_clearance) \
  --dump-tables=$O/tables_$1.json > $O/cue_eval_$1.log 2>&1
