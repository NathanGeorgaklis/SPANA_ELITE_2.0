#!/bin/zsh
# usage: run_option.sh <tag> <model record>. DEV stage 2 of tools/detector_options.py: the labelled DEV clips
# (detect under the DEV record, track, benchmark, target_score with the simulated reviewer), then judge vs base.
set -e
cd /Users/nathangeorgaklis/SPANA_ELITE_2.0
R=runs/detector_options/labelled; C=config/default.yaml; V=~/Downloads/SPANA_fresh_clips
typeset -A NAME
NAME=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper \
      F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
for c in F02 F05 F07 F08; do
  n=$NAME[$c]
  .venv/bin/python -m spana.cli detect --backend local --model-record $2 --dataset-config config/roboflow.yaml \
    --runtime torch --device mps --video $V/$n.mp4 --config $C --out $R/${c}_$1.det.json > $R/${c}_$1.det.log 2>&1
  .venv/bin/python -m spana.cli track --video $V/$n.mp4 --config $C --detections $R/${c}_$1.det.json --out $R/${c}_$1.trk.json > $R/${c}_$1.trk.log 2>&1
  .venv/bin/python -m tools.benchmark --labels data/labels/$n/v1 --detections $R/${c}_$1.det.json --tracks $R/${c}_$1.trk.json \
    --decisions $R/${c}_$1.trk.decisions.jsonl --out $R/${c}_$1.bench.json > $R/${c}_$1.bench.log 2>&1
done
for sfx in "" _reviewer; do
  [[ -z $sfx ]] && flag="" || flag="--simulate-reviewer"
  .venv/bin/python -m tools.target_score --config $C --out $R/target_$1$sfx.json ${=flag} \
    --clip data/labels/$NAME[F02]/v1 $R/F02_$1.trk.json --clip data/labels/$NAME[F05]/v1 $R/F05_$1.trk.json \
    --clip data/labels/$NAME[F07]/v1 $R/F07_$1.trk.json --clip data/labels/$NAME[F08]/v1 $R/F08_$1.trk.json > $R/target_$1$sfx.log 2>&1
done
.venv/bin/python -m tools.detector_options judge --base-dir $R --tag $1 --out runs/detector_options/judge_$1.json
