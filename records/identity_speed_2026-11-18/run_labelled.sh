#!/bin/zsh
# usage: run_labelled.sh <tag> <config>. Labelled DEV clips: baseline (existing fp32 detections, re-tracked with
# current code) vs detections made under <config>.
set -e
cd /Users/nathangeorgaklis/SPANA_ELITE_2.0
R=runs/identity_speed/labelled; mkdir -p $R; C=config/default.yaml; V=~/Downloads/SPANA_fresh_clips
typeset -A NAME
NAME=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper \
      F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
for c in F02 F05 F07 F08; do
  n=$NAME[$c]
  [[ -f $R/${c}_base.det.json ]] || cp runs/person_view_compare/${c}_local.det.json $R/${c}_base.det.json
  .venv/bin/python -m spana.cli detect --backend local --model-record models/rfdetr_v1.json --dataset-config config/roboflow.yaml \
    --runtime torch --device mps --video $V/$n.mp4 --config $2 --out $R/${c}_$1.det.json > $R/${c}_$1.det.log 2>&1
  for k in base $1; do
    [[ -f $R/${c}_$k.bench.json ]] && continue
    .venv/bin/python -m spana.cli track --video $V/$n.mp4 --config $C --detections $R/${c}_$k.det.json --out $R/${c}_$k.trk.json > $R/${c}_$k.trk.log 2>&1
    .venv/bin/python -m tools.benchmark --labels data/labels/$n/v1 --detections $R/${c}_$k.det.json --tracks $R/${c}_$k.trk.json \
      --decisions $R/${c}_$k.trk.decisions.jsonl --out $R/${c}_$k.bench.json > $R/${c}_$k.bench.log 2>&1
  done
done
for k in base $1; do
  [[ -f $R/target_${k}_reviewer.json ]] && continue
  .venv/bin/python -m tools.target_score --config $C --out $R/target_$k.json \
    --clip data/labels/$NAME[F02]/v1 $R/F02_$k.trk.json --clip data/labels/$NAME[F05]/v1 $R/F05_$k.trk.json \
    --clip data/labels/$NAME[F07]/v1 $R/F07_$k.trk.json --clip data/labels/$NAME[F08]/v1 $R/F08_$k.trk.json > $R/target_$k.log 2>&1
  .venv/bin/python -m tools.target_score --config $C --out $R/target_${k}_reviewer.json --simulate-reviewer \
    --clip data/labels/$NAME[F02]/v1 $R/F02_$k.trk.json --clip data/labels/$NAME[F05]/v1 $R/F05_$k.trk.json \
    --clip data/labels/$NAME[F07]/v1 $R/F07_$k.trk.json --clip data/labels/$NAME[F08]/v1 $R/F08_$k.trk.json > $R/target_${k}_reviewer.log 2>&1
done
echo done
