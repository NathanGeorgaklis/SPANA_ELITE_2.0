#!/bin/zsh
# usage: run_pass.sh <tag> <config>  — DEV G1 window 45000..49500 (2.5 min), detect + track with profiling
set -e
cd /Users/nathangeorgaklis/SPANA_ELITE_2.0
R=runs/identity_speed; V=~/Downloads/Sydney_United_APIA.mp4
.venv/bin/python -m spana.cli detect --backend local --model-record models/rfdetr_v1.json --dataset-config config/roboflow.yaml \
  --runtime torch --device mps --video $V --config $2 --start 45000 --stop 49500 --out $R/$1.det.json > $R/$1.det.log 2>&1
.venv/bin/python -m spana.cli track --video $V --config $2 --detections $R/$1.det.json --out $R/$1.trk.json > $R/$1.trk.log 2>&1
echo done
