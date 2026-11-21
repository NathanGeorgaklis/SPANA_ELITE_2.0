#!/bin/bash
# after run_all.sh: held-out TRAIN folds -> fold-pooled ("borrowed") margin -> DEV report + 2026-11-19 floored gate
set -e
W=~/spana-reid-v4; R=/Users/nathangeorgaklis/SPANA_ELITE_2.0; PY=$R/.venv/bin/python
C1=~/spana-reid-run/crops_M001; C2=~/spana-confirmed-labels/crops_M002; C3=~/spana-retrain-v3/crops_M003
cd $R
$PY -m tools.reid_heldout eval --model $W/runs/fold_M001/spana_reid.onnx --crops $C1 --out $W/out/fold_M001.json > $W/out/fold_M001.log
$PY -m tools.reid_heldout eval --model $W/runs/fold_M002/spana_reid.onnx --crops $C2 --out $W/out/fold_M002.json > $W/out/fold_M002.log
$PY -m tools.reid_heldout eval --model $W/runs/fold_M003/spana_reid.onnx --crops $C3 --out $W/out/fold_M003.json > $W/out/fold_M003.log
$PY -m tools.reid_heldout calibrate --folds $W/out/fold_M001.json $W/out/fold_M002.json $W/out/fold_M003.json --out $W/out/calibration_train_folds.json > /dev/null
mkdir -p $W/models_v4 && cp $W/runs/final/spana_reid.onnx $W/models_v4/
$W/dev_tables.sh v4 $W/models_v4
M=$($PY -c "import json;print(json.load(open('$W/out/calibration_train_folds.json'))['final']['margin'])")
$PY records/reid_episode_resolution_2026-11-20/distributions.py $W/out/tables_v4.json $M > $W/out/dist_v4_borrowed.json
$PY records/reid_episode_resolution_2026-11-20/distributions.py $W/out/tables_v4.json 0.517 > $W/out/dist_v4_0517.json
$PY -m tools.dev.reid_calibrate --tables $W/out/tables_v4.json --out $W/out/calibration_dev_v4.json > $W/out/calibration_dev_v4.log
echo POST_DONE margin=$M
