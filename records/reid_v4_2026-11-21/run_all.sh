#!/bin/bash
set -e
cd /Users/nathangeorgaklis/spana-reid-v4
run() { name=$1; shift
  until /Users/nathangeorgaklis/SPANA_ELITE_2.0/.venv/bin/python -m tools.reid_train train --crops "$@" --backbone /Users/nathangeorgaklis/spana-models/dinov2-small --out runs/$name --device mps --max-minutes 100 && grep -q '"epoch": 12' runs/$name/log.jsonl; do echo "resume $name"; done
  [ -f runs/$name/spana_reid.onnx ] || /Users/nathangeorgaklis/SPANA_ELITE_2.0/.venv/bin/python -m tools.reid_train export --run runs/$name --backbone /Users/nathangeorgaklis/spana-models/dinov2-small --out runs/$name/spana_reid.onnx
  echo "DONE $name"; }
run fold_M003 /Users/nathangeorgaklis/spana-reid-run/crops_M001 /Users/nathangeorgaklis/spana-confirmed-labels/crops_M002
run fold_M001 /Users/nathangeorgaklis/spana-confirmed-labels/crops_M002 /Users/nathangeorgaklis/spana-retrain-v3/crops_M003
run fold_M002 /Users/nathangeorgaklis/spana-reid-run/crops_M001 /Users/nathangeorgaklis/spana-retrain-v3/crops_M003
run final /Users/nathangeorgaklis/spana-reid-run/crops_M001 /Users/nathangeorgaklis/spana-confirmed-labels/crops_M002 /Users/nathangeorgaklis/spana-retrain-v3/crops_M003
echo ALL_DONE
