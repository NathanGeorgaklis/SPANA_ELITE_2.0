#!/usr/bin/env bash
# TRAIN sanity check (Nathan, 2026-11-22): the labelled tracklets of TRAIN_M001-M003 only (10-min windows, the tracks
# the re-ID labels were made on, sha256-checked). Team sides only: the TRAIN detections are COCO person-only.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=~/SPANA_ELITE_2.0/.venv/bin/python
S=~/Downloads/SPANA_reid
OUT=${OUT:-~/spana-team-score-run}/train
mkdir -p "$OUT"
[ -f "$OUT/TRAIN_M001.trk.json" ] || xz -dkc $S/TRAIN_M001/TRAIN_M001.trk.json.xz > "$OUT/TRAIN_M001.trk.json"
[ -f "$OUT/TRAIN_M001.det.json" ] || xz -dkc $S/TRAIN_M001/TRAIN_M001.det.json.xz > "$OUT/TRAIN_M001.det.json"
$P -m tools.team_score train --labels data/reid/TRAIN/TRAIN_M001_MORETON_CITY/v1 \
  --video $S/TRAIN_M001/TRAIN_M001_MORETON_CITY_w3000-3600_h264.mp4 --tracks "$OUT/TRAIN_M001.trk.json" \
  --detections "$OUT/TRAIN_M001.det.json" --config config/default.yaml --out "$OUT/TRAIN_M001.score.json"
$P -m tools.team_score train --labels data/reid/TRAIN/TRAIN_M002_EASTERN_SUBURBS/v1 \
  --video $S/TRAIN_M002/work/TRAIN_M002_EASTERN_SUBURBS_w3000-3600_h264.mp4 --tracks $S/TRAIN_M002/work/TRAIN_M002.trk.json \
  --detections $S/TRAIN_M002/work/TRAIN_M002.det.json --config config/default.yaml --out "$OUT/TRAIN_M002.score.json"
$P -m tools.team_score train --labels data/reid/TRAIN/TRAIN_M003_SYDNEY_OLYMPIC/v1 \
  --video $S/TRAIN_M003/work/TRAIN_M003_SYDNEY_OLYMPIC_w3000-3600_h264.mp4 --tracks $S/TRAIN_M003/work/TRAIN_M003.trk.json \
  --detections $S/TRAIN_M003/work/TRAIN_M003.det.json --config config/default.yaml --out "$OUT/TRAIN_M003.score.json"
