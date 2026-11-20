#!/bin/zsh
# Tracker bench v1 (DEV FRESH_02/05/07/08). Code: ~/SPANA_ELITE_2.0 on tracking/tracker-bench-v1-botsort (base main d6511e2).
# Inputs: cached local-detector detections runs/person_view_compare/F0x_local.det.json (same as bytetrack_compare_2026-11-18).
set -euo pipefail
R=~/SPANA_ELITE_2.0; O=~/spana-botsort-run/out; P=$R/.venv/bin/python
GIT="d6511e2+tracker-bench-v1"
typeset -A CLIP=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
                 F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
cd $R
for c in F02 F05 F07 F08; do
  name=${CLIP[$c]}; det=runs/person_view_compare/${c}_local.det.json; ref=runs/person_view_compare/${c}_local.trk.json
  vid=~/Downloads/SPANA_fresh_clips/$name.mp4; lab=data/labels/$name/v1
  # ours = current main tracker, production config
  PYTHONPATH=. $P -m spana.cli track --video $vid --config config/default.yaml --detections $det --out $O/${c}_ours.trk.json > $O/${c}_ours.track.log 2>&1
  # ByteTrack re-run (supervision 0.25.1, as 2026-11-18) through the shared conversion
  PYTHONPATH=.:$HOME/spana-bytetrack-run/svlib $P -m tools.dev.bytetrack_compare run --detections $det --ref-tracks $ref --out $O/${c}_bytetrack.trk.json --git $GIT > $O/${c}_bytetrack.run.log
  # BoT-SORT (trackers 2.7.0, published defaults, CMC on the decoded frames)
  PYTHONPATH=.:$HOME/spana-botsort-run/trklib $P -m tools.dev.botsort_compare run --video $vid --detections $det --ref-tracks $ref --out $O/${c}_botsort.trk.json --git $GIT > $O/${c}_botsort.run.log
  for t in ours bytetrack botsort; do
    PYTHONPATH=. $P tools/benchmark.py --labels $lab --detections $det --tracks $O/${c}_$t.trk.json \
      --decisions $O/${c}_$t.trk.decisions.jsonl --out $O/${c}_$t.bench.json --rule NESTED_V2 > $O/${c}_$t.bench.log 2>&1
  done
done
for t in ours bytetrack botsort; do
  args=()
  for c in F02 F05 F07 F08; do args+=(--clip data/labels/${CLIP[$c]}/v1 $O/${c}_$t.trk.json); done
  PYTHONPATH=. $P -m tools.target_score --config config/default.yaml $args --out $O/target_${t}_default.json
  PYTHONPATH=. $P -m tools.target_score --config config/default.yaml $args --out $O/target_${t}_reviewer.json --simulate-reviewer
done
$P -m tools.dev.tracker_bench summary --out $O/compare_summary.json \
  $(for t in ours bytetrack botsort; do echo --tracker $t $O/target_${t}_default.json $O/target_${t}_reviewer.json $O/F0{2,5,7,8}_$t.bench.json; done)
echo DONE
