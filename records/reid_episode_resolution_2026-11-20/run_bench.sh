#!/bin/zsh
# DEV FRESH_02/05/07/08: tracker (CFG) -> benchmark -> linker (clearance off) -> benchmark -> follower / reviewer -> resolution check.
set -euo pipefail
R=~/SPANA_ELITE_2.0; O=$OUT; P=$R/.venv/bin/python; C=${CFG:-config/default.yaml}
typeset -A CLIP=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
                 F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
mkdir -p $O; cd $R
for c in F02 F05 F07 F08; do
  name=${CLIP[$c]}; det=runs/person_view_compare/${c}_local.det.json; vid=~/Downloads/SPANA_fresh_clips/$name.mp4
  lab=data/labels/$name/v1
  ( PYTHONPATH=. $P -m spana.cli track --video $vid --config $C --detections $det --out $O/${c}_ours.trk.json > $O/${c}_ours.track.log 2>&1
  PYTHONPATH=. $P tools/benchmark.py --labels $lab --detections $det --tracks $O/${c}_ours.trk.json \
    --decisions $O/${c}_ours.trk.decisions.jsonl --out $O/${c}_ours.bench.json --rule NESTED_V2 > $O/${c}_ours.bench.log 2>&1
  PYTHONPATH=. $P -m spana.cli link --video $vid --config $C --linking-config config/tracking/linking.yaml \
    --tracks $O/${c}_ours.trk.json --decisions $O/${c}_ours.trk.decisions.jsonl --out $O/${c}_linked.trk.json > $O/${c}_linked.link.log 2>&1
  PYTHONPATH=. $P tools/benchmark.py --labels $lab --detections $det --tracks $O/${c}_linked.trk.json \
    --decisions $O/${c}_linked.trk.decisions.jsonl --out $O/${c}_linked.bench.json --rule NESTED_V2 > $O/${c}_linked.bench.log 2>&1 ) &
done
wait
for t in ours linked; do
  args=(); for c in F02 F05 F07 F08; do args+=(--clip data/labels/${CLIP[$c]}/v1 $O/${c}_$t.trk.json); done
  PYTHONPATH=. $P -m tools.target_score --config $C $args --out $O/target_${t}_default.json > /dev/null
  PYTHONPATH=. $P -m tools.target_score --config $C $args --out $O/target_${t}_reviewer.json --simulate-reviewer > /dev/null
  PYTHONPATH=. $P -m tools.dev.resolution_check --out $O/resolution_${t}.json $args > /dev/null
done
echo DONE $O
