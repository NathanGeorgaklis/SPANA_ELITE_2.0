#!/bin/zsh
# Tracklet linking v1 bench (DEV FRESH_02/05/07/08). Code: ~/SPANA_ELITE_2.0 on tracking/global-tracklet-linking-v1.
# Input tracks: the production tracker's output from records/tracker_bench_v1_2026-11-20 (~/spana-botsort-run/out/
# F0x_ours.trk.json, main d6511e2; spana/, config/default.yaml and the scorers are unchanged since), made from the
# cached local-detector detections runs/person_view_compare/F0x_local.det.json. Production tracker not re-run.
set -euo pipefail
R=~/SPANA_ELITE_2.0; B=~/spana-botsort-run/out; O=~/spana-linking-run/out; P=$R/.venv/bin/python
typeset -A CLIP=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
                 F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
mkdir -p $O; cd $R
for c in F02 F05 F07 F08; do
  name=${CLIP[$c]}; det=runs/person_view_compare/${c}_local.det.json; vid=~/Downloads/SPANA_fresh_clips/$name.mp4
  PYTHONPATH=. $P -m spana.cli link --video $vid --config config/default.yaml --linking-config config/tracking/linking.yaml \
    --tracks $B/${c}_ours.trk.json --decisions $B/${c}_ours.trk.decisions.jsonl --out $O/${c}_linked.trk.json > $O/${c}_linked.link.log
  PYTHONPATH=. $P tools/benchmark.py --labels data/labels/$name/v1 --detections $det --tracks $O/${c}_linked.trk.json \
    --decisions $O/${c}_linked.trk.decisions.jsonl --out $O/${c}_linked.bench.json --rule NESTED_V2 > $O/${c}_linked.bench.log 2>&1
done
args=(); for c in F02 F05 F07 F08; do args+=(--clip data/labels/${CLIP[$c]}/v1 $O/${c}_linked.trk.json); done
PYTHONPATH=. $P -m tools.target_score --config config/default.yaml $args --out $O/target_linked_default.json
PYTHONPATH=. $P -m tools.target_score --config config/default.yaml $args --out $O/target_linked_reviewer.json --simulate-reviewer
$P -m tools.dev.tracker_bench summary --out $O/compare_summary.json \
  $(for t in ours bytetrack botsort; do echo --tracker $t $B/target_${t}_default.json $B/target_${t}_reviewer.json $B/F0{2,5,7,8}_$t.bench.json; done) \
  --tracker ours_linked $O/target_linked_default.json $O/target_linked_reviewer.json $O/F0{2,5,7,8}_linked.bench.json
PYTHONPATH=. $P -m tools.dev.link_bench transfers --out $O/link_transfers.json \
  $(for c in F02 F05 F07 F08; do echo --clip $c $B/${c}_ours.bench.json $O/${c}_linked.bench.json $O/${c}_linked.trk.links.jsonl; done)
echo DONE
