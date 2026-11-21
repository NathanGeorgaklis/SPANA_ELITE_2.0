#!/bin/zsh
# Re-score stored DEV runs with the 2026-11-20 transfer rule (scoring only: no tracker / linker re-run).
# Runs: records/tracker_bench_v1_2026-11-20 (ours, bytetrack, botsort in ~/spana-botsort-run/out) and
# records/tracklet_linking_v1_2026-11-20 (clearance on ~/spana-linking-run/out, off ~/spana-linking-run/out_noclear).
set -euo pipefail
R=~/SPANA_ELITE_2.0; O=~/spana-flagging-run; P=$R/.venv/bin/python
typeset -A CLIP=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
                 F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
typeset -A DIR=(ours ~/spana-botsort-run/out bytetrack ~/spana-botsort-run/out botsort ~/spana-botsort-run/out
                linked_on ~/spana-linking-run/out linked_off ~/spana-linking-run/out_noclear)
typeset -A STEM=(ours ours bytetrack bytetrack botsort botsort linked_on linked linked_off linked)
mkdir -p $O; cd $R; runs=()
for s in ours bytetrack botsort linked_on linked_off; do
  mkdir -p $O/$s
  for c in F02 F05 F07 F08; do
    d=${DIR[$s]}; t=$d/${c}_${STEM[$s]}
    PYTHONPATH=. $P tools/benchmark.py --labels data/labels/${CLIP[$c]}/v1 --detections runs/person_view_compare/${c}_local.det.json \
      --tracks $t.trk.json --decisions $t.trk.decisions.jsonl --out $O/$s/${c}.bench.json --rule NESTED_V2 > $O/$s/${c}.bench.log
    runs+=(--run $s $c $t.bench.json $O/$s/${c}.bench.json)
  done
done
PYTHONPATH=. $P -m tools.dev.flag_rescore --out $O/flag_rescore.json $runs
PYTHONPATH=. $P -m tools.dev.link_bench summary --out $O/link_summary_on.json $(for c in F02 F05 F07 F08; do echo --clip $c $O/linked_on/$c.bench.json ~/spana-linking-run/out/${c}_linked.trk.links.jsonl; done) > /dev/null
PYTHONPATH=. $P -m tools.dev.link_bench summary --out $O/link_summary_off.json $(for c in F02 F05 F07 F08; do echo --clip $c $O/linked_off/$c.bench.json ~/spana-linking-run/out_noclear/${c}_linked.trk.links.jsonl; done) > /dev/null
echo DONE
