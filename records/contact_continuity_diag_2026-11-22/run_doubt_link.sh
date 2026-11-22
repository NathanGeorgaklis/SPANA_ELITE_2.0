#!/bin/zsh
# Measurement of the EXISTING tracking.appearance_doubt_link (built 2026-11-18, off by default) on DEV, as evidence for
# the contact-continuity proposal. config/default.yaml unchanged: the run uses a copy with that one switch on.
# Same cached detections as records/player_confirmation_2026-11-20; reviewer = config/reviewer_policy.yaml baseline.
set -euo pipefail
R=${R:-$HOME/SPANA_main}; O=${O:-$HOME/spana-contact-diag}; P=${P:-$HOME/SPANA_ELITE_2.0/.venv/bin/python}
typeset -A CLIP=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
                 F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
cd $R
C=$O/doubt_link.yaml; D=$O/doubt
for c in F02 F05 F07 F08; do
  name=${CLIP[$c]}; det=$HOME/SPANA_ELITE_2.0/runs/person_view_compare/${c}_local.det.json; vid=~/Downloads/SPANA_fresh_clips/$name.mp4
  ( PYTHONPATH=. $P -m spana.cli track --video $vid --config $C --detections $det --out $D/${c}_ours.trk.json > $D/${c}.log 2>&1
    PYTHONPATH=. $P tools/benchmark.py --labels data/labels/$name/v1 --detections $det --tracks $D/${c}_ours.trk.json \
      --decisions $D/${c}_ours.trk.decisions.jsonl --out $D/${c}_ours.bench.json --rule NESTED_V2 >> $D/${c}.log 2>&1
    PYTHONPATH=. $P -m spana.cli crops --video $vid --tracks $D/${c}_ours.trk.json --detections $det --config $C \
      --reid --out $D/${c}_ours.crops.npz >> $D/${c}.log 2>&1 ) &
done
wait
a=(); for c in F02 F05 F07 F08; do a+=(--clip data/labels/${CLIP[$c]}/v1 $D/${c}_ours.trk.json); done
PYTHONPATH=. $P -m tools.target_score --config $C $a --simulate-reviewer --confirmations --crops --out $D/target_policy_correct.json > $D/target.log 2>&1
PYTHONPATH=. $P -m tools.target_score --config $C $a --out $D/target_default.json >> $D/target.log 2>&1
PYTHONPATH=. $P records/contact_continuity_diag_2026-11-22/diagnose.py $D $O/doubt_changes.json \
  $(for c in F02 F05 F07 F08; do echo data/labels/${CLIP[$c]}/v1:$c; done) > $D/diag.log
echo DONE
