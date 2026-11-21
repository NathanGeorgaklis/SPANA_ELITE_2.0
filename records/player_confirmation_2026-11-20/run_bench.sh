#!/bin/zsh
# Player confirmation (CODE TASK identity/player-confirmation), DEV FRESH_02/05/07/08, same cached local-detector
# detections (runs/person_view_compare/F0x_local.det.json). Code: worktree of branch identity/player-confirmation.
# 1. tracks (config/default.yaml, unchanged tracker) -> benchmark; linker (clearance off) -> benchmark
# 2. clean crop features with re-ID v2 (spana.cli crops --reid)
# 3. reviewer: baseline (taps), confirmations always correct, confirmations with config/review_sim.yaml wrong taps,
#    confirmation_scope episode (measurement), linked tracks (claims only)
# 4. separate run (tracks + crops built in parallel with step 1): reid_resolve.enabled + target.gallery_reid_resolve (tracks + crops rebuilt), reviewer + episodes
set -euo pipefail
R=${R:-$HOME/SPANA_pc}; O=$OUT; P=${P:-$HOME/SPANA_ELITE_2.0/.venv/bin/python}
typeset -A CLIP=(F02 FRESH_02_G1_3000_midfield_buildup_passing F05 FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper
                 F07 FRESH_07_G3_1205_midfield_passing_switch F08 FRESH_08_G3_2103_attack_into_box_clearance)
cd $R
track() {  # $1 out dir, $2 config
  for c in F02 F05 F07 F08; do
    name=${CLIP[$c]}; det=runs/person_view_compare/${c}_local.det.json; vid=~/Downloads/SPANA_fresh_clips/$name.mp4
    ( PYTHONPATH=. $P -m spana.cli track --video $vid --config $2 --detections $det --out $1/${c}_ours.trk.json > $1/${c}.log 2>&1
    PYTHONPATH=. $P tools/benchmark.py --labels data/labels/$name/v1 --detections $det --tracks $1/${c}_ours.trk.json \
      --decisions $1/${c}_ours.trk.decisions.jsonl --out $1/${c}_ours.bench.json --rule NESTED_V2 >> $1/${c}.log 2>&1
    PYTHONPATH=. $P -m spana.cli crops --video $vid --tracks $1/${c}_ours.trk.json --detections $det --config $2 \
      --reid --out $1/${c}_ours.crops.npz >> $1/${c}.log 2>&1 ) &
  done
  wait
}
args() { a=(); for c in F02 F05 F07 F08; do a+=(--clip data/labels/${CLIP[$c]}/v1 $1/${c}_$2.trk.json); done; echo $a; }
mkdir -p $O/base $O/reid $O/reid/targets
sed -e 's/^  enabled: false$/  enabled: true/' -e 's/^  gallery_reid_resolve: false$/  gallery_reid_resolve: true/' config/default.yaml > $O/reid_on.yaml
track $O/base config/default.yaml &
track $O/reid $O/reid_on.yaml &
wait
for c in F02 F05 F07 F08; do
  name=${CLIP[$c]}; det=runs/person_view_compare/${c}_local.det.json; vid=~/Downloads/SPANA_fresh_clips/$name.mp4
  ( PYTHONPATH=. $P -m spana.cli link --video $vid --config config/default.yaml --linking-config config/tracking/linking.yaml \
    --tracks $O/base/${c}_ours.trk.json --decisions $O/base/${c}_ours.trk.decisions.jsonl --out $O/base/${c}_linked.trk.json
  PYTHONPATH=. $P tools/benchmark.py --labels data/labels/$name/v1 --detections $det --tracks $O/base/${c}_linked.trk.json \
    --decisions $O/base/${c}_linked.trk.decisions.jsonl --out $O/base/${c}_linked.bench.json --rule NESTED_V2 ) > $O/base/${c}_link.log 2>&1 &
done
wait
T="PYTHONPATH=. $P -m tools.target_score"
eval $T --config config/default.yaml $(args $O/base ours) --out $O/base/target_default.json
eval $T --config config/default.yaml $(args $O/base ours) --out $O/base/target_reviewer.json --simulate-reviewer
eval $T --config config/default.yaml $(args $O/base ours) --out $O/base/target_confirm_correct.json --simulate-reviewer --confirmations --crops
eval $T --config config/default.yaml $(args $O/base ours) --out $O/base/target_confirm_err.json --simulate-reviewer --confirmations --crops --review-sim-config config/review_sim.yaml
sed 's/^  confirmation_scope: range$/  confirmation_scope: episode/' config/default.yaml > $O/episode_scope.yaml
eval $T --config $O/episode_scope.yaml $(args $O/base ours) --out $O/base/target_confirm_episode_scope.json --simulate-reviewer --confirmations --crops
eval $T --config config/default.yaml $(args $O/base linked) --out $O/base/target_linked_reviewer.json --simulate-reviewer
eval $T --config config/default.yaml $(args $O/base linked) --out $O/base/target_linked_confirm_correct.json --simulate-reviewer --confirmations
# separate run with re-ID on (tracker reid_resolve + target gallery re-ID); reid_on.yaml = default + those two switches
eval $T --config $O/reid_on.yaml $(args $O/reid ours) --out $O/reid/target_reviewer.json --simulate-reviewer
eval $T --config $O/reid_on.yaml $(args $O/reid ours) --out $O/reid/target_confirm_correct.json --simulate-reviewer --confirmations --crops --artifacts-dir $O/reid/targets
PYTHONPATH=. $P -m tools.dev.reid_episode_report --out $O/reid/episodes.json $(for c in F02 F05 F07 F08; do echo --clip data/labels/${CLIP[$c]}/v1 $O/reid/${c}_ours.trk.json; done)
$P records/player_confirmation_2026-11-20/summarize.py $O > records/player_confirmation_2026-11-20/compare_summary.json
# stress: 30 % wrong IS_ME taps (records/.../review_sim_stress.yaml; more wrong taps to see blocking / containment)
eval $T --config config/default.yaml $(args $O/base ours) --out $O/base/target_confirm_err30.json --simulate-reviewer --confirmations --crops --review-sim-config records/player_confirmation_2026-11-20/review_sim_stress.yaml
$P records/player_confirmation_2026-11-20/summarize.py $O > records/player_confirmation_2026-11-20/compare_summary.json
