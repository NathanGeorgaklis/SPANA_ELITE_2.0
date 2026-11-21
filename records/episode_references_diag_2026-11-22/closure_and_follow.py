"""DEV analysis only (records/episode_references_diag_2026-11-22): closure episodes with clean evidence, the truth among
closure roots, and where the NO_CLEAN_CROP player is while the uncertain track is unobserved. Run from the repo root."""
import json, sys
from collections import Counter
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("dg", "records/episode_references_diag_2026-11-22/diagnose.py")
dg = importlib.util.module_from_spec(spec); spec.loader.exec_module(dg)
from tools.benchmark import load_frozen
from spana.evaluation import NESTED_V2, _match
run = Path.home() / "spana-player-confirmation-run/reid"
CL = {"F02": "FRESH_02_G1_3000_midfield_buildup_passing", "F05": "FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper",
      "F07": "FRESH_07_G3_1205_midfield_passing_switch", "F08": "FRESH_08_G3_2103_attack_into_box_clearance"}
D = json.load(open(Path.home() / "spana-epref-diag.json"))
tot = Counter(); notin = []
for key, name in CL.items():
    ld = f"data/labels/{name}/v1"
    td, rows, ev, history, ov, own, t_of, first_ref, spawn = dg.clip_diag(run, ld, key)
    _, gt = load_frozen(Path(ld))
    trk = {fr["frame"]: fr for fr in td["frames"]}
    def player(tid):
        c = Counter(own.get(tid, {}).values()).most_common(1); return c[0][0] if c else None
    def roots(tid, at, seen=()):
        seen = set(seen)
        if tid in seen: return {("CYCLE", tid)}
        seen.add(tid)
        if first_ref[tid] is not None and first_ref[tid] < at: return {("REF", tid)}
        h = history(tid, at)
        if h["why"] == "BORN_UNCERTAIN_NEVER_CERTAIN" and h["spawn_predecessors"]:
            o = set()
            for p in h["spawn_predecessors"]: o |= roots(p, rows[tid][0][0], seen)
            return o
        return {("NOREF", tid)}
    for e in D["NO_REFERENCE"]:
        if e["clip"] != name[:8] or not e["at_spawn"]: continue
        R = set()
        for p in spawn[e["track"]]["possible_predecessors"]: R |= roots(p, e["start"])
        if all(k == "REF" for k, _ in R):
            tot["closure_ok_episodes"] += 1
            tot["closure_ok_with_clean_evidence_frames>=3" if e["evidence"].get("EVALUATED", 0) >= 3 else "closure_ok_clean_frames<3"] += 1
            tot["closure_ok_clean_frames_total"] += e["evidence"].get("EVALUATED", 0)
            rp = {player(t) for _, t in R}
            if e["player"] is not None and e["player"] not in rp:
                notin.append({"clip": key, "track": e["track"], "player": e["player"], "roots": sorted(t for _, t in R), "root_players": sorted(map(str, rp))})
    # NO_CLEAN_CROP: on labelled frames inside the episode where the track is not observed, where is the player?
    for e in D["NO_CLEAN_CROP"]:
        if e["clip"] != name[:8]: continue
        tid, me = e["track"], player(e["track"])
        if me is None: tot["ncc_player_unknown_frames"] += 0; continue
        for f, r in rows[tid]:
            if not (e["start"] <= f <= e["end"]) or r["source"] == "OBSERVED" or f not in gt.players: continue
            if me not in gt.players[f]: tot["ncc_unobs:player_not_labelled(out_of_view)"] += 1; continue
            occl = me in gt.occluded.get(f, set())
            obs = [t for t in trk[f]["tracks"] if t["source"] == "OBSERVED"]
            m, _, amb = _match(sorted(gt.players[f]), gt.players[f], [t["box"] for t in obs], NESTED_V2, frozenset(gt.occluded.get(f, ())))
            hit = [obs[c]["track_id"] for c, g in m.items() if g == me]
            if not hit: tot["ncc_unobs:labelled_no_box_matched" + ("_occluded" if occl else "_visible")] += 1
            else:
                o = hit[0]; born = rows[o][0][0]
                kind = "rival" if o in e["rivals"] else ("spawned_after_episode_start" if born > e["start"] else "other_older_track")
                names = tid in spawn.get(o, {}).get("possible_predecessors", [])
                tot["ncc_unobs:on_" + kind + ("_naming_this_as_predecessor" if names else "")] += 1
                tot["ncc_unobs_on_other_track_ep:" + e["track"] + key] += 0
for k in sorted(tot): print(k, tot[k])
print(json.dumps(notin))
