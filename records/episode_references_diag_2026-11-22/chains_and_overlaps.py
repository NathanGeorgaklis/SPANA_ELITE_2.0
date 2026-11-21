"""DEV analysis only (records/episode_references_diag_2026-11-22): spawn-chain closure, clean-crop overlaps by player,
NO_CLEAN_CROP track ends / respawns. Needs ~/spana-epref-diag.json from diagnose.py. Run from the repo root."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("dg", "records/episode_references_diag_2026-11-22/diagnose.py")
dg = importlib.util.module_from_spec(spec); spec.loader.exec_module(dg)
run = Path.home() / "spana-player-confirmation-run/reid"
CL = {"F02": "FRESH_02_G1_3000_midfield_buildup_passing", "F05": "FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper",
      "F07": "FRESH_07_G3_1205_midfield_passing_switch", "F08": "FRESH_08_G3_2103_attack_into_box_clearance"}
D = json.load(open(Path.home() / "spana-epref-diag.json"))
tot = Counter()
for key, name in CL.items():
    td, rows, ev, history, ov, own, t_of, first_ref, spawn = dg.clip_diag(run, f"data/labels/{name}/v1", key)
    def player(tid):
        c = Counter(own.get(tid, {}).values()).most_common(1); return c[0][0] if c else None
    def roots(tid, at, seen=None):
        """predecessor closure: a reference-less, born-uncertain predecessor is replaced by its own predecessors."""
        seen = set(seen or ())
        if tid in seen: return {("CYCLE", tid)}
        seen.add(tid)
        if first_ref[tid] is not None and first_ref[tid] < at: return {("REF", tid)}
        h = history(tid, at)
        if h["why"] == "BORN_UNCERTAIN_NEVER_CERTAIN" and h["spawn_predecessors"]:
            out = set()
            for p in h["spawn_predecessors"]:
                out |= roots(p, rows[tid][0][0], seen)
            return out
        return {("NOREF:" + h["why"], tid)}
    for e in D["NO_REFERENCE"]:
        if e["clip"] != name[:8]: continue
        if not e["at_spawn"]:
            tot["nonspawn"] += 1; continue
        preds = spawn[e["track"]]["possible_predecessors"]
        R = set()
        for p in preds: R |= roots(p, e["start"])
        ok = all(k == "REF" for k, _ in R)
        me = e["player"]
        root_players = {player(t) for _, t in R}
        direct_players = {player(p) for p in preds}
        tot["spawn"] += 1
        tot["closure_all_ref" if ok else "closure_still_missing"] += 1
        if ok: tot["closure_ok_true_player_in_roots" if me in root_players else ("closure_ok_player_unknown" if me is None else "closure_ok_true_player_NOT_in_roots")] += 1
        tot[f"direct_true_in_preds_{me in direct_players if me else 'unk'}"] += 1
        for k, _ in R:
            if k != "REF": tot["closure_missing:" + k] += 1
        tot["n_roots_" + str(min(len(R), 6))] += 1
    # NO_CLEAN_CROP: overlap by overall player, IoU, end reason
    for e in D["NO_CLEAN_CROP"]:
        if e["clip"] != name[:8]: continue
        tid = e["track"]; me = player(tid)
        for f, r in rows[tid]:
            if not (e["start"] <= f <= e["end"]) or r["source"] != "OBSERVED": continue
            iou, oth = ov[(f, tid)]
            if iou <= dg.CLEAN: tot["ncc_obs_clean_frame?"] += 1; continue
            them = player(oth)
            tot["ncc_overlap:" + ("unknown" if me is None or them is None else ("same_player_dup" if me == them else "other_player"))] += 1
            tot["ncc_iou:" + ("<=0.10" if iou <= .10 else "<=0.20" if iou <= .2 else "<=0.40" if iou <= .4 else ">0.40")] += 1
        h = history(tid, e["end"] + 1)
        tot["ncc_end:" + str(h["end_reason"])] += 1
        dup = sum(1 for f, r in rows[tid] if e["start"] <= f <= e["end"] and r["source"] == "OBSERVED" and ov[(f, tid)][0] > dg.CLEAN and me is not None and player(ov[(f, tid)][1]) == me)
        tot["ncc_ep_dup_majority" if dup * 2 > e["observed"] else "ncc_ep_other_majority"] += 1
        # is the player picked up again by a spawned track within 3 s after this track ends, and was that spawn ambiguous?
        endf = rows[tid][-1][0]
        if me is not None:
            nxt = sorted((rows[t2][0][0], t2) for t2 in rows if t2 != tid and player(t2) == me and 0 < rows[t2][0][0] - endf <= 90)
            if nxt:
                f2, t2 = nxt[0]; amb = len(spawn.get(t2, {}).get("possible_predecessors", [])) >= 2
                tot["ncc_player_respawned_" + ("ambiguous" if amb else "unambiguous") + ("_names_this_track" if tid in spawn.get(t2, {}).get("possible_predecessors", []) else "")] += 1
            else:
                tot["ncc_player_not_respawned_3s"] += 1
        tot["ncc_first_rule:" + str(next((d for d in [r for r in []]), None))] += 0
for k in sorted(tot): print(k, tot[k])
