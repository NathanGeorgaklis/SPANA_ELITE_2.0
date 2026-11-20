"""STOP check (DEV analysis only): re-ID v2 cosine similarity, same player vs different players, split by look.
Teammate proxy: DEV labels carry no team field, so a player pair is 'same look' when the mean torso-colour similarity
of its different-player crop pairs is >= appearance.min_similarity (0.70), the tracker's own same-look test.
Rival tests (query A vs references of A and B): margin = sim(q, A) - sim(q, B), clean = no occlusion, max overlap
<= clean_max_iou (0.05), as the tracker's evidence."""
import json, sys, collections
import numpy as np
T = json.load(open(sys.argv[1])); MS, CLEAN, M = 0.70, 0.05, float(sys.argv[2]) if len(sys.argv) > 2 else 0.517
def q(x):
    x = np.asarray(x); return {"n": int(len(x)), **({k: round(float(np.percentile(x, p)), 3) for k, p in
            (("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))} if len(x) else {})}
out = {"margin": M, "clips": {}}; pool = collections.defaultdict(list)
for t in T["tables"]:
    c = t["match"][:8]; P = t["pairs"]["all"]
    hp = collections.defaultdict(list)
    for h, pl in zip(P["diff"]["scores"]["hist"], P["diff"]["players"]):
        hp[tuple(sorted(pl.split("|")))].append(h)
    look = {k: float(np.mean(v)) >= MS for k, v in hp.items()}
    same = P["same"]["scores"]["spana"]
    sl = [s for s, pl in zip(P["diff"]["scores"]["spana"], P["diff"]["players"]) if look[tuple(sorted(pl.split("|")))]]
    dl = [s for s, pl in zip(P["diff"]["scores"]["spana"], P["diff"]["players"]) if not look[tuple(sorted(pl.split("|")))]]
    R = t["rival"]; mg = {"same_look": [], "diff_look": []}
    for a, b, pr, occ, ov in zip(R["qa"]["spana"], R["qb"]["spana"], R["pair"], R["any_occluded"], R["max_overlap"]):
        if occ or ov > CLEAN: continue
        k = tuple(sorted(pr.split("|"))); key = "same_look" if look.get(k, True) else "diff_look"   # unknown pair -> same look (conservative)
        mg[key].append(a - b)
    o = {"same_player": q(same), "same_look_diff_player": q(sl), "diff_look_diff_player": q(dl),
         "same_look_player_pairs": sum(look.values()), "player_pairs": len(look)}
    for k, v in mg.items():
        v = np.asarray(v)
        o[f"clean_rival_{k}"] = {**q(v), "right_at_margin": int((v >= M).sum()), "wrong_at_margin": int((v <= -M).sum()),
                                 "undecided": int(((v > -M) & (v < M)).sum())}
        pool[k] += list(v)
    pool["same"] += same; pool["sl"] += sl; pool["dl"] += dl
    out["clips"][c] = o
out["pooled"] = {"same_player": q(pool["same"]), "same_look_diff_player": q(pool["sl"]), "diff_look_diff_player": q(pool["dl"])}
for k in ("same_look", "diff_look"):
    v = np.asarray(pool[k]); out["pooled"][f"clean_rival_{k}"] = {**q(v), "right_at_margin": int((v >= M).sum()),
        "wrong_at_margin": int((v <= -M).sum()), "undecided": int(((v > -M) & (v < M)).sum())}
# overlap of same-player vs same-look-different-player similarity: share of same-look pairs above the same-player median
s, d = np.asarray(pool["same"]), np.asarray(pool["sl"])
out["pooled"]["same_look_above_same_player_p25"] = round(float((d >= np.percentile(s, 25)).mean()), 3)
out["pooled"]["auc_same_vs_same_look"] = round(float((s[:, None] > d[None, :2000]).mean()), 3)
print(json.dumps(out, indent=1))
