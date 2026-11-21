"""DEV same player vs same-look different player AUC (pooled pairs of distributions.py), per clip and pooled."""
import json, sys, collections
import numpy as np
def auc(p, n):
    p, n = np.asarray(p), np.asarray(n); o = np.concatenate([p, n]).argsort(kind="stable").argsort() + 1.0
    return round(float((o[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))), 4)
T = json.load(open(sys.argv[1])); out = {}; P_same, P_sl = [], []
for t in T["tables"]:
    P = t["pairs"]["all"]; hp = collections.defaultdict(list)
    for h, pl in zip(P["diff"]["scores"]["hist"], P["diff"]["players"]): hp[tuple(sorted(pl.split("|")))].append(h)
    look = {k: float(np.mean(v)) >= 0.70 for k, v in hp.items()}
    same = P["same"]["scores"]["spana"]
    sl = [s for s, pl in zip(P["diff"]["scores"]["spana"], P["diff"]["players"]) if look[tuple(sorted(pl.split("|")))]]
    out[t["match"][:8]] = auc(same, sl) if sl else None; P_same += same; P_sl += sl
out["pooled"] = auc(P_same, P_sl); print(json.dumps(out))
