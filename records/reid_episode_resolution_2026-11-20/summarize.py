"""Summarise one DEV run dir: tracking, follower, reviewer, uncertain share, resolutions (DEV analysis only)."""
import json, sys, collections
from pathlib import Path
CL = ["F02", "F05", "F07", "F08"]
def run(d):
    d = Path(d); o = {}
    for t in ("ours", "linked"):
        s = collections.Counter()
        for c in CL:
            b = json.load(open(d / f"{c}_{t}.bench.json"))["summary"]
            for k in ("identity_transfers", "unflagged_transfers", "transfers_on_linked_tracks", "fragments_total", "id_switches", "link_caused_transfers"):
                s[k] += b[k]
        o[t + "_tracking"] = dict(s)
        for m in ("default", "reviewer"):
            p = json.load(open(d / f"target_{t}_{m}.json"))
            S = collections.Counter(); fs = 0.0; pc = {}; pa = {}
            for cl in p["clips"]:
                q = cl["pooled"]
                for k in ("correct", "wrong", "abstain", "lost", "unverified", "candidate_correct", "candidate_wrong", "candidate_unverified"):
                    S[k] += q[k]
                S["actions"] += sum(q["actions"].values()); fs += q["followed_s"]
                pc[cl["clip"][:8]] = q["correct_time"]; pa[cl["clip"][:8]] = q["abstain"]
            sc = sum(S[k] for k in ("correct", "wrong", "abstain", "lost", "unverified", "candidate_correct", "candidate_wrong", "candidate_unverified"))
            o[f"{t}_{m}"] = {"correct": round(S["correct"] / sc, 3), "wrong": S["wrong"], "abstain": S["abstain"], "lost": S["lost"],
                             "actions_per_min": round(S["actions"] / (fs / 60), 2), "scored": sc,
                             "per_clip_correct": pc, "per_clip_abstain": pa}
        rp = d / f"resolution_{t}.json"; r = json.load(open(rp)) if rp.exists() else {"pooled": None}
        o[t + "_resolutions"] = r["pooled"]
    unc = {}
    for c in CL:
        tr = json.load(open(d / f"{c}_ours.trk.json"))
        n = u = 0
        for fr in tr["frames"]:
            for x in fr["tracks"]:
                if x["source"] == "OBSERVED" and x["confirmed"] and x.get("state", "ACTIVE") == "ACTIVE":
                    n += 1; u += x["identity"] != "CERTAIN"
        unc[c] = round(u / n, 3)
    o["uncertain_share"] = unc
    dec = collections.Counter()
    for c in CL:
        for l in open(d / f"{c}_ours.trk.decisions.jsonl"):
            x = json.loads(l)
            if x["kind"].startswith("identity_resol") or x["kind"].startswith("reid_"):
                dec[(x["kind"], x.get("resolved_by") or x.get("reason"))] += 1
    o["decision_counts"] = {f"{a}:{b}": v for (a, b), v in sorted(dec.items(), key=str)}
    return o
if __name__ == "__main__":
    print(json.dumps({Path(a).name: run(a) for a in sys.argv[1:]}, indent=1))
