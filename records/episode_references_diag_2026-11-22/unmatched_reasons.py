"""Why the NO_CLEAN_CROP tracks go unmatched inside their episodes (decision-log reasons). DEV analysis only."""
import json
from collections import Counter
from pathlib import Path
run = Path.home() / "spana-player-confirmation-run/reid"
D = json.load(open(Path.home() / "spana-epref-diag.json"))
K = {"FRESH_02": "F02", "FRESH_05": "F05", "FRESH_07": "F07", "FRESH_08": "F08"}
tot, first = Counter(), Counter()
for key in K.values():
    dec = [json.loads(l) for l in open(run / f"{key}_ours.trk.decisions.jsonl")]
    for e in D["NO_CLEAN_CROP"]:
        if K[e["clip"]] != key:
            continue
        seen = False
        for d in dec:
            if d.get("track_id") != e["track"] or not (e["start"] <= d["frame"] <= e["end"]):
                continue
            if d["kind"] == "unmatched":
                r = d["reason"] if d["reason"].startswith("GATE_FAILED") else d["reason"].split(":")[0]
                tot["unmatched:" + r] += 1
                if not seen:
                    first["first_unmatched:" + r] += 1
                    seen = True
            elif d["kind"] in ("recovery_refused", "match_refused"):
                tot[d["kind"] + ":" + d["reason"]] += 1
for c in (tot, first):
    for k, v in c.most_common():
        print(k, v)
