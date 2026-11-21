import json, sys
from collections import Counter
from pathlib import Path
from spana.artifacts import read_json, sha256_doc
from spana.config import load_config
from spana.identity.confirmation import CropFeatures, claims_from_targets
from spana.identity.follower import Selection, follow
from tools.benchmark import load_frozen
import tools.target_score as TS
cfg = load_config("config/default.yaml")
B = Path.home() / "spana-player-confirmation-run/base"
CL = {"F02": "FRESH_02_G1_3000_midfield_buildup_passing", "F05": "FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper",
      "F07": "FRESH_07_G3_1205_midfield_passing_switch", "F08": "FRESH_08_G3_2103_attack_into_box_clearance"}
out = {}
for key, name in CL.items():
    _, gt = load_frozen(Path(f"data/labels/{name}/v1")); tp = B / f"{key}_ours.trk.json"; tdoc = read_json(tp)
    attr = TS.attributions(gt, tdoc)
    ids = sorted({p for f in gt.frames for p in gt.players.get(f, {})}, key=int)
    crops = CropFeatures.load(tp.with_name(tp.name.replace(".trk.json", ".crops.npz")), sha256_doc(tdoc))
    base = {}
    for p in ids:
        s = TS.first_selection(p, gt, attr)
        if s is None: continue
        sp = Selection.track_at_frame(s[2], s[1]); acts = TS.simulated_review(tdoc, sp, cfg, gt, attr, p)
        base[p] = {"frames": follow(tdoc, sp, cfg, acts)[0]}
    allc = claims_from_targets(base.items())
    for p in ids:
        s = TS.first_selection(p, gt, attr)
        if s is None: continue
        sel = Selection.track_at_frame(s[2], s[1])
        cl = {q: c for q, c in allc.items() if q != p}
        acts = TS.simulated_review(tdoc, sel, cfg, gt, attr, p, confirm=True, crops=crops, claims=cl)
        frames, info = follow(tdoc, sel, cfg, acts, None, crops, cl)
        # answered sequence: runs of (state, track, reason)
        runs = []
        for a in frames:
            k = (a["state"], a.get("track_id"), a["reason"])
            if runs and runs[-1][0] == k: runs[-1][2] = a["frame"]
            else: runs.append([k, a["frame"], a["frame"]])
        out[f"{key}/{p}"] = {"sel": [s[1], s[2]], "actions": [{k: v for k, v in a.items() if k in ("kind", "track_id", "frame", "frame_end", "to_track", "proposed_frame")} for a in acts],
                             "runs": [[*r[0], r[1], r[2]] for r in runs]}
json.dump(out, open(Path.home() / "spana-stretch-seq.json", "w"), indent=0)
