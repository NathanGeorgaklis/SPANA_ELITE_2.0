import json, sys
from collections import Counter
from pathlib import Path
from spana.artifacts import read_json
from spana.config import load_config
from spana.identity.confirmation import CropFeatures, claims_from_targets
from spana.identity.follower import Selection, follow
from spana.artifacts import sha256_doc
from tools.benchmark import load_frozen
import tools.target_score as TS
cfg = load_config("config/default.yaml")
B = Path.home() / "spana-player-confirmation-run/base"
CL = {"F02": "FRESH_02_G1_3000_midfield_buildup_passing", "F05": "FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper",
      "F07": "FRESH_07_G3_1205_midfield_passing_switch", "F08": "FRESH_08_G3_2103_attack_into_box_clearance"}
tot = Counter()
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
    trk = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in tdoc["frames"]}
    frames_sorted = [fr["frame"] for fr in tdoc["frames"]]
    for p in ids:
        s = TS.first_selection(p, gt, attr)
        if s is None: continue
        sel = Selection.track_at_frame(s[2], s[1])
        acts = TS.simulated_review(tdoc, sel, cfg, gt, attr, p, confirm=True, crops=crops,
                                   claims={q: c for q, c in allc.items() if q != p})
        for k, a in enumerate(acts):
            tot["kind:" + a["kind"]] += 1
            if a["kind"] != "player_confirmation": continue
            tid, f, fe = a["track_id"], a["frame"], a["frame_end"]
            eps = trk[f][tid]["uncertain_episodes"]
            # why did the range stop at fe? look at the next frame after fe
            nxt = next((x for x in frames_sorted if x > fe), None)
            lab_after = next((x for x in gt.frames if x > fe), None)
            if nxt is None: why = "clip_end"
            elif tid not in trk[nxt]: why = "track_ended"
            elif trk[nxt][tid]["uncertain_episodes"] != eps: why = "new_episode_same_track"
            else:
                # attribution broke on next labelled frame, or gap end
                x = lab_after
                if x is not None and x in attr and (attr[x].get(tid, (None, False))[0] != p or attr[x].get(tid, (None, False))[1]):
                    pp, amb = attr[x].get(tid, (None, False))
                    why = "attr:" + ("ambiguous" if amb else ("not_observed_or_unlabelled" if pp is None else "other_player"))
                else: why = "gap_end_or_other"
            tot["stop:" + why] += 1
            # next action of this reviewer: same track continuing?
            if k + 1 < len(acts):
                b = acts[k + 1]
                same = b.get("track_id") == tid
                tot["next:" + b["kind"] + (":same_track" if same else ":other_track") + ":" + why] += 1
            # frames covered
            tot["covered_frames"] += fe - f + 1
for k in sorted(tot): print(k, tot[k])
