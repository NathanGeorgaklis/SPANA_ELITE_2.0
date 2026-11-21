import json
from collections import Counter
from pathlib import Path
from spana.artifacts import read_json
from tools.benchmark import load_frozen
import tools.target_score as TS
B = Path.home() / "spana-player-confirmation-run/base"
CL = {"F02": "FRESH_02_G1_3000_midfield_buildup_passing", "F05": "FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper",
      "F07": "FRESH_07_G3_1205_midfield_passing_switch", "F08": "FRESH_08_G3_2103_attack_into_box_clearance"}
seq = json.load(open(Path.home() / "spana-stretch-seq.json"))
tot = Counter()
for key, name in CL.items():
    _, gt = load_frozen(Path(f"data/labels/{name}/v1")); tdoc = read_json(B / f"{key}_ours.trk.json")
    attr = TS.attributions(gt, tdoc)
    trk = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in tdoc["frames"]}
    for k, v in seq.items():
        if not k.startswith(key): continue
        pid = k.split("/")[1]
        acts = [a for a in v["actions"] if a["kind"] in ("player_confirmation", "track_at_frame")]
        for a, b in zip(acts, acts[1:]):
            if a["track_id"] != b["track_id"]: continue
            tid = a["track_id"]; e = a.get("frame_end", a["frame"])
            ea, eb = trk[a["frame"]][tid]["uncertain_episodes"], trk[b["frame"]][tid]["uncertain_episodes"]
            why = []
            if ea != eb: why.append("new_episode")
            for x in sorted(f for f in attr if e < f < b["frame"]):
                p, amb = attr[x].get(tid, (None, False))
                if p is not None and p != pid and not amb: why.append("other_player_on_track"); break
                if any(q == pid and not am for t2, (q, am) in attr[x].items() if t2 != tid): why.append("player_on_other_track"); break
            if trk[b["frame"]][tid]["identity"] == "CERTAIN": why.append("next_is_tap_certain")
            tot[a["kind"] + "->" + b["kind"] + ":" + "+".join(why or ["none"])] += 1
for k, v in tot.most_common(): print(v, k)
