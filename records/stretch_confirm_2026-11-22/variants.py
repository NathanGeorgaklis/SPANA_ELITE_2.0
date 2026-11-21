import json, sys
from collections import Counter
from pathlib import Path
from spana.artifacts import read_json, sha256_doc
from spana.config import load_config
from spana.identity.confirmation import CropFeatures
import tools.target_score as TS
cfg = load_config("config/default.yaml")
B = Path.home() / "spana-player-confirmation-run/base"
CL = {"F02": "FRESH_02_G1_3000_midfield_buildup_passing", "F05": "FRESH_05_G2_5705_midfield_1v1_duel_backpass_keeper",
      "F07": "FRESH_07_G3_1205_midfield_passing_switch", "F08": "FRESH_08_G3_2103_attack_into_box_clearance"}
orig = TS.confirmation_end
def make(skip_amb):
    def stretch_end(tracks_doc, gt, attr, pid, tid, f, g):
        trk = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in tracks_doc["frames"]}
        eps = trk[f][tid]["uncertain_episodes"]; end = f
        for fr in tracks_doc["frames"]:
            x = fr["frame"]
            if x <= f: continue
            if x > g["end_frame"] or tid not in trk[x] or trk[x][tid]["uncertain_episodes"] != eps: break
            if x in attr:
                p, amb = attr[x].get(tid, (None, False))
                if p == pid and not amb: end = x; continue
                if p is not None and not amb and p != pid: break          # the track shows another player
                if amb and not skip_amb: break
                if any(q == pid and not a for t2, (q, a) in attr[x].items() if t2 != tid): break   # player elsewhere
        return end
    return stretch_end
mode = sys.argv[1]
if mode != "orig": TS.confirmation_end = make(mode == "v2")
res = TS.main(["--config", "config/default.yaml", *sum((["--clip", f"data/labels/{n}/v1", str(B / f"{k}_ours.trk.json")] for k, n in CL.items()), []),
               "--out", str(Path.home() / f"spana-stretch-{mode}.json"), "--simulate-reviewer", "--confirmations", "--crops"])
