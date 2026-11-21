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
MIN_COVER = int(sys.argv[1])
from tools.target_score import *
from tools.target_score import follow, review_list, gaps_of, confirm_relink, reject_candidate, offer_verdict, reselect, player_confirmation, wrong_track, draw
def simulated_review(tracks_doc: dict, sel: Selection, cfg, gt, attr, pid: str, max_steps: int = 1000,
                     emb: Optional[dict] = None, confirm: bool = False, sim: Optional[ReviewSimConfig] = None,
                     clip: str = "", crops: Optional[CropFeatures] = None, claims: Optional[dict] = None,
                     stats: Optional[dict] = None) -> List[dict]:
    """An after-match reviewer working through the gap list (decision 2026-11-16), knowing the truth: for each gap in
    turn, it confirms an offer judged right or rejects one judged wrong (on the offer's first frame); otherwise it taps
    the player on the gap's first labelled frame where they have an unambiguous track. Best case of the review UI.
    Only the gaps the review list shows (target.review_min_gap_s) are worked through."""
    actions: List[dict] = []
    handled: set = set()
    retried: set = set()
    trk = {fr["frame"]: fr for fr in tracks_doc["frames"]}
    wrong_ids: set = set()
    gap_of: Dict[str, int] = {}            # confirmation id -> start frame of the gap it was made in
    for _ in range(max_steps):
        frames, info = follow(tracks_doc, sel, cfg, actions, emb, crops, claims)
        last = actions[-1] if actions else None
        if confirm and last is not None and last["kind"] == "player_confirmation":
            e = next(x for x in info["confirmation_log"] if x["confirmation_id"] == last["confirmation_id"])
            gs = gap_of[last["confirmation_id"]]
            if e["status"] == "CONFLICT" and gs not in retried:     # surfaced for re-confirmation: once more
                retried.add(gs)
                handled.discard(gs)
        todo = [g for g in review_list(gaps_of(frames, info["relinks"]), cfg) if g["start_frame"] not in handled]
        if not todo:
            if stats is not None:
                stats["wrong_ids"] = wrong_ids
            return actions
        g = todo[0]
        handled.add(g["start_frame"])
        act = None
        for key in g["offers"]:
            r = next(r for r in info["relinks"] if [r["from_track"], r["to_track"], r["proposed_frame"]] == key)
            if r["outcome"] == "CONFIRMED" or r["rejection"] is not None:
                continue
            verdict = offer_verdict(r, gt, attr, pid)
            if verdict == "right":
                act = confirm_relink(r, r["proposed_frame"])
            elif verdict == "wrong":
                act = reject_candidate(r, r["proposed_frame"])
            if act is not None:
                break
        if act is None:
            ans = {a["frame"]: a for a in frames}
            taken = {a["frame"] for a in actions if a["kind"] in ("confirm_relink", "track_at_frame",
                                                                  "player_confirmation")}
            for f in gt.frames:
                if not (g["start_frame"] <= f <= g["end_frame"]) or pid not in gt.players.get(f, {}) or f in taken:
                    continue
                hit = [tid for tid, (p, amb) in attr[f].items() if p == pid and not amb]
                if hit and not (ans[f]["track_id"] == hit[0] and ans[f]["reason"] == "NOT_CONFIRMED"):
                    tr = next(t for t in trk[f]["tracks"] if t["track_id"] == hit[0])
                    if confirm and tr["identity"] != "CERTAIN":
                        cid = f"{pid}@{f}#{len(actions)}"
                        tid, f_end = hit[0], confirmation_end(tracks_doc, gt, attr, pid, hit[0], f, g)
                        if sum(1 for x in attr if f <= x <= f_end) < MIN_COVER:
                            continue                     # effort bound: too short a stretch to be worth an action
                        if sim is not None and draw(sim, clip, pid, f, len(actions)) < sim.wrong_is_me_rate:
                            w = wrong_track(trk[f], attr[f], hit[0], pid)
                            if w is not None:
                                tid, f_end = w, f
                                wrong_ids.add(cid)
                        act = player_confirmation(cid, tid, f, "IS_ME", f_end)
                        gap_of[cid] = g["start_frame"]
                    else:
                        act = reselect(hit[0], f)
                    break
        if act is not None:
            actions.append(act)
    raise RuntimeError(f"simulated review did not settle for player {pid}")

TS.simulated_review = simulated_review
TS.confirmation_end = make(len(sys.argv) < 3)
confirmation_end = TS.confirmation_end
TS.main(["--config", "config/default.yaml", *sum((["--clip", f"data/labels/{n}/v1", str(B / f"{k}_ours.trk.json")] for k, n in CL.items()), []),
         "--out", str(Path.home() / f"spana-frontier-{MIN_COVER}{sys.argv[2] if len(sys.argv) > 2 else ""}.json"), "--simulate-reviewer", "--confirmations", "--crops"])
