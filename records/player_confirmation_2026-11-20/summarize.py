"""Summarise the player-confirmation DEV runs (DEV analysis only): python summarize.py OUT_DIR > compare_summary.json"""
import json
import sys
from collections import Counter
from pathlib import Path

K = ("correct", "wrong", "unverified", "candidate_correct", "candidate_wrong", "candidate_unverified", "abstain", "lost")
CL = ("F02", "F05", "F07", "F08")


def tracking(d: Path, kind: str) -> dict:
    s = Counter()
    for c in CL:
        p = d / f"{c}_{kind}.bench.json"
        if not p.exists():
            return {}
        b = json.load(open(p))["summary"]
        for k in ("identity_transfers", "unflagged_transfers", "transfers_on_linked_tracks", "fragments_total",
                  "id_switches"):
            s[k] += b[k]
    return dict(s)


def target(p: Path) -> dict:
    if not p.exists():
        return {}
    r = json.load(open(p))
    tot, conf, reasons, undo, byr = Counter(), Counter(), Counter(), Counter(), Counter()
    act, fs, gal, per_clip, wrong_events, greid = 0, 0.0, 0, {}, [], []
    for cl in r["clips"]:
        q = cl["pooled"]
        n = sum(q[k] for k in K)
        for k in K:
            tot[k] += q[k]
        act += sum(q["actions"].values())
        fs += q["followed_s"]
        conf.update(q.get("confirmations", {}))
        reasons.update(q.get("conflict_reasons", {}))
        undo.update(q.get("undo", {}))
        byr.update(q.get("answers_by_reason", {}))
        gal += q.get("gallery_crops", 0)
        per_clip[cl["clip"][:8]] = {"correct": round(q["correct"] / n, 3), "wrong": q["wrong"], "abstain": q["abstain"],
                                    "confirmations": q.get("confirmations")}
        for p_ in cl["per_player"]:
            for e in (p_.get("confirmation_stats") or {}).get("wrong_events", []):
                wrong_events.append({"clip": cl["clip"][:8], "player": p_["player"], **e})
            for g in p_.get("gallery_reid") or []:
                greid.append({"clip": cl["clip"][:8], "player": p_["player"], **g})
    n = sum(tot.values())
    return {"correct": round(tot["correct"] / n, 3), "wrong": tot["wrong"], "unverified": tot["unverified"],
            "abstain_frames": tot["abstain"], "lost_frames": tot["lost"], "scored": n,
            "actions_per_min": round(act / (fs / 60), 2), "actions": act, "followed_min": round(fs / 60, 2),
            "confirmations": dict(conf), "conflict_reasons": dict(reasons), "undo": dict(undo),
            "answers_by_reason": dict(byr), "gallery_crops": gal, "per_clip": per_clip,
            "wrong_confirmations": wrong_events,
            "gallery_reid": {"episodes_evaluated": len(greid),
                             "episodes_answered": sum(1 for g in greid if g["answered"]),
                             "frames_answered": sum(g["answered"] for g in greid),
                             "hold_reasons": dict(sum((Counter(g["hold"]) for g in greid), Counter())),
                             "against": sum(g["against"] for g in greid), "episodes": greid}}


def containment(correct: Path, err: Path) -> dict:
    """Wrong confirmations stay with the player who made them: every player whose own taps were all right has the
    same scored answers as in the always-correct run; players whose answers changed all made a wrong tap."""
    if not err.exists():
        return {}
    a = {(c["clip"][:8], p["player"]): p for c in json.load(open(correct))["clips"] for p in c["per_player"]}
    changed, with_wrong = [], []
    for c in json.load(open(err))["clips"]:
        for p in c["per_player"]:
            key = (c["clip"][:8], p["player"])
            if not p.get("selected"):
                continue
            if (p.get("confirmation_stats") or {}).get("wrong"):
                with_wrong.append(key)
            if any(p[k] != a[key][k] for k in K):
                changed.append(key)
    return {"players_with_wrong_taps": len(with_wrong), "players_changed": len(changed),
            "changed_without_own_wrong_tap": [list(k) for k in changed if k not in with_wrong]}


def main(out: str) -> None:
    o = Path(out)
    res = {"tracking": {"ours": tracking(o / "base", "ours"), "linked": tracking(o / "base", "linked"),
                        "reid_on": tracking(o / "reid", "ours")}}
    for name in ("target_default", "target_reviewer", "target_confirm_correct", "target_confirm_err",
                 "target_confirm_err30", "target_confirm_episode_scope", "target_linked_reviewer",
                 "target_linked_confirm_correct"):
        res[name] = target(o / "base" / f"{name}.json")
    for name in ("target_confirm_err", "target_confirm_err30"):
        res[name]["containment"] = containment(o / "base" / "target_confirm_correct.json", o / "base" / f"{name}.json")
    res["reid_on"] = {k: target(o / "reid" / f"{k}.json") for k in ("target_reviewer", "target_confirm_correct")}
    ep = o / "reid" / "episodes.json"
    if ep.exists():
        e = json.load(open(ep))
        res["reid_on"]["tracker_episodes"] = e.get("pooled", e.get("summary"))
    json.dump(res, sys.stdout, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
