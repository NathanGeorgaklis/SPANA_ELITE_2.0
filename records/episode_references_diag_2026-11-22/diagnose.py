"""Diagnose the NO_REFERENCE / NO_CLEAN_CROP refused IDENTITY_UNCERTAIN episodes on DEV (analysis only, no product
code). Inputs: the reid_resolve-on tracker run (tracks + decision logs) and records' episodes.json
(tools/dev/reid_episode_report.py), plus the frozen DEV labels for who each box is.

    python records/episode_references_diag_2026-11-22/diagnose.py RUN_DIR OUT.json LABELS_DIR:CLIP_KEY ...

Reference reconstruction mirrors spana/tracking/tracker.py (config 2.14.0+): a track's re-ID reference is set on a
frame where it is OBSERVED, CERTAIN, confirmed, uncontested and clean (IoU <= appearance.clean_max_iou with every
other OBSERVED box of the frame); a rival's / predecessor's reference is the snapshot taken at the contest / spawn.
The reconstruction is checked against the tracker's own reid_evidence rows (agreement reported).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from spana.config import load_config
from spana.geometry import iou_matrix
from tools.benchmark import load_frozen
from tools.dev.resolution_check import owners

CFG = load_config("config/default.yaml")
CLEAN = CFG.appearance.clean_max_iou


def load(run, key):
    td = json.load(open(run / f"{key}_ours.trk.json"))
    dec = [json.loads(l) for l in open(run / f"{key}_ours.trk.decisions.jsonl")]
    return td, dec


def frame_overlaps(td):
    """(frame, track) -> (max IoU with another OBSERVED box, that track id)."""
    out = {}
    for fr in td["frames"]:
        obs = [t for t in fr["tracks"] if t["source"] == "OBSERVED"]
        if len(obs) < 2:
            for t in obs:
                out[(fr["frame"], t["track_id"])] = (0.0, None)
            continue
        b = np.array([t["box"] for t in obs], float)
        m = iou_matrix(b, b)
        np.fill_diagonal(m, 0.0)
        for k, t in enumerate(obs):
            j = int(m[k].argmax())
            out[(fr["frame"], t["track_id"])] = (float(m[k, j]), obs[j]["track_id"])
    return out


def clip_diag(run, label_dir, key):
    td, dec = load(run, key)
    _, gt = load_frozen(Path(label_dir))
    own = owners(gt, td)
    t_of = {fr["frame"]: fr["t"] for fr in td["frames"]}
    ov = frame_overlaps(td)
    rows = defaultdict(list)
    for fr in td["frames"]:
        for t in fr["tracks"]:
            rows[t["track_id"]].append((fr["frame"], t))
    spawn = {d["track_id"]: d for d in dec if d["kind"] == "spawn"}
    unc = defaultdict(list)
    for d in dec:
        if d["kind"] == "identity_uncertain":
            unc[d["track_id"]].append(d)
    ends = {d["track_id"]: d for d in dec if d["kind"] == "end"}
    ev = defaultdict(list)
    for d in dec:
        if d["kind"] == "reid_evidence":
            ev[d["track_id"]].append(d)

    def eligible(f, r):
        return (r["source"] == "OBSERVED" and r["identity"] == "CERTAIN" and r["confirmed"] and not r["contested"]
                and ov[(f, r["track_id"])][0] <= CLEAN)

    first_ref = {tid: next((f for f, r in rs if eligible(f, r)), None) for tid, rs in rows.items()}

    def player(tid):
        o = own.get(tid, {})
        c = Counter(o.values()).most_common(1)
        return c[0][0] if c else None

    def history(tid, at):
        """Why track tid had no re-ID reference at frame `at`."""
        rs = [(f, r) for f, r in rows[tid] if f < at] or rows[tid][:1]
        sp = spawn.get(tid, {})
        preds = sp.get("possible_predecessors", [])
        born_unc = rs[0][1]["identity"] != "CERTAIN"
        cert = [(f, r) for f, r in rs if r["identity"] == "CERTAIN"]
        cert_conf = [(f, r) for f, r in cert if r["confirmed"]]
        cert_conf_obs = [(f, r) for f, r in cert_conf if r["source"] == "OBSERVED"]
        cert_conf_obs_unc = [(f, r) for f, r in cert_conf_obs if not r["contested"]]
        if first_ref[tid] is not None and first_ref[tid] < at:
            why = "HAD_REFERENCE?"            # reconstruction disagrees with the tracker
        elif born_unc and not cert:
            why = "BORN_UNCERTAIN_NEVER_CERTAIN"
        elif not cert:
            why = "NEVER_CERTAIN"
        elif not cert_conf:
            why = "CERTAIN_NEVER_CONFIRMED"
        elif not cert_conf_obs:
            why = "CERTAIN_NOT_OBSERVED"
        elif not cert_conf_obs_unc:
            why = "CERTAIN_ONLY_CONTESTED"
        else:
            why = "CERTAIN_NEVER_CLEAN"
        ious = [ov[(f, tid)][0] for f, r in cert_conf_obs_unc]
        first_unc = next((d["rule"] for d in unc[tid]), None)
        # did the track become uncertain before it could confirm? (a contest in its first certain stretch)
        return {"track": tid, "why": why, "player": player(tid), "born_frame": rows[tid][0][0],
                "frames_before": len(rs), "certain_frames_before": len(cert),
                "certain_confirmed_observed_frames_before": len(cert_conf_obs),
                "min_iou_when_certain": round(min(ious), 3) if ious else None,
                "spawn_predecessors": preds, "born_uncertain": born_unc, "first_uncertain_rule": first_unc,
                "ref_later_frame": first_ref[tid] if first_ref[tid] is not None and first_ref[tid] >= at else None,
                "end_reason": ends.get(tid, {}).get("reason")}

    return td, rows, ev, history, ov, own, t_of, first_ref, spawn


def main(run_dir, out_path, *specs):
    run = Path(run_dir)
    epd = json.load(open(run / "episodes.json"))
    eps = {c["clip"][:8]: c["episodes"] for c in epd["clips"]}
    out = {"clean_max_iou": CLEAN, "NO_REFERENCE": [], "NO_CLEAN_CROP": [], "check": Counter()}
    for spec in specs:
        label_dir, key = spec.split(":")
        clip = Path(label_dir).parent.name[:8]
        td, rows, ev, history, ov, own, t_of, first_ref, spawn = clip_diag(run, label_dir, key)
        # reconstruction check: own reference present whenever the tracker evaluated a PAIR rival with own score
        for tid, es in ev.items():
            for e in es:
                has = first_ref[tid] is not None and first_ref[tid] < e["frame"]
                if e["status"] == "NO_OWN_REFERENCE":
                    out["check"]["own_missing_agree" if not has else "own_missing_DISAGREE"] += 1
                elif any("own" in v for v in e["rivals"].values()):
                    out["check"]["own_present_agree" if has else "own_present_DISAGREE"] += 1
        for ep in eps[clip]:
            r = ep.get("refusal")
            if r not in ("NO_REFERENCE", "NO_CLEAN_CROP"):
                continue
            tid, s, e = ep["track_id"], ep["start_frame"], ep["end_frame"]
            es = [x for x in ev[tid] if s <= x["frame"] <= e]
            ep_rows = [(f, x) for f, x in rows[tid] if s <= f <= e]
            obs = [(f, x) for f, x in ep_rows if x["source"] == "OBSERVED"]
            base = {"clip": clip, "track": tid, "start": s, "end": e, "frames": len(ep_rows), "observed": len(obs),
                    "outcome": ep["outcome"], "at_spawn": ep["at_spawn"], "rivals": ep["rivals"],
                    "player": history(tid, e)["player"], "evidence": ep["evidence_frames"]}
            if r == "NO_REFERENCE":
                missing = {}
                for x in es:
                    if x["status"] == "NO_OWN_REFERENCE":
                        missing.setdefault(tid, ("own", x["frame"]))
                    for rid, v in x["rivals"].items():
                        if v.get("refused") == "NO_RIVAL_REFERENCE":
                            missing.setdefault(rid, ("rival", s))
                        for p in v.get("missing", []):
                            missing.setdefault(p, ("predecessor", s))
                base["missing"] = [{"role": role, **history(m, at)} for m, (role, at) in missing.items()]
                # would every other reference exist? (i.e. is the missing one the only blocker)
                out["NO_REFERENCE"].append(base)
            else:
                st = Counter(x["status"] for x in es)
                nc = [x for x in es if x["status"] == "NOT_CLEAN"]
                kinds = Counter()
                ious = []
                for x in nc:
                    iou, other = ov.get((x["frame"], tid), (None, None))
                    ious.append(iou)
                    me = own.get(tid, {}).get(x["frame"])
                    them = own.get(other, {}).get(x["frame"]) if other else None
                    if x["frame"] not in own.get(tid, {}) and other not in own or (me is None and them is None):
                        kinds["unlabelled_frame_or_unmatched"] += 1
                    elif them is None:
                        kinds["other_box_not_a_labelled_player"] += 1
                    elif me is not None and me == them:
                        kinds["duplicate_of_same_player"] += 1
                    else:
                        kinds["other_labelled_player"] += 1
                base.update(statuses=dict(st), not_clean_overlap=dict(kinds),
                            not_clean_iou_median=round(float(np.median(ious)), 3) if ious else None,
                            not_clean_iou_le_0_15=sum(1 for i in ious if i is not None and i <= 0.15),
                            not_observed_frames=len(ep_rows) - len(obs))
                # references: would a clean crop have been usable?
                refs_ok = first_ref[tid] is not None and first_ref[tid] < s
                base["own_reference_before"] = refs_ok
                base["rival_references"] = {rv: first_ref.get(rv) is not None and first_ref[rv] < s
                                            for rv in ep["rivals"]}
                out["NO_CLEAN_CROP"].append(base)
    # summaries
    nr = out["NO_REFERENCE"]
    miss = [m for e in nr for m in e["missing"]]
    out["summary"] = {
        "NO_REFERENCE": {"episodes": len(nr), "at_spawn": sum(e["at_spawn"] for e in nr),
                         "missing_refs": len(miss), "missing_by_role": dict(Counter(m["role"] for m in miss)),
                         "why": dict(Counter(m["why"] for m in miss)),
                         "why_by_role": {ro: dict(Counter(m["why"] for m in miss if m["role"] == ro))
                                         for ro in ("own", "rival", "predecessor")},
                         "first_uncertain_rule_of_missing": dict(Counter(m["first_uncertain_rule"] for m in miss)),
                         "missing_gets_reference_later": sum(1 for m in miss if m["ref_later_frame"] is not None),
                         "missing_unlabelled": sum(1 for m in miss if m["player"] is None),
                         "outcome": dict(Counter(e["outcome"] for e in nr)),
                         "episode_frames_median": float(np.median([e["frames"] for e in nr])) if nr else None},
        "NO_CLEAN_CROP": {"episodes": len(out["NO_CLEAN_CROP"]),
                          "statuses": dict(sum((Counter(e["statuses"]) for e in out["NO_CLEAN_CROP"]), Counter())),
                          "not_clean_overlap": dict(sum((Counter(e["not_clean_overlap"]) for e in out["NO_CLEAN_CROP"]),
                                                        Counter())),
                          "observed_frames_median": float(np.median([e["observed"] for e in out["NO_CLEAN_CROP"]])),
                          "frames_median": float(np.median([e["frames"] for e in out["NO_CLEAN_CROP"]])),
                          "evidence_frames_median": float(np.median([sum(e["statuses"].values())
                                                                     for e in out["NO_CLEAN_CROP"]])),
                          "own_reference_before": sum(e["own_reference_before"] for e in out["NO_CLEAN_CROP"]),
                          "all_rival_references_before": sum(all(e["rival_references"].values())
                                                             for e in out["NO_CLEAN_CROP"]),
                          "outcome": dict(Counter(e["outcome"] for e in out["NO_CLEAN_CROP"]))}}
    out["check"] = dict(out["check"])
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"check": out["check"], "summary": out["summary"]}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
