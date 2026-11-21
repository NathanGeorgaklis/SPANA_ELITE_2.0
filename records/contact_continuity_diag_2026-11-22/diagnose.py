"""Contact losses on DEV (analysis only, no product code): every time a labelled player's track changes (track A on
one labelled frame, B on the next labelled frame that attributes the player unambiguously, NESTED_V2), find what
happened to A in between and the first gate that failed, with the evidence the tracker had on that frame.

    python records/contact_continuity_diag_2026-11-22/diagnose.py RUN_DIR OUT.json LABELS_DIR:KEY ...

RUN_DIR holds KEY_ours.trk.json + .trk.decisions.jsonl (the default tracker; records/player_confirmation_2026-11-20).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from spana.config import load_config
from tools.benchmark import load_frozen
from tools.target_score import attributions

CFG = load_config("config/default.yaml")


def first_gate(c):
    """Order of the gates as the tracker applies them (spana/tracking/tracker.py)."""
    for g in ("centre_ok", "mahalanobis_ok", "height_ok", "low_score_iou_ok", "appearance_veto"):
        if g in c["failed"]:
            return g
    return None


def clip(run, label_dir, key):
    td = json.load(open(run / f"{key}_ours.trk.json"))
    dec = [json.loads(l) for l in open(run / f"{key}_ours.trk.decisions.jsonl")]
    _, gt = load_frozen(Path(label_dir))
    attr = attributions(gt, td)
    rows = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in td["frames"]}
    first_row = {}
    for fr in td["frames"]:
        for t in fr["tracks"]:
            first_row.setdefault(t["track_id"], fr["frame"])
    by = defaultdict(list)
    for d in dec:
        by[(d["kind"], d.get("track_id"))].append(d)
    spawn = {d["track_id"]: d for d in dec if d["kind"] == "spawn"}
    unc = defaultdict(set)
    for d in dec:
        if d["kind"] == "identity_uncertain":
            for r in ([d["rival"]] if d.get("rival") else d.get("rivals", [])):
                unc[d["track_id"]].add(r)
                unc[r].add(d["track_id"])
    out = []
    players = sorted({p for f in gt.frames for p in gt.players.get(f, {})}, key=int)
    for pid in players:
        seq = []
        for f in sorted(attr):
            hits = [t for t, (p, amb) in attr[f].items() if p == pid and not amb]
            if len(hits) == 1:
                seq.append((f, hits[0]))
        for (f1, a), (f2, b) in zip(seq, seq[1:]):
            if a == b:
                continue
            e = {"clip": key, "player": pid, "from": a, "to": b, "f1": f1, "f2": f2}
            # A between f1 and f2: matched to a detection every frame? on f2: alive, and whose box?
            a_f2 = rows.get(f2, {}).get(a)
            e["a_at_f2"] = None if a_f2 is None else {"state": a_f2["state"], "source": a_f2["source"],
                                                      "identity": a_f2["identity"],
                                                      "player": attr[f2].get(a, (None, False))[0]}
            um = [d for d in by[("unmatched", a)] if f1 < d["frame"] <= f2]
            if a_f2 is not None and a_f2["source"] == "OBSERVED" and not um:
                e["what"] = "A_OBSERVED_ON_ANOTHER_BOX"      # A never missed: it moved to another player's box
                pa = attr[f2].get(a, (None, False))
                e["a_player_at_f2"] = pa[0]
                e["a_ambiguous_at_f2"] = pa[1]
            elif not um:
                end = [d for d in by[("end", a)] if f1 < d["frame"] <= f2]
                e["what"] = "A_ENDED:" + (end[0]["reason"] if end else "?")
            else:
                d = um[0]
                g = d["frame"]
                r = rows.get(g, {}).get(a, {})
                e.update(what="A_UNMATCHED", miss_frame=g, reason=d["reason"].split(":")[0], stage=d["stage"],
                         prior_state=d["prior_state"], gap_s=d["gap_s"], a_identity=r.get("identity"),
                         a_confirmed=d["confirmed"])
                cands = sorted(d["candidates"], key=lambda c: c["dist_h"])
                if cands:
                    c = cands[0]          # the nearest detection: where the player most likely is
                    e["nearest"] = {k: c.get(k) for k in ("detection", "dist_h", "gate_radius_h", "iou", "height_ratio",
                                                          "mahalanobis2", "appearance_similarity", "score",
                                                          "taken_by", "taken_contested", "failed")}
                    e["first_failed_gate"] = first_gate(c)
                    e["nearest_taken_by_rival"] = c.get("taken_by") in unc[a]
                    e["nearest_taken_by_b"] = c.get("taken_by") == b
                e["n_candidates"] = len(cands)
                e["first_reasons"] = [x["reason"].split(":")[0] if not x["reason"].startswith("GATE") else x["reason"]
                                      for x in um[:4]]
                e["recovery_refused"] = [x["reason"] for x in by[("recovery_refused", a)] if g <= x["frame"] <= f2]
            # B's origin and relation to A
            sp = spawn.get(b)
            born = first_row.get(b)
            e["b_born"] = born
            e["b_spawned_after_f1"] = born is not None and born > f1
            e["b_preds"] = sp["possible_predecessors"] if sp else None
            rel = []
            if sp and a in sp["possible_predecessors"]:
                rel.append("B_NAMES_A_PREDECESSOR" + ("_UNIQUE" if len(sp["possible_predecessors"]) == 1 else ""))
            if b in unc[a]:
                rel.append("B_IS_A_RIVAL")
            if e.get("nearest", {}).get("taken_by") == b:
                rel.append("B_TOOK_A_DETECTION")
            e["relation"] = rel or ["NONE"]
            e["b_identity_at_f2"] = rows[f2][b]["identity"]
            out.append(e)
    return out


def main(run_dir, out_path, *specs):
    run = Path(run_dir)
    ch = []
    for s in specs:
        ld, key = s.split(":")
        ch += clip(run, ld, key)
    S = lambda f: dict(Counter(f(e) for e in ch).most_common())  # noqa: E731
    um = [e for e in ch if e["what"] == "A_UNMATCHED"]
    summary = {
        "track_changes": len(ch),
        "what_happened_to_A": S(lambda e: e["what"]),
        "relation_A_B": S(lambda e: "+".join(e["relation"])),
        "unmatched_first_reason": dict(Counter(e["reason"] for e in um).most_common()),
        "unmatched_stage_prior_state": dict(Counter(f'{e["stage"]}/{e["prior_state"]}' for e in um).most_common()),
        "unmatched_first_failed_gate_of_nearest": dict(Counter(str(e.get("first_failed_gate")) for e in um).most_common()),
        "unmatched_by_reason_and_gate": dict(Counter(f'{e["reason"]}|{e.get("first_failed_gate")}' for e in um).most_common()),
        "taken_by_rival": sum(1 for e in um if e["reason"] == "TAKEN_BY_OTHER" and e.get("nearest_taken_by_rival")),
        "taken_by_b": sum(1 for e in um if e["reason"] == "TAKEN_BY_OTHER" and e.get("nearest_taken_by_b")),
        "a_identity_at_miss": dict(Counter(str(e.get("a_identity")) for e in um).most_common()),
        "b_identity_at_f2": S(lambda e: e["b_identity_at_f2"]),
    }
    Path(out_path).write_text(json.dumps({"summary": summary, "changes": ch}, indent=1, default=str))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
