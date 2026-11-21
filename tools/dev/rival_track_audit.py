"""DEV diagnostic (not product code): who are the "rivals" in the tracker's RIVAL_TRACK recovery refusals?
(records/rival_track_log_2026-11-19; Nathan's FRESH_08 review, player 10.)

    python -m tools.dev.rival_track_audit --out OUT.json \
        --clip LABELS_DIR TRACKS.json DECISIONS.jsonl [--clip ...]

S4 recovery refuses a lost track X its only unexplained detection when another unmatched track also gates it
(RIVAL_TRACK); the detection then usually spawns a new track. Owners as tools/dev/reid_reentry_eval.py: the player a
track is attributed to (NESTED_V2, unambiguous) on most of its labelled frames. Per refusal:
  rival class   same      every rival has X's owner (the tracker's own duplicate of the player)
                different some rival is another labelled player
                unknown   X or a rival has no owner
  fragment      a track is spawned on that frame on one of X's in-gate detections and names X as a possible
                predecessor; its owner is reported (same as X = the player was split)
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from spana.artifacts import read_json, validate_tracks
from tools.benchmark import load_frozen
from tools.target_score import attributions


def owners(gt, tdoc):
    attr = attributions(gt, tdoc)
    votes = collections.defaultdict(collections.Counter)
    for f in gt.frames:
        for tid, (p, amb) in attr[f].items():
            if p is not None and not amb:
                votes[tid][p] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def audit_clip(label_dir: str, trk_p: str, dec_p: str) -> dict:
    _, gt = load_frozen(Path(label_dir))
    tdoc = read_json(trk_p)
    validate_tracks(tdoc)
    own = owners(gt, tdoc)
    decs = [json.loads(l) for l in Path(dec_p).read_text().splitlines() if l.strip()]
    spawns = collections.defaultdict(list)
    for d in decs:
        if d["kind"] == "spawn":
            spawns[d["frame"]].append(d)
    rows = []
    for d in decs:
        if d["kind"] != "recovery_refused" or d.get("reason") != "RIVAL_TRACK":
            continue
        x, rivals = d["track_id"], d.get("rival_tracks")
        if rivals is None:
            raise SystemExit("decisions log has no rival_tracks: re-track with code >= 33584c5")
        ox, orv = own.get(x), [own.get(r) for r in rivals]
        cls = "unknown" if ox is None or any(o is None for o in orv) else \
            "same" if all(o == ox for o in orv) else "different"
        frag = [s for s in spawns[d["frame"]] if s["detection"] in d["detections_in_gate"]
                and x in s.get("possible_predecessors", [])]
        rows.append({"frame": d["frame"], "track": x, "owner": ox, "rivals": rivals, "rival_owners": orv,
                     "class": cls, "gap_s": d["gap_s"],
                     "spawned": [{"track": s["track_id"], "owner": own.get(s["track_id"])} for s in frag]})
    return {"refusals": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", nargs=3, action="append", required=True, metavar=("LABELS", "TRACKS", "DECISIONS"))
    a = ap.parse_args()
    per = {Path(c[0]).parent.name: audit_clip(*c) for c in a.clip}
    rows = [dict(r, clip=k) for k, v in per.items() for r in v["refusals"]]
    cls = collections.Counter(r["class"] for r in rows)
    split = [r for r in rows if r["spawned"]]
    summary = {"rival_track_refusals": len(rows), "by_class": dict(cls),
               "followed_by_a_spawn_naming_the_track": len(split),
               "spawn_is_the_same_player": sum(1 for r in split if r["owner"] is not None and
                                               any(s["owner"] == r["owner"] for s in r["spawned"])),
               "by_class_with_spawn": dict(collections.Counter(r["class"] for r in split)),
               "same_class_spawn_same_player": sum(1 for r in split if r["class"] == "same" and
                                                   any(s["owner"] == r["owner"] for s in r["spawned"])),
               "per_clip": {k: dict(collections.Counter(r["class"] for r in v["refusals"])) for k, v in per.items()}}
    Path(a.out).write_text(json.dumps({"summary": summary, "refusals": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
