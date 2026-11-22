"""QA table for possession / touches v1 over the scripted scenes (tests/touches_synth.py; no footage).

    python -m tools.touches_scenarios [--config config/default.yaml]

Per scene: frames, frames as scripted, edge frames accepted, other mismatches, WRONG (CONTROL on a player other than
the one on the ball, or CONTROL when nobody is: the only safety failure), CONTROL frames scripted / found, and the
possessions found vs scripted (player, first, last).
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from spana.artifacts import sha256_doc, validate_touches
from spana.config import load_config
from spana.possession.touches import touches_document
from tests.touches_synth import SCENES, Scene, ball_doc, tracks_doc


def produced_matches(truth: tuple, fr: dict, pid: Optional[str]) -> bool:
    kind = truth[0]
    if kind == "CONTROL":
        return fr["state"] == "CONTROL" and pid == truth[1]
    if kind == "AIR":                                     # in the air: FREE away from players, UNKNOWN near one
        return fr["state"] in ("FREE", "UNKNOWN")
    if kind == "UNSURE":
        return fr["state"] == "UNKNOWN"
    return fr["state"] == kind                              # FREE / CONTESTED


def score_scene(s: Scene, cfg) -> dict:
    tdoc, bdoc = tracks_doc(s), ball_doc(s)
    doc, _ = touches_document(tdoc, bdoc, cfg.possession, {"note": "touches_scenarios"}, sha256_doc(tdoc),
                              sha256_doc(bdoc))
    validate_touches(doc, tdoc, bdoc)
    row = {"scene": s.name, "frames": len(doc["frames"]), "as_scripted": 0, "edge_ok": 0, "mismatch": 0,
           "wrong": 0, "control_scripted": 0, "control_found": 0, "mismatch_frames": [], "wrong_frames": []}
    for fr in doc["frames"]:
        f, truth = fr["frame"], s.truth(fr["frame"])
        pid = s.pid_of(fr["track_id"]) if fr["track_id"] else None
        row["control_scripted"] += truth[0] == "CONTROL"
        if fr["state"] == "CONTROL" and not (truth[0] == "CONTROL" and pid == truth[1]):
            row["wrong"] += 1
            row["wrong_frames"].append(f)
        elif produced_matches(truth, fr, pid):
            row["as_scripted"] += 1
            row["control_found"] += fr["state"] == "CONTROL"
        elif f in s.edge:
            row["edge_ok"] += 1
        else:
            row["mismatch"] += 1
            row["mismatch_frames"].append((f, fr["state"], fr["reason"]))
    found = [(s.pid_of(e["track_id"]), e["first_frame"], e["last_frame"]) for e in doc["possessions"]]
    row["possessions_found"] = found
    row["possessions_scripted"] = s.possessions
    row["possessions_ok"] = None if s.possessions is None else found == s.possessions
    row["doc"] = doc
    return row


def table(rows: List[dict]) -> str:
    head = f"{'scene':<20}{'frames':>7}{'scripted':>9}{'edge':>6}{'mism':>6}{'WRONG':>7}{'ctrl s/f':>10}  possessions"
    out = [head]
    for r in rows:
        p = "-" if r["possessions_ok"] is None else "ok" if r["possessions_ok"] else \
            f"FOUND {r['possessions_found']} != {r['possessions_scripted']}"
        out.append(f"{r['scene']:<20}{r['frames']:>7}{r['as_scripted']:>9}{r['edge_ok']:>6}{r['mismatch']:>6}"
                   f"{r['wrong']:>7}{r['control_scripted']:>5}/{r['control_found']:<4}  {p} {r['possessions_found']}")
    tot = {k: sum(r[k] for r in rows) for k in ("frames", "as_scripted", "edge_ok", "mismatch", "wrong")}
    out.append(f"TOTAL {tot}, possessions ok "
               f"{sum(bool(r['possessions_ok']) for r in rows)}/{sum(r['possessions_ok'] is not None for r in rows)}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    rows = [score_scene(make(), cfg) for make in SCENES]
    print(table(rows))
    for r in rows:
        if r["mismatch_frames"] or r["wrong_frames"]:
            print(r["scene"], "mismatch", r["mismatch_frames"], "wrong", r["wrong_frames"])
    return 0 if all(r["wrong"] == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
