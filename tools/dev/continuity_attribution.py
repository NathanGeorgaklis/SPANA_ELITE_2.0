"""DEV diagnostic (not product code): why our tracks broke where ByteTrack kept one track on the same player.

    python -m tools.dev.continuity_attribution --config config/default.yaml --out OUT.json \
        --clip NAME LABEL_DIR DETECTIONS.json OURS.trk.json OURS.trk.decisions.jsonl BYTETRACK.trk.json [--clip ...]

Attribution uses the benchmark's own rule (NESTED_V2, ignore regions, ambiguous attributions skipped), so a "break"
here is exactly one of the benchmark's ID-switch events: on consecutive attributed labelled frames f0 < f1 the
player's track changes A -> B. It is a FRAGMENT event when B is new to that player (what `fragments` counts).

A break is in scope when ByteTrack kept ONE continuous track on the same player across it: ByteTrack's attribution
for that player is the same track X at its last attributed frame <= f0 and at its first attributed frame >= f1.

Cause (first divergence): walk the frames f0+1 .. f1 and, on every frame where X observed a detection d, look at what
our tracker did with d. The first frame where A does not own d decides:
  DUPLICATE_SPAWN_M2   A is observed on another detection d' and one of d / d' lies inside the other
                       (IoA >= tracking.confirm_nested_ioa): a nested part/full-body pair carried two tracks;
  LOW_SCORE_UNMATCHED  d is a low-score detection (< detection.score_high) that A did not take (S2 refused it: IoU
                       below low_score_min_iou, or A unconfirmed, or A not observed on the previous frame);
  IDENTITY_GATE        A was refused by an appearance / identity rule: appearance veto, hidden-rival refusal,
                       recovery refused for RIVAL_TRACK / CONTESTED_NEIGHBOUR / AMBIGUOUS_DETECTIONS / APPEARANCE_*;
  LOST_TIMEOUT         A was unobserved and its re-association window had closed (NOT_ELIGIBLE_GAP: gap beyond
                       rematch_max_s / occlusion_max_s) or A had already ended (LOST_EXPIRED, UNCONFIRMED_MISSED);
  OTHER:<detail>       anything else (motion gate, taken by another track, camera cut, left frame, ...).
If A owned X's detection on every such frame, the attribution switched to a parallel track B on the same player:
  DUPLICATE_SPAWN_M2 when A's and B's boxes are nested at f1, else OTHER:PARALLEL_TRACK.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from spana.artifacts import read_json
from spana.config import load_config
from spana.evaluation import NESTED_V2, _match
from tools.benchmark import IGNORE_IOA, _ioa, build_frames, load_frozen

CAUSES = ("LOST_TIMEOUT", "IDENTITY_GATE", "LOW_SCORE_UNMATCHED", "DUPLICATE_SPAWN_M2", "OTHER")
IDENTITY_RECOVERY_REASONS = {"RIVAL_TRACK", "CONTESTED_NEIGHBOUR", "AMBIGUOUS_DETECTIONS", "APPEARANCE_MISMATCH",
                             "APPEARANCE_UNKNOWN", "APPEARANCE_NOT_DISTINCT"}


def attributions(gt, dets_doc, trk_doc, decisions) -> Dict[str, List[Tuple[int, str]]]:
    """player -> [(labelled frame, track id)], exactly as spana.evaluation.evaluate attributes (NESTED_V2)."""
    seq: Dict[str, List[Tuple[int, str]]] = collections.defaultdict(list)
    for ef in build_frames(gt, dets_doc, trk_doc, decisions):
        gids = sorted(ef.gt)
        tr_gt, _, amb = _match(gids, ef.gt, [t.box for t in ef.tracks], NESTED_V2, ef.occluded)
        for c, g in tr_gt.items():
            if c not in amb:
                seq[g].append((ef.frame, ef.tracks[c].track_id))
    return seq


def nested(a, b, ioa_min: float) -> bool:
    return _ioa(a, b) >= ioa_min or _ioa(b, a) >= ioa_min


def classify(a_id: str, b_id: str, f0: int, f1: int, x_id: str, ours: dict, bt: dict, dets: dict,
             dec: dict, ends: dict, cfg) -> Tuple[str, dict]:
    high = cfg.detection.score_high
    nest = cfg.tracking.confirm_nested_ioa
    for f in range(f0 + 1, f1 + 1):
        x = bt[f].get(x_id)
        if x is None or x["source"] != "OBSERVED" or x["detection_index"] is None:
            continue
        d = x["detection_index"]
        a = ours[f].get(a_id)
        owner = next((t for t in ours[f].values() if t["detection_index"] == d and t["source"] == "OBSERVED"), None)
        if a is not None and a["source"] == "OBSERVED" and a["detection_index"] == d:
            continue
        row = dets[f][d]
        info = {"frame": f, "detection": d, "score": row[4], "owner": owner["track_id"] if owner else None,
                "a_state": None if a is None else a["state"], "a_confirmed": None if a is None else a["confirmed"]}
        if a is not None and a["source"] == "OBSERVED":
            other = dets[f][a["detection_index"]]
            info["a_detection"] = a["detection_index"]
            if nested(row[:4], other[:4], nest):
                return "DUPLICATE_SPAWN_M2", info
            return "OTHER:A_ON_OTHER_DETECTION", info
        if a is None:                           # A had already ended before X's next detection
            end = ends.get(a_id)
            info["a_end"] = end
            if end and end[1] in ("LOST_EXPIRED", "UNCONFIRMED_MISSED"):
                return "LOST_TIMEOUT", info
            return f"OTHER:ENDED_{end[1] if end else 'UNKNOWN'}", info
        u = dec[f].get(("unmatched", a_id))
        reason = u["reason"] if u else "NO_UNMATCHED_LOG"
        info["reason"] = reason
        info["gap_s"] = None if u is None else u["gap_s"]
        cand = next((c for c in (u or {}).get("candidates", []) if c["detection"] == d), None)
        info["candidate"] = None if cand is None else {k: cand[k] for k in (
            "dist_h", "gate_radius_h", "mahalanobis2", "iou", "appearance_similarity", "failed", "taken_by")}
        if reason.startswith("REFUSED:match_refused") or (cand and "appearance_veto" in cand["failed"]):
            return "IDENTITY_GATE", info
        if reason.startswith("REFUSED:recovery_refused"):
            r = reason.split(":")[-1]
            if r in IDENTITY_RECOVERY_REASONS:
                return "IDENTITY_GATE", info
            if r == "LOW_SCORE":
                return "LOW_SCORE_UNMATCHED", info
            return f"OTHER:RECOVERY_{r}", info
        if reason == "NOT_ELIGIBLE_GAP":
            return "LOST_TIMEOUT", info
        if row[4] < high:
            return "LOW_SCORE_UNMATCHED", info
        if reason.startswith("GATE_FAILED"):
            return "OTHER:" + reason, info
        return "OTHER:" + reason, info
    # A kept X's detection throughout: the attribution moved to a parallel track B on the same player.
    a, b = ours[f1].get(a_id), ours[f1].get(b_id)
    info = {"frame": f1}
    if a and b and nested(a["box"], b["box"], nest):
        return "DUPLICATE_SPAWN_M2", info
    return "OTHER:PARALLEL_TRACK", info


def analyse_clip(name: str, label_dir: str, det_p: str, trk_p: str, dec_p: str, bt_p: str, cfg) -> dict:
    _, gt = load_frozen(Path(label_dir))
    dets_doc, ours_doc, bt_doc = read_json(det_p), read_json(trk_p), read_json(bt_p)
    decisions = [json.loads(l) for l in Path(dec_p).read_text().splitlines() if l.strip()]
    ours_seq = attributions(gt, dets_doc, ours_doc, decisions)
    bt_seq = attributions(gt, dets_doc, bt_doc, [])
    dets = {fr["frame"]: fr["detections"] for fr in dets_doc["frames"]}
    ours = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in ours_doc["frames"]}
    bt = {fr["frame"]: {t["track_id"]: t for t in fr["tracks"]} for fr in bt_doc["frames"]}
    dec: Dict[int, dict] = collections.defaultdict(dict)
    ends: Dict[str, Tuple[int, str]] = {}
    for d in decisions:
        if d["kind"] == "unmatched":
            dec[d["frame"]][("unmatched", d["track_id"])] = d
        elif d["kind"] == "end":
            ends[d["track_id"]] = (d["frame"], d["reason"])
    rows = []
    for g, seq in sorted(ours_seq.items()):
        seen = {seq[0][1]} if seq else set()
        bts = bt_seq.get(g, [])
        for (f0, a), (f1, b) in zip(seq, seq[1:]):
            if a == b:
                continue
            fragment = b not in seen
            seen.add(b)
            before = [t for f, t in bts if f <= f0]
            after = [t for f, t in bts if f >= f1]
            kept = bool(before and after and before[-1] == after[0])
            row = {"clip": name, "player": g, "f0": f0, "f1": f1, "from": a, "to": b, "fragment": fragment,
                   "bytetrack_continuous": kept}
            if kept:
                cause, info = classify(a, b, f0, f1, before[-1], ours, bt, dets, dec, ends, cfg)
                row.update({"bytetrack_track": before[-1], "cause": cause, "evidence": info})
            rows.append(row)
    return {"clip": name, "rows": rows}


def summarise(rows: List[dict]) -> dict:
    def count(rs):
        c = collections.Counter(r["cause"].split(":")[0] for r in rs)
        return {k: c.get(k, 0) for k in CAUSES}
    kept = [r for r in rows if r["bytetrack_continuous"]]
    frag = [r for r in kept if r["fragment"]]
    detail = collections.Counter(r["cause"] for r in frag)
    return {"breaks": len(rows), "fragments": sum(r["fragment"] for r in rows),
            "bytetrack_continuous_breaks": len(kept), "bytetrack_continuous_fragments": len(frag),
            "causes_fragments": count(frag), "causes_breaks": count(kept),
            "detail_fragments": dict(detail.most_common())}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clip", nargs=6, action="append", required=True,
                   metavar=("NAME", "LABELS", "DETECTIONS", "TRACKS", "DECISIONS", "BYTETRACK"))
    a = p.parse_args()
    cfg = load_config(a.config)
    clips = [analyse_clip(*c, cfg=cfg) for c in a.clip]
    rows = [r for c in clips for r in c["rows"]]
    out = {"note": "DEV diagnostic; NESTED_V2 attribution; ignore IoA %.2f" % IGNORE_IOA,
           "per_clip": {c["clip"]: summarise(c["rows"]) for c in clips}, "pooled": summarise(rows), "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"per_clip": out["per_clip"], "pooled": out["pooled"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
