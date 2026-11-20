"""DEV report of IDENTITY_UNCERTAIN episodes and their re-ID resolutions / refusals (not product code).

    python -m tools.dev.reid_episode_report --out OUT.json --clip LABELS_DIR TRACKS.json [--clip ...] [--window-s 2]

Episodes: per track, each run of rows that are not CERTAIN; it ends RESOLVED (the next row is CERTAIN; resolved_by
REID / APPEARANCE), TRACK_ENDED (the track's rows stop while uncertain) or CLIP_END.
Judged against the frozen labels as tools/dev/resolution_check.py (NESTED_V2 attribution of OBSERVED boxes, majority
of labelled frames in --window-s, a scoring convention): right / wrong / unknown = the player before the episode vs
after it. An episode that starts at the track's spawn (ambiguous spawn) has no "before" on its own track; when re-ID
resolved it, the predecessor it picked (decision log) is judged instead: right if the picked predecessor carried, in
--window-s before the spawn, the player the track carries after the resolution.
Unresolved episodes get one refusal reason from the tracker's `reid_evidence` log rows inside the episode, the
first that applies of: RUN_TOO_SHORT (re-ID cleared every rival on some frame but the run never reached
identity_resolve_s / min_evidence_frames), AGAINST (re-ID or clear colour evidence against), BELOW_MARGIN (same-look
rivals, no clear margin), NO_REFERENCE (own / rival / predecessor re-ID reference missing), AT_CONTEST,
APPEARANCE_DOUBT, NO_EMBEDDING, NO_CLEAN_CROP (only contested / overlapping crops), NOT_OBSERVED (no matched frame).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from spana.artifacts import read_json, validate_tracks
from tools.benchmark import load_frozen
from tools.dev.resolution_check import majority, owners

REASONS = ("RUN_TOO_SHORT", "AGAINST", "BELOW_MARGIN", "NO_REFERENCE", "AT_CONTEST", "APPEARANCE_DOUBT",
           "NO_EMBEDDING", "NO_CLEAN_CROP", "NOT_OBSERVED")


def episodes(tracks_doc) -> List[dict]:
    rows: Dict[str, List[tuple]] = defaultdict(list)
    for fr in tracks_doc["frames"]:
        for t in fr["tracks"]:
            rows[t["track_id"]].append((fr["frame"], fr["t"], t))
    last_frame = tracks_doc["frames"][-1]["frame"]
    out = []
    for tid, rs in rows.items():
        ep = None
        for k, (f, t, r) in enumerate(rs):
            if r["identity"] != "CERTAIN" and ep is None:
                ep = {"track_id": tid, "start_frame": f, "start_t": t, "at_spawn": k == 0,
                      "rivals": sorted(r["identity_rivals"])}
            elif r["identity"] != "CERTAIN":
                ep["rivals"] = sorted(set(ep["rivals"]) | set(r["identity_rivals"]))
            elif ep is not None:
                out.append({**ep, "end_frame": f, "end_t": t, "outcome": "RESOLVED",
                            "resolved_by": r.get("resolved_by") or "APPEARANCE"})
                ep = None
        if ep is not None:
            f, t, _ = rs[-1]
            out.append({**ep, "end_frame": f, "end_t": t, "resolved_by": None,
                        "outcome": "CLIP_END" if f == last_frame else "TRACK_ENDED"})
    return sorted(out, key=lambda e: (e["start_frame"], e["track_id"]))


def refusal(rows: List[dict]) -> str:
    st = Counter(r["status"] for r in rows)
    refused = Counter(v.get("refused") for r in rows for v in r["rivals"].values() if v.get("refused"))
    verdict = Counter(v.get("verdict") for r in rows for v in r["rivals"].values() if v.get("verdict"))
    if any(r["cleared"] for r in rows):
        return "RUN_TOO_SHORT"
    if verdict.get("AGAINST") or verdict.get("COLOUR_CONTRARY") or verdict.get("PICK_CHANGED"):
        return "AGAINST"
    if refused.get("BELOW_MARGIN"):
        return "BELOW_MARGIN"
    if st.get("NO_OWN_REFERENCE") or refused.get("NO_RIVAL_REFERENCE") or refused.get("NO_PREDECESSOR_REFERENCE"):
        return "NO_REFERENCE"
    if refused.get("AT_CONTEST"):
        return "AT_CONTEST"
    if st.get("APPEARANCE_DOUBT"):
        return "APPEARANCE_DOUBT"
    if st.get("NO_EMBEDDING"):
        return "NO_EMBEDDING"
    if st.get("CONTESTED") or st.get("NOT_CLEAN"):
        return "NO_CLEAN_CROP"
    return "NOT_OBSERVED"


def report_clip(label_dir, tracks_path, window_s: float) -> dict:
    _, gt = load_frozen(Path(label_dir))
    td = read_json(tracks_path)
    validate_tracks(td)
    t_of = {fr["frame"]: fr["t"] for fr in td["frames"]}
    own = owners(gt, td)
    ev: Dict[str, List[dict]] = defaultdict(list)
    res: Dict[tuple, dict] = {}
    dec = Path(tracks_path).with_suffix(".decisions.jsonl")
    for line in dec.read_text().splitlines():
        d = json.loads(line)
        if d["kind"] == "reid_evidence":
            ev[d["track_id"]].append(d)
        elif d["kind"] == "identity_resolved":
            res[(d["track_id"], d["frame"])] = d
    out = []
    for ep in episodes(td):
        o = own.get(ep["track_id"], {})
        after = majority(o, ep["end_t"], ep["end_t"] + window_s, t_of) if ep["outcome"] == "RESOLVED" else None
        before = None if ep["at_spawn"] else majority(o, ep["start_t"] - window_s, ep["start_t"], t_of)
        row = {**ep, "before": before, "after": after}
        rows = [r for r in ev[ep["track_id"]] if ep["start_frame"] <= r["frame"] < ep["end_frame"] or
                (ep["outcome"] != "RESOLVED" and r["frame"] == ep["end_frame"])]
        row["evidence_frames"] = dict(Counter(r["status"] for r in rows))
        margins = [v["margin"] for r in rows for v in r["rivals"].values() if v.get("margin") is not None]
        row["max_margin"] = max(margins) if margins else None
        if ep["outcome"] == "RESOLVED":
            d = res.get((ep["track_id"], ep["end_frame"] - 0)) or next(
                (v for (tid, f), v in res.items() if tid == ep["track_id"] and ep["start_frame"] <= f <= ep["end_frame"]),
                None)
            row["decision"] = d
            if d is not None and d.get("predecessor"):
                po = own.get(d["predecessor"], {})
                row["predecessor_player"] = majority(po, ep["start_t"] - window_s, ep["start_t"], t_of)
                b = row["predecessor_player"]
            else:
                b = before
            row["verdict"] = "unknown" if b is None or after is None else ("right" if b == after else "wrong")
        else:
            row["refusal"] = refusal(rows)
        out.append(row)
    return {"clip": Path(label_dir).parent.name, "episodes": out}


def summarise(clips: List[dict]) -> dict:
    s = {"episodes": 0, "resolved": Counter(), "verdicts": defaultdict(Counter), "refused": Counter(),
         "refusal_reasons": Counter()}
    for c in clips:
        for e in c["episodes"]:
            s["episodes"] += 1
            if e["outcome"] == "RESOLVED":
                s["resolved"][e["resolved_by"]] += 1
                s["verdicts"][e["resolved_by"]][e["verdict"]] += 1
            else:
                s["refused"][e["outcome"]] += 1
                s["refusal_reasons"][e["refusal"]] += 1
    return {k: (dict(v) if isinstance(v, Counter) else ({kk: dict(vv) for kk, vv in v.items()} if isinstance(v, dict)
                                                       else v)) for k, v in s.items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--clip", nargs=2, action="append", required=True, metavar=("LABELS_DIR", "TRACKS_JSON"))
    p.add_argument("--window-s", type=float, default=2.0, help="scoring convention: seconds before / after")
    a = p.parse_args(argv)
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    clips = [report_clip(ld, tp, a.window_s) for ld, tp in a.clip]
    out = {"window_s": a.window_s, "pooled": summarise(clips),
           "per_clip": {c["clip"][:8]: summarise([c]) for c in clips}, "clips": clips}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"pooled": out["pooled"], "per_clip": out["per_clip"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
