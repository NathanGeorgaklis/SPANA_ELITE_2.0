"""DEV check of identity resolutions (not product code): was each IDENTITY_UNCERTAIN episode the tracker cleared
cleared on the right player?

    python -m tools.dev.resolution_check --out OUT.json --clip LABELS_DIR TRACKS.json [--clip ...] [--window-s 2]

Per clip, observed tracks are attributed to labelled players on labelled frames (NESTED_V2, as tools/benchmark.py).
For every episode of a track that ends CERTAIN again (rows: IDENTITY_UNCERTAIN -> CERTAIN), the player the track
was on in the --window-s before the episode started is compared with the player it is on in the --window-s after
it was cleared (majority of attributed labelled frames; scoring convention, not a tracker setting):
  right    same player     wrong    different players (the tracker cleared a swap)
  unknown  no attribution on one side
Reported by resolved_by (tracks 2.2.0; 2.1.0 = APPEARANCE).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from spana.artifacts import read_json, validate_tracks
from spana.evaluation import NESTED_V2, _match
from tools.benchmark import load_frozen


def owners(gt, tracks_doc) -> Dict[str, Dict[int, str]]:
    """track id -> {labelled frame: player} for unambiguous NESTED_V2 attributions of OBSERVED boxes."""
    trk = {fr["frame"]: fr for fr in tracks_doc["frames"]}
    out: Dict[str, Dict[int, str]] = {}
    for f in gt.frames:
        if f not in trk or not gt.players.get(f):
            continue
        obs = [t for t in trk[f]["tracks"] if t["source"] == "OBSERVED"]
        if not obs:
            continue
        m, _, amb = _match(sorted(gt.players[f]), gt.players[f], [t["box"] for t in obs], NESTED_V2,
                           frozenset(gt.occluded.get(f, ())))
        for c, g in m.items():
            if c not in amb:
                out.setdefault(obs[c]["track_id"], {})[f] = g
    return out


def majority(own: Dict[int, str], lo: float, hi: float, t_of: Dict[int, float]) -> Optional[str]:
    xs = [p for f, p in own.items() if lo <= t_of[f] <= hi]
    if not xs:
        return None
    (p, n), *rest = Counter(xs).most_common()
    return p if not rest or rest[0][1] < n else None


def episodes(tracks_doc) -> List[dict]:
    """Every IDENTITY_UNCERTAIN -> CERTAIN transition: track, episode start / clear times, resolved_by."""
    prev: Dict[str, str] = {}
    start: Dict[str, float] = {}
    out = []
    for fr in tracks_doc["frames"]:
        for t in fr["tracks"]:
            tid, idn = t["track_id"], t["identity"]
            if idn != "CERTAIN" and prev.get(tid) in ("CERTAIN", None):
                start[tid] = fr["t"]
            if idn == "CERTAIN" and prev.get(tid) not in ("CERTAIN", None):
                out.append({"track_id": tid, "start_t": start.get(tid), "clear_t": fr["t"], "clear_frame": fr["frame"],
                            "resolved_by": t.get("resolved_by") or "APPEARANCE"})
            prev[tid] = idn
    return out


def judge(own: Dict[int, str], ep: dict, t_of: Dict[int, float], window_s: float) -> dict:
    before = majority(own, ep["start_t"] - window_s, ep["start_t"], t_of) if ep["start_t"] is not None else None
    after = majority(own, ep["clear_t"], ep["clear_t"] + window_s, t_of)
    verdict = "unknown" if before is None or after is None else ("right" if before == after else "wrong")
    return {**ep, "before": before, "after": after, "verdict": verdict}


def check_clip(label_dir, tracks_path, window_s: float) -> dict:
    fz, gt = load_frozen(Path(label_dir))
    td = read_json(tracks_path)
    validate_tracks(td)
    t_of = {fr["frame"]: fr["t"] for fr in td["frames"]}
    own = owners(gt, td)
    rows = [judge(own.get(ep["track_id"], {}), ep, t_of, window_s) for ep in episodes(td)]
    counts: Dict[str, Counter] = {}
    for r in rows:
        counts.setdefault(r["resolved_by"], Counter())[r["verdict"]] += 1
    return {"clip": fz["clip"]["name"], "resolutions": rows, "counts": {k: dict(v) for k, v in counts.items()},
            "uncertain_observed_track_frames": sum(1 for fr in td["frames"] for t in fr["tracks"]
                                                   if t["source"] == "OBSERVED" and t["identity"] != "CERTAIN")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--clip", nargs=2, action="append", required=True, metavar=("LABELS_DIR", "TRACKS_JSON"))
    p.add_argument("--window-s", type=float, default=2.0, help="scoring convention: seconds before / after")
    a = p.parse_args(argv)
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    res = [check_clip(ld, tp, a.window_s) for ld, tp in a.clip]
    pooled: Dict[str, Counter] = {}
    for r in res:
        for k, v in r["counts"].items():
            pooled.setdefault(k, Counter()).update(v)
    out = {"window_s": a.window_s, "clips": res, "pooled": {k: dict(v) for k, v in pooled.items()},
           "uncertain_observed_track_frames": sum(r["uncertain_observed_track_frames"] for r in res)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"pooled": out["pooled"], "uncertain_observed_track_frames": out["uncertain_observed_track_frames"],
                      "per_clip": {r["clip"][:8]: r["counts"] for r in res}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
