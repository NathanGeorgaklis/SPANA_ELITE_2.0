"""DEV comparison (not product code): Roboflow supervision ByteTrack on OUR detections, written as a spana.tracks
artifact so the same scorers (tools/benchmark.py, tools/target_score.py) judge both trackers on identical input.

    PYTHONPATH=<dir with supervision==0.25.1> python -m tools.dev.bytetrack_compare run \
        --detections DET.json --ref-tracks OUR.trk.json --out BT.trk.json [--config config/bytetrack_compare.yaml]
    python -m tools.dev.bytetrack_compare life --tracks A.trk.json [B.trk.json ...]

ByteTrack is fed every person-view detection of the artifact (class-agnostic, as our tracker uses them) with the
supervision 0.25.1 defaults from config. Conversion (ByteTrack has no notion of the extra states we report):
- a ByteTrack output = an ACTIVE, OBSERVED row on the exact detection row (index kept through supervision's data);
- frames between two outputs of a track = LOST (box held), as ByteTrack keeps it in its lost pool;
- after a track's last output it stays LOST for ByteTrack's max_time_lost = int(fps / 30 * lost_track_buffer)
  frames, then ends (LOST_EXPIRED), exactly when ByteTrack removes it;
- confirmed from its second output frame (ByteTrack outputs only activated tracks; our artifact needs a fresh start);
- identity is always CERTAIN: ByteTrack never flags an uncertain identity (no rivals, no occluder, no predecessors).
Camera status / affine / time are copied from our tracks artifact of the same detections; a camera cut is refused
(ByteTrack would carry identities across it).
Also writes OUT.decisions.jsonl (empty: ByteTrack makes no recovery decisions) and OUT.provenance.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from spana.artifacts import read_json, sha256_doc, validate_detections, validate_tracks

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "bytetrack_compare.yaml"


def load_cfg(path=DEFAULT_CONFIG) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    if c.get("schema") != "spana.bytetrack_compare_config":
        raise SystemExit(f"{path}: not a spana.bytetrack_compare_config")
    return c


def max_time_lost(fps: float, lost_track_buffer: int) -> int:
    return int(fps / 30.0 * lost_track_buffer)                     # supervision ByteTrack.__init__


def run_bytetrack(dets_doc: dict, cfg: dict) -> Dict[int, List[Tuple[int, int]]]:
    """{frame: [(bytetrack id, detection index)]} from supervision's ByteTrack."""
    import supervision as sv
    if sv.__version__ != cfg["supervision_version"]:
        raise SystemExit(f"supervision {sv.__version__} found, config requires {cfg['supervision_version']}")
    bc = cfg["bytetrack"]
    fps = float(dets_doc["video"]["fps"])
    bt = sv.ByteTrack(track_activation_threshold=bc["track_activation_threshold"],
                      lost_track_buffer=bc["lost_track_buffer"],
                      minimum_matching_threshold=bc["minimum_matching_threshold"],
                      frame_rate=int(round(fps)), minimum_consecutive_frames=bc["minimum_consecutive_frames"])
    out = {}
    for fr in dets_doc["frames"]:
        rows = fr["detections"]
        n = len(rows)
        det = sv.Detections(xyxy=np.array([r[:4] for r in rows], float).reshape(n, 4),
                            confidence=np.array([r[4] for r in rows], float),
                            class_id=np.zeros(n, int) if cfg["class_agnostic"] else np.array([r[5] for r in rows], int),
                            data={"idx": np.arange(n)})
        res = bt.update_with_detections(det)
        out[fr["frame"]] = [(int(t), int(i)) for t, i in zip(res.tracker_id, res.data["idx"])] if len(res) else []
    return out


def to_tracks(dets_doc: dict, ref: dict, assoc: Dict[int, List[Tuple[int, int]]], max_lost: int,
              provenance: dict, removed_at: Optional[Dict[int, int]] = None) -> dict:
    """removed_at (optional, used by tools/dev/botsort_compare.py): {external id: frame on which the external tracker
    no longer holds the track}, observed from the tracker itself; ids missing from it are still held at the clip end.
    Without it, removal follows ByteTrack's fixed max_lost after the last output."""
    fps = float(dets_doc["video"]["fps"])
    frames = [fr["frame"] for fr in dets_doc["frames"]]
    rows_of = {fr["frame"]: fr["detections"] for fr in dets_doc["frames"]}
    refs = {fr["frame"]: fr for fr in ref["frames"]}
    if [fr["frame"] for fr in ref["frames"]] != frames:
        raise SystemExit("reference tracks do not cover the same frames as the detections")
    if any(fr["camera"]["status"] in ("CUT", "UNKNOWN") for fr in ref["frames"]):
        raise SystemExit("camera cut / unknown motion in this clip: not supported by the comparison")
    pos = {f: k for k, f in enumerate(frames)}
    name: Dict[int, str] = {}
    seen: Dict[str, Dict[int, int]] = {}
    for f in frames:
        for ext, di in sorted(assoc.get(f, []), key=lambda x: x[1]):
            tid = name.setdefault(ext, f"T{len(name) + 1:06d}")
            seen.setdefault(tid, {})[f] = di
    ext_of = {tid: ext for ext, tid in name.items()}
    span = {}
    for tid, s in seen.items():
        first, last = min(s, key=pos.get), max(s, key=pos.get)
        if removed_at is None:
            end_pos = pos[last] + max_lost + 1                     # removed from ByteTrack's lost pool here
        elif ext_of[tid] in removed_at:
            end_pos = pos[removed_at[ext_of[tid]]]
            if end_pos <= pos[last]:
                raise SystemExit(f"track {ext_of[tid]} removed on frame {removed_at[ext_of[tid]]} before its last output")
        else:
            end_pos = len(frames)                                  # still held at the clip end
        span[tid] = (pos[first], pos[last], end_pos)
    out_frames, state = [], {}
    for k, f in enumerate(frames):
        rf = refs[f]
        ended = sorted([tid, "LOST_EXPIRED"] for tid, (a, b, e) in span.items() if e == k)
        rows = []
        for tid in sorted(t for t, (a, b, e) in span.items() if a <= k < e):
            st = state.setdefault(tid, {"age": -1, "obs": 0, "miss": 0, "box": None, "score": None})
            st["age"] += 1
            di = seen[tid].get(f)
            observed = di is not None
            if observed:
                r = rows_of[f][di]
                st["box"], st["score"] = list(r[:4]), r[4]
                st["obs"] += 1
                st["miss"] = 0
            else:
                st["miss"] += 1
            rows.append({
                "track_id": tid, "state": "ACTIVE" if observed else "LOST",
                "confirmed": st["age"] > 0, "source": "OBSERVED" if observed else "HELD", "box": st["box"],
                "detection_index": di if observed else None, "detection_score": st["score"] if observed else None,
                "confidence": float(st["score"]), "contested": False, "age_frames": st["age"],
                "missed_frames": st["miss"], "time_since_observed_s": 0 if observed else round(st["miss"] / fps, 6),
                "observed_frames": st["obs"], "epoch": 0, "occluder": None, "max_overlap_iou": 0.0,
                "overlap_with": None, "recoveries": 0, "last_gap_s": 0, "possible_predecessors": [],
                "identity": "CERTAIN", "identity_rivals": [], "uncertain_episodes": 0})
        out_frames.append({"frame": f, "t": rf["t"], "camera": rf["camera"], "tracks": rows, "ended": ended})
    tracklets = []
    for tid, (a, b, e) in span.items():
        tracklets.append({"track_id": tid, "epoch": 0, "first_frame": frames[a],
                          "last_frame": frames[min(e, len(frames)) - 1], "last_observed_frame": frames[b],
                          "observed_frames": len(seen[tid]), "confirmed": b > a,
                          "end_reason": "LOST_EXPIRED" if e < len(frames) else None, "recoveries": 0,
                          "possible_predecessors": []})
    doc = {"schema": "spana.tracks", "schema_version": "2.1.0", "provenance": provenance, "video": ref["video"],
           "detections_sha256": sha256_doc(dets_doc), "frames": out_frames, "tracklets": tracklets}
    validate_tracks(doc, dets_doc)
    return doc


def life(tracks_doc: dict) -> dict:
    """Confirmed track life (first frame to last observation, seconds) and how tracks end."""
    fps = float(tracks_doc["video"]["fps"])
    pos = {fr["frame"]: k for k, fr in enumerate(tracks_doc["frames"])}
    ls = [(pos[t["last_observed_frame"]] - pos[t["first_frame"]]) / fps for t in tracks_doc["tracklets"] if t["confirmed"]]
    ends: Dict[str, int] = {}
    for t in tracks_doc["tracklets"]:
        if t["confirmed"]:
            ends[str(t["end_reason"])] = ends.get(str(t["end_reason"]), 0) + 1
    q = lambda p: round(float(np.percentile(ls, p)), 2) if ls else None   # noqa: E731
    return {"confirmed_tracks": len(ls), "life_s_median": round(statistics.median(ls), 2) if ls else None,
            "life_s_p25": q(25), "life_s_p75": q(75), "life_s_max": q(100), "ends": dict(sorted(ends.items())),
            "uncertain_track_frames": sum(1 for fr in tracks_doc["frames"] for t in fr["tracks"]
                                          if t["source"] == "OBSERVED" and t["identity"] != "CERTAIN")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--detections", required=True)
    r.add_argument("--ref-tracks", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config", default=str(DEFAULT_CONFIG))
    r.add_argument("--git", default=None, help="commit of this tool, recorded in provenance (tools/benchmark.py reads it)")
    lf = sub.add_parser("life")
    lf.add_argument("--tracks", nargs="+", required=True)
    a = p.parse_args(argv)
    if a.cmd == "life":
        for t in a.tracks:
            print(json.dumps({"tracks": t, **life(read_json(t))}))
        return 0
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    cfg = load_cfg(a.config)
    dets, ref = read_json(a.detections), read_json(a.ref_tracks)
    validate_detections(dets)
    validate_tracks(ref, dets)
    fps = float(dets["video"]["fps"])
    ml = max_time_lost(fps, cfg["bytetrack"]["lost_track_buffer"])
    prov = {"tool": "tools.dev.bytetrack_compare", "git": a.git, "tracker": f"supervision ByteTrack {cfg['supervision_version']}",
            "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), "bytetrack": cfg["bytetrack"],
            "frame_rate": int(round(fps)), "max_time_lost_frames": ml,
            "reference_tracks_sha256": hashlib.sha256(Path(a.ref_tracks).read_bytes()).hexdigest()}
    doc = to_tracks(dets, ref, run_bytetrack(dets, cfg), ml, prov)
    out.write_text(json.dumps(doc, separators=(",", ":")))
    Path(str(out).replace(".trk.json", "") + ".trk.decisions.jsonl").write_text("")
    print(json.dumps({"out": str(out), **life(doc)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
