"""Choose DEV labelling windows from track logs, by generic event criteria, and build their label packs.

    python -m tools.dev.label_windows select --out WINDOWS.json [--config config/label_windows.yaml] \
        [--tracker-config config/default.yaml] \
        --source NAME VIDEO TRACKS.json DECISIONS.jsonl [--source ...] \
        [--exclude NAME START END ...] [--wrong-from REPORT.json CLIPS_DIR ...]
    python -m tools.dev.label_windows build --windows WINDOWS.json --out-dir PACKS_DIR

select:
  Events per source (confirmed tracks only; config/label_windows.yaml says what each type is): occlusion, exit,
  entry, flagged, similar_rival. Candidate windows of window_s start every stride_s inside the tracked frames;
  score = sum over types of weight * min(count, cap); a window needs min_event_types types and may not overlap an
  --exclude range (footage already labelled) or a chosen window. The best `count` are chosen greedily, at most
  max_per_source per source; ties go to the earlier source / frame (deterministic).
  Dense windows: every wrong TARGET answer in the given target_score reports (events of kind "wrong") gets a window
  from half_s before to half_s after it, every frame, on that clip (CLIPS_DIR/<clip>.mp4).
  Nothing is chosen by hand: the only inputs are track logs, scorer reports and config.
build:
  One label pack per window (tools/label_pack.py range pack, DEV only, never overwrites).
DEV only: sources must be DEV footage; the scorer reports are DEV by construction (tools/target_score.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from spana.artifacts import read_json, validate_tracks, write_json_atomic

SCHEMA = ("spana.label_windows", "1.0.0")
TYPES = ("occlusion", "exit", "entry", "flagged", "similar_rival")


def load_cfg(path) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    if c.get("schema") != "spana.label_windows_config":
        raise SystemExit(f"{path}: not a spana.label_windows_config")
    if set(c["weights"]) != set(TYPES):
        raise SystemExit(f"{path}: weights must name exactly {TYPES}")
    for k in ("window_s", "step_s", "stride_s", "edge_frac"):
        if not c[k] > 0:
            raise SystemExit(f"{path}: {k} must be > 0")
    if not (c["count"] >= 1 and c["max_per_source"] >= 1 and c["cap"] >= 1 and 1 <= c["min_event_types"] <= len(TYPES)):
        raise SystemExit(f"{path}: count / max_per_source / cap / min_event_types out of range")
    if not (c["dense"]["half_s"] > 0 and c["dense"]["step_frames"] >= 1):
        raise SystemExit(f"{path}: dense.half_s > 0 and dense.step_frames >= 1 needed")
    return c


def events(tracks: dict, decisions: Sequence[dict], cfg: dict, min_similarity: float) -> Dict[str, List[int]]:
    """Frames of each event type in one tracks document (sorted, one entry per event)."""
    confirmed = {r["track_id"] for r in tracks["tracklets"] if r["confirmed"]}
    w, h = tracks["video"]["width"], tracks["video"]["height"]
    ex, ey = cfg["edge_frac"] * w, cfg["edge_frac"] * h
    out: Dict[str, List[int]] = {t: [] for t in TYPES}
    prev_state: Dict[str, str] = {}
    first_obs: Dict[str, int] = {}
    for fr in tracks["frames"]:
        for t in fr["tracks"]:
            tid = t["track_id"]
            if tid in confirmed:
                if t["state"] == "OCCLUDED" and prev_state.get(tid) != "OCCLUDED":
                    out["occlusion"].append(fr["frame"])
                if t["source"] == "OBSERVED" and tid not in first_obs:
                    first_obs[tid] = fr["frame"]
                    x1, y1, x2, y2 = t["box"]
                    if x1 <= ex or y1 <= ey or x2 >= w - ex or y2 >= h - ey:
                        out["entry"].append(fr["frame"])
            prev_state[tid] = t["state"]
    for d in decisions:
        if d.get("track_id") not in confirmed:
            continue
        if d["kind"] == "end" and d.get("reason") == "LEFT_FRAME":
            out["exit"].append(d["frame"])
        elif d["kind"] == "identity_uncertain":
            out["flagged"].append(d["frame"])
            rs = d.get("rival_similarity")
            if rs is not None and rs >= min_similarity:
                out["similar_rival"].append(d["frame"])
    return {k: sorted(v) for k, v in out.items()}


def candidates(name: str, ev: Dict[str, List[int]], first: int, last: int, fps: float, cfg: dict,
               excluded: Sequence[Tuple[int, int]]) -> List[dict]:
    """Every candidate window [start, end) of this source with its event counts and score."""
    n = int(round(cfg["window_s"] * fps))
    stride = max(1, int(round(cfg["stride_s"] * fps)))
    out = []
    start = first
    while start + n <= last + 1:
        end = start + n
        if not any(start < b and a < end for a, b in excluded):
            counts = {t: sum(1 for f in ev[t] if start <= f < end) for t in TYPES}
            types = sum(1 for v in counts.values() if v)
            if types >= cfg["min_event_types"]:
                score = sum(cfg["weights"][t] * min(counts[t], cfg["cap"]) for t in TYPES)
                out.append({"source": name, "start_frame": start, "end_frame": end, "score": round(score, 6),
                            "event_types": types, "events": counts})
        start += stride
    return out


def choose(cands: List[dict], source_order: Sequence[str], cfg: dict) -> List[dict]:
    rank = {s: k for k, s in enumerate(source_order)}
    pool = sorted(cands, key=lambda c: (-c["score"], -c["event_types"], rank[c["source"]], c["start_frame"]))
    chosen: List[dict] = []
    for c in pool:
        if len(chosen) == cfg["count"]:
            break
        mine = [x for x in chosen if x["source"] == c["source"]]
        if len(mine) >= cfg["max_per_source"]:
            continue
        if any(c["start_frame"] < x["end_frame"] and x["start_frame"] < c["end_frame"] for x in mine):
            continue
        chosen.append(c)
    return chosen


def dense_windows(report: dict, clips_dir: Path, cfg: dict) -> List[dict]:
    """A window around every wrong TARGET answer of a target_score report (clip frames; DEV labels only)."""
    out = []
    for c in report["clips"]:
        fz = json.loads((Path(c["labels"]["dir"]) / "FROZEN.json").read_text())
        if fz["split"] != "DEV":
            raise SystemExit("refusing: a report on non-DEV labels")
        fps = fz["clip"]["fps"]
        half = int(round(cfg["dense"]["half_s"] * fps))
        rec = json.loads((Path(c["labels"]["dir"]) / "labels_frozen.json").read_text())
        last = max(rec["frames_labelled"]) + int(rec["frame_step"])
        for player, pp in sorted(c["per_player"].items(), key=lambda kv: str(kv[0])):
            for e in pp.get("events", []):
                if e.get("kind") != "wrong":
                    continue
                f = int(e["frame"])
                a, b = max(0, f - half), min(last, f + half + 1)
                out.append({"clip": c["clip"], "video": str(clips_dir / f"{c['clip']}.mp4"),
                            "video_sha256": fz["clip"]["sha256"], "player": str(player),
                            "attributed_to": e.get("attributed_to"), "track_id": e.get("track_id"),
                            "wrong_frame": f, "start_frame": a, "end_frame": b,
                            "step_frames": cfg["dense"]["step_frames"],
                            "name": f"{c['clip']}_dense_f{a}_{b}"})
    seen, uniq = set(), []
    for w in out:                                        # one window per (clip, range) even if two players share it
        k = (w["clip"], w["start_frame"], w["end_frame"])
        if k not in seen:
            seen.add(k)
            uniq.append(w)
    return uniq


def select(a) -> int:
    cfg = load_cfg(a.config)
    tcfg = yaml.safe_load(Path(a.tracker_config).read_text())
    min_sim = float(tcfg["appearance"]["min_similarity"])
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    names = [s[0] for s in a.source]
    if len(set(names)) != len(names):
        raise SystemExit("source names must be unique")
    excl: Dict[str, List[Tuple[int, int]]] = {}
    for n, s, e in a.exclude or []:
        if n not in names:
            raise SystemExit(f"--exclude names unknown source {n}")
        excl.setdefault(n, []).append((int(s), int(e)))
    sources, cands = [], []
    for name, video, tpath, dpath in a.source:
        tdoc = read_json(tpath)
        validate_tracks(tdoc)
        decisions = [json.loads(line) for line in Path(dpath).read_text().splitlines() if line.strip()]
        ev = events(tdoc, decisions, cfg, min_sim)
        first, last, fps = tdoc["frames"][0]["frame"], tdoc["frames"][-1]["frame"], tdoc["video"]["fps"]
        if any(fr["frame"] != first + k for k, fr in enumerate(tdoc["frames"])):
            raise SystemExit(f"{name}: tracks are not every frame (windows need a contiguous run)")
        sources.append({"name": name, "video": str(video), "video_sha256": tdoc["video"]["sha256"], "fps": fps,
                        "first_frame": first, "last_frame": last, "excluded": excl.get(name, []),
                        "tracks_sha256": hashlib.sha256(Path(tpath).read_bytes()).hexdigest(),
                        "event_totals": {t: len(v) for t, v in ev.items()}})
        for c in candidates(name, ev, first, last, fps, cfg, excl.get(name, [])):
            c["events_at"] = {t: [f for f in ev[t] if c["start_frame"] <= f < c["end_frame"]] for t in TYPES}
            cands.append(c)
    chosen = choose(cands, names, cfg)
    by_name = {s["name"]: s for s in sources}
    for c in chosen:
        s = by_name[c["source"]]
        c.update({"video": s["video"], "video_sha256": s["video_sha256"],
                  "step_frames": max(1, int(round(cfg["step_s"] * s["fps"]))),
                  "name": f"{c['source']}_f{c['start_frame']}_{c['end_frame']}"})
    dense = []
    for rpath, cdir in a.wrong_from or []:
        dense += dense_windows(read_json(rpath), Path(cdir), cfg)
    raw = Path(a.config).read_bytes()
    doc = {"schema": SCHEMA[0], "schema_version": SCHEMA[1], "config": cfg,
           "config_sha256": hashlib.sha256(raw).hexdigest(), "appearance_min_similarity": min_sim,
           "sources": sources, "candidates_scored": len(cands), "windows": chosen, "dense": dense}
    write_json_atomic(a.out, doc)
    for w in chosen:
        print(json.dumps({k: w[k] for k in ("name", "score", "events")}))
    for w in dense:
        print(json.dumps({k: w[k] for k in ("name", "player", "attributed_to", "wrong_frame")}))
    return 0


def build(a) -> int:
    from tools.label_pack import make_pack
    doc = read_json(a.windows)
    if (doc.get("schema"), doc.get("schema_version")) != SCHEMA:
        raise SystemExit("not a spana.label_windows document")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for w in doc["windows"] + doc["dense"]:
        m = make_pack(w["video"], "DEV", None, str(out / w["name"]), name=w["name"], start_frame=w["start_frame"],
                      end_frame=w["end_frame"], step_frames=w["step_frames"])
        if m["clip"]["sha256"] != w["video_sha256"]:
            raise SystemExit(f"{w['name']}: video sha256 differs from the tracks / labels it was chosen on")
        print(f"pack {w['name']}: {len(m['frames'])} frames")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s.add_argument("--config", default="config/label_windows.yaml")
    s.add_argument("--tracker-config", default="config/default.yaml")
    s.add_argument("--source", nargs=4, action="append", required=True,
                   metavar=("NAME", "VIDEO", "TRACKS", "DECISIONS"))
    s.add_argument("--exclude", nargs=3, action="append", metavar=("NAME", "START", "END"))
    s.add_argument("--wrong-from", nargs=2, action="append", metavar=("REPORT", "CLIPS_DIR"))
    b = sub.add_parser("build")
    b.add_argument("--windows", required=True)
    b.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    return select(a) if a.cmd == "select" else build(a)


if __name__ == "__main__":
    sys.exit(main())
