"""Tracker bench v1: same frozen DEV detections, every candidate, same scorers.

    # 1. once per DEV clip: freeze the detector's output (+ re-ID v2 embeddings for C / D)
    python -m bench.run cache --clip FRESH_02 --detections DET.json --video CLIP.mp4 --embed
    # 2. run every candidate on every cached clip (videos only for trackers that need pixels)
    python -m bench.run track --video FRESH_02=CLIP.mp4 [--video ...]
    # 3. score against the frozen DEV labels -> bench/results/tracker_bench_v1.csv
    python -m bench.run score

Caches and runs live under runs/tracker_bench_v1/ (ignored by Git); nothing is ever overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from spana.tracking.protocol import FrameInput, TrackOutput, check_outputs
from tools.benchmark import load_frozen

from .adapters import make_tracker, refine
from .cache import DetectionCache, build_cache
from .config import REPO, Bench, Candidate, load_bench
from .metrics import (TargetStats, target_stats, trackeval_combine, trackeval_data, trackeval_row,
                      trackeval_sequence)

RUN_SCHEMA = ("spana.tracker_bench_run", "1.0.0")
WORK = REPO / "runs" / "tracker_bench_v1"
CSV_COLUMNS = ["candidate", "tracker", "base", "clip", "licence", "commercial_ok", "hota", "deta", "assa", "idf1",
               "idsw", "mota", "target_players", "target_untapped", "target_idsw", "target_fragments",
               "target_coverage", "target_longest_seg_s_median", "target_wrong_frames",
               "target_wrong_frames_unflagged", "frames", "fps"]


def run_online(tracker, cache: DetectionCache, video_path: Optional[str], needs_image: bool) -> tuple:
    """[(frame, outputs)] and the seconds spent inside tracker.update only (decoding / embedding excluded)."""
    images = None
    if needs_image:
        from spana.video import VideoSource
        if video_path is None:
            raise SystemExit(f"{tracker.name} needs the clip's video (--video NAME=PATH)")
        images = ((i, img) for i, _, img in VideoSource(video_path, expected_sha256=cache.video_sha256)
                  .frames(cache.frames[0].index, cache.frames[-1].index + 1))
    run, spent = [], 0.0
    for cf in cache.frames:
        img = None
        if images is not None:
            for i, img in images:
                if i == cf.index:
                    break
            else:
                raise SystemExit("video ended before the last cached frame")
        frame = FrameInput(cf.index, cf.t, img, cf.embeddings)
        t0 = time.perf_counter()
        outs = tracker.update(cf.detections, frame)
        spent += time.perf_counter() - t0
        check_outputs(outs, cf.detections)
        run.append((cf.index, outs))
    return run, spent


def save_run(path: Path, cand: str, clip: str, cache: DetectionCache, run, seconds: float, base: Optional[str]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": RUN_SCHEMA[0], "schema_version": RUN_SCHEMA[1], "candidate": cand, "base": base, "clip": clip,
           "cache_fingerprint": cache.fingerprint(), "seconds": round(seconds, 4), "frames": len(run),
           "run": [[f, [[o.track_id, o.detection_index, o.identity_certain] for o in outs]] for f, outs in run]}
    path.write_text(json.dumps(doc, separators=(",", ":")))


def load_run(path: Path, cache: DetectionCache) -> tuple:
    doc = json.loads(path.read_text())
    if (doc["schema"], doc["schema_version"]) != RUN_SCHEMA:
        raise SystemExit(f"{path}: not a {RUN_SCHEMA[0]} {RUN_SCHEMA[1]}")
    if doc["cache_fingerprint"] != cache.fingerprint():
        raise SystemExit(f"{path}: made from different cached detections")
    by = {cf.index: cf for cf in cache.frames}
    run = [(f, [TrackOutput(t, k, tuple(by[f].detections[k].box), c) for t, k, c in outs]) for f, outs in doc["run"]]
    return run, doc


def _run_name(cand: Candidate, base: Optional[str]) -> str:
    return f"{cand.name}+{base}" if base else cand.name


def cmd_cache(bench: Bench, a) -> None:
    clip = next((c for c in bench.clips if c.name == a.clip), None)
    if clip is None:
        raise SystemExit(f"{a.clip} is not a bench clip")
    frozen, _ = load_frozen(clip.labels)                     # refuses non-DEV labels
    embedder = None
    if a.embed:
        from spana.identity.reid import ReidEmbedder, load_pinned
        embedder = ReidEmbedder(load_pinned(bench.embedder["model_record"], bench.embedder["model_sha256"]))
    out = build_cache(Path(a.cache_root) / clip.name, clip.name, a.detections, frozen["clip"]["sha256"], a.video,
                      embedder)
    c = DetectionCache.load(out)
    print(json.dumps({"cache": str(out), "frames": len(c.frames), "fingerprint": c.fingerprint(),
                      "embedder": c.manifest["embedder"]}))


def cmd_track(bench: Bench, a) -> None:
    videos = dict(v.split("=", 1) for v in a.video)
    only = set(a.only) if a.only else None
    for clip in bench.clips:
        cache = DetectionCache.load(Path(a.cache_root) / clip.name)
        done: Dict[str, tuple] = {}
        for cand in bench.candidates:
            if cand.kind != "online" or (only and cand.name not in only):
                continue
            if cand.needs_embeddings and not cache.has_embeddings:
                raise SystemExit(f"{clip.name}: {cand.name} needs embeddings; rebuild the cache with --embed")
            run, secs = run_online(make_tracker(cand, cache, bench.external_dir), cache, videos.get(clip.name),
                                   cand.needs_image)
            save_run(Path(a.out_root) / cand.name / f"{clip.name}.json", cand.name, clip.name, cache, run, secs, None)
            done[cand.name] = (run, secs)
            print(f"{clip.name} {cand.name}: {len(run) / max(secs, 1e-9):.1f} fps")
        for cand in bench.candidates:
            if cand.kind != "offline" or (only and cand.name not in only):
                continue
            for base in cand.applies_to:
                if base not in done:
                    p = Path(a.out_root) / base / f"{clip.name}.json"
                    if not p.exists():
                        continue
                    run, doc = load_run(p, cache)
                    done[base] = (run, doc["seconds"])
                t0 = time.perf_counter()
                run = refine(cand, done[base][0], cache, bench.external_dir)
                secs = done[base][1] + time.perf_counter() - t0    # online + offline time
                for (f, outs), cf in zip(run, cache.frames):
                    check_outputs(outs, cf.detections)
                save_run(Path(a.out_root) / _run_name(cand, base) / f"{clip.name}.json", cand.name, clip.name, cache,
                         run, secs, base)
                print(f"{clip.name} {_run_name(cand, base)}: {len(run) / max(secs, 1e-9):.1f} fps")


def score_rows(bench: Bench, cache_root: Path, out_root: Path) -> List[dict]:
    rows = []
    units = [(c, None) for c in bench.candidates if c.kind == "online"] + \
            [(c, b) for c in bench.candidates if c.kind == "offline" for b in c.applies_to]
    for cand, base in units:
        name = _run_name(cand, base)
        seqs, pooled, frames, secs = [], TargetStats(), 0, 0.0
        lic = cand.licence["spdx"]
        ok = cand.commercial_ok
        if base:
            b = next(c for c in bench.candidates if c.name == base)
            lic, ok = f"{lic}+{b.licence['spdx']}", ok and b.commercial_ok
        for clip in bench.clips:
            path = out_root / name / f"{clip.name}.json"
            if not path.exists():
                raise SystemExit(f"missing run {path}")
            cache = DetectionCache.load(cache_root / clip.name)
            frozen, gt = load_frozen(clip.labels)
            run, doc = load_run(path, cache)
            te = trackeval_sequence(trackeval_data(gt, run))
            ts = target_stats(gt, run, frozen["frame_step"], cache.fps, bench.scoring["target_match_rule"])
            seqs.append(te)
            pooled.add(ts)
            frames += doc["frames"]
            secs += doc["seconds"]
            rows.append({"candidate": cand.candidate, "tracker": name, "base": base, "clip": clip.name, "licence": lic,
                         "commercial_ok": ok, **trackeval_row(te), **ts.row(), "frames": doc["frames"],
                         "fps": round(doc["frames"] / max(doc["seconds"], 1e-9), 1)})
        rows.append({"candidate": cand.candidate, "tracker": name, "base": base, "clip": "ALL", "licence": lic,
                     "commercial_ok": ok, **trackeval_row(trackeval_combine(seqs)), **pooled.row(), "frames": frames,
                     "fps": round(frames / max(secs, 1e-9), 1)})
    return rows


def rank(rows: List[dict], ranking) -> List[dict]:
    pooled = [r for r in rows if r["clip"] == "ALL"]
    def key(r):
        return tuple(float("inf") if r[k] is None else (r[k] if o == "asc" else -r[k]) for k, o in ranking)
    return sorted(pooled, key=key)


def cmd_score(bench: Bench, a) -> None:
    rows = score_rows(bench, Path(a.cache_root), Path(a.out_root))
    out = Path(a.csv) if a.csv else bench.output_csv
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    cols = ["tracker", "hota", "idf1", "idsw", "target_idsw", "target_fragments", "target_coverage",
            "target_longest_seg_s_median", "target_wrong_frames", "target_wrong_frames_unflagged", "fps", "licence"]
    print(" | ".join(cols))
    for r in rank(rows, bench.ranking):
        print(" | ".join(str(r[c]) for c in cols))
    best = next((r for r in rank(rows, bench.ranking) if r["commercial_ok"]), None)
    print(f"recommended by the pre-registered rule (commercial-compatible only): {best['tracker'] if best else None}")
    print(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=None)
    p.add_argument("--cache-root", default=str(WORK / "cache"))
    p.add_argument("--out-root", default=str(WORK / "out"))
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--clip", required=True)
    c.add_argument("--detections", required=True)
    c.add_argument("--video", default=None)
    c.add_argument("--embed", action="store_true")
    t = sub.add_parser("track")
    t.add_argument("--video", action="append", default=[], help="CLIP=PATH, for candidates that need pixels")
    t.add_argument("--only", nargs="*", default=None)
    s = sub.add_parser("score")
    s.add_argument("--csv", default=None)
    a = p.parse_args(argv)
    bench = load_bench(a.bench) if a.bench else load_bench()
    {"cache": cmd_cache, "track": cmd_track, "score": cmd_score}[a.cmd](bench, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
