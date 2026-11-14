"""Read-only preflight before a dataset version is generated. Nothing is written to Roboflow.

    python -m tools.roboflow_preflight [--config config/roboflow.yaml]

Reads the project listing and the saved annotation of every frame that would enter the version (in the Dataset, not
tagged excluded), then checks, failing closed (exit 3, every reason listed) if any does not hold:
  * the version gate of tools/roboflow_version.py (reviewed, licence allow-list, no EVAL, no match in two splits,
    nothing in "test", train and valid not empty);
  * every such frame has a saved annotation, its boxes use only the configured classes and lie inside the frame;
  * each split holds exactly the matches in preflight.split_matches, and each frame's split:<name> tag is its split;
  * the preflight.ball_quantile ball short side, scaled to the configured version size, is at least
    preflight.ball_min_version_px.
Reports per-split, per-class box counts and the ball short-side distribution.

Outputs (paths.version_dir; no pixels):
  labels_<stamp>.json     the annotations exactly as read: the frozen ground truth a model is later scored against
  preflight_<stamp>.json  checks, counts, ball statistics, sha256 of the labels file
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from spana.artifacts import sha256_doc, write_json_atomic
from spana.dataset.config import DatasetConfig, PolicyError, load_dataset_config
from spana.dataset.prep import list_all, match_tag, open_project, sort_key
from spana.dataset.review_flags import boxes_from_annotation
from spana.dataset.rf_project import ProjectImage
from spana.secrets import load_secret
from tools.roboflow_version import check_ready, version_images

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
EDGE_TOL_PX = 1.0     # saved boxes are centre/size numbers; a box on the border can round just outside it
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def version_scale(cfg: DatasetConfig, width: int, height: int) -> float:
    """Source pixel -> version pixel factor of an aspect-preserving resize (all formats except "Stretch to")."""
    v = cfg.version
    if v.resize_format == "Stretch to":
        raise PolicyError("version.resize_format 'Stretch to' does not keep the aspect ratio")
    return min(v.resize_width / width, v.resize_height / height)


def size_bucket(short_side: float, edges: Sequence[float]) -> str:
    lo = 0.0
    for e in edges:
        if short_side < e:
            return f"{lo:g}-{e:g}"
        lo = e
    return f">={lo:g}"


def bucket_names(edges: Sequence[float]) -> List[str]:
    return [size_bucket(v, edges) for v in (0.0, *edges)]


def read_labels(cfg: DatasetConfig, project, images: List[ProjectImage]) -> List[dict]:
    """One entry per version frame with the annotation as saved on the server (boxes in source pixels)."""
    out = []
    for img in sorted(version_images(cfg, images), key=sort_key):
        rec = project.get_image(img.id)
        ann = rec.get("annotation")
        entry = {"id": img.id, "name": img.name, "match_id": match_tag(img), "split": img.split,
                 "record_split": rec.get("split"),
                 "split_tags": sorted(t[len("split:"):] for t in img.tags if t.startswith("split:")), "width": img.width, "height": img.height,
                 "annotated": ann is not None, "boxes": None, "error": None}
        try:
            entry["boxes"] = boxes_from_annotation(ann)
        except ValueError as e:
            entry["error"] = str(e)
        out.append(entry)
    return out


def check_labels(cfg: DatasetConfig, entries: List[dict]) -> dict:
    """Raises PolicyError listing every failed check; returns the statistics otherwise."""
    problems = []
    pf, ball = cfg.preflight, cfg.prelabel.ball_class

    def some(names):
        return f"{len(names)} (e.g. {names[:3]})"

    unlabelled = [e["name"] for e in entries if not e["annotated"] or e["boxes"] is None]
    if unlabelled:
        problems.append(f"frames without a readable saved annotation: {some(unlabelled)}")
    no_size = [e["name"] for e in entries if not e["width"] or not e["height"]]
    if no_size:
        problems.append(f"frames without a recorded size: {some(no_size)}")
    moved = [e["name"] for e in entries if e["record_split"] != e["split"]]
    if moved:
        problems.append(f"frames whose split differs between the listing and the image record: {some(moved)}")
    expected = pf.expected()

    def home(e):       # the split a frame's tag must name: its own, or for a platform test frame its match's split
        return expected.get(e["match_id"]) if e["split"] == "test" else e["split"]
    mistagged = [e["name"] for e in entries if e["split_tags"] != [home(e)]]
    if mistagged:
        problems.append(f"frames whose split:<name> tag is not exactly their Roboflow split (platform test frames: "
                        f"their match's split): {some(mistagged)}")
    counts: Dict[str, Counter] = defaultdict(Counter)
    frames: Counter = Counter()
    empty: Counter = Counter()
    bad_class, bad_box = [], []
    balls = defaultdict(list)                       # split -> (short side source px, short side version px)
    for e in entries:
        frames[e["split"]] += 1
        if not e["boxes"]:
            empty[e["split"]] += 1
            continue
        for b in e["boxes"]:
            x1, y1, x2, y2 = b["box"]
            if b["class"] not in cfg.classes:
                bad_class.append(f"{e['name']}: {b['class']!r}")
                continue
            if e["width"] and e["height"] and not (-EDGE_TOL_PX <= x1 < x2 <= e["width"] + EDGE_TOL_PX
                                                       and -EDGE_TOL_PX <= y1 < y2 <= e["height"] + EDGE_TOL_PX):
                bad_box.append(f"{e['name']}: {b['class']} {[round(v, 1) for v in b['box']]}")
                continue
            counts[e["split"]][b["class"]] += 1
            if b["class"] == ball and e["width"] and e["height"]:
                s = min(x2 - x1, y2 - y1)
                balls[e["split"]].append((s, s * version_scale(cfg, e["width"], e["height"])))
    if bad_class:
        problems.append(f"boxes with a class outside {list(cfg.classes)}: {some(bad_class)}")
    if bad_box:
        problems.append(f"empty boxes or boxes outside the frame: {some(bad_box)}")
    seen = defaultdict(set)
    for e in entries:
        if e["split"] != "test":               # platform test frames: counted and limited by the version gate
            seen[e["split"]].add(e["match_id"])
    for split in sorted(set(seen) | set(expected.values())):
        want = {m for m, s in expected.items() if s == split}
        if seen.get(split, set()) != want:
            problems.append(f"split {split!r} holds {sorted(map(str, seen.get(split, set())))}, "
                            f"preflight.split_matches expects {sorted(want)}")
    for cls in cfg.classes:
        for split in sorted(set(expected.values())):
            if frames[split] and not counts[split][cls]:
                problems.append(f"no {cls!r} box in the {split} split")

    def ball_stats(rows):
        if not rows:
            return None
        src, ver = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        return {"boxes": len(rows),
                "short_side_px": {f"q{int(q * 100):02d}": round(float(np.quantile(src, q)), 2) for q in QUANTILES},
                "short_side_version_px": {f"q{int(q * 100):02d}": round(float(np.quantile(ver, q)), 2)
                                          for q in QUANTILES},
                "share_below_small_px": round(float((src < pf.ball_small_px).mean()), 4),
                "buckets_px": {n: int(sum(size_bucket(v, pf.ball_size_buckets_px) == n for v in src))
                               for n in bucket_names(pf.ball_size_buckets_px)},
                "at_version_quantile_px": round(float(np.quantile(ver, pf.ball_quantile)), 2)}

    every = [r for rows in balls.values() for r in rows]
    stats = {"all": ball_stats(every), **{s: ball_stats(r) for s, r in sorted(balls.items())}}
    if every and stats["all"]["at_version_quantile_px"] < pf.ball_min_version_px:
        problems.append(f"ball short side at quantile {pf.ball_quantile} is {stats['all']['at_version_quantile_px']} px "
                        f"at the version size {cfg.version.resize_width}x{cfg.version.resize_height}, below "
                        f"preflight.ball_min_version_px {pf.ball_min_version_px}")
    if problems:
        raise PolicyError("preflight failed:\n  - " + "\n  - ".join(problems))
    return {"frames": dict(sorted(frames.items())), "frames_without_boxes": dict(sorted(empty.items())),
            "boxes": {s: {c: counts[s][c] for c in cfg.classes} for s in sorted(counts)},
            "ball_small_px": pf.ball_small_px, "ball": stats}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    project = open_project(cfg, load_secret(cfg.roboflow.api_key_env))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / cfg.paths.version_dir
    images = list_all(cfg, project)
    doc = {"kind": "spana.roboflow_preflight", "created_utc": stamp, "project": cfg.roboflow.project,
           "config_sha256": {n: cfg.section_sha256(n) for n in ("version", "preflight")},
           "project_images": len(images), "passed": False, "problems": None}
    try:
        doc["gate"] = check_ready(cfg, images)
        entries = read_labels(cfg, project, images)
        labels = {"kind": "spana.roboflow_labels", "created_utc": stamp, "project": cfg.roboflow.project,
                  "classes": list(cfg.classes), "frames": entries}
        labels_path = out / f"labels_{stamp}.json"
        write_json_atomic(labels_path, labels)
        doc["labels_file"] = labels_path.name
        doc["labels_sha256"] = sha256_doc(labels)
        doc["stats"] = check_labels(cfg, entries)
        doc["passed"] = True
    except PolicyError as e:
        doc["problems"] = str(e)
    write_json_atomic(out / f"preflight_{stamp}.json", doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "problems"}, indent=1, sort_keys=True))
    print(f"record: {out / f'preflight_{stamp}.json'}")
    if not doc["passed"]:
        print(f"ERROR: {doc['problems']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
