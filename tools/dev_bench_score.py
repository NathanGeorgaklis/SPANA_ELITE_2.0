"""DEV generalisation benchmark: freeze the reviewed labels, then score detector.model_id against them. No tuning.

    python -m tools.dev_bench_score freeze --expect N      # once, after the reviewer has approved the frames
    python -m tools.dev_bench_score score                  # any number of times; reads the frozen labels only

freeze (read-only on Roboflow): every frame of dev_bench.project carrying dev_bench.bench_tag AND
review.reviewed_tag, in the project's Dataset, passing the licence gate as a DEV match. The annotations are written
exactly as saved on the server to <paths.dev_bench_labels_dir>/<bench tag>/labels.json with FROZEN.json (sha256,
counts, the bench frames left out because they are not reviewed). Refused when the folder already exists (labels are
never edited or re-frozen in place), when the count is not --expect, or when any frame cannot be read.

score: the labels file must match FROZEN.json's sha256. Pixels are the sha-verified project images; every frame passes
the licence gate again. Predictions are requested once at detector_eval.request_confidence and cached under
paths.dev_bench_score_dir; everything else uses the same configuration and code as tools/detector_eval.py
(detector.confidence, detector_eval.match_iou, preflight.ball_size_buckets_px), overall and per match:
  * AP50 / AP50-95, precision, recall per class, confusion, ball recall by short side;
  * off-pitch false positives (analysis only, nothing is filtered): predictions at detector.confidence that match no
    label of any class (the confusion's BACKGROUND row), split by the existing grass mask (prelabel.grass) into
    off_pitch (bottom-centre outside the mask), on_pitch, and foot_unobserved (box cut by the frame's bottom border,
    prelabel.grass.frame_edge_margin_frac: the mask says nothing about it).
Record: <paths.detector_record_dir>/dev_bench_eval_<model>_<stamp>.json (no pixels; committed). Annotated frames
(detector_eval.render_frames, worst per detector_eval.worst_classes first) under paths.dev_bench_score_dir.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2

from spana.artifacts import sha256_doc, write_json_atomic
from spana.dataset import det_metrics as dm
from spana.dataset.config import DatasetConfig, PolicyError, load_dataset_config, match_of
from spana.dataset.prefilter import foot_unobserved, grass_mask, on_grass
from spana.dataset.prep import list_all, load_image, match_tag
from spana.dataset.review_flags import boxes_from_annotation
from spana.detection.roboflow_backend import RoboflowDetector
from spana.secrets import load_secret
from tools.detector_eval import draw, evaluate, pick_frames, predict
from tools.dev_bench import open_bench_project

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
LABELS_KIND = "spana.dev_bench_labels"
FROZEN_KIND = "spana.dev_bench_frozen"


def labels_dir(cfg: DatasetConfig, root: Path) -> Path:
    return root / cfg.paths.dev_bench_labels_dir / cfg.dev_bench.bench_tag.replace(":", "_")


# ------------------------------------------------------------------------------------------------------ freeze ----
def bench_frames(cfg: DatasetConfig, images) -> tuple:
    """(reviewed benchmark frames, names of benchmark frames that are not reviewed). Fails closed on a reviewed frame
    outside the Dataset or failing the licence gate as a DEV match."""
    b, rev = cfg.dev_bench.bench_tag, cfg.review.reviewed_tag
    bench = [i for i in images if b in i.tags]
    reviewed = [i for i in bench if rev in i.tags]
    problems = []
    for i in reviewed:
        entry, reason = match_of(cfg, i.tags, i.metadata)
        if reason:
            problems.append(f"{i.name}: {reason}")
        elif entry.split != "dev":
            problems.append(f"{i.name}: match {entry.match_id} is not a DEV match")
        if not i.in_dataset:
            problems.append(f"{i.name}: tagged {rev!r} but not in the project's Dataset")
    if problems:
        raise PolicyError("refused to freeze:\n  - " + "\n  - ".join(problems))
    left_out = sorted(i.name for i in bench if rev not in i.tags)
    return sorted(reviewed, key=lambda i: (match_tag(i) or "", i.name)), left_out


def freeze(cfg: DatasetConfig, project, expect: int, root: Path = REPO, now: Optional[dt.datetime] = None) -> Path:
    out = labels_dir(cfg, root)
    if out.exists():
        raise PolicyError(f"refused: {out} exists; frozen labels are never re-frozen or edited in place")
    images = list_all(cfg, project)
    reviewed, left_out = bench_frames(cfg, images)
    if len(reviewed) != expect:
        raise PolicyError(f"refused: {len(reviewed)} reviewed benchmark frames, --expect {expect}")
    frames = []
    for i in reviewed:
        rec = project.get_image(i.id)
        try:
            boxes = boxes_from_annotation(rec.get("annotation"))
        except ValueError as e:
            raise PolicyError(f"refused: {i.name}: {e}") from None
        unknown = sorted({bx["class"] for bx in boxes} - set(cfg.classes))
        if unknown:
            raise PolicyError(f"refused: {i.name}: classes {unknown} are not configured classes")
        if sorted(rec.get("tags") or []) != sorted(i.tags):
            raise PolicyError(f"refused: {i.name}: tags changed while freezing; run again")
        frames.append({"id": i.id, "name": i.name, "match_id": match_tag(i),
                       "frame_index": i.metadata.get("frame_index"), "image_sha256": i.metadata.get("image_sha256"),
                       "width": i.width, "height": i.height, "boxes": [{"class": bx["class"],
                                                                        "box": [round(v, 3) for v in bx["box"]]}
                                                                       for bx in boxes]})
    labels = {"kind": LABELS_KIND, "workspace": cfg.roboflow.workspace, "project": cfg.dev_bench.project,
              "bench_tag": cfg.dev_bench.bench_tag, "reviewed_tag": cfg.review.reviewed_tag, "frames": frames}
    frozen = {"kind": FROZEN_KIND, "created_utc": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds"),
              "labels_file": "labels.json", "labels_sha256": sha256_doc(labels), "n_frames": len(frames),
              "frames_per_match": dict(sorted(Counter(f["match_id"] for f in frames).items())),
              "boxes_per_class": dict(sorted(Counter(bx["class"] for f in frames for bx in f["boxes"]).items())),
              "bench_frames_not_reviewed": left_out}
    out.mkdir(parents=True)
    write_json_atomic(out / "labels.json", labels)
    write_json_atomic(out / "FROZEN.json", frozen)
    return out


def load_frozen(cfg: DatasetConfig, root: Path = REPO) -> tuple:
    d = labels_dir(cfg, root)
    frozen = json.loads((d / "FROZEN.json").read_text())
    labels = json.loads((d / frozen["labels_file"]).read_text())
    if frozen.get("kind") != FROZEN_KIND or sha256_doc(labels) != frozen["labels_sha256"]:
        raise PolicyError(f"refused: {d / frozen['labels_file']} does not match its FROZEN.json sha256")
    if labels["project"] != cfg.dev_bench.project or labels["bench_tag"] != cfg.dev_bench.bench_tag:
        raise PolicyError("refused: frozen labels belong to another benchmark project / tag")
    return labels, frozen


# ------------------------------------------------------------------------------------------------------- score ----
def off_pitch(cfg: DatasetConfig, frames: List[dict], masks: Dict[str, object], heights: Dict[str, int]) -> dict:
    """{class: {false_positives, off_pitch, on_pitch, foot_unobserved}} for BACKGROUND predictions at
    detector.confidence. masks / heights per frame id."""
    g = cfg.prelabel.grass
    out = {c: {"false_positives": 0, "off_pitch": 0, "on_pitch": 0, "foot_unobserved": 0} for c in cfg.classes}
    for f in dm.at_confidence(frames, cfg.detector.confidence):
        for pi in dm.background_predictions(f, cfg.detector_eval.match_iou):
            p = f["pred"][pi]
            row = out[p["class"]]
            row["false_positives"] += 1
            if foot_unobserved(p["box"], heights[f["id"]], g.frame_edge_margin_frac):
                row["foot_unobserved"] += 1
            elif on_grass(masks[f["id"]], p["box"]):
                row["on_pitch"] += 1
            else:
                row["off_pitch"] += 1
    return out


def score_frames(cfg: DatasetConfig, frames: List[dict], masks, heights) -> dict:
    by_match: Dict[str, List[dict]] = {}
    for f in frames:
        by_match.setdefault(f["match_id"], []).append(f)
    def one(fs):
        return {**evaluate(cfg, fs), "off_pitch_false_positives": off_pitch(cfg, fs, masks, heights)}
    return {"overall": one(frames), "per_match": {m: one(fs) for m, fs in sorted(by_match.items())}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("freeze")
    fr.add_argument("--expect", type=int, required=True, help="number of reviewed frames the reviewer approved")
    sub.add_parser("score")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    key = load_secret(cfg.roboflow.api_key_env)
    try:
        _, project = open_bench_project(cfg, key)
        if a.cmd == "freeze":
            out = freeze(cfg, project, a.expect)
            print(json.dumps(json.loads((out / "FROZEN.json").read_text()), indent=1))
            print(f"frozen: {out}")
            return 0
        labels, frozen = load_frozen(cfg)
        model = cfg.detector.model_id.replace("/", "_")
        out_dir = REPO / cfg.paths.dev_bench_score_dir / model
        request = {"model_id": cfg.detector.model_id, "api_url": cfg.detector.api_url, "iou": cfg.detector.iou,
                   "request_confidence": cfg.detector_eval.request_confidence, "project": cfg.dev_bench.project,
                   "labels_sha256": frozen["labels_sha256"]}
        images = {i.id: i for i in list_all(cfg, project)}
        cache_dir = REPO / cfg.paths.image_cache_dir
        cache_file = out_dir / "predictions.json"
        if cache_file.is_file() and json.loads(cache_file.read_text())["request"] == request:
            frames = json.loads(cache_file.read_text())["frames"]
        else:
            detector = RoboflowDetector.from_detector_config(cfg, key, confidence=cfg.detector_eval.request_confidence)
            frames = predict(cfg, project, detector, labels["frames"], images, cache_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"request": request, "frames": frames,
                                              "dropped_labels": dict(detector.dropped_labels)}))
        meta = {e["id"]: e for e in labels["frames"]}
        for f in frames:
            f["match_id"] = meta[f["id"]]["match_id"]
        pixels = {f["id"]: load_image(project, images[f["id"]], cache_dir) for f in frames}
        masks = {i: grass_mask(px, cfg.prelabel.grass) for i, px in pixels.items()}
        heights = {i: px.shape[0] for i, px in pixels.items()}
        result = score_frames(cfg, frames, masks, heights)
        picks = pick_frames(cfg, frames)
        (out_dir / "frames").mkdir(parents=True, exist_ok=True)
        for k, p in enumerate(picks):
            f = frames[p["index"]]
            p["name"] = f["name"]
            cv2.imwrite(str(out_dir / "frames" / f"{k:02d}_{p['why']}_{Path(f['name']).stem}.jpg"),
                        draw(pixels[f["id"]], f, list(cfg.classes), cfg.detector.confidence))
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    doc = {"kind": "spana.dev_bench_eval", "created_utc": stamp, "request": request,
           "frozen_labels": {"dir": str(labels_dir(cfg, Path("."))), "labels_sha256": frozen["labels_sha256"],
                             "frozen_utc": frozen["created_utc"]},
           "config_sha256": {n: cfg.section_sha256(n) for n in ("detector", "detector_eval", "dev_bench", "prelabel")},
           "result": result, "rendered": picks}
    path = REPO / cfg.paths.detector_record_dir / f"dev_bench_eval_{model}_{stamp}.json"
    write_json_atomic(path, doc)
    print(json.dumps(result, indent=1))
    print(f"record: {path}\nframes: {out_dir / 'frames'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
