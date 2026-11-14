"""Score the trained detector (detector.model_id) against the frozen labels of one Roboflow split. No tuning.

    python -m tools.detector_eval --labels records/roboflow_version/labels_<stamp>.json --version N

* Labels: the snapshot written by tools/roboflow_preflight.py for dataset version N; its sha256 must equal the one
  in that version's manifest, so the model is scored against exactly the labels frozen before it was trained.
* Frames: detector_eval.split only (never "test"); each frame passes the licence gate again; a hosted api_url is
  third-party processing and needs it confirmed for the match. Pixels are the sha-verified project images.
* Predictions are requested once at detector_eval.request_confidence and cached under paths.detector_render_dir;
  precision, recall, confusion and worst frames use detector.confidence and detector_eval.match_iou as configured.

Outputs:
  <paths.detector_record_dir>/eval_<model>_<stamp>.json   metrics (no pixels; committed)
  <paths.detector_render_dir>/<model>/predictions.json    cached raw predictions
  <paths.detector_render_dir>/<model>/frames/*.jpg        detector_eval.render_frames annotated frames: the worst
                                                          per detector_eval.worst_classes, the rest evenly spaced
Drawing: labels thin white, predictions in the class colour with the score.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import cv2

from spana.artifacts import sha256_doc, write_json_atomic
from spana.dataset import det_metrics as dm
from spana.dataset.config import DatasetConfig, PolicyError, load_dataset_config, match_of
from spana.dataset.prep import evenly_spaced, list_all, load_image, open_project
from spana.detection.roboflow_backend import RoboflowDetector
from spana.secrets import load_secret
from tools.roboflow_preflight import bucket_names, size_bucket

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
_PALETTE = [(60, 200, 60), (0, 165, 255), (230, 60, 230), (0, 255, 255), (255, 128, 0), (128, 128, 255)]


def load_frozen_labels(cfg: DatasetConfig, labels_path: Path, version: int) -> List[dict]:
    """Label entries of detector_eval.split, after checking the file is the one frozen for dataset version N."""
    manifest = json.loads((REPO / cfg.paths.version_dir / f"version_{version}.json").read_text())
    doc = json.loads(Path(labels_path).read_text())
    if version != cfg.detector.dataset_version:
        raise PolicyError(f"refused: version {version} is not detector.dataset_version {cfg.detector.dataset_version}")
    if sha256_doc(doc) != manifest["labels_sha256"]:
        raise PolicyError(f"refused: {Path(labels_path).name} is not the label snapshot frozen for version {version}")
    return [e for e in doc["frames"] if e["split"] == cfg.detector_eval.split]


def pick_frames(cfg: DatasetConfig, frames: List[dict]) -> List[dict]:
    """[{index, why}]: the worst frames per configured class, then evenly spaced others, render_frames in all."""
    e = cfg.detector_eval
    op = dm.at_confidence(frames, cfg.detector.confidence)
    chosen, why = [], {}
    for cls in e.worst_classes:
        ranked = sorted(range(len(op)), key=lambda i: (-dm.frame_errors(op[i], cls, e.match_iou), i))
        picked = [i for i in ranked if i not in why and dm.frame_errors(op[i], cls, e.match_iou) > 0]
        for i in picked[:e.worst_per_class]:
            why[i] = f"worst-{cls}-{dm.frame_errors(op[i], cls, e.match_iou)}-errors"
            chosen.append(i)
    rest = [i for i in range(len(op)) if i not in why]
    for i in evenly_spaced(rest, max(0, e.render_frames - len(chosen))):
        why[i] = "sample"
        chosen.append(i)
    return [{"index": i, "why": why[i]} for i in chosen]


def draw(frame, entry: dict, classes: Sequence[str], confidence: float):
    img = frame.copy()
    for g in entry["gt"]:
        x1, y1, x2, y2 = (int(round(v)) for v in g["box"])
        cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)
    for p in entry["pred"]:
        if p["score"] < confidence:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in p["box"])
        col = _PALETTE[classes.index(p["class"]) % len(_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{p['class'][:3]} {p['score']:.2f}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    col, 1, cv2.LINE_AA)
    return img


def predict(cfg: DatasetConfig, project, detector, labels: List[dict], images: dict, cache: Path) -> List[dict]:
    """Per label entry: {"name", "id", "gt", "pred"}. Fails closed on any frame that cannot be scored."""
    out = []
    for e in labels:
        img = images.get(e["id"])
        if img is None or img.name != e["name"]:
            raise PolicyError(f"refused: {e['name']} is no longer in the project under the frozen id")
        _, reason = match_of(cfg, img.tags, img.metadata)
        if reason:
            raise PolicyError(f"refused: {e['name']}: {reason}")
        frame = load_image(project, img, cache)
        dets = detector.detect(frame)
        out.append({"name": e["name"], "id": e["id"], "gt": e["boxes"],
                    "pred": [{"class": detector.class_names[d.class_id], "box": [round(v, 2) for v in d.box],
                              "score": round(d.score, 4)} for d in dets]})
        if cfg.detector_eval.min_interval_s:
            time.sleep(cfg.detector_eval.min_interval_s)
    return out


def evaluate(cfg: DatasetConfig, frames: List[dict]) -> dict:
    e, edges = cfg.detector_eval, cfg.preflight.ball_size_buckets_px
    op = dm.at_confidence(frames, cfg.detector.confidence)
    by_size = {}
    for t in e.size_recall_ious:
        got = dm.recall_by_size(op, e.size_recall_class, t, lambda s: size_bucket(s, edges))
        by_size[f"iou_{t:g}"] = {n: got[n] for n in bucket_names(edges) if n in got}
    return {**dm.summarize(frames, cfg.classes, cfg.detector.confidence, e.match_iou),
            "frames": len(frames), "confidence": cfg.detector.confidence, "match_iou": e.match_iou,
            "size_recall_class": e.size_recall_class, "recall_by_short_side_px": by_size}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--labels", required=True)
    ap.add_argument("--version", type=int, required=True)
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    key = load_secret(cfg.roboflow.api_key_env)
    try:
        labels = load_frozen_labels(cfg, Path(a.labels), a.version)
        detector = RoboflowDetector.from_detector_config(cfg, key, confidence=cfg.detector_eval.request_confidence)
        out_dir = REPO / cfg.paths.detector_render_dir / cfg.detector.model_id.replace("/", "_")
        request = {"model_id": cfg.detector.model_id, "api_url": cfg.detector.api_url, "iou": cfg.detector.iou,
                   "request_confidence": cfg.detector_eval.request_confidence, "split": cfg.detector_eval.split,
                   "labels_sha256": sha256_doc(json.loads(Path(a.labels).read_text()))}
        cache_file = out_dir / "predictions.json"
        project = open_project(cfg, key)
        images = {i.id: i for i in list_all(cfg, project)}
        if cache_file.is_file() and json.loads(cache_file.read_text())["request"] == request:
            frames = json.loads(cache_file.read_text())["frames"]
        else:
            frames = predict(cfg, project, detector, labels, images, REPO / cfg.paths.image_cache_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"request": request, "frames": frames,
                                              "dropped_labels": dict(detector.dropped_labels)}))
        result = evaluate(cfg, frames)
        picks = pick_frames(cfg, frames)
        (out_dir / "frames").mkdir(parents=True, exist_ok=True)
        for k, p in enumerate(picks):
            f = frames[p["index"]]
            frame = load_image(project, images[f["id"]], REPO / cfg.paths.image_cache_dir)
            p["name"] = f["name"]
            cv2.imwrite(str(out_dir / "frames" / f"{k:02d}_{p['why']}_{Path(f['name']).stem}.jpg"),
                        draw(frame, f, list(cfg.classes), cfg.detector.confidence))
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    doc = {"kind": "spana.detector_eval", "created_utc": stamp, "request": request, "dataset_version": a.version,
           "config_sha256": {n: cfg.section_sha256(n) for n in ("detector", "detector_eval")},
           "result": result, "rendered": picks}
    path = REPO / cfg.paths.detector_record_dir / f"eval_{cfg.detector.model_id.replace('/', '_')}_{stamp}.json"
    write_json_atomic(path, doc)
    print(json.dumps(result, indent=1))
    print(f"record: {path}\nframes: {out_dir / 'frames'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
