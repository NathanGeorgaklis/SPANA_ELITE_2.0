"""Step 4: pre-label project frames with a pretrained general detector, as annotations for HUMAN REVIEW.

    python -m tools.roboflow_prelabel --limit 20 [--match MATCH_ID]          # sample (<= prelabel.sample_size)
    python -m tools.roboflow_prelabel --all --confirm-full-run              # only after Nathan's go-ahead

Model, model->class map and per-class thresholds come from config (prelabel section). Only classes in
prelabel.class_map are auto-labelled; the rest (e.g. goalkeeper, referee) come from human review. A detection is
kept when its score >= its class threshold. Per frame, fail closed and logged:
  * licence gate (allow-list, EVAL, tag/metadata consistency) -> skipped with the reason
  * excluded (near-duplicate), already pre-labelled, already annotated, already reviewed -> skipped
  * pixels must match the sha256 recorded at upload, decoded size must match the recorded size -> else skipped
Every processed frame first gets prelabel.tag (also frames with no box above threshold: a human still has to look
at them), then its annotation is saved without overwriting (an existing annotation is never replaced).
Selection is deterministic: frames sorted by (match, frame index), then evenly spaced.

Writes paths.prelabel_dir/prelabel_<stamp>.json (per-frame log, no pixels) and annotated renders to
paths.render_dir/<stamp>/ for visual review; prints a summary.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from spana.artifacts import write_json_atomic
from spana.dataset.config import DatasetConfig, PolicyError, load_dataset_config, match_of
from spana.dataset.prep import SkipFrame, evenly_spaced, load_image, match_tag, open_project, sort_key
from spana.dataset.rf_project import ProjectImage, RoboflowApiError
from spana.detection.roboflow_backend import HttpInferenceClient, RoboflowError
from spana.secrets import load_secret

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
LOG_SCHEMA = "spana.roboflow_prelabel_log"
LOG_SCHEMA_VERSION = "1.0.0"
_PALETTE = [(60, 200, 60), (0, 165, 255), (230, 60, 230), (0, 255, 255)]


def eligibility(cfg: DatasetConfig, img: ProjectImage) -> Optional[str]:
    """None if the frame may be pre-labelled, else the skip reason."""
    _, reason = match_of(cfg, img.tags, img.metadata)
    if reason:
        return f"licence gate: {reason}"
    if cfg.splits.excluded_tag in img.tags:
        return "excluded (near-duplicate)"
    if cfg.prelabel.tag in img.tags:
        return "already pre-labelled"
    if cfg.review.reviewed_tag in img.tags:
        return "already reviewed"
    if img.annotated:
        return "already annotated (existing labels are never overwritten)"
    return None


def select(cfg: DatasetConfig, images: List[ProjectImage], *, limit: Optional[int], match: Optional[str],
           confirm_full_run: bool):
    """-> (selected images, skipped [{id, name, reason}]). Raises PolicyError for a run larger than allowed."""
    if limit is None and not confirm_full_run:
        raise PolicyError("refused: a full run needs --confirm-full-run (Nathan's go-ahead after the sample check)")
    if limit is not None and limit > cfg.prelabel.sample_size and not confirm_full_run:
        raise PolicyError(f"refused: --limit {limit} > prelabel.sample_size {cfg.prelabel.sample_size} "
                          "without --confirm-full-run")
    if limit is not None and limit <= 0:
        raise PolicyError("--limit must be positive")
    eligible, skipped = [], []
    for img in sorted(images, key=sort_key):
        if match is not None and match_tag(img) != match:
            continue
        reason = eligibility(cfg, img)
        if reason:
            skipped.append({"id": img.id, "name": img.name, "reason": reason})
        else:
            eligible.append(img)
    chosen = eligible if limit is None else evenly_spaced(eligible, limit)
    return chosen, skipped


def map_predictions(cfg: DatasetConfig, resp: dict, width: int, height: int):
    """Raw model response -> (boxes [{class, class_id, box, score}], dropped labels Counter, below-threshold Counter).
    Fails closed (RoboflowError) on a malformed response or a size mismatch."""
    if not isinstance(resp, dict) or not isinstance(resp.get("predictions"), list):
        raise RoboflowError("malformed response: no predictions list")
    img = resp.get("image") or {}
    if (img.get("width"), img.get("height")) != (width, height):
        raise RoboflowError(f"response size {img.get('width')}x{img.get('height')} != image {width}x{height}")
    cmap, thr = cfg.prelabel_class_map(), cfg.prelabel_thresholds()
    boxes, dropped, below = [], Counter(), Counter()
    for p in resp["predictions"]:
        try:
            x, y, w, h, s = (float(p[k]) for k in ("x", "y", "width", "height", "confidence"))
            label = str(p["class"])
        except (KeyError, TypeError, ValueError):
            raise RoboflowError(f"malformed prediction {p!r}") from None
        if not all(np.isfinite([x, y, w, h, s])):
            raise RoboflowError(f"non-finite prediction {p!r}")
        cls = cmap.get(label)
        if cls is None:
            dropped[label] += 1
            continue
        if s < thr[cls]:
            below[cls] += 1
            continue
        x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x2, y2 = min(float(width), x + w / 2), min(float(height), y + h / 2)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"class": cls, "class_id": cfg.classes.index(cls), "box": [round(v, 2) for v in (x1, y1, x2, y2)],
                      "score": round(s, 4)})
    boxes.sort(key=lambda b: (b["class_id"], -b["score"], b["box"]))
    return boxes, dropped, below


def yolo_annotation(cfg: DatasetConfig, boxes: List[dict], width: int, height: int):
    """Darknet TXT lines + labelmap over the full configured class list (keeps class ids stable)."""
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        lines.append(f"{b['class_id']} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return "\n".join(lines) + "\n", {str(i): c for i, c in enumerate(cfg.classes)}


def render(frame: np.ndarray, boxes: List[dict]) -> np.ndarray:
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b["box"])
        col = _PALETTE[b["class_id"] % len(_PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{b['class']} {b['score']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col,
                    1, cv2.LINE_AA)
    return out


def run_prelabel(cfg: DatasetConfig, project, infer, images: List[ProjectImage], *, limit: Optional[int],
                 match: Optional[str], confirm_full_run: bool, load, render_dir: Optional[Path]) -> dict:
    """`infer(jpeg_bytes) -> raw response`; `load(img) -> BGR frame` (raises SkipFrame). Returns the log document."""
    chosen, skipped = select(cfg, images, limit=limit, match=match, confirm_full_run=confirm_full_run)
    entries, dropped_all = [], Counter()
    for img in chosen:
        e = {"id": img.id, "name": img.name, "match_id": match_tag(img),
             "frame_index": img.metadata.get("frame_index"), "boxes_per_class": {}, "n_boxes": 0,
             "below_threshold": {}, "status": None, "reason": None}
        try:
            frame = load(img)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise SkipFrame("JPEG encoding failed")
            boxes, dropped, below = map_predictions(cfg, infer(buf.tobytes()), frame.shape[1], frame.shape[0])
            dropped_all.update(dropped)
            e["boxes_per_class"] = {c: sum(b["class"] == c for b in boxes) for c in cfg.prelabel_class_map().values()}
            e["n_boxes"] = len(boxes)
            e["below_threshold"] = dict(sorted(below.items()))
            e["boxes"] = boxes
            # tag first: if the annotation save then fails, the frame is still marked for review (fail safe)
            project.update_tags(img.id, add=[cfg.prelabel.tag])
            if boxes:
                content, labelmap = yolo_annotation(cfg, boxes, frame.shape[1], frame.shape[0])
                saved = project.save_annotation(img.id, name=Path(img.name).stem + ".txt", content=content,
                                                labelmap=labelmap, add_to_dataset=cfg.prelabel.add_to_dataset)
                if saved == "already_annotated":
                    raise SkipFrame("already annotated on the server (kept, not overwritten)")
                e["status"] = "labelled"
            else:
                e["status"] = "no_boxes"
                e["reason"] = "no detection above its class threshold"
            if render_dir is not None:
                render_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(render_dir / img.name), render(frame, boxes))
        except SkipFrame as ex:
            e["status"], e["reason"] = "skipped", str(ex)
        except (RoboflowError, RoboflowApiError) as ex:
            e["status"], e["reason"] = "failed", str(ex)[:300]
        entries.append(e)
    per_class = Counter()
    for e in entries:
        per_class.update(e["boxes_per_class"])
    status = Counter(e["status"] for e in entries)
    return {"schema": LOG_SCHEMA, "schema_version": LOG_SCHEMA_VERSION,
            "model_id": cfg.prelabel.model_id, "model_licence": cfg.prelabel.model_licence,
            "prelabel_config": cfg.to_dict()["prelabel"], "prelabel_config_sha256": cfg.section_sha256("prelabel"),
            "request": {"limit": limit, "match": match, "confirm_full_run": confirm_full_run},
            "entries": entries, "not_selected_skips": skipped,
            "summary": {"selected": len(chosen), "status": dict(sorted(status.items())),
                        "boxes_per_class": dict(sorted(per_class.items())),
                        "frames_with_ball": sum(e["boxes_per_class"].get("ball", 0) > 0 for e in entries),
                        "dropped_model_labels": dict(sorted(dropped_all.items())),
                        "skip_reasons": dict(sorted(Counter(s["reason"] for s in skipped).items()))}}


def summary_text(doc: dict) -> str:
    s = doc["summary"]
    lines = [f"pre-label {doc['model_id']}: selected {s['selected']}, status {s['status']}",
             f"  boxes per class {s['boxes_per_class']}; frames with ball {s['frames_with_ball']}",
             f"  dropped model labels {s['dropped_model_labels']}",
             f"  not selected (skipped) {s['skip_reasons']}"]
    for e in doc["entries"]:
        lines.append(f"  {e['name']}: {e['status']} {e['boxes_per_class']}"
                     + (f" below-threshold {e['below_threshold']}" if e["below_threshold"] else "")
                     + (f" ({e['reason']})" if e["reason"] else ""))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--limit", type=int)
    g.add_argument("--all", action="store_true")
    ap.add_argument("--match", help="only frames tagged match:<id>")
    ap.add_argument("--confirm-full-run", action="store_true")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    key = load_secret(cfg.roboflow.api_key_env)
    project = open_project(cfg, key)
    client = HttpInferenceClient(cfg.prelabel.api_url, key, cfg.prelabel.timeout_s)
    infer = lambda jpg: client.infer(jpg, model_id=cfg.prelabel.model_id, confidence=cfg.prelabel.request_confidence,
                                     iou=cfg.prelabel.iou)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    cache = REPO / cfg.paths.image_cache_dir
    try:
        doc = run_prelabel(cfg, project, infer, project.list_images(cfg.audit.page_size),
                           limit=None if a.all else a.limit, match=a.match, confirm_full_run=a.confirm_full_run,
                           load=lambda img: load_image(project, img, cache), render_dir=REPO / cfg.paths.render_dir / stamp)
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    doc["created_utc"] = now.isoformat(timespec="seconds")
    out = REPO / cfg.paths.prelabel_dir / f"prelabel_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, doc)
    print(summary_text(doc))
    print(f"log: {out}\nrenders: {REPO / cfg.paths.render_dir / stamp}")
    return 0 if not doc["summary"]["status"].get("failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
