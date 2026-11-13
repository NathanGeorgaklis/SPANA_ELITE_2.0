"""Step 4: pre-label project frames with a pretrained general detector, as annotations for HUMAN REVIEW (v2).

    python -m tools.roboflow_prelabel --diagnose-ball --ids-from LOG [LOG ...]       # read-only ball score check
    python -m tools.roboflow_prelabel --limit 20 [--match MATCH_ID]                  # sample (<= prelabel.sample_size)
    python -m tools.roboflow_prelabel --migrate --limit 20 --ids-from LOG [LOG ...]  # replace older pre-labels
    python -m tools.roboflow_prelabel --all --confirm-full-run --approved-sample SAMPLE_LOG   # after Nathan's go-ahead

Model, class map, per-class thresholds and every filter setting come from config (prelabel section). Only classes in
prelabel.class_map are auto-labelled; the rest (goalkeeper, referee) come from human review.

Run control (fail closed):
  * at most prelabel.sample_size frames, unless --confirm-full-run AND --approved-sample names a sample log made
    with this exact prelabel section (sha256 must match): approval of an older version never carries over;
  * every Roboflow write goes through spana.dataset.guard.GuardedProject, which re-reads the image and re-checks the
    licence allow-list and the reviewed tag immediately before writing.

Per frame:
  * eligibility: licence gate, excluded (near-duplicate / blur), reviewed, already pre-labelled -> skipped;
    a frame carrying a superseded tag (prelabel.superseded_tags) is touched only with --migrate;
  * pixels must match the sha256 recorded at upload;
  * blur gate: variance of Laplacian < prelabel.blur.laplacian_var_min -> tagged prelabel.blur.excluded_tag, no labels;
  * full-frame inference (players and ball, thresholds per class), optional ball-only tiled pass merged by NMS;
  * ball boxes larger than prelabel.ball_max_size_frac x frame height are dropped + logged;
  * grass filter: a box (prelabel.grass.apply_to) whose bottom-centre is outside the grass mask is dropped + logged;
  * --migrate: every selected frame's current annotation is backed up to paths.prelabel_backup_dir BEFORE any write;
    the annotation is then overwritten (empty for blurred / box-less frames), read back and compared, and only then
    is the superseded tag swapped for prelabel.tag in one call.

Writes paths.prelabel_dir/prelabel_<stamp>.json (per-frame log, no pixels), annotated renders to
paths.render_dir/<stamp>/ and prints a per-match summary.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from spana.artifacts import read_json, write_json_atomic
from spana.dataset.config import DatasetConfig, PolicyError, load_dataset_config, match_of
from spana.dataset.guard import GuardedProject
from spana.dataset.prefilter import blur_score, bottom_centre, grass_mask, merge_boxes, on_grass, tiles
from spana.dataset.prep import SkipFrame, evenly_spaced, list_all, load_image, match_tag, open_project, sort_key
from spana.dataset.rf_project import ProjectImage, RoboflowApiError
from spana.detection.roboflow_backend import HttpInferenceClient, RoboflowError
from spana.secrets import load_secret

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
LOG_SCHEMA = "spana.roboflow_prelabel_log"
LOG_SCHEMA_VERSION = "2.0.0"
BACKUP_SCHEMA = "spana.roboflow_annotation_backup"
DIAG_SCHEMA = "spana.roboflow_ball_diagnostic"
_PALETTE = [(60, 200, 60), (0, 165, 255), (230, 60, 230), (0, 255, 255)]
_DROPPED = (60, 60, 230)
VERIFY_TOL_PX = 1.5   # read-back box centre / size tolerance: YOLO text keeps 6 decimals of the normalised values


# --------------------------------------------------------------------------------------------------- selection ---
def eligibility(cfg: DatasetConfig, img: ProjectImage, *, migrate: bool = False) -> Optional[str]:
    """None if the frame may be pre-labelled (or, with migrate, re-labelled), else the skip reason."""
    _, reason = match_of(cfg, img.tags, img.metadata)
    if reason:
        return f"licence gate: {reason}"
    if cfg.review.reviewed_tag in img.tags:
        return "already reviewed (never touched)"
    if cfg.splits.excluded_tag in img.tags:
        return "excluded (near-duplicate)"
    if cfg.prelabel.blur.excluded_tag in img.tags:
        return "excluded (blur)"
    if cfg.prelabel.tag in img.tags:
        return "already pre-labelled"
    old = [t for t in cfg.prelabel.superseded_tags if t in img.tags]
    if migrate:
        return None if old else "not an older pre-label (--migrate only replaces superseded pre-labels)"
    if old:
        return f"older pre-label {old} (replace only with --migrate)"
    if img.annotated:
        return "already annotated (existing labels are never overwritten)"
    return None


def check_approved_sample(cfg: DatasetConfig, doc: Optional[dict]) -> None:
    """A run beyond prelabel.sample_size needs the log of a reviewed sample made with this exact prelabel section."""
    if doc is None:
        raise PolicyError("refused: a run beyond prelabel.sample_size needs --approved-sample <sample log> "
                          "(the sample Nathan reviewed) as well as --confirm-full-run")
    if doc.get("schema") != LOG_SCHEMA or doc.get("schema_version") != LOG_SCHEMA_VERSION:
        raise PolicyError("refused: --approved-sample is not a pre-label log of this version")
    req = doc.get("request") or {}
    if req.get("dry_run"):
        raise PolicyError("refused: --approved-sample is a dry run; approve a sample that was actually written")
    if req.get("confirm_full_run") or req.get("limit") is None or req["limit"] > cfg.prelabel.sample_size:
        raise PolicyError("refused: --approved-sample must be a sample run (limit <= prelabel.sample_size)")
    if doc.get("prelabel_config_sha256") != cfg.section_sha256("prelabel"):
        raise PolicyError("refused: --approved-sample was made with different prelabel settings "
                          "(config sha256 differs); run and review a new sample first")
    if (doc.get("summary") or {}).get("status", {}).get("failed"):
        raise PolicyError("refused: --approved-sample has failed frames")


def select(cfg: DatasetConfig, images: List[ProjectImage], *, limit: Optional[int], match: Optional[str],
           confirm_full_run: bool, migrate: bool = False, ids: Optional[Sequence[str]] = None,
           approved_sample: Optional[dict] = None):
    """-> (selected images, skipped [{id, name, reason}]). Raises PolicyError for a run larger than allowed."""
    if limit is not None and limit <= 0:
        raise PolicyError("--limit must be positive")
    if limit is None or limit > cfg.prelabel.sample_size:
        if not confirm_full_run:
            raise PolicyError(f"refused: more than prelabel.sample_size {cfg.prelabel.sample_size} frames needs "
                              "--confirm-full-run (Nathan's go-ahead after the sample check)")
        check_approved_sample(cfg, approved_sample)
    wanted = None if ids is None else set(ids)
    eligible, skipped = [], []
    for img in sorted(images, key=sort_key):
        if match is not None and match_tag(img) != match:
            continue
        if wanted is not None and img.id not in wanted:
            continue
        reason = eligibility(cfg, img, migrate=migrate)
        if reason:
            skipped.append({"id": img.id, "name": img.name, "reason": reason})
        else:
            eligible.append(img)
    chosen = eligible if limit is None else evenly_spaced(eligible, limit)
    return chosen, skipped


def ids_from_logs(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        doc = read_json(p)
        if doc.get("schema") != LOG_SCHEMA:
            raise PolicyError(f"--ids-from {p}: not a pre-label log")
        out += [e["id"] for e in doc.get("entries", []) if e.get("status") in ("labelled", "no_boxes")]
    if not out:
        raise PolicyError("--ids-from: no labelled frames in the given logs")
    return out


# ------------------------------------------------------------------------------------------------- predictions ---
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


def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise SkipFrame("JPEG encoding failed")
    return buf.tobytes()


def tiled_ball_boxes(cfg: DatasetConfig, frame: np.ndarray, infer: Callable[[bytes], dict], tile_px: int):
    """Ball-class boxes from overlapping tiles, in frame coordinates (not yet merged). -> (boxes, tiles used)."""
    ball = cfg.prelabel.ball_class
    h, w = frame.shape[:2]
    out, windows = [], tiles(w, h, tile_px, cfg.prelabel.ball_tiling.overlap_frac)
    for (x1, y1, x2, y2) in windows:
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
        boxes, _, _ = map_predictions(cfg, infer(_jpeg(crop)), x2 - x1, y2 - y1)
        for b in boxes:
            if b["class"] != ball:
                continue
            bx = b["box"]
            out.append(dict(b, box=[round(bx[0] + x1, 2), round(bx[1] + y1, 2), round(bx[2] + x1, 2),
                                    round(bx[3] + y1, 2)], source="tile"))
    return out, len(windows)


def detect(cfg: DatasetConfig, frame: np.ndarray, infer: Callable[[bytes], dict]):
    """Full-frame boxes for every auto-labelled class plus, if enabled, ball-only tiled boxes merged by NMS.
    Player boxes are always the full-frame ones (settings unchanged by tiling). -> (boxes, info)."""
    h, w = frame.shape[:2]
    boxes, dropped, below = map_predictions(cfg, infer(_jpeg(frame)), w, h)
    boxes, too_big = ball_size_filter(cfg, h, boxes)       # before merging: an oversize box must not absorb a ball
    info = {"dropped_model_labels": dropped, "below_threshold": below, "tiles": 0, "ball_merged": 0,
            "too_big": too_big}
    t = cfg.prelabel.ball_tiling
    if t.enabled:
        ball = cfg.prelabel.ball_class
        tiled, info["tiles"] = tiled_ball_boxes(cfg, frame, infer, t.tile_px)
        tiled, big = ball_size_filter(cfg, h, tiled)
        info["too_big"] += big
        balls = [dict(b, source="full") for b in boxes if b["class"] == ball] + tiled
        merged, info["ball_merged"] = merge_boxes(balls, t.merge_iou)
        boxes = [b for b in boxes if b["class"] != ball] + merged
        boxes.sort(key=lambda b: (b["class_id"], -b["score"], b["box"]))
    return boxes, info


def ball_size_filter(cfg: DatasetConfig, frame_height: int, boxes: List[dict]):
    """-> (kept, dropped [{class, box, score, reason}]): ball boxes larger than ball_max_size_frac x frame height."""
    limit = cfg.prelabel.ball_max_size_frac * frame_height
    kept, dropped = [], []
    for b in boxes:
        side = max(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1])
        if b["class"] == cfg.prelabel.ball_class and side > limit:
            dropped.append({"class": b["class"], "box": b["box"], "score": b["score"],
                            "reason": f"ball box side {side:.1f} px > {limit:.1f} px (ball_max_size_frac)"})
        else:
            kept.append(b)
    return kept, dropped


def grass_filter(cfg: DatasetConfig, frame: np.ndarray, boxes: List[dict]):
    """-> (kept, dropped [{class, box, score, reason}]). Pre-labels only."""
    g = cfg.prelabel.grass
    if not g.enabled:
        return list(boxes), []
    mask = grass_mask(frame, g)
    kept, dropped = [], []
    for b in boxes:
        if b["class"] in g.apply_to and not on_grass(mask, b["box"]):
            x, y = bottom_centre(b["box"])
            dropped.append({"class": b["class"], "box": b["box"], "score": b["score"],
                            "reason": f"bottom-centre ({x:.1f}, {y:.1f}) outside grass mask"})
        else:
            kept.append(b)
    return kept, dropped


def yolo_annotation(cfg: DatasetConfig, boxes: List[dict], width: int, height: int):
    """Darknet TXT lines + labelmap over the full configured class list (keeps class ids stable)."""
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        lines.append(f"{b['class_id']} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return "\n".join(lines) + "\n", {str(i): c for i, c in enumerate(cfg.classes)}


def render(frame: np.ndarray, boxes: List[dict], dropped: Sequence[dict] = (), note: str = "") -> np.ndarray:
    out = frame.copy()
    for b in dropped:
        x1, y1, x2, y2 = (int(round(v)) for v in b["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), _DROPPED, 1)
        cv2.line(out, (x1, y1), (x2, y2), _DROPPED, 1)
        cv2.putText(out, f"dropped {b['class']}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _DROPPED,
                    1, cv2.LINE_AA)
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b["box"])
        col = _PALETTE[b["class_id"] % len(_PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{b['class']} {b['score']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col,
                    1, cv2.LINE_AA)
    if note:
        cv2.putText(out, note, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------------------------------- backup ----
def backup_annotations(guard: GuardedProject, chosen: List[ProjectImage]) -> dict:
    """Current annotation + tags of every chosen frame, read from the server. Any read failure aborts the run."""
    frames = []
    for img in chosen:
        try:
            raw = guard.get_image(img.id)
        except RoboflowApiError as e:
            raise PolicyError(f"refused: backup read failed for {img.name} ({e}); nothing was changed") from None
        frames.append({"id": img.id, "name": img.name, "match_id": match_tag(img), "tags": sorted(raw.get("tags") or []),
                       "annotation": raw.get("annotation")})
    return {"schema": BACKUP_SCHEMA, "schema_version": "1.0.0", "frames": frames}


# ------------------------------------------------------------------------------------------------------- run -----
def _verify_saved(guard: GuardedProject, image_id: str, boxes: List[dict]) -> None:
    if guard.dry_run:
        return
    ann = guard.get_image(image_id).get("annotation") or {}
    got = ann.get("boxes") if isinstance(ann, dict) else None
    if not isinstance(got, list):
        got = []
    try:
        have = sorted((str(g["label"]), float(g["x"]), float(g["y"]), float(g["width"]), float(g["height"]))
                      for g in got)
    except (KeyError, TypeError, ValueError):
        raise RoboflowApiError("read-back: unexpected annotation box shape") from None
    want = sorted((b["class"], (b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2,
                   b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]) for b in boxes)
    if len(have) != len(want) or any(h[0] != w[0] or max(abs(a - c) for a, c in zip(h[1:], w[1:])) > VERIFY_TOL_PX
                                     for h, w in zip(have, want)):
        raise RoboflowApiError(f"read-back mismatch: server has {dict(Counter(h[0] for h in have))}, "
                               f"expected {dict(Counter(w[0] for w in want))} (classes or box positions differ)")


def _write_labels(cfg: DatasetConfig, guard: GuardedProject, img: ProjectImage, boxes: List[dict], w: int, h: int,
                  *, migrate: bool, add_tags: List[str]) -> None:
    old = [t for t in cfg.prelabel.superseded_tags if t in guard.current(img.id).tags]
    if not migrate:
        guard.update_tags(img.id, add=add_tags)          # tag first: a failed save still leaves it marked for review
        if boxes:
            content, labelmap = yolo_annotation(cfg, boxes, w, h)
            if guard.save_annotation(img.id, name=Path(img.name).stem + ".txt", content=content, labelmap=labelmap,
                                     add_to_dataset=cfg.prelabel.add_to_dataset) == "already_annotated":
                raise SkipFrame("already annotated on the server (kept, not overwritten)")
        return
    # migrate: the frame is still marked by its old tag until the new annotation is verified on the server
    content, labelmap = yolo_annotation(cfg, boxes, w, h)
    res = guard.save_annotation(img.id, name=Path(img.name).stem + ".txt", content=content, labelmap=labelmap,
                                add_to_dataset=cfg.prelabel.add_to_dataset, overwrite=True)
    if res != "saved":
        raise RoboflowApiError(f"overwrite not applied ({res}); old pre-label kept")
    _verify_saved(guard, img.id, boxes)
    guard.update_tags(img.id, add=add_tags, remove=old)


def process_frame(cfg: DatasetConfig, guard: GuardedProject, infer, img: ProjectImage, *, migrate: bool, load,
                  render_dir: Optional[Path]) -> dict:
    e = {"id": img.id, "name": img.name, "match_id": match_tag(img), "frame_index": img.metadata.get("frame_index"),
         "status": None, "reason": None, "blur_score": None, "blurred": None, "boxes_per_class": {}, "n_boxes": 0,
         "players_kept": 0, "players_dropped": 0, "balls": 0, "balls_dropped": 0, "ball_merged": 0, "tiles": 0,
         "below_threshold": {}, "dropped_boxes": [], "boxes": []}
    fresh = guard.refresh(img.id)                    # current tags: a frame reviewed / tagged mid-run is left alone
    reason = eligibility(cfg, fresh, migrate=migrate)
    if reason:
        raise SkipFrame(f"changed since listing: {reason}")
    frame = load(img)
    h, w = frame.shape[:2]
    b = cfg.prelabel.blur
    e["blur_score"] = round(blur_score(frame, b.analysis_width_px), 2)
    e["blurred"] = e["blur_score"] < b.laplacian_var_min
    note = f"blur {e['blur_score']:.0f}"
    def recheck():
        """Re-read the image right before the first write (inference can take a while)."""
        r = eligibility(cfg, guard.refresh(img.id), migrate=migrate)
        if r:
            raise SkipFrame(f"changed during processing: {r}")
    if e["blurred"]:
        recheck()
        if migrate:                                   # old labels are cleared (after the backup), then tags swapped
            _write_labels(cfg, guard, img, [], w, h, migrate=True, add_tags=[b.excluded_tag])
        else:
            guard.update_tags(img.id, add=[b.excluded_tag])
        e["status"], e["reason"] = "excluded_blur", f"variance of Laplacian {e['blur_score']} < {b.laplacian_var_min}"
        boxes, dropped = [], []
        note += " EXCLUDED (blur)"
    else:
        boxes, info = detect(cfg, frame, infer)
        boxes, dropped = grass_filter(cfg, frame, boxes)
        dropped = info["too_big"] + dropped
        recheck()
        e.update(below_threshold=dict(sorted(info["below_threshold"].items())), tiles=info["tiles"],
                 ball_merged=info["ball_merged"], dropped_model_labels=dict(sorted(info["dropped_model_labels"].items())))
        e["boxes"], e["dropped_boxes"] = boxes, dropped
        _write_labels(cfg, guard, img, boxes, w, h, migrate=migrate, add_tags=[cfg.prelabel.tag])
        e["status"] = "labelled" if boxes else "no_boxes"
        if not boxes:
            e["reason"] = "no detection above its class threshold on the pitch"
    ball = cfg.prelabel.ball_class
    e["boxes_per_class"] = {c: sum(x["class"] == c for x in boxes) for c in sorted(set(cfg.prelabel_class_map().values()))}
    e["n_boxes"] = len(boxes)
    e["balls"] = e["boxes_per_class"].get(ball, 0)
    e["players_kept"] = sum(x["class"] != ball for x in boxes)
    e["players_dropped"] = sum(x["class"] != ball for x in dropped)
    e["balls_dropped"] = sum(x["class"] == ball for x in dropped)
    if render_dir is not None:
        try:
            render_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(render_dir / img.name), render(frame, boxes, dropped, note)):
                raise OSError("imwrite returned false")
        except Exception as ex:                      # the frame's writes are done: record, do not fail it
            e["render_error"] = f"{type(ex).__name__}: {ex}"[:200]
    return e


def run_prelabel(cfg: DatasetConfig, project, infer, images: List[ProjectImage], *, limit: Optional[int],
                 match: Optional[str], confirm_full_run: bool, load, render_dir: Optional[Path],
                 migrate: bool = False, ids: Optional[Sequence[str]] = None, approved_sample: Optional[dict] = None,
                 write_backup: Optional[Callable[[dict], dict]] = None, dry_run: bool = False) -> dict:
    """`infer(jpeg_bytes) -> raw response`; `load(img) -> BGR frame` (raises SkipFrame); `write_backup(doc) -> path`
    (required with migrate; called BEFORE any write). Every write goes through GuardedProject. dry_run: every check
    and inference runs, nothing is written to Roboflow and no backup is written. Returns the log."""
    chosen, skipped = select(cfg, images, limit=limit, match=match, confirm_full_run=confirm_full_run,
                             migrate=migrate, ids=ids, approved_sample=approved_sample)
    guard = GuardedProject(cfg, project, images, dry_run=dry_run)
    backup = None
    if migrate and chosen:
        if write_backup is None:
            raise PolicyError("refused: --migrate needs a backup writer")
        doc = backup_annotations(guard, chosen)
        backup = {"dry_run": True, "frames": len(doc["frames"])} if dry_run else write_backup(doc)
    entries, dropped_all = [], Counter()
    for k, img in enumerate(chosen):
        if k and cfg.prelabel.min_interval_s:
            time.sleep(cfg.prelabel.min_interval_s)
        try:
            e = process_frame(cfg, guard, infer, img, migrate=migrate, load=load, render_dir=render_dir)
            dropped_all.update(e.get("dropped_model_labels", {}))
        except SkipFrame as ex:
            e = {"id": img.id, "name": img.name, "match_id": match_tag(img), "status": "skipped", "reason": str(ex)}
        except PolicyError as ex:
            e = {"id": img.id, "name": img.name, "match_id": match_tag(img), "status": "refused", "reason": str(ex)}
        except (RoboflowError, RoboflowApiError) as ex:
            e = {"id": img.id, "name": img.name, "match_id": match_tag(img), "status": "failed", "reason": str(ex)[:300]}
        except Exception as ex:                      # anything else fails this frame only; the run and its log go on
            e = {"id": img.id, "name": img.name, "match_id": match_tag(img), "status": "failed",
                 "reason": f"{type(ex).__name__}: {ex}"[:300]}
        entries.append(e)
        print(f"[{k + 1}/{len(chosen)}] {e['name']}: {e['status']}" + (f" ({e['reason']})" if e.get('reason') else ""),
              file=sys.stderr, flush=True)
    return {"schema": LOG_SCHEMA, "schema_version": LOG_SCHEMA_VERSION,
            "model_id": cfg.prelabel.model_id, "model_licence": cfg.prelabel.model_licence,
            "prelabel_config": cfg.to_dict()["prelabel"], "prelabel_config_sha256": cfg.section_sha256("prelabel"),
            "request": {"limit": limit, "match": match, "confirm_full_run": confirm_full_run, "migrate": migrate,
                        "dry_run": dry_run, "ids": list(ids) if ids is not None else None,
                        "approved_sample_sha256": (approved_sample or {}).get("_sha256")},
            "backup": backup, "writes": len(guard.writes),
            "entries": entries, "not_selected_skips": skipped,
            "summary": summarize(cfg, entries, skipped, dropped_all)}


def summarize(cfg: DatasetConfig, entries: List[dict], skipped: List[dict], dropped_all: Counter) -> dict:
    done = ("labelled", "no_boxes", "excluded_blur")
    per_match: Dict[str, Counter] = defaultdict(Counter)
    for e in entries:
        m = per_match[e.get("match_id") or "?"]
        m["frames"] += 1
        m[e["status"]] += 1
        if e["status"] in done:                         # only frames whose writes completed
            m["players_kept"] += e["players_kept"]
            m["players_dropped_off_pitch"] += e["players_dropped"]
            m["balls"] += e["balls"]
            m["balls_dropped"] += e["balls_dropped"]
            m["frames_with_ball"] += e["balls"] > 0
    total = Counter()
    for m in per_match.values():
        total.update(m)
    per_class = Counter()
    for e in entries:
        if e["status"] in done:
            per_class.update(e["boxes_per_class"])
    return {"selected": len(entries), "status": dict(sorted(Counter(e["status"] for e in entries).items())),
            "per_match": {k: dict(sorted(v.items())) for k, v in sorted(per_match.items())},
            "total": dict(sorted(total.items())), "boxes_per_class": dict(sorted(per_class.items())),
            "dropped_model_labels": dict(sorted(dropped_all.items())),
            "skip_reasons": dict(sorted(Counter(s["reason"] for s in skipped).items()))}


def summary_text(doc: dict) -> str:
    s = doc["summary"]
    lines = [f"{'DRY RUN (nothing written) ' if doc['request'].get('dry_run') else ''}"
             f"pre-label {doc['model_id']} ({doc['prelabel_config']['tag']}): selected {s['selected']}, "
             f"status {s['status']}, writes {doc['writes']}" + (f", backup {doc['backup']}" if doc.get("backup") else "")]
    for e in doc["entries"]:
        if e["status"] in ("labelled", "no_boxes", "excluded_blur"):
            lines.append(f"  {e['name']}: {e['status']} players kept {e['players_kept']} dropped {e['players_dropped']}"
                         f", balls {e['balls']} (dropped {e['balls_dropped']}, merged {e['ball_merged']}, tiles {e['tiles']}), blur {e['blur_score']}"
                         f"{' BLURRED' if e['blurred'] else ''}")
            for d in e["dropped_boxes"]:
                lines.append(f"      dropped {d['class']} {d['box']} {d['score']:.2f}: {d['reason']}")
        else:
            lines.append(f"  {e['name']}: {e['status']} ({e['reason']})")
    lines.append("per match:")
    for mid, m in s["per_match"].items():
        lines.append(f"  {mid}: {m}")
    lines.append(f"  total: {s['total']}")
    lines.append(f"  dropped model labels {s['dropped_model_labels']}; not selected {s['skip_reasons']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- ball diagnostic -----
def diagnose_ball(cfg: DatasetConfig, infer_raw: Callable[[bytes, float], dict], images: List[ProjectImage],
                  ids: Sequence[str], load) -> dict:
    """Read-only. Top-k raw ball scores per frame before any class threshold, full frame and per tile size.
    `infer_raw(jpeg, request_confidence)`. Licence gate applies (hosted inference is third-party processing)."""
    d, ball = cfg.prelabel.ball_diagnostic, cfg.prelabel.ball_class
    labels = {lbl for lbl, c in cfg.prelabel_class_map().items() if c == ball}
    by_id = {i.id: i for i in images}
    rows = []
    for k, iid in enumerate(ids):
        img = by_id.get(iid)
        row = {"id": iid, "name": img.name if img else None, "match_id": match_tag(img) if img else None}
        if img is None:
            row["status"] = "not in project"
            rows.append(row)
            continue
        _, reason = match_of(cfg, img.tags, img.metadata)
        if reason:
            row["status"] = f"licence gate: {reason}"
            rows.append(row)
            continue
        if k and cfg.prelabel.min_interval_s:
            time.sleep(cfg.prelabel.min_interval_s)
        try:
            frame = load(img)
            h, w = frame.shape[:2]
            passes = [("full", [(0, 0, w, h)])] + [(f"tile{t}", tiles(w, h, t, cfg.prelabel.ball_tiling.overlap_frac))
                                                  for t in d.tile_px_candidates]
            for name, windows in passes:
                scores = []
                for (x1, y1, x2, y2) in windows:
                    resp = infer_raw(_jpeg(np.ascontiguousarray(frame[y1:y2, x1:x2])), d.request_confidence)
                    for p in resp.get("predictions", []):
                        if str(p.get("class")) in labels:
                            scores.append({"score": round(float(p["confidence"]), 4),
                                           "x": round(float(p["x"]) + x1, 1), "y": round(float(p["y"]) + y1, 1),
                                           "w": round(float(p["width"]), 1), "h": round(float(p["height"]), 1)})
                scores.sort(key=lambda s: -s["score"])
                row[name] = {"windows": len(windows), "n": len(scores), "top": scores[:d.top_k]}
            row["status"] = "ok"
        except (SkipFrame, RoboflowError) as ex:
            row["status"] = f"failed: {str(ex)[:200]}"
        rows.append(row)
    thr = cfg.prelabel_thresholds()[ball]
    summ = {}
    for name in ["full"] + [f"tile{t}" for t in d.tile_px_candidates]:
        best = [r[name]["top"][0]["score"] if r.get(name, {}).get("top") else 0.0 for r in rows if r.get("status") == "ok"]
        summ[name] = {"frames": len(best), "frames_any_ball": sum(b > 0 for b in best),
                      "frames_best_ge_threshold": sum(b >= thr for b in best),
                      "frames_best_ge_0.5": sum(b >= 0.5 for b in best),
                      "median_best": round(float(np.median(best)), 4) if best else None}
    return {"schema": DIAG_SCHEMA, "schema_version": "1.0.0", "model_id": cfg.prelabel.model_id,
            "ball_threshold": thr, "request_confidence": d.request_confidence,
            "overlap_frac": cfg.prelabel.ball_tiling.overlap_frac, "frames": rows, "summary": summ}


# ------------------------------------------------------------------------------------------------------- CLI -----
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--limit", type=int)
    g.add_argument("--all", action="store_true")
    g.add_argument("--diagnose-ball", action="store_true", help="read-only raw ball scores (needs --ids-from)")
    ap.add_argument("--match", help="only frames tagged match:<id>")
    ap.add_argument("--ids-from", nargs="+", help="only the frames listed in these pre-label logs")
    ap.add_argument("--migrate", action="store_true", help="replace superseded pre-labels (backup first)")
    ap.add_argument("--dry-run", action="store_true", help="run every check and inference; write nothing")
    ap.add_argument("--confirm-full-run", action="store_true")
    ap.add_argument("--approved-sample", help="sample log Nathan reviewed (needed beyond prelabel.sample_size)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    key = load_secret(cfg.roboflow.api_key_env)
    project = open_project(cfg, key)
    client = HttpInferenceClient(cfg.prelabel.api_url, key, cfg.prelabel.timeout_s)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    cache = REPO / cfg.paths.image_cache_dir
    load = lambda img: load_image(project, img, cache)
    try:
        ids = ids_from_logs(a.ids_from) if a.ids_from else None
        if a.diagnose_ball:
            if ids is None:
                raise PolicyError("--diagnose-ball needs --ids-from")
            doc = diagnose_ball(cfg, lambda jpg, conf: client.infer(jpg, model_id=cfg.prelabel.model_id,
                                                                     confidence=conf, iou=cfg.prelabel.iou),
                                list_all(cfg, project), ids, load)
            doc["created_utc"] = now.isoformat(timespec="seconds")
            out = REPO / cfg.paths.prelabel_dir / f"ball_diagnostic_{stamp}.json"
            write_json_atomic(out, doc)
            for r in doc["frames"]:
                print(f"  {r['name']}: " + "; ".join(f"{n} {[t['score'] for t in r[n]['top']]}"
                                                    for n in r if isinstance(r.get(n), dict)) + f" [{r['status']}]")
            print(f"summary (ball threshold {doc['ball_threshold']}): {doc['summary']}\nlog: {out}")
            return 0
        approved = None
        if a.approved_sample:
            approved = read_json(a.approved_sample)
            approved["_sha256"] = hashlib.sha256(Path(a.approved_sample).read_bytes()).hexdigest()
        backup_dir = REPO / cfg.paths.prelabel_backup_dir

        def write_backup(doc):
            doc["created_utc"] = now.isoformat(timespec="seconds")
            path = backup_dir / f"backup_{stamp}.json"
            sha = write_json_atomic(path, doc)
            return {"path": str(path.relative_to(REPO)), "sha256": sha, "frames": len(doc["frames"])}
        doc = run_prelabel(cfg, project, lambda jpg: client.infer(jpg, model_id=cfg.prelabel.model_id,
                                                                  confidence=cfg.prelabel.request_confidence,
                                                                  iou=cfg.prelabel.iou),
                           list_all(cfg, project), limit=None if a.all else a.limit, match=a.match,
                           confirm_full_run=a.confirm_full_run, load=load,
                           render_dir=REPO / cfg.paths.render_dir / stamp, migrate=a.migrate, ids=ids,
                           approved_sample=approved, write_backup=write_backup, dry_run=a.dry_run)
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    doc["created_utc"] = now.isoformat(timespec="seconds")
    out = REPO / cfg.paths.prelabel_dir / f"prelabel_{stamp}{'_dryrun' if a.dry_run else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, doc)
    print(summary_text(doc))
    print(f"log: {out}\nrenders: {REPO / cfg.paths.render_dir / stamp}")
    return 0 if not (doc["summary"]["status"].get("failed") or doc["summary"]["status"].get("refused")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
