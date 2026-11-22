"""Contact sheets for labelling a side (A / B / GK_A / GK_B / REF / OTHER) per frozen DEV GT id, one match at a time.

    python -m tools.team_label_sheet --match-id DEV_G3 --clip data/labels/<clip>/v1 CLIP.mp4 [--clip ...] \
        --out-dir DIR [--config config/team_labels.yaml]

- DEV only. The frozen labels are read as they are, with their hashes checked (FROZEN.json gt_sha256), and the video
  must match the frozen clip sha256.
- Writes DIR/<match>_sheet_<page>.png: one row per GT id ("<clip> id <n>"), `crops_per_id` crops spread evenly over
  its labelled frames (clean, fully visible crops first; others orange-bordered) and one downscaled full frame with
  the box drawn.
- Writes DIR/<clip>.sides.json per clip: a template `spana.team_sides` 1.0.0 with every id's side null. Nathan fills
  in `annotator` and every side; the same letter means the same kit in every clip of the match. Then
  `tools.freeze_team_labels` freezes it. Nothing is inferred by the system.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

from spana.artifacts import refuse_existing, write_json_atomic
from spana.geometry import iou_matrix
from spana.labels import LabelError, parse_mot
from spana.reid_labels import SIDES
from spana.video import VideoSource

SIDES_SCHEMA = ("spana.team_sides", "1.0.0")
_CFG_KEYS = ("schema_version", "crops_per_id", "clean_max_iou", "thumb_height_px", "cell_width_px",
             "context_width_px", "rows_per_page")
ORANGE, WHITE, GREEN = (0, 140, 255), (255, 255, 255), (0, 220, 0)


def load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or set(cfg) != set(_CFG_KEYS) or cfg["schema_version"] != "1.0.0":
        raise LabelError(f"{path}: keys must be exactly {_CFG_KEYS}, schema 1.0.0")
    if not all(isinstance(cfg[k], int) and cfg[k] > 0 for k in _CFG_KEYS[1:] if k != "clean_max_iou") \
            or not 0.0 <= cfg["clean_max_iou"] <= 1.0:
        raise LabelError(f"{path}: bad values")
    return cfg


def load_frozen_dev(label_dir: Path):
    """(FROZEN.json, GroundTruth) of a frozen DEV label version, hashes checked."""
    frozen = json.loads((label_dir / "FROZEN.json").read_text())
    record = json.loads((label_dir / "labels_frozen.json").read_text())
    gt_bytes = (label_dir / "gt.txt").read_bytes()
    if frozen.get("split") != "DEV" or record.get("split") != "DEV":
        raise LabelError(f"{label_dir}: only DEV labels")
    if hashlib.sha256(gt_bytes).hexdigest() != frozen["gt_sha256"]:
        raise LabelError(f"{label_dir}: gt.txt no longer matches its frozen hash")
    if hashlib.sha256((label_dir / "labels_frozen.json").read_bytes()).hexdigest() != frozen["labels_record_sha256"]:
        raise LabelError(f"{label_dir}: labels_frozen.json no longer matches its frozen hash")
    clip = frozen["clip"]
    return frozen, parse_mot(gt_bytes.decode(), clip["width"], clip["height"], record["frames_labelled"])


def pick_crops(gt, k: int, clean_max_iou: float) -> Dict[str, List[Tuple[int, tuple, bool]]]:
    """Per id up to k (frame, box, clean) spread evenly over its frames; clean, fully visible ones preferred."""
    rows: Dict[str, List[Tuple[int, tuple, bool]]] = {}
    for f in sorted(gt.players):
        ids = sorted(gt.players[f], key=int)
        boxes = np.asarray([gt.players[f][i] for i in ids], dtype=float).reshape(-1, 4)
        ious = iou_matrix(boxes, boxes) if len(ids) else np.zeros((0, 0))
        if len(ids):
            np.fill_diagonal(ious, 0.0)
        for n, pid in enumerate(ids):
            clean = pid not in gt.occluded.get(f, set()) and (len(ids) < 2 or ious[n].max() <= clean_max_iou)
            rows.setdefault(pid, []).append((f, gt.players[f][pid], clean))

    def spread(xs):
        if len(xs) <= k:
            return list(xs)
        return [xs[round(j * (len(xs) - 1) / (k - 1))] for j in range(k)] if k > 1 else [xs[len(xs) // 2]]
    out = {}
    for pid, xs in rows.items():
        good = spread([x for x in xs if x[2]])
        rest = spread([x for x in xs if not x[2]])[: k - len(good)]
        out[pid] = sorted(good + rest, key=lambda x: (not x[2], x[0]))
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _cell(img, box, cfg, clean: bool) -> np.ndarray:
    th, cw = cfg["thumb_height_px"], cfg["cell_width_px"]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    crop = img[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    s = min(th / crop.shape[0], cw / crop.shape[1])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
    cell = np.full((th, cw, 3), 30, np.uint8)
    oy, ox = (th - crop.shape[0]) // 2, (cw - crop.shape[1]) // 2
    cell[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    if not clean:
        cv2.rectangle(cell, (0, 0), (cw - 1, th - 1), ORANGE, 3)
    return cell


def build_sheets(match_id: str, clips: List[Tuple[Path, str]], cfg: dict, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[np.ndarray] = []
    th, label_w = cfg["thumb_height_px"], 260
    templates = []
    for label_dir, video_path in clips:
        frozen, gt = load_frozen_dev(label_dir)
        clip = frozen["clip"]
        video = VideoSource(video_path, expected_sha256=clip["sha256"])
        picks = pick_crops(gt, cfg["crops_per_id"], cfg["clean_max_iou"])
        need = {f for xs in picks.values() for f, _, _ in xs}
        imgs = {}
        for i, _t, img in video.frames(0, max(need) + 1 if need else 0):
            if i in need:
                imgs[i] = img
        for pid, xs in picks.items():
            row = np.full((th, label_w + cfg["crops_per_id"] * cfg["cell_width_px"] + cfg["context_width_px"], 3),
                          30, np.uint8)
            cv2.putText(row, f"{clip['name'][:8]}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            cv2.putText(row, f"id {pid}", (8, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.4, GREEN, 3)
            for n, (f, box, clean) in enumerate(xs):
                x0 = label_w + n * cfg["cell_width_px"]
                row[:, x0:x0 + cfg["cell_width_px"]] = _cell(imgs[f], box, cfg, clean)
            f, box, _ = xs[len(xs) // 2]
            ctx = imgs[f].copy()
            cv2.rectangle(ctx, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), GREEN, 6)
            s = min(cfg["context_width_px"] / ctx.shape[1], th / ctx.shape[0])
            ctx = cv2.resize(ctx, (int(ctx.shape[1] * s), int(ctx.shape[0] * s)))
            x0 = label_w + cfg["crops_per_id"] * cfg["cell_width_px"]
            row[:ctx.shape[0], x0:x0 + ctx.shape[1]] = ctx
            cv2.line(row, (0, th - 1), (row.shape[1], th - 1), (90, 90, 90), 1)
            rows.append(row)
        templates.append((out_dir / f"{clip['name']}.sides.json", {
            "schema": SIDES_SCHEMA[0], "schema_version": SIDES_SCHEMA[1], "split": "DEV", "match_id": match_id,
            "clip": {"name": clip["name"], "sha256": clip["sha256"]}, "label_version": frozen["label_version"],
            "gt_sha256": frozen["gt_sha256"], "annotator": "", "sides_allowed": list(SIDES),
            "sides": {pid: None for pid in picks}, "notes": ""}))
    pages = []
    per = cfg["rows_per_page"]
    for p in range(0, len(rows), per):
        path = out_dir / f"{match_id}_sheet_{p // per + 1}.png"
        refuse_existing(path)
        cv2.imwrite(str(path), np.vstack(rows[p:p + per]))
        pages.append(path)
    for path, doc in templates:
        refuse_existing(path)
        write_json_atomic(path, doc)
    return pages + [p for p, _ in templates]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--clip", nargs=2, action="append", required=True, metavar=("LABEL_DIR", "VIDEO"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default="config/team_labels.yaml")
    a = ap.parse_args()
    for p in build_sheets(a.match_id, [(Path(d), v) for d, v in a.clip], load_cfg(a.config), Path(a.out_dir)):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
