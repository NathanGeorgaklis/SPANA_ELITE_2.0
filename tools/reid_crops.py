"""Cut re-ID training crops from a frozen TRAIN label version and its hashed video.

    python -m tools.reid_crops --labels data/reid/TRAIN/<match>/v<N> --video CLIP.mp4 --out CROPS_DIR
                               [--config config/reid_train.yaml]

- Labels are loaded through spana.reid_labels.load_frozen(..., "train"): every file hash is checked and only TRAIN
  labels are accepted (DEV / EVAL are refused).
- The video must match the label manifest's sha256.
- Boxes: the MOT export (`boxes.mot.txt`, 1-based frames, id = players.json id). Per player, a crop at most every
  `crops.min_gap_s`; boxes lower than `crops.min_height_px` are skipped.
- The last `crops.holdout_tail` of the labelled frames is marked "holdout" (kept out of training; a sanity check).
- Output (a new folder, never overwritten): `<player>/<frame>.jpg` at `crops.size_hw`, `crops.jsonl` (one row per
  crop: player, side, frame, t, height, max IoU with another labelled box, split, file, sha256) and `crops.json`
  (provenance: label and video hashes, config, counts, index hash). Crops stay on the machine that cut them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

from spana.geometry import iou_matrix
from spana.reid_labels import load_frozen

CROPS_SCHEMA = ("spana.reid_crops", "1.0.0")
Box = Tuple[float, float, float, float]


def load_train_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("schema") != "spana.reid_train_config":
        raise SystemExit(f"{path}: not a spana.reid_train_config")
    return cfg


def parse_mot(text: str) -> Dict[int, List[Tuple[int, Box]]]:
    """MOT rows (frame 1-based, id, x, y, w, h, ...) -> {video frame index (0-based): [(id, (x1, y1, x2, y2))]}."""
    out: Dict[int, List[Tuple[int, Box]]] = defaultdict(list)
    for line in text.splitlines():
        if not line.strip():
            continue
        f, i, x, y, w, h = line.split(",")[:6]
        x, y, w, h = float(x), float(y), float(w), float(h)
        out[int(f) - 1].append((int(i), (x, y, x + w, y + h)))
    return dict(out)


def sample(boxes: Dict[int, List[Tuple[int, Box]]], fps: float, min_gap_s: float, min_h: float,
           holdout_tail: float) -> List[dict]:
    """Deterministic per-player sampling: frames in order, a crop when >= min_gap_s after the player's last crop."""
    frames = sorted(boxes)
    if not frames:
        return []
    cut = frames[min(len(frames) - 1, int(round(len(frames) * (1.0 - holdout_tail))))] if holdout_tail > 0 else None
    last: Dict[int, int] = {}
    rows = []
    for f in frames:
        ids = [i for i, _ in boxes[f]]
        arr = np.array([b for _, b in boxes[f]], float)
        ious = iou_matrix(arr, arr)
        np.fill_diagonal(ious, 0.0)
        for k, (pid, b) in enumerate(boxes[f]):
            if b[3] - b[1] < min_h:
                continue
            if pid in last and (f - last[pid]) / fps < min_gap_s - 1e-9:
                continue
            last[pid] = f
            rows.append({"id": pid, "frame": f, "t": round(f / fps, 6), "box": [round(v, 1) for v in b],
                         "h": round(b[3] - b[1], 1), "overlap": round(float(ious[k].max()) if len(ids) > 1 else 0.0, 4),
                         "split": "holdout" if cut is not None and f >= cut else "train"})
    return rows


def cut_crop(img: np.ndarray, box: Box, pad: float, size_hw: Tuple[int, int]) -> np.ndarray:
    """Padded box, clipped to the image, resized to size_hw (BGR in, BGR out)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    xa, ya = int(max(0, round(x1 - pad * w))), int(max(0, round(y1 - pad * h)))
    xb, yb = int(min(W, round(x2 + pad * w))), int(min(H, round(y2 + pad * h)))
    c = img[ya:yb, xa:xb]
    if c.size == 0:
        raise ValueError(f"empty crop for box {box}")
    return cv2.resize(c, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA if c.shape[0] > size_hw[0]
                      else cv2.INTER_LINEAR)


def main(argv=None) -> int:
    from spana.video import VideoSource
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config/reid_train.yaml")
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    cfg_bytes = Path(a.config).read_bytes()
    cc = load_train_config(a.config)["crops"]
    lab = load_frozen(Path(a.labels), "train")
    man, frozen = lab["manifest"], lab["frozen"]
    by_id = {v["id"]: (k, v["side"]) for k, v in lab["players"].items()}
    boxes = parse_mot((Path(a.labels) / "boxes.mot.txt").read_text())
    video = VideoSource(a.video, expected_sha256=man["video_sha256"])
    rows = sample(boxes, video.meta.fps, cc["min_gap_s"], cc["min_height_px"], cc["holdout_tail"])
    want: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        if r["id"] not in by_id:
            raise SystemExit(f"MOT id {r['id']} is not in players.json")
        want[r["frame"]].append(r)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise SystemExit(f"{tmp} exists from an interrupted run: remove it first")
    tmp.mkdir(parents=True)
    index = []
    size = tuple(cc["size_hw"])
    for f, _, img in video.frames(min(want), max(want) + 1):
        for r in want.get(f, ()):
            player, side = by_id[r["id"]]
            crop = cut_crop(img, tuple(r["box"]), cc["pad"], size)
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(cc["jpeg_quality"])])
            assert ok
            rel = f"{player}/{f:07d}.jpg"
            (tmp / player).mkdir(exist_ok=True)
            (tmp / rel).write_bytes(buf.tobytes())
            index.append({"player": player, "side": side, "frame": f, "t": r["t"], "h": r["h"],
                          "overlap": r["overlap"], "split": r["split"], "file": rel,
                          "sha256": hashlib.sha256(buf.tobytes()).hexdigest()})
    if len(index) != len(rows):
        raise SystemExit(f"video ended early: {len(index)} of {len(rows)} crops cut")
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in index)
    (tmp / "crops.jsonl").write_text(lines)
    counts = defaultdict(lambda: {"train": 0, "holdout": 0})
    for r in index:
        counts[r["player"]][r["split"]] += 1
    (tmp / "crops.json").write_text(json.dumps({
        "schema": CROPS_SCHEMA[0], "schema_version": CROPS_SCHEMA[1], "match_id": man["match_id"],
        "split": frozen["split"], "label_version": frozen["label_version"], "label_files": frozen["files"],
        "video_sha256": man["video_sha256"], "fps": video.meta.fps, "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
        "crops_config": cc, "crops": len(index), "players": len(counts), "per_player": dict(sorted(counts.items())),
        "index_sha256": hashlib.sha256(lines.encode()).hexdigest(), "argv": ["tools.reid_crops", *(argv or sys.argv[1:])],
    }, indent=1, sort_keys=True))
    tmp.rename(out)
    print(json.dumps({"out": str(out), "crops": len(index), "players": len(counts)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
