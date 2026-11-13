"""Contact sheets of the ball pre-label boxes in pre-label logs, for a quick human check of false positives.

Offline and read-only: reads pre-label logs and the local image cache (paths.image_cache_dir); makes no API call and
changes nothing in the project. One sheet per match, boxes in log order, each cell a magnified crop around the box
with its frame name, score and size. Output: <paths.render_dir>/ball_sheets/<match>_<n>.jpg (not committed).

    python -m tools.roboflow_ball_sheet records/roboflow_prelabel/prelabel_<stamp>.json [...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from spana.dataset.config import load_dataset_config

REPO = Path(__file__).resolve().parents[1]
CROP_FRAC = 0.09      # crop side as a fraction of frame height (context around the ball)
CELL_PX = 240         # magnified cell side
COLS, ROWS = 8, 6


def ball_cells(doc: dict, ball_class: str, cache: Path):
    """-> (match_id, cell image) for every ball box of every written entry in a pre-label log."""
    for e in doc.get("entries", []):
        balls = [b for b in e.get("boxes", []) if b["class"] == ball_class]
        if not balls:
            continue
        frame = cv2.imread(str(cache / f"{e['id']}.jpg"))
        if frame is None:
            raise SystemExit(f"ERROR: {e['name']} is not in the image cache ({cache})")
        h, w = frame.shape[:2]
        side = max(16, int(round(CROP_FRAC * h)))
        for b in balls:
            x1, y1, x2, y2 = b["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ox = int(min(max(0, cx - side / 2), w - side))
            oy = int(min(max(0, cy - side / 2), h - side))
            cell = cv2.resize(frame[oy:oy + side, ox:ox + side], (CELL_PX, CELL_PX), interpolation=cv2.INTER_CUBIC)
            s = CELL_PX / side
            cv2.rectangle(cell, (int((x1 - ox) * s) - 2, int((y1 - oy) * s) - 2),
                          (int((x2 - ox) * s) + 2, int((y2 - oy) * s) + 2), (0, 255, 255), 1)
            frame_no = Path(e["name"]).stem.rsplit("_f", 1)[-1]
            for k, text in enumerate((f"f{frame_no}", f"{b['score']:.2f} {x2 - x1:.0f}x{y2 - y1:.0f}px")):
                org = (4, 14 + 15 * k)
                cv2.putText(cell, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
                cv2.putText(cell, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            yield e["match_id"], cell


def sheets(cells, cols: int = COLS, rows: int = ROWS):
    """Pack cells into pages of rows x cols (the last page padded with black)."""
    for i in range(0, len(cells), cols * rows):
        page = cells[i:i + cols * rows]
        page += [np.zeros_like(page[0])] * (-len(page) % cols)
        yield np.vstack([np.hstack(page[r:r + cols]) for r in range(0, len(page), cols)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--config", default=str(REPO / "config" / "roboflow.yaml"))
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    by_match = defaultdict(list)
    for p in a.logs:
        for match, cell in ball_cells(json.loads(Path(p).read_text()), cfg.prelabel.ball_class,
                                      REPO / cfg.paths.image_cache_dir):
            by_match[match].append(cell)
    out = REPO / cfg.paths.render_dir / "ball_sheets"
    out.mkdir(parents=True, exist_ok=True)
    for match, cells in sorted(by_match.items()):
        for n, sheet in enumerate(sheets(cells), 1):
            path = out / f"{match}_{n}.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"{path}  ({match}: {len(cells)} ball boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
