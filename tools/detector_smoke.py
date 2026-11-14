"""Smoke test of the trained detector's adapter on ONE cached DEV frame. No scoring, no tuning.

    python -m tools.detector_smoke --image FRAME.png --match-id DEV_MATCH_ID

The match must be on the licence allow-list with split dev (EVAL and TRAIN matches are refused); a hosted
detector.api_url is third-party processing and needs it confirmed for the match. Prints the detections per class
and writes one annotated image under paths.detector_render_dir/dev_smoke/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

import cv2

from spana.dataset.config import DatasetConfig, PolicyError, authorize, load_dataset_config
from spana.detection.roboflow_backend import RoboflowDetector
from spana.secrets import load_secret

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
_PALETTE = [(60, 200, 60), (0, 165, 255), (230, 60, 230), (0, 255, 255), (255, 128, 0), (128, 128, 255)]


def run_smoke(cfg: DatasetConfig, frame, match_id: str, detector=None, api_key: Optional[str] = None):
    """-> (summary, annotated frame). Raises PolicyError before any pixel leaves this machine."""
    hosted = urlparse(cfg.detector.api_url).hostname not in cfg.local_api_hosts
    authorize(cfg, match_id, "dev", third_party=hosted)
    detector = detector or RoboflowDetector.from_detector_config(cfg, api_key)
    dets = detector.detect(frame)
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        col = _PALETTE[d.class_id % len(_PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{detector.class_names[d.class_id][:3]} {d.score:.2f}", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    counts = Counter(detector.class_names[d.class_id] for d in dets)
    return {"model": detector.model_id, "match_id": match_id, "frame_size": [frame.shape[1], frame.shape[0]],
            "detections": {c: counts.get(c, 0) for c in detector.class_names},
            "dropped_labels": dict(detector.dropped_labels)}, out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--image", required=True)
    ap.add_argument("--match-id", required=True)
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    frame = cv2.imread(a.image)
    if frame is None:
        print(f"ERROR: cannot read {a.image}", file=sys.stderr)
        return 2
    try:
        summary, out = run_smoke(cfg, frame, a.match_id, api_key=load_secret(cfg.roboflow.api_key_env))
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    dest = REPO / cfg.paths.detector_render_dir / "dev_smoke" / f"{Path(a.image).stem}_pred.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), out)
    print(json.dumps(summary, indent=1))
    print(f"image: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
