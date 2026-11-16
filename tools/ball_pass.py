"""One detector pass that keeps a single class (e.g. the ball): positions for DEV analysis only.

    python -m tools.ball_pass --video V --dataset-config config/roboflow.yaml --model-record models/rfdetr_v1.json \
        --runtime torch --device mps --class-name ball --out BALL.json

The tracking artifacts carry persons only (the person view drops every other class), so analyses that need
the ball (e.g. tools/guard_cost.py "near-ball" moments) read it from here. Same local detector and pre-processing
as `detect --backend local`; boxes of --class-name at the detector's configured evaluation operating point
(`detector.confidence` in the dataset config), every frame. Frames stay on this machine. Not a product layer: ball
tracking is a later layer.

Output: {tool, schema_version, provenance, video, model, class_name, confidence, frames: [{frame, t, boxes:
[[x1, y1, x2, y2, score], ...]}]}; canonical JSON, never overwritten.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from spana.artifacts import refuse_existing, write_json_atomic
from spana.provenance import build_provenance
from spana.video import VideoSource, sha256_file

SCHEMA = ("spana.class_pass", "1.0.0")


def main(argv: Optional[List[str]] = None) -> int:
    from spana.dataset.config import load_dataset_config
    from spana.detection.rfdetr_local import LocalModelRecord, LocalRFDETRDetector, make_runner
    p = argparse.ArgumentParser()
    for k in ("video", "dataset-config", "model-record", "out", "class-name"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--runtime", choices=("torch", "onnx"), required=True)
    p.add_argument("--device", choices=("mps", "cpu"), required=True)
    a = p.parse_args(argv)
    refuse_existing(a.out)
    dcfg = load_dataset_config(a.dataset_config)
    if a.class_name not in dcfg.classes:
        raise SystemExit(f"--class-name {a.class_name!r} is not a configured class {list(dcfg.classes)}")
    rec = LocalModelRecord.load(a.model_record)
    conf = dcfg.detector.confidence
    det = LocalRFDETRDetector(rec, make_runner(rec, a.runtime, a.device), dcfg.classes, conf)
    cid = dcfg.classes.index(a.class_name)
    video = VideoSource(a.video)
    frames = [{"frame": i, "t": round(t, 6),
               "boxes": [[*(round(v, 1) for v in d.box), round(d.score, 4)] for d in det.detect(img)
                         if d.class_id == cid]}
              for i, t, img in video.frames()]
    doc = {"tool": "tools.ball_pass", "schema": SCHEMA[0], "schema_version": SCHEMA[1],
           "provenance": build_provenance(config_sha256=dcfg.section_sha256("detector"), argv=sys.argv,
                                          inputs={"video": video.meta.sha256, "model_record": sha256_file(a.model_record)}),
           "video": video.meta.to_dict(), "model": det.model_id, "class_name": a.class_name, "confidence": conf,
           "frames": frames}
    print(write_json_atomic(a.out, doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
