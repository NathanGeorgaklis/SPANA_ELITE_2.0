"""Re-apply the post-NMS detection rules of the CURRENT config to a cached detections artifact (no model re-run).

Valid only for rules that act on post-NMS boxes (currently: union-box suppression). Refuses if any setting that
acts before or during NMS (scores, NMS, size, tiling) differs between the cached artifact and the config.
The output records the source artifact's sha256 in its provenance.

    python -m tools.refilter_detections --detections IN --config C --out OUT
"""
from __future__ import annotations

import argparse
import dataclasses
import sys

import numpy as np

from spana.artifacts import (ArtifactError, read_json, sha256_doc, validate_detections, write_json_atomic)
from spana.config import load_config
from spana.detection.postprocess import suppress_unions
from spana.provenance import build_provenance

POST_NMS_KEYS = ("union_member_ioa", "union_member_max_iou")


def main() -> int:
    p = argparse.ArgumentParser()
    for k in ("detections", "config", "out"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    src = read_json(a.detections)
    validate_detections(src)
    new_cfg = dataclasses.asdict(cfg.detection)
    pre = {k: v for k, v in new_cfg.items() if k not in POST_NMS_KEYS}
    if {k: v for k, v in src["detection_config"].items() if k not in POST_NMS_KEYS} != pre:
        raise ArtifactError("cached detections differ in pre-NMS settings: re-run the detector instead")
    frames, removed = [], 0
    for fr in src["frames"]:
        rows = fr["detections"]
        if rows:
            b = np.array([r[:4] for r in rows], float)
            s = np.array([r[4] for r in rows], float)
            keep = suppress_unions(b, s, cfg.detection, list(range(len(rows))))
            removed += len(rows) - len(keep)
            rows = [rows[k] for k in keep]
        frames.append({**fr, "detections": rows})
    doc = {**src, "detection_config": new_cfg, "frames": frames,
           "provenance": build_provenance(config_sha256=cfg.sha256(), argv=sys.argv,
                                          inputs={"video": src["video"]["sha256"],
                                                  "refiltered_from_detections": sha256_doc(src)})}
    validate_detections(doc)
    print(write_json_atomic(a.out, doc), f"removed={removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
