"""Merge detections artifacts produced in consecutive frame ranges of the SAME video (e.g. a long window detected
in chunks) into one artifact. Refuses parts that differ in video, model, detection config or stride, or whose
frames do not continue each other exactly at the stride.

    python -m tools.merge_detections --out MERGED.det.json PART0.det.json PART1.det.json ...
"""
from __future__ import annotations

import argparse
import sys

from spana.artifacts import ArtifactError, read_json, sha256_doc, validate_detections, write_json_atomic

SAME = ("schema", "schema_version", "video", "model", "detection_config", "frame_stride")


def merge(parts):
    docs = [read_json(p) for p in parts]
    for d in docs:
        validate_detections(d)
    base = docs[0]
    for p, d in zip(parts[1:], docs[1:]):
        for k in SAME:
            if d[k] != base[k]:
                raise ArtifactError(f"{p}: {k} differs from {parts[0]}")
    frames = []
    for p, d in zip(parts, docs):
        if frames and d["frames"][0]["frame"] != frames[-1]["frame"] + base["frame_stride"]:
            raise ArtifactError(f"{p}: does not continue the previous part at stride {base['frame_stride']}")
        if frames and d["frames"][0]["t"] <= frames[-1]["t"]:
            raise ArtifactError(f"{p}: time does not increase across the part boundary")
        frames.extend(d["frames"])
    out = {**base, "frames": frames,
           "provenance": {**base["provenance"], "merged_from": [{"path": str(p), "sha256": sha256_doc(d),
                                                                 "provenance": d["provenance"]}
                                                                for p, d in zip(parts, docs)]}}
    validate_detections(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("parts", nargs="+")
    a = ap.parse_args()
    try:
        print(write_json_atomic(a.out, merge(a.parts)))
    except ArtifactError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
