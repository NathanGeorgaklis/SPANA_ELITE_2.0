"""Command-line runner.

  python -m spana.cli detect --video V --config C --model-record M --out detections.json
  python -m spana.cli track  --video V --config C --detections detections.json --out tracks.json

Detection and tracking are separate passes joined by a versioned artifact, so tracking can be re-run on cached
detections without re-running the model. Every input is hash-checked; nothing falls back to a default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .artifacts import (DETECTIONS_SCHEMA, TRACKS_SCHEMA, ArtifactError, detection_frame, detections_from_frame,
                        read_json, sha256_doc, tracks_frame, validate_detections, validate_tracks,
                        write_json_atomic, write_jsonl_atomic)
from .config import load_config
from .provenance import build_provenance
from .tracking.camera import CameraMotionEstimator
from .tracking.tracker import MultiPlayerTracker
from .video import VideoSource, sha256_file


def cmd_detect(a: argparse.Namespace) -> int:
    from .detection.onnx_backend import ModelRecord, OnnxPersonDetector

    cfg = load_config(a.config)
    video = VideoSource(a.video, expected_sha256=a.video_sha256)
    rec = ModelRecord.load(a.model_record)
    det = OnnxPersonDetector(rec, cfg.detection)
    frames = [detection_frame(i, t, det.detect(img), cfg.output)
              for i, t, img in video.frames(a.start, a.stop)]
    doc = {
        "schema": DETECTIONS_SCHEMA[0], "schema_version": DETECTIONS_SCHEMA[1],
        "provenance": build_provenance(config_sha256=cfg.sha256(), argv=sys.argv,
                                       inputs={"video": video.meta.sha256,
                                               "model_record": sha256_file(a.model_record),
                                               "weights": rec.sha256}),
        "video": video.meta.to_dict(), "model": det.model_id, "frames": frames,
    }
    validate_detections(doc)
    print(write_json_atomic(a.out, doc))
    return 0


def cmd_track(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    video = VideoSource(a.video, expected_sha256=a.video_sha256)
    ddoc = read_json(a.detections)
    validate_detections(ddoc)
    if ddoc["video"]["sha256"] != video.meta.sha256:
        raise ArtifactError("detections artifact was produced from a different video")
    det_frames = {fr["frame"]: fr for fr in ddoc["frames"]}
    cam = CameraMotionEstimator(cfg.camera, video.meta.width, video.meta.height)
    trk = MultiPlayerTracker(cfg, video.meta.width, video.meta.height)
    frames, log = [], []
    first, last = min(det_frames), max(det_frames)
    for i, t, img in video.frames(first, last + 1):
        if i not in det_frames:
            raise ArtifactError(f"frame {i} missing from the detections artifact")
        dets = detections_from_frame(det_frames[i])
        ft = trk.update(i, t, dets, cam.observe(img, [d.box for d in dets]))
        frames.append(tracks_frame(ft, cfg.output))
        log.extend({"frame": i, **d} for d in ft.decisions)
    if frames[-1]["frame"] != last:
        raise ArtifactError("video ended before the last detection frame")
    doc = {
        "schema": TRACKS_SCHEMA[0], "schema_version": TRACKS_SCHEMA[1],
        "provenance": build_provenance(config_sha256=cfg.sha256(), argv=sys.argv,
                                       inputs={"video": video.meta.sha256, "detections": sha256_doc(ddoc)}),
        "video": video.meta.to_dict(), "detections_sha256": sha256_doc(ddoc), "frames": frames,
    }
    validate_tracks(doc)
    print(write_json_atomic(a.out, doc))
    print(write_jsonl_atomic(Path(a.out).with_suffix(".decisions.jsonl"), log))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="spana")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("detect", "track"):
        s = sub.add_parser(name)
        s.add_argument("--video", required=True)
        s.add_argument("--video-sha256", default=None, help="refuse to run if the video hash differs")
        s.add_argument("--config", required=True)
        s.add_argument("--out", required=True)
    d = sub.choices["detect"]
    d.add_argument("--model-record", required=True)
    d.add_argument("--start", type=int, default=0)
    d.add_argument("--stop", type=int, default=None)
    sub.choices["track"].add_argument("--detections", required=True)
    a = p.parse_args(argv)
    return cmd_detect(a) if a.cmd == "detect" else cmd_track(a)


if __name__ == "__main__":
    raise SystemExit(main())
