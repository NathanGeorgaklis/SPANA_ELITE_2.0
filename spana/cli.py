"""Command-line runner.

  python -m spana.cli detect --video V --config C --model-record M --out detections.json
  python -m spana.cli detect --backend roboflow --video V --config C --dataset-config config/roboflow.yaml \
      --match-id DEV_MATCH_ID --out detections.json        # football detector as a person view (hosted, gated)
  python -m spana.cli track  --video V --config C --detections detections.json --out tracks.json

Detection and tracking are separate passes joined by a versioned artifact, so tracking can be re-run on cached
detections without re-running the model. Every input is hash-checked; nothing falls back to a default.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import List, Optional

from .artifacts import (DETECTIONS_SCHEMA, TRACKS_SCHEMA, ArtifactError, detection_frame, detections_from_frame,
                        read_json, refuse_existing, sha256_doc, tracklet_rows, tracks_frame, validate_detections,
                        validate_tracks,
                        write_json_atomic, write_jsonl_atomic)
from .appearance import describe
from .config import load_config
from .provenance import build_provenance
from .tracking.camera import CameraMotionEstimator
from .tracking.tracker import MultiPlayerTracker
from .video import VideoSource, sha256_file


def person_only_class_names(person_class_id: int) -> List[str]:
    """class_names of a person-only backend: its single class sits at the model's own class index."""
    return [f"unused_{i}" for i in range(person_class_id)] + ["person"]


def _onnx_detector(a: argparse.Namespace, cfg):
    from .detection.onnx_backend import ModelRecord, OnnxPersonDetector
    if not a.model_record:
        raise ArtifactError("--backend onnx needs --model-record")
    rec = ModelRecord.load(a.model_record)
    return (OnnxPersonDetector(rec, cfg.detection), person_only_class_names(rec.person_class_id),
            {"model_record": sha256_file(a.model_record), "weights": rec.sha256})


def _roboflow_detector(a: argparse.Namespace, cfg):
    """The football detector (config/roboflow.yaml `detector`) as a person view. Hosted inference sends frames to a
    third party: the match must be an allow-listed DEV match with third-party processing confirmed."""
    from urllib.parse import urlparse
    from .dataset.config import authorize, load_dataset_config
    from .detection.person_view import PersonView
    from .detection.roboflow_backend import RoboflowDetector
    from .secrets import load_secret
    if not (a.dataset_config and a.match_id):
        raise ArtifactError("--backend roboflow needs --dataset-config and --match-id")
    dcfg = load_dataset_config(a.dataset_config)
    hosted = urlparse(dcfg.detector.api_url).hostname not in dcfg.local_api_hosts
    authorize(dcfg, a.match_id, "dev", third_party=hosted)
    raw = RoboflowDetector.from_detector_config(dcfg, load_secret(dcfg.roboflow.api_key_env),
                                                confidence=cfg.detection.score_low)
    det = PersonView(raw, dcfg.detector.person_classes, cfg.detection)
    return det, list(det.class_names), {"dataset_config_detector": dcfg.section_sha256("detector"),
                                        "match_id": a.match_id}


def cmd_detect(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    refuse_existing(a.out)
    if a.stride < 1:
        raise ArtifactError("--stride must be >= 1")
    det, class_names, model_inputs = (_roboflow_detector if a.backend == "roboflow" else _onnx_detector)(a, cfg)
    video = VideoSource(a.video, expected_sha256=a.video_sha256)
    frames = [detection_frame(i, t, det.detect(img), cfg.output)
              for i, t, img in video.frames(a.start, a.stop) if (i - a.start) % a.stride == 0]
    if not frames:
        raise ArtifactError(f"no frames read from {a.video} in [{a.start}, {a.stop})")
    doc = {
        "schema": DETECTIONS_SCHEMA[0], "schema_version": DETECTIONS_SCHEMA[1],
        "provenance": build_provenance(config_sha256=cfg.sha256(), argv=sys.argv,
                                       inputs={"video": video.meta.sha256, **model_inputs}),
        "video": video.meta.to_dict(), "model": det.model_id, "class_names": class_names,
        "detection_config": dataclasses.asdict(cfg.detection), "frame_stride": a.stride, "frames": frames,
    }
    validate_detections(doc)
    print(write_json_atomic(a.out, doc))
    return 0


def cmd_track(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    decisions_path = Path(a.out).with_suffix(".decisions.jsonl")
    refuse_existing(a.out, decisions_path)
    video = VideoSource(a.video, expected_sha256=a.video_sha256)
    ddoc = read_json(a.detections)
    validate_detections(ddoc)
    if ddoc["video"]["sha256"] != video.meta.sha256:
        raise ArtifactError("detections artifact was produced from a different video")
    if ddoc["detection_config"] != dataclasses.asdict(cfg.detection):
        raise ArtifactError("detections artifact was produced under a different detection config")
    det_frames = {fr["frame"]: fr for fr in ddoc["frames"]}
    cam = CameraMotionEstimator(cfg.camera, video.meta.width, video.meta.height, video.meta.fps)
    trk = MultiPlayerTracker(cfg, video.meta.width, video.meta.height)
    frames, log = [], []
    first, last, stride = min(det_frames), max(det_frames), ddoc["frame_stride"]
    if sorted(det_frames) != list(range(first, last + 1, stride)):
        raise ArtifactError("detections artifact frames are not a contiguous run at its frame_stride")
    for i, t, img in video.frames(first, last + 1):
        if (i - first) % stride:
            continue
        dets = detections_from_frame(det_frames[i])
        boxes = [d.box for d in dets]
        ft = trk.update(i, t, dets, cam.observe(img, boxes), describe(img, boxes, cfg.appearance))
        frames.append(tracks_frame(ft, cfg.output))
        log.extend({"frame": i, **d} for d in ft.decisions)
    if not frames or frames[-1]["frame"] != last:
        raise ArtifactError("video ended before the last detection frame")
    doc = {
        "schema": TRACKS_SCHEMA[0], "schema_version": TRACKS_SCHEMA[1],
        "provenance": build_provenance(config_sha256=cfg.sha256(), argv=sys.argv,
                                       inputs={"video": video.meta.sha256, "detections": sha256_doc(ddoc)}),
        "video": video.meta.to_dict(), "detections_sha256": sha256_doc(ddoc), "frames": frames,
        "tracklets": tracklet_rows(trk.summaries()),
    }
    validate_tracks(doc, ddoc)
    print(write_jsonl_atomic(decisions_path, log))   # log first: a tracks file never exists without its log
    print(write_json_atomic(a.out, doc))
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
    d.add_argument("--backend", choices=("onnx", "roboflow"), default="onnx",
                   help="onnx: pinned person detector (--model-record); roboflow: football detector as a person view "
                        "(--dataset-config, --match-id; licence-gated)")
    d.add_argument("--model-record")
    d.add_argument("--dataset-config")
    d.add_argument("--match-id")
    d.add_argument("--start", type=int, default=0)
    d.add_argument("--stop", type=int, default=None)
    d.add_argument("--stride", type=int, default=1, help="analyse every N-th frame (tracking is time-based)")
    sub.choices["track"].add_argument("--detections", required=True)
    a = p.parse_args(argv)
    return cmd_detect(a) if a.cmd == "detect" else cmd_track(a)


if __name__ == "__main__":
    raise SystemExit(main())
