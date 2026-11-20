"""DEV comparison (not product code), tracker bench v1: Roboflow `trackers` BoT-SORT on OUR detections, written as a
spana.tracks artifact so the same scorers (tools/benchmark.py, tools/target_score.py) judge it on identical input.

    PYTHONPATH=.:<dir with trackers==2.7.0> python -m tools.dev.botsort_compare run --video CLIP.mp4 \
        --detections DET.json --ref-tracks OUR.trk.json --out BS.trk.json [--config config/botsort_compare.yaml]

`trackers` is a bench-only dependency: it is installed in its own folder and loaded via PYTHONPATH; nothing under
spana/ imports it. BoT-SORT gets every person-view detection of the artifact (class-agnostic, as our tracker uses
them) with the published defaults from config, and the decoded BGR frame for its camera motion compensation.
Conversion (tools/dev/bytetrack_compare.to_tracks, shared with the ByteTrack comparison):
- a BoT-SORT output with a real id (>= 0) = an ACTIVE, OBSERVED row on the exact detection row; -1 rows (unconfirmed
  or unmatched) are not tracks;
- frames between two outputs of a track = LOST (box held): the track is in BoT-SORT's lost pool;
- the track ends (LOST_EXPIRED) on the first frame BoT-SORT no longer holds it, read from the tracker's own track list
  after each update (its removal depends on the frame rate and its time budget, so it is observed, not recomputed);
- identity is always CERTAIN: BoT-SORT never flags an uncertain identity.
Camera status / time are copied from our tracks artifact of the same detections; a camera cut is refused.
Also writes OUT.decisions.jsonl (empty) and the run summary on stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from spana.artifacts import read_json, validate_detections, validate_tracks
from spana.video import VideoSource
from tools.dev.bytetrack_compare import life, to_tracks

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "botsort_compare.yaml"


def load_cfg(path=DEFAULT_CONFIG) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    if c.get("schema") != "spana.botsort_compare_config":
        raise SystemExit(f"{path}: not a spana.botsort_compare_config")
    return c


def make_tracker(cfg: dict, fps: float):
    import trackers
    from importlib.metadata import version
    from trackers import BoTSORTTracker
    found = version("trackers")
    if found != cfg["trackers_version"]:
        raise SystemExit(f"trackers {found} found ({trackers.__file__}), config requires {cfg['trackers_version']}")
    return BoTSORTTracker(frame_rate=float(fps), **cfg["botsort"])


def run_tracker(tracker, dets_doc: dict, frames_bgr, class_agnostic: bool = True
                ) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, int]]:
    """({frame: [(external id, detection index)]}, {external id: frame on which the tracker no longer holds it}).

    frames_bgr yields (frame index, BGR image or None) in the detections' frame order."""
    import supervision as sv
    out: Dict[int, List[Tuple[int, int]]] = {}
    removed: Dict[int, int] = {}
    held: set = set()
    det_frames = dets_doc["frames"]
    for fr, (fi, img) in zip(det_frames, frames_bgr):
        if fi != fr["frame"]:
            raise SystemExit(f"video frame {fi} does not match detections frame {fr['frame']}")
        rows = fr["detections"]
        n = len(rows)
        det = sv.Detections(xyxy=np.array([r[:4] for r in rows], float).reshape(n, 4),
                            confidence=np.array([r[4] for r in rows], float),
                            class_id=np.zeros(n, int) if class_agnostic else np.array([r[5] for r in rows], int),
                            data={"idx": np.arange(n)})
        res = tracker.update(det, frame=img)
        out[fr["frame"]] = ([(int(t), int(i)) for t, i in zip(res.tracker_id, res.data["idx"]) if int(t) >= 0]
                            if len(res) else [])
        now = {int(t.tracker_id) for t in tracker.tracks if int(t.tracker_id) >= 0}
        now |= {e for e, _ in out[fr["frame"]]}
        for e in held - now:
            removed[e] = fr["frame"]
        held = now
    return out, removed


def video_frames(path: str, sha256: str, dets_doc: dict):
    fs = [fr["frame"] for fr in dets_doc["frames"]]
    if fs != list(range(fs[0], fs[0] + len(fs))):
        raise SystemExit("detections are not on consecutive frames: BoT-SORT's motion model expects every frame")
    for i, _t, img in VideoSource(path, expected_sha256=sha256).frames(fs[0], fs[-1] + 1):
        yield i, img


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--video", required=True)
    r.add_argument("--detections", required=True)
    r.add_argument("--ref-tracks", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config", default=str(DEFAULT_CONFIG))
    r.add_argument("--git", default=None, help="commit of this tool, recorded in provenance (tools/benchmark.py reads it)")
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    cfg = load_cfg(a.config)
    dets, ref = read_json(a.detections), read_json(a.ref_tracks)
    validate_detections(dets)
    validate_tracks(ref, dets)
    fps = float(dets["video"]["fps"])
    trk = make_tracker(cfg, fps)
    assoc, removed = run_tracker(trk, dets, video_frames(a.video, dets["video"]["sha256"], dets), cfg["class_agnostic"])
    prov = {"tool": "tools.dev.botsort_compare", "git": a.git, "tracker": f"trackers BoTSORTTracker {cfg['trackers_version']}",
            "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), "botsort": cfg["botsort"],
            "frame_rate": fps, "max_frames_without_update": int(trk.maximum_frames_without_update),
            "reference_tracks_sha256": hashlib.sha256(Path(a.ref_tracks).read_bytes()).hexdigest()}
    doc = to_tracks(dets, ref, assoc, max_lost=0, provenance=prov, removed_at=removed)
    out.write_text(json.dumps(doc, separators=(",", ":")))
    Path(str(out).replace(".trk.json", "") + ".trk.decisions.jsonl").write_text("")
    print(json.dumps({"out": str(out), **life(doc)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
