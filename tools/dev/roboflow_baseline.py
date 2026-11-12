"""DEV ONLY: baseline measurement of the configured pretrained detector (baseline.model_id) on one DEV clip.

    python -m tools.dev.roboflow_baseline --video DEV_CLIP.mp4 --match-id ID [--config config/roboflow.yaml]

* Runs only on the split in the config's `split` field, which the loader restricts to "dev"; the match must be in
  licensed_matches with split dev. EVAL is never touched.
* Frames: the same deterministic sampling as ingestion (sampling section).
* A hosted api_url sends frames to a third party, so it also needs third_party_processing_confirmed for the match;
  an api_url on local_api_hosts (local inference server) does not.
* No threshold is tuned here: confidence / iou are the config priors, recorded in the output.

Outputs under paths.baseline_dir/<match_id>/<model>__<video sha[:12]>/ (gitignored run outputs):
  detections.json  cached per-frame detections (box, class, score); reused when model, thresholds, video and
                   sampling are identical; a different request never overwrites it
  summary.json / summary.md   detections per class per frame, share of frames with each class (incl. ball),
                   confidence histograms, model labels dropped by class_map
  samples/*.jpg    a few annotated frames for visual review
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from spana.artifacts import write_json_atomic
from spana.dataset.config import DatasetConfig, PolicyError, authorize, load_dataset_config
from spana.dataset.sampling import extract_frames, scan_signals, select_frames
from spana.detection.roboflow_backend import RoboflowDetector
from spana.provenance import build_provenance
from spana.secrets import load_secret
from spana.video import VideoSource

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
DETECTIONS_SCHEMA = "spana.baseline_detections"
SCHEMA_VERSION = "1.0.0"
# BGR colours by class index (display only)
_PALETTE = [(60, 200, 60), (0, 165, 255), (230, 60, 230), (0, 255, 255), (255, 128, 0), (128, 128, 255)]


def _cache_key(cfg: DatasetConfig, video_sha: str, selected) -> dict:
    return {"video_sha256": video_sha, "model_id": cfg.baseline.model_id, "api_url": cfg.baseline.api_url,
            "confidence": cfg.baseline.confidence, "iou": cfg.baseline.iou, "classes": list(cfg.classes),
            "class_map": list(cfg.baseline.class_map), "sampling_sha256": cfg.section_sha256("sampling"),
            "frame_indices": [s.frame_index for s in selected]}


def summarize_detections(frames: List[dict], classes: Sequence[str], bins: int, dropped: dict) -> dict:
    """Pure summary of per-frame detections (see module docstring)."""
    n = len(frames)
    edges = np.linspace(0.0, 1.0, bins + 1)
    per_frame = [{"frame_index": f["frame_index"], "timestamp_s": f["timestamp_s"],
                  "counts": {c: sum(d["class"] == c for d in f["detections"]) for c in classes}} for f in frames]
    per_class = {}
    for c in classes:
        counts = [pf["counts"][c] for pf in per_frame]
        scores = [d["score"] for f in frames for d in f["detections"] if d["class"] == c]
        hist, _ = np.histogram(scores, bins=edges)
        per_class[c] = {
            "total": int(sum(counts)),
            "mean_per_frame": round(sum(counts) / n, 4) if n else None,
            "median_per_frame": statistics.median(counts) if n else None,
            "max_per_frame": max(counts) if n else None,
            "frame_share": round(sum(x > 0 for x in counts) / n, 4) if n else None,
            "confidence_histogram": [int(h) for h in hist],
        }
    all_scores = [d["score"] for f in frames for d in f["detections"]]
    return {"n_frames": n, "bin_edges": [round(float(e), 6) for e in edges], "per_class": per_class,
            "confidence_histogram_all": [int(h) for h in np.histogram(all_scores, bins=edges)[0]],
            "dropped_model_labels": dict(sorted(dropped.items())), "per_frame": per_frame}


def summary_markdown(summary: dict, key: dict) -> str:
    lines = [f"# Baseline detections ({key['model_id']}, DEV)", "",
             f"conf {key['confidence']} / iou {key['iou']} (untuned priors); {summary['n_frames']} sampled frames; "
             f"video sha256 {key['video_sha256'][:12]}", "",
             "| class | total | mean / frame | median | max | share of frames with >= 1 |",
             "|---|---|---|---|---|---|"]
    for c, s in summary["per_class"].items():
        lines.append(f"| {c} | {s['total']} | {s['mean_per_frame']} | {s['median_per_frame']} | "
                     f"{s['max_per_frame']} | {s['frame_share']} |")
    edges = summary["bin_edges"]
    lines += ["", "Confidence histogram (all classes):", "", "| bin | count |", "|---|---|"]
    for i, h in enumerate(summary["confidence_histogram_all"]):
        lines.append(f"| {edges[i]:.1f}-{edges[i + 1]:.1f} | {h} |")
    lines += ["", f"Model labels dropped by class_map: {summary['dropped_model_labels'] or 'none'}", ""]
    return "\n".join(lines)


def render(frame: np.ndarray, dets: List[dict]) -> np.ndarray:
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        col = _PALETTE[d["class_id"] % len(_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{d['class']} {d['score']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)
    return img


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"{path} exists with different content; refusing to overwrite")
        return
    path.write_text(text)


def run_baseline(cfg: DatasetConfig, *, video: str, match_id: str, detector=None, api_key: Optional[str] = None,
                 root: Path = REPO, argv: Sequence[str] = ()) -> Path:
    authorize(cfg, match_id, cfg.split, third_party=not cfg.is_local_api())
    src = VideoSource(video)
    selected = select_frames(scan_signals(src.frames(), cfg.sampling), cfg.sampling)
    key = _cache_key(cfg, src.meta.sha256, selected)
    model_slug = cfg.baseline.model_id.replace("/", "_")
    out = root / cfg.paths.baseline_dir / match_id / f"{model_slug}__{src.meta.sha256[:12]}"
    det_path = out / "detections.json"

    if det_path.exists():
        doc = json.loads(det_path.read_text())
        if doc.get("key") != key:
            raise RuntimeError(f"{det_path} was produced with a different model / thresholds / sampling; "
                               "refusing to overwrite (use another paths.baseline_dir)")
    else:
        if detector is None:
            detector = RoboflowDetector.from_config(cfg, api_key or load_secret(cfg.roboflow.api_key_env))
        names = detector.class_names
        frames = []
        for s, frame in extract_frames(src.frames(), selected):
            dets = detector.detect(frame)
            frames.append({"frame_index": s.frame_index, "timestamp_s": round(s.timestamp_s, 6),
                           "reasons": list(s.reasons),
                           "detections": [{"box": [round(v, 2) for v in d.box], "class": names[d.class_id],
                                           "class_id": d.class_id, "score": round(d.score, 4)} for d in dets]})
        if len(frames) != len(selected):
            raise RuntimeError(f"detected {len(frames)} of {len(selected)} selected frames (decode stopped early)")
        doc = {"schema": DETECTIONS_SCHEMA, "schema_version": SCHEMA_VERSION, "split": cfg.split,
               "match_id": match_id, "key": key, "dropped_model_labels": dict(getattr(detector, "dropped_labels", {})),
               "frames": frames,
               "provenance": build_provenance(config_sha256=cfg.section_sha256("baseline"),
                                              inputs={"video_sha256": src.meta.sha256},
                                              argv=[Path(a).name if a == video else a for a in argv])}
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(det_path, doc)

    summary = summarize_detections(doc["frames"], cfg.classes, cfg.baseline.histogram_bins,
                                   doc["dropped_model_labels"])
    _write_once(out / "summary.json", json.dumps(summary, indent=1, sort_keys=True) + "\n")
    _write_once(out / "summary.md", summary_markdown(summary, key))

    k = min(cfg.baseline.render_samples, len(doc["frames"]))
    if k:
        picks = sorted({int(round(i * (len(doc["frames"]) - 1) / max(1, k - 1))) for i in range(k)})
        by_idx = {doc["frames"][i]["frame_index"]: doc["frames"][i] for i in picks}
        sdir = out / "samples"
        sdir.mkdir(exist_ok=True)
        todo = [s for s in selected if s.frame_index in by_idx
                and not (sdir / f"f{s.frame_index:07d}.jpg").exists()]
        for s, frame in extract_frames(src.frames(), todo):
            cv2.imwrite(str(sdir / f"f{s.frame_index:07d}.jpg"), render(frame, by_idx[s.frame_index]["detections"]))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    try:
        out = run_baseline(cfg, video=a.video, match_id=a.match_id, argv=argv)
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print(f"baseline outputs: {out}  ({datetime.now(timezone.utc).isoformat(timespec='seconds')})")
    print((out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
