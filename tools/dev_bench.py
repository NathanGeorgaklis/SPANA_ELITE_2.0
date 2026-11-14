"""DEV generalisation benchmark: sample one DEV match, upload to the separate benchmark project, pre-label with the
trained detector. Evaluation only: no model, threshold or training-data change.

    python -m tools.dev_bench --match-id DEV_MATCH_ID --video CLIP_A.mp4 [--video CLIP_B.mp4 ...] [--dry-run]
        [--config config/roboflow.yaml]

Give every local clip of the match, in match order. Fails closed before any video is read:
  * the match must be on licensed_matches with split dev (EVAL, TRAIN and unlisted matches are refused);
  * a real run needs third_party_processing_confirmed for the match (upload + hosted inference);
  * an earlier benchmark manifest for the match is refused (one sample per match and bench tag).

Selection (deterministic, no randomness):
  1. per clip, the `sampling` rules (regular frames, motion / cut frames, per-video cap), as for the training data;
  2. frames below prelabel.blur.laplacian_var_min are dropped (same blur rule);
  3. near-duplicates (64-bit pHash within audit.phash_hamming_max bits) are dropped, the earliest frame kept;
  4. dev_bench.frames_per_match of what is left, evenly spread over the whole match (clip order, then time).
Fewer usable frames than dev_bench.frames_per_match: refused, nothing is uploaded.

Upload goes to dev_bench.project only (never roboflow.project), tagged match:<id>, split:dev and dev_bench.bench_tag.
Each frame is then pre-labelled with detector.model_id at detector.confidence / detector.iou, with no grass or size
filter (the benchmark has to see the detector's own errors). The annotation is saved through the licence / review
write guard and read back; only then is dev_bench.prelabel_tag set. A frame tagged review.reviewed_tag is never
written to; a refusal by the write guard stops the run (the manifest is still written). --dry-run does steps 1-4 and writes the manifest under paths.dev_bench_frames_dir; nothing leaves this
machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from spana.artifacts import write_json_atomic
from spana.dataset.config import DatasetConfig, PolicyError, authorize, load_dataset_config
from spana.dataset.guard import GuardedProject
from spana.dataset.manifest import frame_tags
from spana.dataset.phash import near_duplicate_groups, phash, phash_hex
from spana.dataset.prefilter import blur_score
from spana.dataset.prep import evenly_spaced
from spana.dataset.rf_project import ProjectImage, RoboflowApiError, RoboflowProject
from spana.dataset.sampling import extract_frames, scan_signals, select_frames
from spana.provenance import build_provenance
from spana.secrets import load_secret, redact
from spana.video import VideoSource
from tools.roboflow_prelabel import _verify_saved, yolo_annotation

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
BENCH_SCHEMA = "spana.dev_bench"
BENCH_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class Candidate:
    clip_index: int            # position of the clip in the match (order of --video)
    frame_index: int
    timestamp_s: float
    reasons: Tuple[str, ...]
    image_file: str
    image_sha256: str
    blur: float                # variance of Laplacian (prelabel.blur)
    phash: int

    @property
    def key(self) -> Tuple[int, int]:
        return (self.clip_index, self.frame_index)

    def to_dict(self) -> dict:
        return {"clip_index": self.clip_index, "frame_index": self.frame_index,
                "timestamp_s": round(self.timestamp_s, 6), "reasons": list(self.reasons),
                "image_file": self.image_file, "image_sha256": self.image_sha256, "blur": round(self.blur, 3),
                "phash": phash_hex(self.phash)}


# ------------------------------------------------------------------------------------------------- selection ----
def select_bench(cfg: DatasetConfig, candidates: Sequence[Candidate]) -> Tuple[List[Candidate], dict]:
    """Pure. -> (selected frames in match order, {"clip:frame": exclusion reason} for every frame not kept by a rule).
    Raises PolicyError when fewer than dev_bench.frames_per_match frames survive the filters."""
    order = sorted(candidates, key=lambda c: c.key)
    excluded = {}
    sharp = []
    for c in order:
        if c.blur < cfg.prelabel.blur.laplacian_var_min:
            excluded[f"{c.clip_index}:{c.frame_index}"] = "blur"
        else:
            sharp.append(c)
    groups = near_duplicate_groups([(f"{c.clip_index}:{c.frame_index}", c.key, c.phash) for c in sharp],
                                   cfg.audit.phash_hamming_max)
    for g in groups:
        for e in g["exclude"]:
            excluded[e] = f"near-duplicate of {g['keep']}"
    usable = [c for c in sharp if f"{c.clip_index}:{c.frame_index}" not in excluded]
    n = cfg.dev_bench.frames_per_match
    if len(usable) < n:
        raise PolicyError(f"refused: only {len(usable)} usable frames after the blur and near-duplicate rules, "
                          f"dev_bench.frames_per_match is {n}; nothing was uploaded")
    return evenly_spaced(usable, n), excluded


def sample_clip(cfg: DatasetConfig, video: str, clip_index: int, match_id: str, out_dir: Path):
    """Sampling rules on one clip; candidate JPEGs written to out_dir. -> (VideoMeta, candidates, n_analysis)."""
    src = VideoSource(video)
    signals = scan_signals(src.frames(), cfg.sampling)
    selected = select_frames(signals, cfg.sampling)
    out = []
    for s, frame in extract_frames(src.frames(), selected):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.sampling.jpeg_quality])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for clip {clip_index} frame {s.frame_index}")
        data = buf.tobytes()
        name = f"{match_id}_c{clip_index:02d}_f{s.frame_index:07d}.jpg"
        (out_dir / name).write_bytes(data)
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # scored as the uploaded bytes
        out.append(Candidate(clip_index, s.frame_index, s.timestamp_s, s.reasons, name,
                             hashlib.sha256(data).hexdigest(), blur_score(decoded, cfg.prelabel.blur.analysis_width_px),
                             phash(decoded)))
    if len(out) != len(selected):
        raise RuntimeError(f"clip {clip_index}: extracted {len(out)} of {len(selected)} frames (decode stopped early)")
    return src.meta, out, len(signals)


# ------------------------------------------------------------------------------------------- remote clients ----
def open_bench_project(cfg: DatasetConfig, api_key: str):
    """(uploader, project client) for dev_bench.project. Fails closed unless it exists with the expected type."""
    from spana.dataset.upload import RoboflowUploader
    r, b = cfg.roboflow, cfg.dev_bench
    up = RoboflowUploader(api_url=r.api_url, workspace=r.workspace, project=b.project, api_key=api_key,
                          timeout_s=r.upload_timeout_s, rate_limit_retries=r.rate_limit_retries,
                          rate_limit_wait_s=r.rate_limit_wait_s, rate_limit_max_wait_s=r.rate_limit_max_wait_s)
    up.check_project(r.project_type)
    proj = RoboflowProject(api_url=r.api_url, workspace=r.workspace, project=b.project, api_key=api_key,
                           timeout_s=cfg.audit.fetch_timeout_s, rate_limit_retries=r.rate_limit_retries,
                           rate_limit_wait_s=r.rate_limit_wait_s, rate_limit_max_wait_s=r.rate_limit_max_wait_s)
    return up, proj


def boxes_of(detector, dets) -> List[dict]:
    return [{"class": detector.class_names[d.class_id], "class_id": d.class_id, "box": list(d.box),
             "score": float(d.score)} for d in dets]


def upload_and_prelabel(cfg: DatasetConfig, *, match_id: str, frames_dir: Path, chosen: Sequence[Candidate],
                        clips: Sequence[dict], uploader, project, detector, secret: str,
                        sleep: Callable[[float], None] = time.sleep) -> List[dict]:
    """One entry per chosen frame: upload id, status, per-class pre-label counts, error. Never raises per frame."""
    b = cfg.dev_bench
    tags = frame_tags(match_id, "dev") + [b.bench_tag]
    batch = f"bench_{match_id}"
    entries = []
    for k, c in enumerate(chosen):
        if k and b.min_interval_s:
            sleep(b.min_interval_s)
        e = {**c.to_dict(), "upload_id": None, "status": "failed", "prelabel": None, "error": None}
        entries.append(e)
        path = frames_dir / c.image_file
        meta = {"match_id": match_id, "split": "dev", "clip_index": c.clip_index,
                "clip_file": clips[c.clip_index]["file_name"], "frame_index": c.frame_index,
                "timestamp_s": round(c.timestamp_s, 6), "video_sha256": clips[c.clip_index]["sha256"],
                "image_sha256": c.image_sha256, "bench": b.bench_tag}
        try:
            res = uploader.single_upload(image_path=str(path), split=b.roboflow_split, batch_name=batch,
                                         tag_names=tags, metadata=meta, num_retry_uploads=cfg.roboflow.upload_retries)
            uid = str(((res or {}).get("image") or {}).get("id") or "")
            if not uid:
                raise RoboflowApiError(f"no image id in upload response: {str(res)[:200]}")
            e["upload_id"] = uid
            e["status"] = "duplicate" if res["image"].get("duplicate") else "uploaded"
            frame = cv2.imread(str(path))
            h, w = frame.shape[:2]
            pimg = ProjectImage(id=uid, name=c.image_file, split=b.roboflow_split, in_dataset=False,
                                tags=tuple(sorted(tags)), metadata=meta, width=w, height=h, url=None)
            guard = GuardedProject(cfg, project, [pimg])
            cur = guard.refresh(uid)                    # fresh tags: licence + review gate on the server's state
            if b.prelabel_tag in cur.tags:
                e["prelabel"] = "kept (already pre-labelled)"
                continue
            boxes = boxes_of(detector, detector.detect(frame))
            if boxes:
                content, labelmap = yolo_annotation(cfg, boxes, w, h)
                res = guard.save_annotation(uid, name=Path(c.image_file).stem + ".txt", content=content,
                                            labelmap=labelmap, add_to_dataset=b.add_to_dataset)
                if res != "saved":
                    raise RoboflowApiError(f"annotation not saved ({res}); frame left without the pre-label tag")
                _verify_saved(guard, uid, boxes)
            guard.update_tags(uid, add=[b.prelabel_tag])
            e["prelabel"] = dict(sorted(Counter(x["class"] for x in boxes).items()))
        except PolicyError as ex:   # the write guard refused: record it, write nothing more in this run
            e["error"] = f"refused: {ex}"
            entries += [{**r.to_dict(), "upload_id": None, "status": "not_uploaded", "prelabel": None,
                         "error": "not attempted (run stopped by a refusal)"} for r in chosen[k + 1:]]
            break
        except Exception as ex:   # per-frame failure: recorded, run continues; texts may carry the key
            e["error"] = redact(f"{type(ex).__name__}: {ex}", secret)[:500]
    return entries


# ------------------------------------------------------------------------------------------------------ run -----
def prior_manifests(cfg: DatasetConfig, root: Path, match_id: str) -> List[str]:
    d = root / cfg.paths.dev_bench_dir
    out = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        doc = json.loads(p.read_text())
        if doc.get("match_id") == match_id and doc.get("bench_tag") == cfg.dev_bench.bench_tag:
            out.append(p.name)
    return out


def run_bench(cfg: DatasetConfig, *, match_id: str, videos: Sequence[str], dry_run: bool, root: Path = REPO,
              uploader=None, project=None, detector=None, api_key: Optional[str] = None,
              argv: Sequence[str] = (), now: Optional[datetime] = None, sleep=time.sleep) -> Path:
    if not videos:
        raise PolicyError("refused: give every local clip of the match with --video")
    entry = authorize(cfg, match_id, "dev", third_party=not dry_run)   # a real run uploads: always third-party
    prior = prior_manifests(cfg, root, match_id)
    if prior and not dry_run:
        raise PolicyError(f"refused: {match_id} already has benchmark manifest(s) {prior} for "
                          f"{cfg.dev_bench.bench_tag!r}; the sample is not redrawn")
    if not dry_run and not cfg.detector.model_id:
        raise PolicyError("refused: detector.model_id is empty (no trained detector to pre-label with)")
    if not dry_run and (uploader is None or project is None or detector is None):
        from spana.detection.roboflow_backend import RoboflowDetector
        api_key = api_key or load_secret(cfg.roboflow.api_key_env)
        if uploader is None or project is None:
            uploader, project = open_bench_project(cfg, api_key)
        detector = detector or RoboflowDetector.from_detector_config(cfg, api_key)

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    frames_dir = root / cfg.paths.dev_bench_frames_dir / match_id / stamp
    frames_dir.mkdir(parents=True, exist_ok=False)
    clips, cands, n_analysis = [], [], 0
    for ci, v in enumerate(videos):
        meta, cs, na = sample_clip(cfg, v, ci, match_id, frames_dir)
        clips.append({"clip_index": ci, "file_name": Path(v).name, "sha256": meta.sha256, "fps": meta.fps,
                      "width": meta.width, "height": meta.height, "n_candidates": len(cs)})
        cands += cs
        n_analysis += na
    if len({c["sha256"] for c in clips}) != len(clips):
        raise PolicyError("refused: the same clip was given twice")
    chosen, excluded = select_bench(cfg, cands)

    if dry_run:
        frames = [{**c.to_dict(), "upload_id": None, "status": "not_uploaded", "prelabel": None, "error": None}
                  for c in chosen]
    else:
        frames = upload_and_prelabel(cfg, match_id=match_id, frames_dir=frames_dir, chosen=chosen, clips=clips,
                                     uploader=uploader, project=project, detector=detector, secret=api_key or "",
                                     sleep=sleep)
    sections = ("sampling", "dev_bench", "detector")
    doc = {
        "schema": BENCH_SCHEMA, "schema_version": BENCH_SCHEMA_VERSION,
        "created_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "dry_run": dry_run,
        "match_id": match_id, "split": "dev", "permission": entry.permission,
        "workspace": cfg.roboflow.workspace, "project": cfg.dev_bench.project, "bench_tag": cfg.dev_bench.bench_tag,
        "prelabel_tag": cfg.dev_bench.prelabel_tag,
        "detector": {"model_id": cfg.detector.model_id, "confidence": cfg.detector.confidence,
                     "iou": cfg.detector.iou},
        "rules": {"blur_laplacian_var_min": cfg.prelabel.blur.laplacian_var_min,
                  "phash_hamming_max": cfg.audit.phash_hamming_max,
                  "frames_per_match": cfg.dev_bench.frames_per_match},
        "config_sha256": {s: cfg.section_sha256(s) for s in sections},
        "clips": clips, "n_analysis_frames": n_analysis, "n_candidates": len(cands),
        "excluded": dict(sorted(Counter("blur" if r == "blur" else "near-duplicate"
                                        for r in excluded.values()).items())),
        "excluded_frames": excluded, "frames": frames,
        "summary": {"n_frames": len(frames), "status": dict(sorted(Counter(f["status"] for f in frames).items())),
                    "n_errors": sum(1 for f in frames if f["error"]),
                    "prelabel_boxes": dict(sorted(sum((Counter(f["prelabel"]) for f in frames
                                                       if isinstance(f["prelabel"], dict)), Counter()).items()))},
        "provenance": build_provenance(config_sha256=cfg.section_sha256("dev_bench"),
                                       inputs={f"clip_{c['clip_index']}": c["sha256"] for c in clips},
                                       argv=[Path(a).name if a in videos else a for a in argv]),
    }
    out_dir = frames_dir if dry_run else root / cfg.paths.dev_bench_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{match_id}__{cfg.dev_bench.bench_tag.replace(':', '_')}__{stamp}{'__dryrun' if dry_run else ''}.json"
    write_json_atomic(out, doc)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--video", action="append", default=[], help="a clip of the match; repeat, in match order")
    ap.add_argument("--dry-run", action="store_true", help="sample and select only; nothing leaves this machine")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    try:
        out = run_bench(cfg, match_id=a.match_id, videos=a.video, dry_run=a.dry_run, argv=argv)
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    doc = json.loads(out.read_text())
    print(f"manifest {out}\n{json.dumps(doc['summary'], sort_keys=True)}  excluded {doc['excluded']}")
    return 0 if doc["summary"]["n_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
