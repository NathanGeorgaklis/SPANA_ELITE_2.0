"""Upload manifests: one JSON per ingestion run, tracing every frame to its source video, time and upload.

Contains no pixels and no secrets. Validated before it is written; written atomically and never overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

MANIFEST_SCHEMA = "spana.dataset_upload"
MANIFEST_SCHEMA_VERSION = "1.0.0"
FRAME_STATUSES = ("uploaded", "duplicate", "failed", "not_uploaded")
_REASONS = {"interval", "motion", "cut"}


class ManifestError(ValueError):
    pass


def frame_tags(match_id: str, split: str) -> List[str]:
    return [f"match:{match_id}", f"split:{split}"]


def frame_entry(*, frame_index: int, timestamp_s: float, reasons, image_file: str, image_sha256: str,
                tags: List[str], upload_id: Optional[str], status: str, error: Optional[str] = None) -> dict:
    return {"frame_index": int(frame_index), "timestamp_s": round(float(timestamp_s), 6), "reasons": list(reasons),
            "image_file": image_file, "image_sha256": image_sha256, "tags": list(tags), "upload_id": upload_id,
            "status": status, "error": error}


def summarize(frames: List[dict]) -> dict:
    return {"n_frames": len(frames), **{f"n_{s}": sum(f["status"] == s for f in frames) for s in FRAME_STATUSES}}


def validate_manifest(doc: dict) -> None:
    """Structure + policy checks. Raises ManifestError."""
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ManifestError(msg)

    need(isinstance(doc, dict), "manifest must be an object")
    need(doc.get("schema") == MANIFEST_SCHEMA and doc.get("schema_version") == MANIFEST_SCHEMA_VERSION,
         "wrong schema / schema_version")
    for k in ("created_utc", "match_id", "split", "roboflow_split", "workspace", "project", "dry_run", "tags",
              "permission", "video", "sampling", "frames", "summary", "provenance"):
        need(k in doc, f"missing key {k!r}")
    need(doc["split"] in ("train", "dev"), f"split {doc['split']!r} cannot be in an upload manifest")
    need(doc["tags"] == frame_tags(doc["match_id"], doc["split"]), "tags must be [match:<id>, split:<split>]")
    v = doc["video"]
    for k in ("file_name", "sha256", "fps", "width", "height"):
        need(k in v, f"video.{k} missing")
    need(isinstance(v["sha256"], str) and len(v["sha256"]) == 64, "video.sha256 must be a sha256 hex digest")
    need("config" in doc["sampling"] and "config_sha256" in doc["sampling"], "sampling.config / config_sha256 missing")
    frames = doc["frames"]
    need(isinstance(frames, list), "frames must be a list")
    prev = -1
    for f in frames:
        for k in ("frame_index", "timestamp_s", "reasons", "image_file", "image_sha256", "tags", "upload_id",
                  "status", "error"):
            need(k in f, f"frame entry missing {k!r}")
        need(f["frame_index"] > prev, "frame indices must strictly increase")
        prev = f["frame_index"]
        need(f["timestamp_s"] >= 0, "negative timestamp")
        need(bool(f["reasons"]) and set(f["reasons"]) <= _REASONS, f"bad reasons {f['reasons']!r}")
        need(f["tags"] == doc["tags"], "every frame carries the manifest's tags (one match, one split)")
        need(f["status"] in FRAME_STATUSES, f"bad status {f['status']!r}")
        need((f["upload_id"] is not None) == (f["status"] in ("uploaded", "duplicate")),
             "upload_id is present exactly for uploaded / duplicate frames")
        need(not doc["dry_run"] or f["status"] == "not_uploaded", "a dry run uploads nothing")
    need(doc["summary"] == summarize(frames), "summary does not match frames")


def prior_splits(manifest_dir: Path, match_id: str) -> set:
    """Splits that earlier manifests in `manifest_dir` recorded for `match_id` (guards against a registry edit)."""
    out = set()
    d = Path(manifest_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ManifestError(f"unreadable manifest {p}: {e}") from e
        if doc.get("schema") == MANIFEST_SCHEMA and doc.get("match_id") == match_id:
            out.add(doc.get("split"))
    return out
