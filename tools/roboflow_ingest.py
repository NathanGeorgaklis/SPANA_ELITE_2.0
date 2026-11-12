"""Sample frames from one video and upload them to the Roboflow detection project, with a traceable manifest.

    python -m tools.roboflow_ingest --video MATCH.mp4 --match-id ID --split {train,dev,eval} [--dry-run]
        [--config config/roboflow.yaml]

Fails closed, before the video is read:
  * split=eval is always refused (EVAL is never uploaded);
  * the match must be in licensed_matches with exactly this split (one match -> one split, no per-image split);
  * an earlier manifest recording this match under another split is refused;
  * a real upload needs third_party_processing_confirmed: true for the match.
--dry-run samples and extracts frames and writes the manifest under paths.frames_dir, uploading nothing.

Frames are tagged match:<id> and split:<split>, carry metadata (match, split, frame index, time, video sha256) and
all go to the one Roboflow split mapped from --split. The manifest (paths.manifest_dir) records, per frame: index,
timestamp, sampling reasons, image sha256, tags, upload id and status. The API key is read from the environment / .env
(roboflow.api_key_env) and is never printed or stored.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import cv2

from spana.artifacts import write_json_atomic
from spana.dataset.config import DatasetConfig, PolicyError, SPLITS, authorize, load_dataset_config
from spana.dataset.manifest import (MANIFEST_SCHEMA, MANIFEST_SCHEMA_VERSION, frame_entry, frame_tags, prior_splits,
                                    summarize, validate_manifest)
from spana.dataset.sampling import extract_frames, scan_signals, select_frames
from spana.provenance import build_provenance
from spana.secrets import load_secret, redact
from spana.video import VideoSource

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"


def open_project(cfg: DatasetConfig, api_key: str):
    """Upload client for the configured project; fails closed unless the key can read it and its type matches."""
    from spana.dataset.upload import RoboflowUploader
    r = cfg.roboflow
    up = RoboflowUploader(api_url=r.api_url, workspace=r.workspace, project=r.project, api_key=api_key,
                          timeout_s=r.upload_timeout_s, rate_limit_retries=r.rate_limit_retries,
                          rate_limit_wait_s=r.rate_limit_wait_s)
    up.check_project(r.project_type)
    return up


def upload_one(project, image_path: Path, *, roboflow_split: str, batch_name: str, tags, metadata: dict,
               retries: int, secret: str):
    """-> (upload_id | None, status, error | None). Never raises for a per-image failure."""
    try:
        res = project.single_upload(image_path=str(image_path), split=roboflow_split, batch_name=batch_name,
                                    tag_names=list(tags), metadata=metadata, num_retry_uploads=retries)
    except Exception as e:  # SDK errors can include request URLs with the key
        return None, "failed", redact(f"{type(e).__name__}: {e}", secret)[:500]
    image = (res or {}).get("image") or {}
    uid = image.get("id")
    if not uid:
        return None, "failed", redact(f"no image id in response: {str(res)[:300]}", secret)
    return str(uid), ("duplicate" if image.get("duplicate") else "uploaded"), None


def run_ingest(cfg: DatasetConfig, *, video: str, match_id: str, split: str, dry_run: bool,
               project=None, api_key: Optional[str] = None, root: Path = REPO,
               argv: Sequence[str] = (), now: Optional[datetime] = None) -> Path:
    """Returns the manifest path. Raises PolicyError before touching the video when the request is not allowed."""
    entry = authorize(cfg, match_id, split, third_party=not dry_run)
    manifest_dir = root / cfg.paths.manifest_dir
    other = prior_splits(manifest_dir, match_id) - {split}
    if other:
        raise PolicyError(f"refused: earlier manifests put {match_id!r} in split(s) {sorted(other)}; "
                          "a match never spans splits")
    if not dry_run and project is None:
        api_key = api_key or load_secret(cfg.roboflow.api_key_env)
        project = open_project(cfg, api_key)

    src = VideoSource(video)
    signals = scan_signals(src.frames(), cfg.sampling)
    selected = select_frames(signals, cfg.sampling)

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    frames_out = root / cfg.paths.frames_dir / match_id / stamp
    frames_out.mkdir(parents=True, exist_ok=False)
    tags = frame_tags(match_id, split)
    rf_split = cfg.roboflow_split(split)
    batch = f"spana_{match_id}_{split}"
    entries = []
    for s, frame in extract_frames(src.frames(), selected):
        name = f"{match_id}_f{s.frame_index:07d}.jpg"
        path = frames_out / name
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.sampling.jpeg_quality])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for frame {s.frame_index}")
        data = buf.tobytes()
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        if dry_run:
            uid, status, err = None, "not_uploaded", None
        else:
            meta = {"match_id": match_id, "split": split, "frame_index": s.frame_index,
                    "timestamp_s": round(s.timestamp_s, 6), "video_sha256": src.meta.sha256, "image_sha256": sha}
            uid, status, err = upload_one(project, path, roboflow_split=rf_split, batch_name=batch, tags=tags,
                                          metadata=meta, retries=cfg.roboflow.upload_retries, secret=api_key or "")
        entries.append(frame_entry(frame_index=s.frame_index, timestamp_s=s.timestamp_s, reasons=s.reasons,
                                   image_file=name, image_sha256=sha, tags=tags, upload_id=uid, status=status,
                                   error=err))
    if len(entries) != len(selected):
        raise RuntimeError(f"extracted {len(entries)} of {len(selected)} selected frames (decode stopped early)")

    safe_argv = [Path(a).name if a == video else a for a in argv]   # no local directory names in records
    doc = {
        "schema": MANIFEST_SCHEMA, "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "dry_run": dry_run, "match_id": match_id, "split": split, "roboflow_split": rf_split,
        "workspace": cfg.roboflow.workspace, "project": cfg.roboflow.project, "batch_name": batch,
        "tags": tags, "permission": entry.permission,
        "video": {"file_name": Path(video).name, "sha256": src.meta.sha256, "fps": src.meta.fps,
                  "width": src.meta.width, "height": src.meta.height},
        "sampling": {"config": cfg.to_dict()["sampling"], "config_sha256": cfg.section_sha256("sampling"),
                     "n_analysis_frames": len(signals), "n_selected": len(selected)},
        "frames": entries, "summary": summarize(entries),
        "provenance": build_provenance(config_sha256=cfg.section_sha256("sampling"),
                                       inputs={"video_sha256": src.meta.sha256}, argv=safe_argv),
    }
    validate_manifest(doc)
    out_dir = frames_out if dry_run else manifest_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{match_id}__{split}__{stamp}{'__dryrun' if dry_run else ''}.json"
    write_json_atomic(out, doc)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true", help="sample + manifest only; upload nothing")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    try:
        out = run_ingest(cfg, video=a.video, match_id=a.match_id, split=a.split, dry_run=a.dry_run, argv=argv)
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    import json
    s = json.loads(out.read_text())["summary"]
    print(f"manifest {out}\n{s}")
    return 0 if s["n_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
