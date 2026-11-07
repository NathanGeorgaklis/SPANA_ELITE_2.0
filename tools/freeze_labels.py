"""Freeze exported DEV labels into the repository (before any inference is scored against them).

    python -m tools.freeze_labels --pack PACK_DIR --gt CLIP_vN_gt.txt --record CLIP_vN_labels_frozen.json \
        [--video CLIP.mp4]

Checks: DEV split only; the record, pack and (optionally) video belong to the same clip (sha256); the gt file's
sha256 matches the record; every row parses as MOT 1.1 on a labelled frame. Writes
data/labels/<clip>/v<N>/{gt.txt, labels_frozen.json, FROZEN.json} and never overwrites an existing version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from spana.artifacts import write_json_atomic
from spana.labels import LabelError, parse_mot
from spana.video import sha256_file

REPO = Path(__file__).resolve().parent.parent


def freeze(pack: str, gt_path: str, record_path: str, video: str | None, root: Path = REPO) -> Path:
    manifest = json.loads((Path(pack) / "manifest.json").read_text())
    record = json.loads(Path(record_path).read_text())
    gt_bytes = Path(gt_path).read_bytes()
    if manifest.get("split") != "DEV" or record.get("split") != "DEV":
        raise LabelError("only DEV labels can be frozen now")
    sha = manifest["clip"]["sha256"]
    if record["clip"]["sha256"] != sha:
        raise LabelError("labels record and pack are for different clips")
    if video is not None and sha256_file(video) != sha:
        raise LabelError("video does not match the pack's sha256")
    gt_sha = hashlib.sha256(gt_bytes).hexdigest()
    if gt_sha != record["gt_sha256"]:
        raise LabelError("gt file sha256 does not match the frozen record (edited after export?)")
    pack_frames = {f["index"] for f in manifest["frames"]}
    if not set(record["frames_labelled"]) <= pack_frames:
        raise LabelError("record lists frames that are not in the pack")
    if not record.get("annotator", "").strip():
        raise LabelError("annotator missing")
    gt = parse_mot(gt_bytes.decode(), manifest["clip"]["width"], manifest["clip"]["height"],
                   record["frames_labelled"])
    dest = root / "data" / "labels" / manifest["clip"]["name"] / f"v{int(record['label_version'])}"
    if dest.exists():
        raise LabelError(f"{dest} already exists: frozen labels are never overwritten")
    dest.mkdir(parents=True)
    (dest / "gt.txt").write_bytes(gt_bytes)
    (dest / "labels_frozen.json").write_bytes(Path(record_path).read_bytes())
    write_json_atomic(dest / "FROZEN.json", {
        "split": "DEV", "clip": manifest["clip"], "label_version": record["label_version"],
        "annotator": record["annotator"], "exported_utc": record["frozen_utc"],
        "frozen_in_repo_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gt_sha256": gt_sha, "labels_record_sha256": hashlib.sha256(Path(record_path).read_bytes()).hexdigest(),
        "frames_labelled": len(record["frames_labelled"]), "frame_step": record["frame_step"],
        "player_boxes": sum(len(v) for v in gt.players.values()),
        "players": len({p for v in gt.players.values() for p in v}),
        "ignore_regions": sum(len(v) for v in gt.ignore.values()),
        "rule": "Frozen before inference is scored. Never edit; a correction is a new version with a reason.",
    })
    return dest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--record", required=True)
    p.add_argument("--video", default=None)
    a = p.parse_args()
    try:
        dest = freeze(a.pack, a.gt, a.record, a.video)
    except LabelError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"frozen -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
