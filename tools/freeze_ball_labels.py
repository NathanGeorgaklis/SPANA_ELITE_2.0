"""Freeze exported DEV ball labels into the repository (before any ball tracker output is scored against them).

    python -m tools.freeze_ball_labels --pack PACK_DIR --record CLIP_vN_ball_labels.json [--video CLIP.mp4]

Checks: DEV only; record, pack and (optionally) video are the same clip (sha256); every labelled frame is a pack
frame; spana.ball_labels rules (spana/ball_labels.py). Writes data/ball_labels/<clip>/v<N>/{ball_labels_frozen.json,
FROZEN.json} and never overwrites an existing version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from spana.artifacts import write_json_atomic
from spana.ball_labels import BallLabelError, summary, validate_record
from spana.video import sha256_file

REPO = Path(__file__).resolve().parent.parent


def freeze(pack: str, record_path: str, video: str | None, root: Path = REPO) -> Path:
    manifest = json.loads((Path(pack) / "manifest.json").read_text())
    raw = Path(record_path).read_bytes()
    rec = json.loads(raw)
    if manifest.get("split") != "DEV":
        raise BallLabelError("only DEV labels can be frozen now")
    sha = manifest["clip"]["sha256"]
    if rec.get("clip", {}).get("sha256") != sha:
        raise BallLabelError("ball labels and pack are for different clips")
    if video is not None and sha256_file(video) != sha:
        raise BallLabelError("video does not match the pack's sha256")
    labels = validate_record(rec, [f["index"] for f in manifest["frames"]])
    dest = root / "data" / "ball_labels" / manifest["clip"]["name"] / f"v{int(rec['label_version'])}"
    if dest.exists():
        raise BallLabelError(f"{dest} already exists: frozen labels are never overwritten")
    dest.mkdir(parents=True)
    (dest / "ball_labels_frozen.json").write_bytes(raw)
    write_json_atomic(dest / "FROZEN.json", {
        "split": "DEV", "clip": manifest["clip"], "label_version": rec["label_version"],
        "annotator": rec["annotator"], "exported_utc": rec["frozen_utc"],
        "frozen_in_repo_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_sha256": hashlib.sha256(raw).hexdigest(), "frame_step": rec["frame_step"], **summary(labels),
        "rule": "Frozen before inference is scored. Never edit; a correction is a new version with a reason.",
    })
    return dest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--record", required=True)
    p.add_argument("--video", default=None)
    a = p.parse_args()
    try:
        dest = freeze(a.pack, a.record, a.video)
    except BallLabelError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"frozen -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
