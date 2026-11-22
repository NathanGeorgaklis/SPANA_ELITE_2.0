"""Freeze a filled-in side sheet (`spana.team_sides` 1.0.0 from tools.team_label_sheet) next to its frozen DEV labels.

    python -m tools.freeze_team_labels --sides DIR/<clip>.sides.json --labels data/labels/<clip>/v1

Checks (fail closed):
- DEV only (`spana.reid_labels.split_guard`, purpose tune).
- The sheet names the same clip (sha256), label version and gt.txt sha256 as the frozen labels, whose hashes are
  re-checked.
- Every GT id of the frozen labels has exactly one side from `spana.reid_labels.SIDES`, with no extra ids.
- annotator and match_id are set.

Writes data/labels/<clip>/teams_v<N>/{sides.json, FROZEN.json} (N = the label version) and never overwrites. The
existing frozen labels are not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from spana.artifacts import write_json_atomic
from spana.labels import LabelError
from spana.reid_labels import MATCH_ID_RE, SIDES, split_guard
from tools.team_label_sheet import SIDES_SCHEMA, load_frozen_dev

_KEYS = ("schema", "schema_version", "split", "match_id", "clip", "label_version", "gt_sha256", "annotator",
         "sides_allowed", "sides", "notes")


def freeze(sides_path: str | Path, label_dir: str | Path) -> Path:
    label_dir = Path(label_dir)
    raw = Path(sides_path).read_bytes()
    doc = json.loads(raw)
    if not isinstance(doc, dict) or set(doc) != set(_KEYS) or (doc["schema"], doc["schema_version"]) != SIDES_SCHEMA:
        raise LabelError(f"sides file: keys must be exactly {_KEYS}, schema {SIDES_SCHEMA}")
    split_guard(doc["split"], "tune")
    frozen, gt = load_frozen_dev(label_dir)
    if doc["clip"] != {"name": frozen["clip"]["name"], "sha256": frozen["clip"]["sha256"]}:
        raise LabelError("sides file and frozen labels are for different clips")
    if doc["label_version"] != frozen["label_version"] or doc["gt_sha256"] != frozen["gt_sha256"]:
        raise LabelError("sides file was made from another label version / gt.txt")
    if not str(doc["annotator"]).strip():
        raise LabelError("annotator missing")
    if not (isinstance(doc["match_id"], str) and MATCH_ID_RE.match(doc["match_id"])):
        raise LabelError(f"bad match_id {doc['match_id']!r}")
    ids = {p for v in gt.players.values() for p in v}
    if set(doc["sides"]) != ids:
        raise LabelError(f"sides must name exactly the GT ids: missing {sorted(ids - set(doc['sides']), key=int)}, "
                         f"extra {sorted(set(doc['sides']) - ids)}")
    bad = {k: v for k, v in doc["sides"].items() if v not in SIDES}
    if bad:
        raise LabelError(f"sides must be one of {SIDES}: {bad}")
    dest = label_dir.parent / f"teams_v{int(frozen['label_version'])}"
    if dest.exists():
        raise LabelError(f"{dest} already exists: frozen labels are never overwritten")
    dest.mkdir(parents=True)
    (dest / "sides.json").write_bytes(raw)
    counts = {s: sum(v == s for v in doc["sides"].values()) for s in SIDES}
    write_json_atomic(dest / "FROZEN.json", {
        "kind": "spana.team_sides_frozen", "split": "DEV", "match_id": doc["match_id"], "clip": doc["clip"],
        "label_version": doc["label_version"], "gt_sha256": doc["gt_sha256"], "annotator": doc["annotator"],
        "sides_sha256": hashlib.sha256(raw).hexdigest(), "counts": counts, "ids": len(ids),
        "frozen_in_repo_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "Frozen before team classification is scored. Never edit; a correction is a new version with a "
                "reason. The same letter means the same kit in every clip of the match.",
    })
    return dest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sides", required=True)
    p.add_argument("--labels", required=True, help="the frozen DEV label version, e.g. data/labels/<clip>/v1")
    a = p.parse_args()
    print(freeze(a.sides, a.labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
