"""Step 1, READ-ONLY dataset audit of the Roboflow project. Changes nothing on Roboflow.

    python -m tools.roboflow_audit [--config config/roboflow.yaml] [--no-phash]

Reports: frames per match tag, Roboflow split per match (a match in more than one split is flagged), images without
a match tag, images the licence gate rejects (with the reason), and near-duplicate frames: 64-bit pHash, linked when
the Hamming distance is <= audit.phash_hamming_max; each linked group keeps its earliest frame and lists the rest
for exclusion. Pixels are fetched from Roboflow and verified against the sha256 recorded at upload (unverifiable
frames are listed, never hashed).

Writes paths.audit_dir/audit_<UTC stamp>.json (no pixels, no secrets) and prints a short summary.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from spana.artifacts import write_json_atomic
from spana.dataset.config import DatasetConfig, load_dataset_config, match_of
from spana.dataset.phash import near_duplicate_groups, phash, phash_hex
from spana.dataset.prep import SkipFrame, list_all, load_image, match_tag, open_project, sort_key
from spana.dataset.rf_project import ProjectImage
from spana.secrets import load_secret

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
AUDIT_SCHEMA = "spana.roboflow_audit"
AUDIT_SCHEMA_VERSION = "1.0.0"


def audit(cfg: DatasetConfig, images: List[ProjectImage], *, load=None, with_phash: bool = True) -> dict:
    """Pure report over `images`. `load(img) -> BGR frame` (raises SkipFrame) is only used for the pHash."""
    per_match = defaultdict(lambda: {"n": 0, "splits": Counter(), "in_dataset": 0, "annotated": 0})
    untagged, gate_failures = [], []
    for img in images:
        mid = match_tag(img)
        if mid is None:
            untagged.append({"id": img.id, "name": img.name, "tags": list(img.tags)})
        else:
            pm = per_match[mid]
            pm["n"] += 1
            pm["splits"][img.split or "none"] += 1
            pm["in_dataset"] += int(img.in_dataset)
            pm["annotated"] += int(img.annotated)
        _, reason = match_of(cfg, img.tags, img.metadata)
        if reason:
            gate_failures.append({"id": img.id, "name": img.name, "reason": reason})
    matches = {m: {"n": v["n"], "splits": dict(sorted(v["splits"].items())), "in_dataset": v["in_dataset"],
                   "annotated": v["annotated"]} for m, v in sorted(per_match.items())}
    multi_split = sorted(m for m, v in matches.items() if len(v["splits"]) > 1)

    dup = {"max_distance": cfg.audit.phash_hamming_max, "computed": with_phash, "hashes": {}, "unverified": [],
           "groups": [], "n_excluded": 0}
    if with_phash:
        items = []
        for img in sorted(images, key=sort_key):
            try:
                h = phash(load(img))
            except SkipFrame as e:
                dup["unverified"].append({"id": img.id, "name": img.name, "reason": str(e)})
                continue
            dup["hashes"][img.id] = phash_hex(h)
            items.append((img.id, sort_key(img), h))
        dup["groups"] = near_duplicate_groups(items, cfg.audit.phash_hamming_max)
        names = {i.id: i.name for i in images}
        for g in dup["groups"]:
            g["keep_name"] = names[g["keep"]]
            g["exclude_names"] = [names[x] for x in g["exclude"]]
            g["pairs"] = [list(p) for p in g["pairs"]]
        dup["n_excluded"] = sum(len(g["exclude"]) for g in dup["groups"])
    return {"schema": AUDIT_SCHEMA, "schema_version": AUDIT_SCHEMA_VERSION,
            "workspace": cfg.roboflow.workspace, "project": cfg.roboflow.project, "n_images": len(images),
            "matches": matches, "matches_in_several_splits": multi_split, "untagged": untagged,
            "gate_failures": gate_failures, "near_duplicates": dup,
            "audit_config_sha256": cfg.section_sha256("audit")}


def summary_text(doc: dict) -> str:
    lines = [f"{doc['project']}: {doc['n_images']} images"]
    for m, v in doc["matches"].items():
        lines.append(f"  {m}: {v['n']} frames, splits {v['splits']}, in dataset {v['in_dataset']}, "
                     f"annotated {v['annotated']}")
    lines.append(f"  matches in several splits: {doc['matches_in_several_splits'] or 'none'}")
    lines.append(f"  untagged images: {len(doc['untagged'])}")
    reasons = Counter(g["reason"] for g in doc["gate_failures"])
    lines.append(f"  licence-gate rejections: {dict(reasons) or 'none'}")
    d = doc["near_duplicates"]
    if d["computed"]:
        lines.append(f"  near-duplicates (Hamming <= {d['max_distance']}): {len(d['groups'])} groups, "
                     f"{d['n_excluded']} frames to exclude; unverified {len(d['unverified'])}")
    else:
        lines.append("  near-duplicates: not computed (--no-phash)")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-phash", action="store_true", help="skip pixel download and near-duplicate detection")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    project = open_project(cfg, load_secret(cfg.roboflow.api_key_env))
    images = list_all(cfg, project)
    cache = REPO / cfg.paths.image_cache_dir
    now = datetime.now(timezone.utc)
    doc = audit(cfg, images, load=lambda img: load_image(project, img, cache), with_phash=not a.no_phash)
    doc["created_utc"] = now.isoformat(timespec="seconds")
    out = REPO / cfg.paths.audit_dir / f"audit_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, doc)
    print(summary_text(doc))
    print(f"audit: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
