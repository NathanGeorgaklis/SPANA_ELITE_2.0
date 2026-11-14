"""Step 6: generate a Roboflow dataset version for detector training.

    python -m tools.roboflow_version [--config config/roboflow.yaml]            # checks + prints settings
    python -m tools.roboflow_version --generate --preflight <preflight record>   # actually generates
    python -m tools.roboflow_version --record N --preflight <preflight record>   # checks version N, writes manifest

--generate and --record need a passed record of tools/roboflow_preflight.py made under the current version and
preflight config and agreeing with the live project listing. The manifest (paths.version_dir/version_<N>.json)
is written only when the server's split sizes, preprocessing and augmentation equal what was asked for.

Settings from config (version section): auto-orient, resize (width, height, format), images carrying an
excluded tag (near-duplicate, blur) filtered out, and only geometric augmentations (none by default; colour-changing augmentation
is refused by the config loader because kit colour matters for team classification). Splits are used as they are:
no rebalance.

Refuses (nothing is generated) if any of these holds:
  * a frame tagged prelabel.tag has no review.reviewed_tag, or a frame in the Dataset is not reviewed;
  * a frame that would enter the version fails the licence gate (allow-list, EVAL, tag consistency);
  * a match has frames in more than one split, or a frame is in the "test" split;
  * the valid (or train) split of the version would be empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Sequence

from spana.artifacts import write_json_atomic
from spana.dataset.config import DatasetConfig, PolicyError, load_dataset_config, match_of
from spana.dataset.prep import list_all, match_tag, open_project
from spana.dataset.rf_project import ProjectImage
from spana.secrets import load_secret

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"


def version_settings(cfg: DatasetConfig) -> dict:
    v = cfg.version
    pre = {"auto-orient": v.auto_orient,
           "resize": {"width": v.resize_width, "height": v.resize_height, "format": v.resize_format},
           "filter-tags": {t: False for t in cfg.excluded_tags()}}
    aug = {}
    if "flip_horizontal" in v.augmentations:
        aug["flip"] = {"horizontal": True, "vertical": False}
    return {"preprocessing": pre, "augmentation": aug}


def version_images(cfg: DatasetConfig, images: List[ProjectImage]) -> List[ProjectImage]:
    """Frames that enter a version: in the Dataset and carrying no excluded tag (the version filters those out)."""
    return [i for i in images if i.in_dataset and not any(t in i.tags for t in cfg.excluded_tags())]


def check_ready(cfg: DatasetConfig, images: List[ProjectImage]) -> dict:
    """Raises PolicyError listing every reason the version must not be generated; returns counts otherwise."""
    problems = []
    unreviewed = [i.name for i in images if any(t in i.tags for t in cfg.prelabel_tags())
                  and cfg.review.reviewed_tag not in i.tags]
    if unreviewed:
        problems.append(f"{len(unreviewed)} pre-labelled frames are not reviewed (e.g. {unreviewed[:3]})")
    version_imgs = version_images(cfg, images)
    not_reviewed = [i.name for i in version_imgs if cfg.review.reviewed_tag not in i.tags]
    if not_reviewed:
        problems.append(f"{len(not_reviewed)} frames in the Dataset are not tagged {cfg.review.reviewed_tag!r} "
                        f"(e.g. {not_reviewed[:3]})")
    gate = Counter()
    for i in version_imgs:
        _, reason = match_of(cfg, i.tags, i.metadata)
        if reason:
            gate[reason] += 1
    if gate:
        problems.append(f"frames failing the licence gate would enter the version: {dict(gate)}")
    splits_per_match = defaultdict(set)
    for i in images:
        mid = match_tag(i)
        if mid is not None and cfg.splits.excluded_tag not in i.tags:
            splits_per_match[mid].add(i.split)
    multi = {m: sorted(s) for m, s in splits_per_match.items() if len(s) > 1}
    if multi:
        problems.append(f"matches in several splits: {multi}")
    counts = Counter(i.split for i in version_imgs)
    if counts.get("test"):
        problems.append(f"{counts['test']} frames are in the test split (held-out testing is EVAL's job)")
    for need in ("train", "valid"):
        if not counts.get(need):
            problems.append(f"the {need} split of the version would be empty")
    if problems:
        raise PolicyError("refused to generate a version:\n  - " + "\n  - ".join(problems))
    return {"images": len(version_imgs), "splits": dict(sorted(counts.items())),
            "match_splits": {m: sorted(s)[0] for m, s in sorted(splits_per_match.items())}}


def load_preflight(cfg: DatasetConfig, path: Path, ready: dict) -> dict:
    """A passed preflight record made under the current version / preflight config and matching the live listing."""
    doc = json.loads(Path(path).read_text())
    problems = []
    if doc.get("kind") != "spana.roboflow_preflight" or doc.get("passed") is not True:
        problems.append("it is not a passed preflight record")
    if doc.get("config_sha256") != {n: cfg.section_sha256(n) for n in ("version", "preflight")}:
        problems.append("it was made under another version / preflight config")
    if doc.get("gate") != ready:
        problems.append(f"the project changed since: preflight {doc.get('gate')}, now {ready}")
    if problems:
        raise PolicyError(f"refused: preflight record {Path(path).name}: " + "; ".join(problems))
    return doc


def server_preprocessing(pre: Optional[dict]) -> dict:
    """The server's saved preprocessing in the shape it was asked for. Tag filters come back as
    {"tags": {tag: {"omit": true}}} (seen 2026-11-14); any other saved filter shape is left as it is, so it differs."""
    pre = dict(pre or {})
    tags = (pre.get("filter-tags") or {}).get("tags") if isinstance(pre.get("filter-tags"), dict) else None
    if isinstance(tags, dict) and all(v == {"omit": True} for v in tags.values()):
        pre["filter-tags"] = {t: False for t in tags}
    return pre


def build_manifest(cfg: DatasetConfig, number: int, server: dict, settings: dict, preflight: dict,
                   preflight_sha256: str) -> dict:
    """Manifest of a generated version, checked against what was asked for. Fails closed on any difference."""
    v = server.get("version") or {}
    splits = {k: int(n) for k, n in (v.get("splits") or {}).items() if int(n)}
    copies = 1 + (int(bool(settings["augmentation"])))        # Roboflow's default: 2 outputs per train image
    want = dict(preflight["gate"]["splits"])
    want["train"] = want["train"] * copies
    problems = []
    if v.get("generating"):
        problems.append("the version is still generating")
    if splits != want:
        problems.append(f"split sizes {splits} differ from the preflight frames {want}")
    if server_preprocessing(v.get("preprocessing")) != settings["preprocessing"]:
        problems.append(f"preprocessing on the server {v.get('preprocessing')} differs from {settings['preprocessing']}")
    if (v.get("augmentation") or {}) != settings["augmentation"]:
        problems.append(f"augmentation on the server {v.get('augmentation')} differs from {settings['augmentation']}")
    if problems:
        raise PolicyError(f"version {number} does not match what was asked for:\n  - " + "\n  - ".join(problems))
    return {"kind": "spana.roboflow_version", "project": cfg.roboflow.project, "version": number,
            "version_id": v.get("id"), "images": v.get("images"), "splits": splits, "settings": settings,
            "config_sha256": preflight["config_sha256"], "preflight_created_utc": preflight["created_utc"],
            "preflight_sha256": preflight_sha256, "labels_file": preflight["labels_file"],
            "labels_sha256": preflight["labels_sha256"], "match_splits": preflight["gate"]["match_splits"],
            "boxes": preflight["stats"]["boxes"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--preflight", help="passed record of tools/roboflow_preflight.py (needed to generate / record)")
    ap.add_argument("--generate", action="store_true", help="generate the version (otherwise only check)")
    ap.add_argument("--record", type=int, metavar="N", help="check generated version N and write its manifest")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    project = open_project(cfg, load_secret(cfg.roboflow.api_key_env))
    settings = version_settings(cfg)
    try:
        ready = check_ready(cfg, list_all(cfg, project))
        print(json.dumps({"ready": ready, "settings": settings}, indent=1, sort_keys=True))
        if a.generate or a.record is not None:
            if not a.preflight:
                raise PolicyError("refused: --generate / --record need --preflight <passed preflight record>")
            preflight = load_preflight(cfg, Path(a.preflight), ready)
        if a.generate:
            print(json.dumps(project.generate_version(settings), indent=1, sort_keys=True))
        if a.record is not None:
            manifest = build_manifest(cfg, a.record, project.get_version(a.record), settings, preflight,
                                      hashlib.sha256(Path(a.preflight).read_bytes()).hexdigest())
            path = REPO / cfg.paths.version_dir / f"version_{a.record}.json"
            write_json_atomic(path, manifest)
            print(json.dumps(manifest, indent=1, sort_keys=True))
            print(f"manifest: {path}")
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
