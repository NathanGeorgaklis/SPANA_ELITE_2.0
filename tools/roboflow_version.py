"""Step 6: generate a Roboflow dataset version for detector training. NOT RUN YET (code only).

    python -m tools.roboflow_version [--config config/roboflow.yaml]            # checks + prints settings
    python -m tools.roboflow_version --generate                                  # actually generates

Settings from config (version section): auto-orient, resize (width, height, format), images tagged
splits.excluded_tag filtered out, and only geometric augmentations (none by default; colour-changing augmentation
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
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Sequence

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
           "filter-tags": {cfg.splits.excluded_tag: False}}
    aug = {}
    if "flip_horizontal" in v.augmentations:
        aug["flip"] = {"horizontal": True, "vertical": False}
    return {"preprocessing": pre, "augmentation": aug}


def check_ready(cfg: DatasetConfig, images: List[ProjectImage]) -> dict:
    """Raises PolicyError listing every reason the version must not be generated; returns counts otherwise."""
    problems = []
    unreviewed = [i.name for i in images if any(t in i.tags for t in cfg.prelabel_tags())
                  and cfg.review.reviewed_tag not in i.tags]
    if unreviewed:
        problems.append(f"{len(unreviewed)} pre-labelled frames are not reviewed (e.g. {unreviewed[:3]})")
    version_imgs = [i for i in images if i.in_dataset and not any(t in i.tags for t in cfg.excluded_tags())]
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--generate", action="store_true", help="generate the version (otherwise only check)")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    project = open_project(cfg, load_secret(cfg.roboflow.api_key_env))
    settings = version_settings(cfg)
    try:
        ready = check_ready(cfg, list_all(cfg, project))
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print(json.dumps({"ready": ready, "settings": settings}, indent=1, sort_keys=True))
    if a.generate:
        print(json.dumps(project.generate_version(settings), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
