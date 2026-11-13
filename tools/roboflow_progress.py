"""Step 5: labelling progress per match and per class (read-only).

    python -m tools.roboflow_progress [--config config/roboflow.yaml] [--json OUT.json]

Per match: frames, excluded, pre-labelled, annotated, reviewed, pre-labelled-but-not-reviewed, in Dataset, splits;
per class: boxes on annotated frames and on reviewed frames. Frames the licence gate rejects are counted apart.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Sequence

from spana.dataset.config import DatasetConfig, load_dataset_config, match_of
from spana.dataset.prep import list_all, match_tag, open_project
from spana.dataset.rf_project import ProjectImage
from spana.secrets import load_secret

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"


def progress(cfg: DatasetConfig, images: List[ProjectImage], annotation_of=None) -> dict:
    """`annotation_of(image_id) -> (annotated, classes)` reads saved annotations; search results omit them for
    images not yet in the Dataset. Used for frames tagged pre-labelled or reviewed, or reported annotated."""
    rows = defaultdict(lambda: Counter())
    classes = defaultdict(lambda: {"annotated": Counter(), "reviewed": Counter()})
    splits = defaultdict(Counter)
    gate = Counter()
    for img in images:
        if annotation_of is not None and (img.annotated or any(t in img.tags for t in cfg.prelabel_tags())
                                          or cfg.review.reviewed_tag in img.tags):
            annotated, cls = annotation_of(img.id)
            img = ProjectImage(**{**img.__dict__, "annotated": annotated, "classes": cls})
        _, reason = match_of(cfg, img.tags, img.metadata)
        if reason:
            gate[reason] += 1
        mid = match_tag(img) or "(no match tag)"
        r = rows[mid]
        r["frames"] += 1
        excluded = any(t in img.tags for t in cfg.excluded_tags())
        pre = any(t in img.tags for t in cfg.prelabel_tags())
        rev = cfg.review.reviewed_tag in img.tags
        r["excluded"] += excluded
        r["prelabelled"] += pre
        r["annotated"] += img.annotated
        r["reviewed"] += rev
        r["prelabelled_not_reviewed"] += pre and not rev
        r["in_dataset"] += img.in_dataset
        splits[mid][img.split or "none"] += 1
        if img.annotated and not excluded:
            classes[mid]["annotated"].update(img.classes)
            if rev:
                classes[mid]["reviewed"].update(img.classes)
    out = {}
    for mid in sorted(rows):
        out[mid] = {**{k: rows[mid][k] for k in ("frames", "excluded", "prelabelled", "annotated", "reviewed",
                                                 "prelabelled_not_reviewed", "in_dataset")},
                    "splits": dict(sorted(splits[mid].items())),
                    "boxes_annotated": {c: classes[mid]["annotated"].get(c, 0) for c in cfg.classes},
                    "boxes_reviewed": {c: classes[mid]["reviewed"].get(c, 0) for c in cfg.classes},
                    "other_labels": sorted(set(classes[mid]["annotated"]) - set(cfg.classes))}
    return {"matches": out, "licence_gate_rejections": dict(sorted(gate.items()))}


def table(doc: dict) -> str:
    lines = ["match | frames | excl | prelab | annotated | reviewed | prelab-not-reviewed | in dataset | splits"]
    for m, r in doc["matches"].items():
        lines.append(f"{m} | {r['frames']} | {r['excluded']} | {r['prelabelled']} | {r['annotated']} | "
                     f"{r['reviewed']} | {r['prelabelled_not_reviewed']} | {r['in_dataset']} | {r['splits']}")
    lines.append("boxes per class (annotated / reviewed frames):")
    for m, r in doc["matches"].items():
        lines.append(f"  {m}: {r['boxes_annotated']} / {r['boxes_reviewed']}"
                     + (f"  other labels: {r['other_labels']}" if r["other_labels"] else ""))
    lines.append(f"licence-gate rejections: {doc['licence_gate_rejections'] or 'none'}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    project = open_project(cfg, load_secret(cfg.roboflow.api_key_env))
    doc = progress(cfg, list_all(cfg, project), annotation_of=project.get_image_annotation)
    print(table(doc))
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
