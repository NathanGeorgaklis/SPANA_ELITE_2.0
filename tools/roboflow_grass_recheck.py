"""Re-run the pre-label grass filter offline on the detections stored in pre-label logs, with the current config.

Offline and read-only: reads pre-label logs and the local image cache (paths.image_cache_dir); makes no API call and
changes nothing in the project. For every chosen frame the logged candidates (kept boxes plus boxes the grass
filter dropped; size-cap drops are left out) are filtered again and the outcome is compared with the log. Used to
check that a change to prelabel.grass alters only the boxes it is meant to.

    python -m tools.roboflow_grass_recheck LOG [LOG ...] [--frames NAME_SUBSTRING ...] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import cv2

from spana.dataset.config import DatasetConfig, load_dataset_config
from tools.roboflow_prelabel import grass_filter

REPO = Path(__file__).resolve().parents[1]
GRASS_REASON = "outside grass mask"


def recheck_entry(cfg: DatasetConfig, entry: dict, frame) -> dict:
    """-> {name, dropped_before, dropped_now, now_kept, now_dropped, unchanged} for one written log entry."""
    was_dropped = [d for d in entry.get("dropped_boxes", []) if GRASS_REASON in d.get("reason", "")]
    candidates = [{k: v for k, v in b.items() if k != "grass_note"} for b in entry.get("boxes", [])]
    ids = {c: i for i, c in enumerate(cfg.classes)}
    candidates += [{"class": d["class"], "class_id": ids[d["class"]], "box": d["box"], "score": d["score"]}
                   for d in was_dropped]
    kept, dropped = grass_filter(cfg, frame, candidates)
    before = sorted((d["class"], tuple(d["box"])) for d in was_dropped)
    now = sorted((d["class"], tuple(d["box"])) for d in dropped)
    notes = {(k["class"], tuple(k["box"])): k["grass_note"] for k in kept if "grass_note" in k}
    return {"name": entry["name"], "candidates": len(candidates), "dropped_before": len(before),
            "dropped_now": len(now),
            "now_kept": [{"class": c, "box": list(b), "note": notes.get((c, b))} for c, b in before if (c, b) not in now],
            "now_dropped": [{"class": c, "box": list(b)} for c, b in now if (c, b) not in before],
            "unchanged": before == now}


def recheck(cfg: DatasetConfig, docs: Sequence[dict], cache: Path, frames: Optional[Sequence[str]] = None) -> List[dict]:
    out = []
    for doc in docs:
        for e in doc.get("entries", []):
            if e.get("status") not in ("labelled", "no_boxes"):
                continue
            if frames and not any(f in e["name"] for f in frames):
                continue
            frame = cv2.imread(str(cache / f"{e['id']}.jpg"))
            if frame is None:
                raise SystemExit(f"ERROR: {e['name']} is not in the image cache ({cache})")
            out.append(recheck_entry(cfg, e, frame))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--frames", nargs="+", help="only frames whose name contains one of these strings")
    ap.add_argument("--config", default=str(REPO / "config" / "roboflow.yaml"))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    rows = recheck(cfg, [json.loads(Path(p).read_text()) for p in a.logs], REPO / cfg.paths.image_cache_dir, a.frames)
    for r in rows:
        if not r["unchanged"] or a.frames:
            print(f"{r['name']}: grass drops {r['dropped_before']} -> {r['dropped_now']} "
                  f"({'unchanged' if r['unchanged'] else 'CHANGED'})")
            for k in r["now_kept"]:
                print(f"    now kept    {k['class']} {k['box']}: {k['note']}")
            for d in r["now_dropped"]:
                print(f"    now dropped {d['class']} {d['box']}")
    changed = [r for r in rows if not r["unchanged"]]
    print(f"frames {len(rows)}; changed {len(changed)}; grass drops {sum(r['dropped_before'] for r in rows)} -> "
          f"{sum(r['dropped_now'] for r in rows)}; boxes now kept {sum(len(r['now_kept']) for r in rows)}; "
          f"boxes now dropped {sum(len(r['now_dropped']) for r in rows)}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
