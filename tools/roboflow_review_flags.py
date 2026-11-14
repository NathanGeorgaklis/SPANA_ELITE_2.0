"""Step 4b: tag pre-labelled frames with review-priority flags (tags only; annotations are never changed).

    python -m tools.roboflow_review_flags [--match MATCH_ID]            # dry run: read, compute, log, print
    python -m tools.roboflow_review_flags [--match MATCH_ID] --apply    # also add the flag tags
    python -m tools.roboflow_review_flags --resume LOG --apply          # continue an interrupted --apply run

Frames: every licensed frame carrying a pre-label tag that is not excluded and not tagged review.reviewed_tag.
Per frame the CURRENT saved annotation is read from the server (GET images/{id}) and the rules in
spana/dataset/review_flags.py are applied with the settings in config (review_flags):
  review:ball-static, review:ball-offpitch, review:ball-tiny, review:player-edge, review:dense (tag names: config).
A flag orders human review; it never says a box is wrong.

Writes (only with --apply): missing flag tags are ADDED through spana.dataset.guard.GuardedProject, which re-reads
the image and re-checks the licence allow-list and the reviewed tag immediately before each write. Nothing else is
written: no annotation call, no tag removal (a flag that no longer applies is left for the reviewer and reported).
Tag writes go in batches of review_flags.write_batch_size; the run log is saved after every batch. HTTP 429 is
waited out by the client's backoff (roboflow.rate_limit_*); any other write failure stops the run (exit 2) and
`--resume LOG --apply` continues with the frames not yet written, using the flags already computed in LOG.

Log: paths.review_dir/review_flags_<stamp>.json (per frame: boxes read, flags, reasons, write state; no pixels).
Prints a per-match count per flag and the suggested review order (most flags first, then frame order).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from spana.artifacts import canonical_bytes, read_json
from spana.dataset import review_flags as rf
from spana.dataset.config import DatasetConfig, PolicyError, load_dataset_config, match_of
from spana.dataset.guard import GuardedProject
from spana.dataset.prefilter import grass_mask
from spana.dataset.prep import SkipFrame, list_all, load_image, match_tag, open_project, sort_key
from spana.dataset.rf_project import ProjectImage, RoboflowApiError
from spana.secrets import load_secret

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
LOG_SCHEMA = "spana.roboflow_review_flags_log"
LOG_SCHEMA_VERSION = "1.0.0"


def eligibility(cfg: DatasetConfig, img: ProjectImage) -> Optional[str]:
    """None if the frame is flagged / tagged by this tool, else the skip reason."""
    _, reason = match_of(cfg, img.tags, img.metadata)
    if reason:
        return f"licence gate: {reason}"
    if cfg.review.reviewed_tag in img.tags:
        return "already reviewed (never touched)"
    if any(t in img.tags for t in cfg.excluded_tags()):
        return "excluded"
    if not any(t in img.tags for t in cfg.prelabel_tags()):
        return "not pre-labelled"
    return None


def compute_flags(cfg: DatasetConfig, project, images: List[ProjectImage], load, *, match: Optional[str] = None,
                  progress: Callable[[str], None] = lambda s: None):
    """Read-only. -> (entries, skipped). One entry per eligible frame, in (match, frame) order:
    {id, name, match_id, frame_index, status ok|failed, reason, boxes, flags {flag: [reasons]}, tags (flag tags)}."""
    s, ball, tags = cfg.review_flags.ball_static, cfg.prelabel.ball_class, cfg.review_flags.flags()
    entries, skipped = [], []
    prev = None                                # (entry, features, balls, timestamp, frame height) of the last ok frame
    for img in sorted(images, key=sort_key):
        if match is not None and match_tag(img) != match:
            continue
        reason = eligibility(cfg, img)
        if reason:
            skipped.append({"id": img.id, "name": img.name, "reason": reason})
            continue
        e = {"id": img.id, "name": img.name, "match_id": match_tag(img), "frame_index": img.metadata.get("frame_index"),
             "status": "ok", "reason": None, "boxes": [], "flags": {}, "tags": [], "written": None}
        entries.append(e)
        try:
            boxes = rf.boxes_from_annotation(project.get_image(img.id).get("annotation"))
            frame = load(img)
        except (SkipFrame, RoboflowApiError, ValueError) as ex:   # unknown labels: no flags, and the chain breaks
            e["status"], e["reason"], prev = "failed", str(ex)[:300], None
            progress(f"{img.name}: failed ({e['reason']})")
            continue
        h = frame.shape[0]
        e["boxes"] = [{"class": b["class"], "box": [round(v, 2) for v in b["box"]]} for b in boxes]
        e["flags"] = rf.frame_flags(cfg, frame, boxes, grass_mask(frame, cfg.prelabel.grass))
        balls = [b for b in boxes if b["class"] == ball]
        t = img.metadata.get("timestamp_s")
        feats = rf.background_features(frame, boxes, s) if balls else None
        if (prev and balls and prev[2] and e["match_id"] == prev[0]["match_id"] and prev[4] == h
                and isinstance(t, (int, float)) and isinstance(prev[3], (int, float)) and 0 < t - prev[3] <= s.max_gap_s):
            for a, b, resid, shift in rf.ball_static(prev[2], balls, rf.camera_transform(prev[1], feats, h, s), h, s):
                for entry, box, other in ((prev[0], a, e["name"]), (e, b, prev[0]["name"])):
                    entry["flags"].setdefault("ball_static", []).append(
                        f"ball box {[round(v, 1) for v in box['box']]}: same place in the scene as a ball in {other} "
                        f"(residual {resid:.1f} px after a camera shift of {shift:.1f} px)")
        prev = (e, feats, balls, t, h)
        progress(f"{img.name}: {len(boxes)} boxes, flags {sorted(e['flags']) or 'none'}")
    for e in entries:
        e["tags"] = sorted(tags[f] for f in e["flags"])
    return entries, skipped


def apply_tags(cfg: DatasetConfig, guard: GuardedProject, entries: List[dict], *, save: Callable[[], None],
               progress: Callable[[str], None] = lambda s: None, sleep=time.sleep) -> None:
    """Add each entry's missing flag tags (never removes one; never touches annotations). Sets e["written"]:
    "added" | "already_tagged" | "skipped: ..." | "refused: ..." | "failed: ...". `save()` is called after every
    batch of review_flags.write_batch_size writes. An API failure stops the run (re-raised after saving)."""
    r, sent = cfg.review_flags, 0
    todo = [e for e in entries if e["status"] == "ok" and e["tags"] and e.get("written") not in ("added", "already_tagged")]
    try:
        for e in todo:
            try:
                fresh = guard.refresh(e["id"])                    # current tags: reviewed / re-tagged frames are seen
                skip = eligibility(cfg, fresh)
                if skip:
                    e["written"] = f"skipped: changed since listing: {skip}"
                    continue
                missing = [t for t in e["tags"] if t not in fresh.tags]
                if not missing:
                    e["written"] = "already_tagged"
                    continue
                if sent and r.min_interval_s:
                    sleep(r.min_interval_s)
                guard.update_tags(e["id"], add=missing)
                e["written"] = "added"
                sent += 1
                progress(f"{e['name']}: + {missing}")
                if sent % r.write_batch_size == 0:
                    save()
            except PolicyError as ex:
                e["written"] = f"refused: {ex}"[:300]
            except RoboflowApiError as ex:
                e["written"] = f"failed: {ex}"[:300]
                raise
    finally:
        save()


def summarize(cfg: DatasetConfig, entries: List[dict], skipped: List[dict]) -> dict:
    per_match: Dict[str, Counter] = defaultdict(Counter)
    order: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        m = per_match[e["match_id"] or "?"]
        m["frames"] += 1
        m["failed"] += e["status"] != "ok"
        m["flagged"] += bool(e["flags"])
        for f in e["flags"]:
            m[f] += 1
        order[e["match_id"] or "?"].append(e)
    review_order = {}
    for mid, es in sorted(order.items()):
        es = sorted(es, key=lambda e: (-len(e["flags"]), e["frame_index"] if isinstance(e["frame_index"], int) else -1,
                                       e["name"]))
        review_order[mid] = [{"name": e["name"], "flags": sorted(e["flags"])} for e in es]
    written = Counter((e.get("written") or "not written").split(":")[0] for e in entries if e["tags"])
    return {"frames": len(entries),
            "per_match": {m: {k: c[k] for k in ("frames", "flagged", *rf.FLAGS, "failed")} for m, c in sorted(per_match.items())},
            "tag_writes": dict(sorted(written.items())),
            "skip_reasons": dict(sorted(Counter(s["reason"] for s in skipped).items())),
            "review_order": review_order}


def summary_text(doc: dict, top: int = 15) -> str:
    s = doc["summary"]
    lines = [f"review flags ({'APPLY' if doc['request']['apply'] else 'DRY RUN, nothing written'}): "
             f"{s['frames']} frames, tag writes {s['tag_writes'] or 'none'}, not selected {s['skip_reasons'] or 'none'}",
             "match | frames | flagged | " + " | ".join(doc["tags"][f] for f in rf.FLAGS) + " | failed"]
    for m, c in s["per_match"].items():
        lines.append(f"{m} | {c['frames']} | {c['flagged']} | " + " | ".join(str(c[f]) for f in rf.FLAGS)
                     + f" | {c['failed']}")
    lines.append("suggested review order (flagged frames first: most flags, then frame order; rest in frame order):")
    for m, rows in s["review_order"].items():
        flagged = [r for r in rows if r["flags"]]
        lines.append(f"  {m}: {len(flagged)} flagged, then {len(rows) - len(flagged)} unflagged")
        for r in flagged[:top]:
            lines.append(f"    {r['name']}: {', '.join(r['flags'])}")
        if len(flagged) > top:
            lines.append(f"    ... {len(flagged) - top} more flagged frames in the log")
    return "\n".join(lines)


def build_doc(cfg: DatasetConfig, entries: List[dict], skipped: List[dict], *, apply: bool, match: Optional[str],
              resumed_from: Optional[str] = None) -> dict:
    return {"schema": LOG_SCHEMA, "schema_version": LOG_SCHEMA_VERSION, "project": cfg.roboflow.project,
            "review_flags_config": cfg.to_dict()["review_flags"],
            "review_flags_config_sha256": cfg.section_sha256("review_flags"),
            "grass_config_sha256": cfg.section_sha256("prelabel"), "tags": cfg.review_flags.flags(),
            "request": {"apply": apply, "match": match, "resumed_from": resumed_from},
            "entries": entries, "not_selected_skips": skipped, "summary": summarize(cfg, entries, skipped)}


def save_log(path: Path, doc: dict) -> None:
    """Write this run's log, replacing its previous state (it is saved after every batch of tag writes).
    spana.artifacts.write_json_atomic refuses to overwrite, which is right for finished records and wrong for a
    progress log: the second save crashed the first --apply run on 2026-11-14."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(canonical_bytes(doc))
    tmp.replace(path)


def check_resume(cfg: DatasetConfig, doc: dict) -> None:
    if doc.get("schema") != LOG_SCHEMA or doc.get("schema_version") != LOG_SCHEMA_VERSION:
        raise PolicyError("refused: --resume is not a review-flags log of this version")
    if doc.get("project") != cfg.roboflow.project:
        raise PolicyError("refused: --resume log is for another project")
    if (doc.get("review_flags_config_sha256") != cfg.section_sha256("review_flags")
            or doc.get("grass_config_sha256") != cfg.section_sha256("prelabel")):
        raise PolicyError("refused: --resume log was computed with different settings; run again without --resume")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--match", help="only frames tagged match:<id>")
    ap.add_argument("--apply", action="store_true", help="add the flag tags (default: dry run)")
    ap.add_argument("--resume", help="review-flags log of an interrupted run: reuse its flags, write what is missing")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    project = open_project(cfg, load_secret(cfg.roboflow.api_key_env))
    now = datetime.now(timezone.utc)
    out = REPO / cfg.paths.review_dir / f"review_flags_{now.strftime('%Y%m%dT%H%M%SZ')}{'' if a.apply else '_dryrun'}.json"
    say = lambda s: print(s, file=sys.stderr, flush=True)
    cache = REPO / cfg.paths.image_cache_dir
    try:
        images = list_all(cfg, project)
        if a.resume:
            if not a.apply or a.match:
                raise PolicyError("--resume needs --apply and takes its frames from the log (no --match)")
            old = read_json(a.resume)
            check_resume(cfg, old)
            entries, skipped, match = old["entries"], old["not_selected_skips"], old["request"]["match"]
        else:
            entries, skipped = compute_flags(cfg, project, images, lambda i: load_image(project, i, cache),
                                             match=a.match, progress=say)
            match = a.match
        doc = build_doc(cfg, entries, skipped, apply=a.apply, match=match,
                        resumed_from=Path(a.resume).name if a.resume else None)
        doc["created_utc"] = now.isoformat(timespec="seconds")

        def save():
            doc["summary"] = summarize(cfg, entries, skipped)
            save_log(out, doc)
        error = None
        save()                                   # the computed flags are on disk before the first write
        if a.apply:
            guard = GuardedProject(cfg, project, images)
            try:
                apply_tags(cfg, guard, entries, save=save, progress=say)
            except RoboflowApiError as ex:
                error = str(ex)[:300]
            doc["writes"] = len(guard.writes)
        save()
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print(summary_text(doc))
    print(f"log: {out}")
    if error:
        print(f"STOPPED on a write failure: {error}\ncontinue with: python -m tools.roboflow_review_flags "
              f"--resume {out.relative_to(REPO)} --apply", file=sys.stderr)
        return 2
    return 0 if not any(c["failed"] for c in doc["summary"]["per_match"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
