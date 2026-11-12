"""Step 3: match-level split assignment and near-duplicate exclusion for the Roboflow project.

    python -m tools.roboflow_split --assign valid=<match_id> [--audit records/roboflow_audit/audit_X.json]
        [--apply | --apply-exclusions-only] [--config config/roboflow.yaml]

Plan (always written to paths.split_dir as a before/after manifest):
  * every frame of each --assign'ed match goes to that Roboflow split; every other licensed match goes to "train";
    a match is never in two splits. Only splits in splits.assignable are accepted (never "test").
  * refused: an EVAL match (allow-list or split:eval tag), a match not on the allow-list, an assigned match with no
    frames, an empty valid split.
  * frames the licence gate rejects are not moved and are listed with the reason.
  * near-duplicates from the audit (`--audit`) get splits.excluded_tag. They are never deleted.
Default is a dry run (manifest only). --apply-exclusions-only adds the exclusion tags. --apply also performs the
split moves, only with splits.move_method "delete_reupload"; with "none" it refuses whenever a move is planned and
nothing is changed. Roboflow's API has no split-change call for single images (only a random ratio rebalance, which
would split matches), and re-uploading identical bytes keeps the old split (probed 2026-11-13). "delete_reupload":
  1. every frame's original bytes are fetched and verified against the sha256 recorded at upload;
  2. a restore file (paths.split_dir/restore_<stamp>.json: name, sha, tags, metadata, target split, cached file) is
     written BEFORE anything is deleted; `--restore FILE` re-uploads any listed frame missing from the project;
  3. the frames are removed with their upload batch (the match's whole batch when it holds exactly those frames,
     else a new batch made of just those frames), permanently, and the listing must no longer show them;
  4. each frame is uploaded again with the new split, same name, tags and metadata;
  5. the listing must show each name exactly once, in the new split, with the same image sha256.
Any failed check stops the run with an error (fail closed) and the manifest records what happened.
--probe N moves only the first N planned frames (N <= splits.probe_max; no exclusions) to test the method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from spana.artifacts import write_json_atomic
from spana.dataset.config import DatasetConfig, PolicyError, load_dataset_config, match_of
from spana.dataset.prep import SkipFrame, list_all, match_tag, open_project, sort_key, verified_image_file
from spana.dataset.upload import RoboflowUploader
from spana.dataset.rf_project import IncompleteListing, ProjectImage
from spana.secrets import load_secret

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "roboflow.yaml"
SPLIT_SCHEMA = "spana.roboflow_split_plan"
SPLIT_SCHEMA_VERSION = "1.0.0"
DEFAULT_SPLIT = "train"


def parse_assign(values: Sequence[str]) -> Dict[str, str]:
    """["valid=M1"] -> {"M1": "valid"}; a match may be assigned once."""
    out: Dict[str, str] = {}
    for v in values:
        split, sep, mid = v.partition("=")
        if not sep or not split or not mid:
            raise PolicyError(f"--assign {v!r}: expected <split>=<match_id>")
        if mid in out:
            raise PolicyError(f"match {mid!r} assigned twice: a match goes to exactly one split")
        out[mid] = split
    return out


def plan_split(cfg: DatasetConfig, images: List[ProjectImage], assign: Dict[str, str],
               exclude_ids: Sequence[str] = ()) -> dict:
    """Pure planning + policy checks. Raises PolicyError; never returns a plan that splits a match."""
    if DEFAULT_SPLIT not in cfg.splits.assignable:
        raise PolicyError(f"splits.assignable must contain {DEFAULT_SPLIT!r}")
    by_match: Dict[str, List[ProjectImage]] = defaultdict(list)
    for img in images:
        mid = match_tag(img)
        if mid is not None:
            by_match[mid].append(img)
    for mid, split in assign.items():
        if split not in cfg.splits.assignable:
            raise PolicyError(f"refused: split {split!r} not in splits.assignable {cfg.splits.assignable}")
        entry = next((m for m in cfg.licensed_matches if m.match_id == mid), None)
        if entry is not None and entry.split == "eval":
            raise PolicyError(f"refused: {mid!r} is an EVAL match")
        if any("split:eval" in i.tags for i in by_match.get(mid, [])):
            raise PolicyError(f"refused: {mid!r} has frames tagged split:eval")
        if entry is None:
            raise PolicyError(f"refused: {mid!r} is not on the licensed_matches allow-list")
        if not by_match.get(mid):
            raise PolicyError(f"refused: no frames tagged match:{mid} in the project")

    rows, skipped = [], []
    exclude = set(exclude_ids)
    for img in sorted(images, key=sort_key):
        entry, reason = match_of(cfg, img.tags, img.metadata)
        if reason:
            skipped.append({"id": img.id, "name": img.name, "reason": reason, "split": img.split})
            continue
        after = assign.get(entry.match_id, DEFAULT_SPLIT)
        rows.append({"id": img.id, "name": img.name, "match_id": entry.match_id, "before": img.split,
                     "after": after, "move": img.split != after, "exclude": img.id in exclude,
                     "already_excluded": cfg.splits.excluded_tag in img.tags})
    after_splits = defaultdict(set)
    for r in rows:
        after_splits[r["match_id"]].add(r["after"])
    split_matches = sorted(m for m, s in after_splits.items() if len(s) != 1)
    if split_matches:   # cannot happen by construction; checked anyway (fail closed)
        raise PolicyError(f"plan would put matches {split_matches} in several splits")
    for split in set(assign.values()):
        if not any(r["after"] == split and not r["exclude"] for r in rows):
            raise PolicyError(f"refused: split {split!r} would be empty")
    unknown_excl = sorted(exclude - {r["id"] for r in rows})
    counts = Counter((r["match_id"], r["after"]) for r in rows)
    return {"schema": SPLIT_SCHEMA, "schema_version": SPLIT_SCHEMA_VERSION, "assign": dict(sorted(assign.items())),
            "default_split": DEFAULT_SPLIT, "excluded_tag": cfg.splits.excluded_tag,
            "match_splits_after": {m: sorted(s)[0] for m, s in sorted(after_splits.items())},
            "counts_after": [{"match_id": m, "split": s, "n": n} for (m, s), n in sorted(counts.items())],
            "n_moves": sum(r["move"] for r in rows), "n_exclude": sum(r["exclude"] for r in rows),
            "exclude_ids_not_in_plan": unknown_excl, "rows": rows, "skipped": skipped}


def _find_batch(project, name: str) -> dict:
    found = [b for b in project.list_batches() if b["name"] == name]
    if len(found) != 1:
        raise PolicyError(f"stopped: expected exactly one batch named {name!r}, found {len(found)}")
    return found[0]


def write_restore(path: Path, frames: List[dict]) -> None:
    """Restore file: everything needed to upload the frames again. Cached files are named relative to the image
    cache (paths.image_cache_dir), so no local directory names are stored."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in f.items() if k != "file"} for f in frames]
    write_json_atomic(path, {"schema": "spana.roboflow_restore", "schema_version": "1.0.0", "frames": rows})


def upload_frames(cfg: DatasetConfig, uploader, frames: List[dict]) -> List[dict]:
    log = []
    for f in frames:
        res = uploader.single_upload(image_path=f["file"], split=f["split"], batch_name=f["batch_name"],
                                     tag_names=f["tags"], metadata=f["metadata"], num_retry_uploads=0)
        got = (res or {}).get("image") or {}
        log.append({"name": f["name"], "old_id": f["old_id"], "new_id": got.get("id"),
                    "duplicate": bool(got.get("duplicate")), "requested_split": f["split"]})
        if got.get("duplicate"):
            raise PolicyError(f"stopped: {f['name']} was treated as a duplicate after deletion. Log: {log}")
    return log


def retry_check(cfg: DatasetConfig, check) -> None:
    """Run `check()` (raises PolicyError while the state is not yet as expected) up to splits.verify_attempts times,
    waiting splits.verify_wait_s between attempts: Roboflow's search index lags writes."""
    for attempt in range(cfg.splits.verify_attempts):
        try:
            check()
            return
        except (PolicyError, IncompleteListing):
            if attempt == cfg.splits.verify_attempts - 1:
                raise
            time.sleep(cfg.splits.verify_wait_s)


def verify_uploaded(cfg: DatasetConfig, project, frames: List[dict], log: List[dict]) -> None:
    retry_check(cfg, lambda: _verify_uploaded_once(cfg, project, frames, log))


def _verify_uploaded_once(cfg: DatasetConfig, project, frames: List[dict], log: List[dict]) -> None:
    listing = list_all(cfg, project)
    by_name: Dict[str, List[ProjectImage]] = defaultdict(list)
    for i in listing:
        by_name[i.name].append(i)
    for f, e in zip(frames, log):
        hits = by_name.get(f["name"], [])
        e["split_after"] = hits[0].split if len(hits) == 1 else None
        e["copies"] = len(hits)
        if len(hits) != 1 or hits[0].split != f["split"] or hits[0].metadata.get("image_sha256") != f["sha256"]:
            raise PolicyError(f"stopped: {f['name']} is not exactly once in split {f['split']!r} with the same "
                              f"bytes (copies {len(hits)}, split {e['split_after']!r}). Log: {log}")


def move_frames(cfg: DatasetConfig, project, uploader, plan: dict, images: List[ProjectImage], cache_dir: Path,
                restore_path: Path, limit: Optional[int] = None, stamp: str = "") -> List[dict]:
    """delete_reupload (see module docstring). Raises PolicyError on any failed check."""
    by_id = {i.id: i for i in images}
    rows = [r for r in plan["rows"] if r["move"]]
    if limit is not None:
        rows = rows[:limit]
    frames = []
    for r in rows:                                      # 1. verify every frame's bytes before touching anything
        img = by_id[r["id"]]
        try:
            src = verified_image_file(project, img, cache_dir)
        except SkipFrame as e:
            raise PolicyError(f"refused: cannot verify {img.name} before moving it: {e}") from None
        named = cache_dir / "move" / img.name
        named.parent.mkdir(parents=True, exist_ok=True)
        named.write_bytes(src.read_bytes())
        spana_split = img.metadata.get("split", "train")
        frames.append({"name": img.name, "old_id": img.id, "match_id": r["match_id"], "sha256": img.metadata["image_sha256"],
                       "tags": list(img.tags), "metadata": dict(img.metadata), "split": r["after"],
                       "file": str(named), "cached_file": f"move/{img.name}",
                       "old_batch": f"spana_{r['match_id']}_{spana_split}",
                       "batch_name": f"spana_{r['match_id']}_{spana_split}_{r['after']}"})
    write_restore(restore_path, frames)                 # 2. restore file before any deletion
    per_match_total = Counter(i.metadata.get("match_id") for i in images)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for f in frames:
        groups[f["old_batch"]].append(f)
    for batch_name, fs in sorted(groups.items()):       # 3. remove with the batch, permanently
        src = _find_batch(project, batch_name)
        ids = [f["old_id"] for f in fs]
        whole = len(ids) == src["images"] == per_match_total[fs[0]["match_id"]]
        if whole:
            target = src
        else:
            move_name = f"spana_move_{fs[0]['match_id']}_{stamp}"
            project.create_batch(src["id"], ids, move_name)
            target = _find_batch(project, move_name)
            if target["images"] != len(ids):
                raise PolicyError(f"stopped: batch {move_name!r} holds {target['images']} images, expected {len(ids)} "
                                  "(nothing deleted)")
        project.delete_batch(target["id"], permanent=True)
    def gone():
        still = {i.id for i in list_all(cfg, project)} & {f["old_id"] for f in frames}
        if still:
            raise PolicyError(f"stopped: {len(still)} frames still in the project after the batch delete; nothing "
                              f"uploaded. Restore file: {restore_path}")
    retry_check(cfg, gone)
    log = upload_frames(cfg, uploader, frames)          # 4. upload with the new split
    verify_uploaded(cfg, project, frames, log)          # 5. exactly once, new split, same bytes
    return log


def restore(cfg: DatasetConfig, project, uploader, restore_doc: dict, cache_dir: Path) -> List[dict]:
    """Re-upload every frame of a restore file that is missing from the project, then verify."""
    present = {i.name for i in list_all(cfg, project)}
    todo = [dict(f, file=str(cache_dir / f["cached_file"])) for f in restore_doc["frames"] if f["name"] not in present]
    for f in todo:
        if not Path(f["file"]).is_file() or hashlib.sha256(Path(f["file"]).read_bytes()).hexdigest() != f["sha256"]:
            raise PolicyError(f"refused: local copy of {f['name']} does not match its sha256")
    log = upload_frames(cfg, uploader, todo)
    verify_uploaded(cfg, project, todo, log)
    return log


def apply_plan(cfg: DatasetConfig, project, plan: dict, *, exclusions_only: bool, uploader=None,
               images: Sequence[ProjectImage] = (), cache_dir: Optional[Path] = None,
               probe: Optional[int] = None, restore_path: Optional[Path] = None, stamp: str = "") -> List[dict]:
    """Executes a plan. Refuses (changing nothing) if a split move is needed and no move method is enabled."""
    if probe is not None:
        if not 1 <= probe <= cfg.splits.probe_max:
            raise PolicyError(f"refused: --probe must be in [1, splits.probe_max={cfg.splits.probe_max}]")
        if not plan["n_moves"]:
            raise PolicyError("refused: nothing to move, nothing to probe")
        return [dict(m, action="probe_move") for m in
                move_frames(cfg, project, uploader, plan, list(images), cache_dir, restore_path, limit=probe,
                            stamp=stamp)]
    if not exclusions_only and plan["n_moves"] and cfg.splits.move_method == "none":
        raise PolicyError(f"refused: {plan['n_moves']} frames need a split move, but Roboflow's API cannot move "
                          "single images between splits (splits.move_method: none). Nothing was changed. "
                          "Use --apply-exclusions-only for the exclusion tags, or --probe to test 'delete_reupload'.")
    log = []
    for r in plan["rows"]:
        if r["exclude"] and not r["already_excluded"]:
            project.update_tags(r["id"], add=[cfg.splits.excluded_tag])
            log.append({"id": r["id"], "action": "tag", "tag": cfg.splits.excluded_tag})
    if not exclusions_only and plan["n_moves"]:
        log += [dict(m, action="move") for m in
                move_frames(cfg, project, uploader, plan, list(images), cache_dir, restore_path, stamp=stamp)]
    return log


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--assign", action="append", default=[], help="<split>=<match_id>, e.g. valid=MATCH_ID")
    ap.add_argument("--audit", help="audit JSON whose near-duplicate groups are excluded")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--apply-exclusions-only", action="store_true")
    g.add_argument("--probe", type=int, help="move only the first N planned frames (<= splits.probe_max), verify")
    g.add_argument("--restore", help="re-upload frames of a restore file that are missing from the project")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    if a.restore:
        key = load_secret(cfg.roboflow.api_key_env)
        r = cfg.roboflow
        uploader = RoboflowUploader(api_url=r.api_url, workspace=r.workspace, project=r.project, api_key=key,
                                    timeout_s=r.upload_timeout_s)
        try:
            log = restore(cfg, open_project(cfg, key), uploader, json.loads(Path(a.restore).read_text()),
                          REPO / cfg.paths.image_cache_dir)
        except PolicyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        print(f"restored {len(log)} frames: {log}")
        return 0
    plan = None
    try:
        assign = parse_assign(a.assign)
        exclude = []
        if a.audit:
            doc = json.loads(Path(a.audit).read_text())
            if doc.get("project") != cfg.roboflow.project or not doc["near_duplicates"]["computed"]:
                raise PolicyError("audit is for another project or has no near-duplicate result")
            exclude = [x for g_ in doc["near_duplicates"]["groups"] for x in g_["exclude"]]
        key = load_secret(cfg.roboflow.api_key_env)
        project = open_project(cfg, key)
        images = list_all(cfg, project)
        plan = plan_split(cfg, images, assign, exclude)
        now = datetime.now(timezone.utc)
        plan["created_utc"] = now.isoformat(timespec="seconds")
        plan["audit_file"] = Path(a.audit).name if a.audit else None
        plan["mode"] = ("apply" if a.apply else "apply_exclusions_only" if a.apply_exclusions_only
                        else f"probe{a.probe}" if a.probe is not None else "dry_run")
        r = cfg.roboflow
        uploader = RoboflowUploader(api_url=r.api_url, workspace=r.workspace, project=r.project, api_key=key,
                                    timeout_s=r.upload_timeout_s)
        log, error = [], None
        if a.apply or a.apply_exclusions_only or a.probe is not None:
            try:
                stamp = now.strftime('%Y%m%dT%H%M%SZ')
                log = apply_plan(cfg, project, plan, exclusions_only=a.apply_exclusions_only, uploader=uploader,
                                 images=images, cache_dir=REPO / cfg.paths.image_cache_dir, probe=a.probe,
                                 restore_path=REPO / cfg.paths.split_dir / f"restore_{stamp}.json", stamp=stamp)
            except Exception as e:   # any failure: record it in the manifest, then exit non-zero (fail closed)
                error = f"{type(e).__name__}: {e}"
        plan["applied"] = log
        plan["error"] = error
        out = REPO / cfg.paths.split_dir / f"split_{now.strftime('%Y%m%dT%H%M%SZ')}_{plan['mode']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out, plan)
        if error:
            raise PolicyError(error + f" (manifest: {out})")
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print(f"plan: {out}")
    print(f"after: {plan['match_splits_after']}; moves {plan['n_moves']}; exclusions {plan['n_exclude']}; "
          f"skipped {len(plan['skipped'])}; applied {len(log)} changes")
    for entry in log:
        print(f"  {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
