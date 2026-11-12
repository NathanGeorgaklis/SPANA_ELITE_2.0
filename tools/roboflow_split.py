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
Default is a dry run (manifest only). --apply-exclusions-only adds the exclusion tags. --apply also needs every
split move to be possible: Roboflow's API cannot move single images between splits (only a random ratio rebalance,
which would split matches), so with splits.move_method "none" --apply refuses whenever a move is planned and
nothing is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from spana.artifacts import write_json_atomic
from spana.dataset.config import DatasetConfig, PolicyError, load_dataset_config, match_of
from spana.dataset.prep import match_tag, open_project, sort_key
from spana.dataset.rf_project import ProjectImage
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


def apply_plan(cfg: DatasetConfig, project, plan: dict, *, exclusions_only: bool) -> List[dict]:
    """Executes a plan. Refuses (changing nothing) if a split move is needed and no move method exists."""
    if not exclusions_only and plan["n_moves"] and cfg.splits.move_method == "none":
        raise PolicyError(f"refused: {plan['n_moves']} frames need a split move, but Roboflow's API cannot move "
                          "single images between splits (splits.move_method: none). Nothing was changed. "
                          "Use --apply-exclusions-only for the exclusion tags.")
    log = []
    for r in plan["rows"]:
        if r["exclude"] and not r["already_excluded"]:
            project.update_tags(r["id"], add=[cfg.splits.excluded_tag])
            log.append({"id": r["id"], "action": "tag", "tag": cfg.splits.excluded_tag})
    return log


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--assign", action="append", default=[], help="<split>=<match_id>, e.g. valid=MATCH_ID")
    ap.add_argument("--audit", help="audit JSON whose near-duplicate groups are excluded")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--apply-exclusions-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_dataset_config(a.config)
    try:
        assign = parse_assign(a.assign)
        exclude = []
        if a.audit:
            doc = json.loads(Path(a.audit).read_text())
            if doc.get("project") != cfg.roboflow.project or not doc["near_duplicates"]["computed"]:
                raise PolicyError("audit is for another project or has no near-duplicate result")
            exclude = [x for g_ in doc["near_duplicates"]["groups"] for x in g_["exclude"]]
        project = open_project(cfg, load_secret(cfg.roboflow.api_key_env))
        plan = plan_split(cfg, project.list_images(cfg.audit.page_size), assign, exclude)
        now = datetime.now(timezone.utc)
        plan["created_utc"] = now.isoformat(timespec="seconds")
        plan["audit_file"] = Path(a.audit).name if a.audit else None
        plan["mode"] = "apply" if a.apply else ("apply_exclusions_only" if a.apply_exclusions_only else "dry_run")
        log = []
        if a.apply or a.apply_exclusions_only:
            log = apply_plan(cfg, project, plan, exclusions_only=a.apply_exclusions_only)
        plan["applied"] = log
        out = REPO / cfg.paths.split_dir / f"split_{now.strftime('%Y%m%dT%H%M%SZ')}_{plan['mode']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out, plan)
    except PolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print(f"plan: {out}")
    print(f"after: {plan['match_splits_after']}; moves {plan['n_moves']}; exclusions {plan['n_exclude']}; "
          f"skipped {len(plan['skipped'])}; applied {len(log)} changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
