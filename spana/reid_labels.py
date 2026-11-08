"""Re-ID identity labels: per-match manifests, tracklet identities, split guard, frozen-label loading.

Identities are match-local: (match_id, player). Splits are assigned per MATCH (never per frame) and recorded
with the footage rights and the consent basis before any labelling. EVAL is refused for training and tuning.
Frozen label versions are never edited: a correction is a new version with a reason.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

SPLITS = ("TRAIN", "DEV", "EVAL")
SIDES = ("A", "B", "GK_A", "GK_B", "REF", "OTHER")
PURPOSE_SPLIT = {"train": "TRAIN", "tune": "DEV", "eval": "EVAL"}
MATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
PLAYER_RE = re.compile(r"^[A-Za-z0-9_-]{1,16}$")
MANIFEST_SCHEMA = ("spana.reid_match", "1.1.0")
IDENTITIES_SCHEMA = ("spana.reid_identities", "1.0.0")
MANIFEST_FIELDS = ("schema", "schema_version", "match_id", "split", "video_sha256", "footage_licence",
                   "rights_holder", "consent_basis", "training_permitted", "agreement_reference", "permissions",
                   "notes")
# What a written permission must explicitly grant before a match may be TRAIN (Nathan, 2026-11-08). Each is
# recorded as true only when the agreement says so; anything else keeps the match out of TRAIN.
REQUIRED_PERMISSIONS = (
    "copy_and_process",          # copy / process the footage
    "ml_development",            # develop, train, test and improve ML / computer-vision models
    "commercial_use",            # that use may be commercial (SPANA is a commercial product)
    "player_consent",            # consent / authority covering the players' images for tracking / re-ID development
    "minors_covered",            # parent / guardian consent or another legal basis for any minors (or none filmed)
    "retain_derived_data",       # keep crops, annotations, embeddings and training outputs
)
RECOMMENDED_PERMISSIONS = ("model_survives_footage_removal",)


class ReidLabelError(ValueError):
    pass


def validate_manifest(m: dict) -> dict:
    """A match manifest is complete and consistent, or the label workflow refuses it."""
    missing = [k for k in MANIFEST_FIELDS if k not in m]
    if missing:
        raise ReidLabelError(f"match manifest missing fields: {missing}")
    if (m["schema"], m["schema_version"]) != MANIFEST_SCHEMA:
        raise ReidLabelError(f"unsupported match manifest schema {m['schema']} {m['schema_version']}")
    if not MATCH_ID_RE.match(str(m["match_id"])):
        raise ReidLabelError("match_id must be 3-64 chars of letters, digits, '_' or '-'")
    if m["split"] not in SPLITS:
        raise ReidLabelError(f"split must be one of {SPLITS}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(m["video_sha256"])):
        raise ReidLabelError("video_sha256 must be a sha256 hex digest")
    for k in ("footage_licence", "rights_holder", "consent_basis"):
        if not str(m[k]).strip():
            raise ReidLabelError(f"{k} is required for every match")
    if not isinstance(m["training_permitted"], bool):
        raise ReidLabelError("training_permitted must be true or false")
    perms = m["permissions"]
    keys = REQUIRED_PERMISSIONS + RECOMMENDED_PERMISSIONS
    if not isinstance(perms, dict) or set(perms) != set(keys) or not all(isinstance(perms[k], bool) for k in keys):
        raise ReidLabelError(f"permissions must give true/false for exactly: {list(keys)}")
    if m["split"] == "TRAIN":
        missing = [k for k in REQUIRED_PERMISSIONS if not perms[k]]
        if not m["training_permitted"] or missing:
            raise ReidLabelError("a TRAIN match needs training_permitted = true and written permission for: "
                                 f"{missing or REQUIRED_PERMISSIONS}")
        if not str(m["agreement_reference"]).strip():
            raise ReidLabelError("a TRAIN match needs agreement_reference (the written permission it relies on)")
    return m


def manifest_template(match_id: str, split: str, video_sha256: str) -> dict:
    return {"schema": MANIFEST_SCHEMA[0], "schema_version": MANIFEST_SCHEMA[1], "match_id": match_id,
            "split": split, "video_sha256": video_sha256, "footage_licence": "", "rights_holder": "",
            "consent_basis": "", "training_permitted": False, "agreement_reference": "",
            "permissions": {k: False for k in REQUIRED_PERMISSIONS + RECOMMENDED_PERMISSIONS}, "notes": ""}


def split_guard(split: str, purpose: str, eval_approval: Optional[str] = None) -> None:
    """train -> TRAIN only, tune -> DEV only, eval -> EVAL only and only with a recorded approval."""
    if purpose not in PURPOSE_SPLIT:
        raise ReidLabelError(f"unknown purpose {purpose!r}")
    if split == "EVAL" and purpose != "eval":
        raise ReidLabelError("refusing: EVAL labels can never be used for training or tuning")
    if PURPOSE_SPLIT[purpose] != split:
        raise ReidLabelError(f"refusing: purpose {purpose!r} may only use {PURPOSE_SPLIT[purpose]} labels, got {split}")
    if purpose == "eval" and not (eval_approval or "").strip():
        raise ReidLabelError("refusing: EVAL use needs an explicit, recorded approval")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_frozen(label_dir: Path, purpose: str, eval_approval: Optional[str] = None) -> Dict[str, object]:
    """Load a frozen re-ID label version after checking every file hash and the split guard."""
    label_dir = Path(label_dir)
    frozen = json.loads((label_dir / "FROZEN.json").read_text())
    split_guard(frozen["split"], purpose, eval_approval)
    for name, sha in frozen["files"].items():
        if sha256_bytes((label_dir / name).read_bytes()) != sha:
            raise ReidLabelError(f"{name} no longer matches its frozen hash")
    manifest = validate_manifest(json.loads((label_dir / "manifest.json").read_text()))
    if manifest["split"] != frozen["split"] or manifest["match_id"] != frozen["match_id"]:
        raise ReidLabelError("FROZEN.json and manifest disagree")
    rows: List[dict] = [json.loads(l) for l in (label_dir / "identities.jsonl").read_text().splitlines() if l]
    return {"frozen": frozen, "manifest": manifest, "identities": rows,
            "players": json.loads((label_dir / "players.json").read_text())}
