"""Test helper: rewrite a current spana.target document in the shape of an older version (fields added since)."""
from __future__ import annotations

import copy

ADDED = {"2.4.0": ("reentry_offer", "reentry_max_gap_s", "reentry_min_similarity", "reentry_min_margin"),
         "2.3.0": ("resume_after_doubt_resolved",), "2.2.0": ("tap_vouches_doubt",)}


def older(doc: dict, version: str) -> dict:
    """Drop the target_config keys, embeddings_sha256 and relink evidence kinds added after `version`."""
    d = copy.deepcopy(doc)
    order = ["2.2.0", "2.3.0", "2.4.0"]
    for v in order:
        if v > version:
            for k in ADDED[v]:
                d["target_config"].pop(k, None)
    if version < "2.4.0":
        d.pop("embeddings_sha256", None)
        for r in d["relinks"]:
            r["evidence"].pop("kind", None)
    d["schema_version"] = version
    return d
