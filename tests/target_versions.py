"""Test helper: rewrite a current spana.target document in the shape of an older version (fields added since)."""
from __future__ import annotations

import copy

ADDED = {"2.10.0": ("confirm_contradict_reid_margin", "confirm_contradict_colour_margin", "gallery_reid_resolve",
                    "confirmation_scope", "reid_resolve_enabled", "reid_margin"),
         "2.9.0": ("resume_after_reid",),
         "2.8.0": ("offer_hold_s",),
         "2.7.0": ("reentry_exit_edge", "reentry_exit_edge_max_frac"),
         "2.6.0": ("reentry_signature", "reentry_wait_samples", "reentry_withdraw"),
         "2.5.0": ("reentry_mutual", "reentry_position_weight"),
         "2.4.0": ("reentry_offer", "reentry_max_gap_s", "reentry_min_similarity", "reentry_min_margin"),
         "2.3.0": ("resume_after_doubt_resolved",), "2.2.0": ("tap_vouches_doubt",)}


def older(doc: dict, version: str) -> dict:
    """Drop the target_config keys, embeddings_sha256 and relink evidence kinds added after `version`."""
    d = copy.deepcopy(doc)
    order = ["2.2.0", "2.3.0", "2.4.0", "2.5.0", "2.6.0", "2.7.0", "2.8.0", "2.9.0", "2.10.0"]
    vt = lambda v: tuple(int(x) for x in v.split("."))      # noqa: E731
    for v in order:
        if vt(v) > vt(version):
            for k in ADDED[v]:
                d["target_config"].pop(k, None)
    if vt(version) < vt("2.10.0"):
        for k in ("crop_features", "claims_identities", "confirmation_log", "gallery", "gallery_reid_log"):
            d.pop(k, None)
    if vt(version) < vt("2.7.0"):
        for e in d.get("reentry_log", []):
            e.pop("exit_excluded", None)
    if vt(version) < vt("2.5.0"):
        d.pop("reentry_log", None)
        for r in d["relinks"]:
            ev = r["evidence"]
            if ev.get("kind") == "reid":
                ev["second_similarity"] = ev.pop("second_combined")
                for k in ("position", "combined", "position_informative"):
                    ev.pop(k)
    if vt(version) < vt("2.4.0"):
        d.pop("embeddings_sha256", None)
        for r in d["relinks"]:
            r["evidence"].pop("kind", None)
    d["schema_version"] = version
    return d
