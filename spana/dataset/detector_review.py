"""Review file of mined detector-training candidates: schema `spana.detector_review` 1.0.0 (docs/INTERFACES.md).

One verdict per candidate object. Only MATCH_BALL, NOT_BALL and SPARE_BALL candidates are ever used; UNSURE and SKIP
are never used, and a frame they share with a used candidate is left out too (fail closed).

This module checks the STRUCTURE only (types, enums, required fields per verdict). Licence, split, pixels, hashes,
ball-box rules and category counts are the preflight's job (tools/dataset_v3.py).
"""
from __future__ import annotations

import re
from typing import Any, List

SCHEMA = ("spana.detector_review", "1.0.0")
MATCH_BALL, NOT_BALL, SPARE_BALL, UNSURE, SKIP = "MATCH_BALL", "NOT_BALL", "SPARE_BALL", "UNSURE", "SKIP"
VERDICTS = (MATCH_BALL, NOT_BALL, SPARE_BALL, UNSURE, SKIP)
USED = (MATCH_BALL, NOT_BALL, SPARE_BALL)
EXISTING_VERDICTS = (MATCH_BALL, SPARE_BALL)     # verdicts on a ball box of the base labels: it stays / it is removed
TRAIN = "TRAIN"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.jpg$")


class ReviewError(ValueError):
    pass


def is_sha256(v: Any) -> bool:
    return isinstance(v, str) and bool(_SHA.match(v))


def is_box(v: Any) -> bool:
    return (isinstance(v, list) and len(v) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == x for x in v)
            and v[0] < v[2] and v[1] < v[3])


def validate_review(doc: Any) -> List[str]:
    """Every structural problem of a review document (empty list = well formed). Missing image hashes are NOT a
    structural problem: they are reported by the preflight, per frame."""
    if not isinstance(doc, dict):
        return ["the review is not a JSON object"]
    bad: List[str] = []
    if (doc.get("schema"), doc.get("schema_version")) != SCHEMA:
        bad.append(f"schema / schema_version must be {SCHEMA[0]!r} / {SCHEMA[1]!r}")
    v = doc.get("review_version")
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        bad.append("review_version must be an integer >= 1")
    for k in ("reviewer", "reviewed_utc"):
        if not isinstance(doc.get(k), str) or not doc[k].strip():
            bad.append(f"{k} must be a non-empty string")
    if not is_sha256(doc.get("base_labels_sha256")):
        bad.append("base_labels_sha256 must be the sha256 of the frozen labels the review was made against")
    cands, existing = doc.get("candidates"), doc.get("existing_labels")
    if not isinstance(cands, list) or not isinstance(existing, list):
        return bad + ["candidates and existing_labels must be lists"]
    seen = set()
    for n, c in enumerate(cands):
        w = f"candidates[{n}]"
        if not isinstance(c, dict):
            bad.append(f"{w}: not an object")
            continue
        cid = c.get("id")
        if not isinstance(cid, str) or not cid.strip() or cid in seen:
            bad.append(f"{w}: id must be a non-empty string, unique in the review")
        seen.add(cid)
        w = f"candidate {cid!r}"
        for k in ("match_id", "split"):
            if not isinstance(c.get(k), str) or not c[k].strip():
                bad.append(f"{w}: {k} must be a non-empty string")
        verdict = c.get("verdict")
        if verdict not in VERDICTS:
            bad.append(f"{w}: verdict must be one of {list(VERDICTS)}")
            continue
        cat = c.get("category")
        if verdict == NOT_BALL and (not isinstance(cat, str) or not cat.strip()):
            bad.append(f"{w}: NOT_BALL needs a look-alike category")
        if verdict != NOT_BALL and cat is not None:
            bad.append(f"{w}: only NOT_BALL carries a category")
        frames = c.get("frames")
        if not isinstance(frames, list) or not frames:
            bad.append(f"{w}: frames must be a non-empty list")
            continue
        names = set()
        for f in frames:
            if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not _NAME.match(f["name"]):
                bad.append(f"{w}: every frame needs a name (a plain .jpg file name)")
                continue
            fw = f"{w} frame {f['name']}"
            if f["name"] in names:
                bad.append(f"{fw}: listed twice")
            names.add(f["name"])
            if f.get("sha256") is not None and not is_sha256(f["sha256"]):
                bad.append(f"{fw}: sha256 is not a sha256")
            for k in ("object_box", "match_ball_box"):
                if f.get(k) is not None and not is_box(f[k]):
                    bad.append(f"{fw}: {k} must be [x1, y1, x2, y2] with x1 < x2 and y1 < y2, or null")
            if not isinstance(f.get("match_ball_absent", False), bool):
                bad.append(f"{fw}: match_ball_absent must be true or false")
    for n, e in enumerate(existing):
        w = f"existing_labels[{n}]"
        if not isinstance(e, dict) or not isinstance(e.get("image_id"), str) or not e["image_id"].strip():
            bad.append(f"{w}: image_id must be a non-empty string")
            continue
        if not is_box(e.get("box")):
            bad.append(f"{w}: box must be [x1, y1, x2, y2]")
        if e.get("verdict") not in EXISTING_VERDICTS:
            bad.append(f"{w}: verdict must be one of {list(EXISTING_VERDICTS)} (anything else: settle it first)")
    return bad
