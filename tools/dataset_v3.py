"""Detector dataset version 3: from the review of the mined candidates to a staged upload
(docs/DETECTOR_BALL_RETRAINING_PLAN.md, task 3). Nothing here uploads, labels or trains.

    # read-only: is this review file ready? (nothing is written)
    python -m tools.dataset_v3 check --review REVIEW.json

    # freeze the review, preflight it, assemble the new frames and the scale copies, stage the upload
    python -m tools.dataset_v3 build --review REVIEW.json

The review file is `spana.detector_review` 1.0.0 (spana/dataset/detector_review.py, docs/INTERFACES.md): one verdict
per mined candidate (MATCH_BALL / NOT_BALL + look-alike category / SPARE_BALL / UNSURE / SKIP) and one verdict per
audited ball box of the base labels (MATCH_BALL / SPARE_BALL). UNSURE and SKIP candidates are never used, and a
frame they share with a used candidate is left out as well.

Freeze (`build` only): the review bytes go to <review_dir>/v<review_version>/review_frozen.json with FROZEN.json
(sha256, provenance). A frozen version is never overwritten; the same bytes again are accepted as the same freeze.

Preflight. Refuses (exit 3, every reason listed, nothing staged) on:
  * a candidate whose match is not a TRAIN match on the licence allow-list with third-party processing confirmed,
    or is not in preflight.split_matches (config/roboflow.yaml);
  * a candidate not declared TRAIN, or a frame whose name or image sha256 is in a frozen DEV label set;
  * a frame without a sha256, not found in frames_dir, unreadable, or whose bytes do not match its sha256;
  * a ball box on a spare ball or on a look-alike;
  * two ball boxes on one frame (new frames, and the base labels after the review's removals);
  * a MATCH_BALL frame without a ball box; a look-alike / spare-ball frame that neither boxes the match ball nor
    states it is not visible (an unboxed real ball would be taught as background);
  * a look-alike category with fewer frames than min_per_category (config/detector_mine.yaml). The shortfall is
    reported; nothing is padded;
  * a review made against other base labels, an audited box that is not in the base labels, a frame name already
    used by the base labels, a box outside its frame, an unknown category, a base frame missing from the image cache.

Assembly, into <stage_dir>/v<dataset_version>_review<review_version>/ (never overwritten):
  * mined frames: the match ball boxed as the ball class; spare balls and look-alikes get no box; tags for why the
    frame was mined, `has-spare-ball`, and "person labels pending". They are staged WITHOUT the review tag: the
    other classes are labelled and reviewed in the project, and the existing version gate keeps them out until then;
  * base-label changes: each ball box the review calls SPARE_BALL is listed for removal by its project image id and
    box, with the `has-spare-ball` tag for that frame;
  * scale copies of base frames (after those removals), in their source's split;
  * stage.json: every staged file with sha256, split, tags and boxes, the project changes, counts. `uploaded` false.
The same inputs give byte-identical staged output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from spana.artifacts import write_json_atomic
from spana.config import ConfigError, typed_from_dict
from spana.dataset import detector_review as rv
from spana.dataset.config import DatasetConfig, PolicyError, authorize, load_dataset_config
from spana.dataset.manifest import frame_tags

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "config" / "dataset_v3.yaml"
CONFIG_VERSION = "1.0.0"
STAGE_SCHEMA = ("spana.dataset_stage", "1.0.0")
FROZEN_RULE = "Frozen before the dataset is assembled. Never edit; a correction is a new review_version."


@dataclass(frozen=True)
class Tags:
    spare_ball: str
    small_ball: str
    negative_prefix: str
    persons_pending: str
    version: str
    scale_copy_prefix: str
    source_prefix: str


@dataclass(frozen=True)
class ScaleCopies:
    enabled: bool
    require_ball: bool
    shrink_factors: Tuple[float, ...]
    pad_value: int
    tile_frac_height: float
    tile_overlap_frac: float
    tile_enlarge: Tuple[float, ...]
    min_box_visible_frac: float
    jpeg_quality: int


@dataclass(frozen=True)
class V3Config:
    schema_version: str
    dataset_version: int
    base_labels: str
    base_labels_sha256: str
    dev_labels: Tuple[str, ...]
    review_dir: str
    frames_dir: str
    stage_dir: str
    same_ball_iou: float
    object_overlap_iou: float
    not_lookalike_categories: Tuple[str, ...]
    tags: Tags
    scale_copies: ScaleCopies


def load_config(path=DEFAULT_CONFIG) -> V3Config:
    cfg = typed_from_dict(V3Config, yaml.safe_load(Path(path).read_text(encoding="utf-8")), "dataset_v3")
    sc = cfg.scale_copies
    if cfg.schema_version != CONFIG_VERSION:
        raise ConfigError(f"dataset_v3 schema_version {cfg.schema_version!r} != {CONFIG_VERSION!r}")
    if not (cfg.dataset_version >= 1 and rv.is_sha256(cfg.base_labels_sha256) and cfg.dev_labels
            and 0 < cfg.same_ball_iou <= 1 and 0 < cfg.object_overlap_iou <= 1):
        raise ConfigError("dataset_v3: need dataset_version >= 1, a base_labels_sha256, dev_labels and IoUs in (0, 1]")
    if any(not str(t).strip() for t in vars(cfg.tags).values()):
        raise ConfigError("dataset_v3.tags: every tag must be a non-empty string")
    if not (all(0 < f < 1 for f in sc.shrink_factors) and all(e > 0 for e in sc.tile_enlarge)
            and 0 < sc.tile_frac_height <= 1 and 0 <= sc.tile_overlap_frac < 1 and 0 < sc.min_box_visible_frac <= 1
            and 0 <= sc.pad_value <= 255 and 1 <= sc.jpeg_quality <= 100):
        raise ConfigError("dataset_v3.scale_copies: shrink factors in (0, 1), enlarge > 0, fractions in range")
    return cfg


@dataclass(frozen=True)
class Context:
    root: Path
    cfg: V3Config
    dcfg: DatasetConfig
    min_per_category: int
    categories: Tuple[str, ...]            # look-alike categories a NOT_BALL verdict may name
    config_sha256: Dict[str, str]


def load_context(config, dataset_config, mine_config, diag_config, root: Path = REPO) -> Context:
    from tools import detector_mine as dm
    from tools.diagnostics import ball_tiling_fp as fp
    cfg, dcfg = load_config(config), load_dataset_config(dataset_config)
    cats = tuple(c for c in fp.load_config(diag_config).categories if c not in cfg.not_lookalike_categories)
    shas = {k: sha256_bytes(Path(p).read_bytes()) for k, p in (
        ("dataset_v3", config), ("dataset", dataset_config), ("detector_mine", mine_config), ("ball_fp_diag", diag_config))}
    return Context(Path(root), cfg, dcfg, dm.load_config(mine_config).min_per_category, cats, shas)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Refused(Exception):
    """The build stops here. `reasons`: (code, text, items) in the order found."""

    def __init__(self, reasons: Sequence[Tuple[str, str, List[str]]]):
        self.reasons = list(reasons)
        self.codes = {c for c, _, _ in self.reasons}
        super().__init__("\n".join(self.lines()))

    def lines(self) -> List[str]:
        out = []
        for code, text, items in self.reasons:
            more = f" (+{len(items) - 5} more)" if len(items) > 5 else ""
            out.append(f"[{code}] {text}" + (f": {', '.join(items[:5])}{more}" if items else ""))
        return out


class _Reasons:
    def __init__(self) -> None:
        self._r: Dict[Tuple[str, str], List[str]] = {}

    def add(self, code: str, text: str, item: Optional[str] = None) -> None:
        items = self._r.setdefault((code, text), [])
        if item is not None and item not in items:
            items.append(item)

    def __bool__(self) -> bool:
        return bool(self._r)

    def refused(self) -> Refused:
        return Refused([(c, t, i) for (c, t), i in self._r.items()])


# ------------------------------------------------------------------------------------------------------ freeze
def freeze(review_path, ctx: Context) -> Tuple[Path, dict]:
    """Freeze the review bytes under review_dir (same pattern as tools.freeze_ball_labels). Returns (dir, document).
    A malformed review is refused and not frozen."""
    raw = Path(review_path).read_bytes()
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise Refused([("SCHEMA", f"the review is not valid JSON ({e})", [])]) from e
    bad = rv.validate_review(doc)
    if bad:
        raise Refused([("SCHEMA", b, []) for b in bad])
    dest = ctx.root / ctx.cfg.review_dir / f"v{doc['review_version']}"
    sha = sha256_bytes(raw)
    if dest.exists():
        if load_frozen(dest)[1]["record_sha256"] != sha:
            raise Refused([("FROZEN_EXISTS", f"{dest} already holds a different review: frozen reviews are never "
                            "overwritten; a correction is a new review_version with a reason", [])])
        return dest, doc
    verdicts = [c["verdict"] for c in doc["candidates"]]
    dest.mkdir(parents=True)
    (dest / "review_frozen.json").write_bytes(raw)
    write_json_atomic(dest / "FROZEN.json", {
        "schema": rv.SCHEMA[0], "schema_version": rv.SCHEMA[1], "split": rv.TRAIN,
        "review_version": doc["review_version"], "reviewer": doc["reviewer"], "reviewed_utc": doc["reviewed_utc"],
        "frozen_in_repo_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_sha256": sha, "base_labels_sha256": doc["base_labels_sha256"], "tool": "tools.dataset_v3",
        "candidates": {v: verdicts.count(v) for v in rv.VERDICTS},
        "existing_labels": {v: sum(e["verdict"] == v for e in doc["existing_labels"]) for v in rv.EXISTING_VERDICTS},
        "rule": FROZEN_RULE})
    return dest, doc


def load_frozen(dest: Path) -> Tuple[dict, dict]:
    """(review document, FROZEN.json) of a frozen review; refuses one whose bytes changed after freezing."""
    raw, frozen = (dest / "review_frozen.json").read_bytes(), json.loads((dest / "FROZEN.json").read_text())
    if sha256_bytes(raw) != frozen["record_sha256"]:
        raise Refused([("FROZEN_CHANGED", f"{dest}: review_frozen.json no longer matches its frozen sha256", [])])
    return json.loads(raw), frozen


# --------------------------------------------------------------------------------------------------- preflight
def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _on(ball: Sequence[float], obj: Sequence[float], min_iou: float) -> bool:
    cx, cy = (ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2
    return (obj[0] <= cx <= obj[2] and obj[1] <= cy <= obj[3]) or _iou(ball, obj) >= min_iou


def _box(b: Sequence[float]) -> List[float]:
    return [round(float(x), 2) for x in b]


def preflight(doc: dict, ctx: Context) -> dict:
    """Check a (well-formed) review against policy, pixels and the base labels. Returns the plan the assembly follows;
    raises Refused with every reason if anything does not hold. Reads only."""
    import cv2
    cfg, dcfg, root, why = ctx.cfg, ctx.dcfg, ctx.root, _Reasons()
    ball = dcfg.prelabel.ball_class
    base_path = root / cfg.base_labels
    if not base_path.is_file():
        raise Refused([("BASE_LABELS", f"base labels not found: {cfg.base_labels}", [])])
    base_raw = base_path.read_bytes()
    if sha256_bytes(base_raw) != cfg.base_labels_sha256:
        why.add("BASE_LABELS", f"{cfg.base_labels} does not match base_labels_sha256 of the config")
    if doc["base_labels_sha256"] != cfg.base_labels_sha256:
        why.add("BASE_LABELS", "the review was made against other base labels than the configured ones")
    base = json.loads(base_raw)
    base_by_id = {f["id"]: f for f in base["frames"]}
    base_names = {f["name"] for f in base["frames"]}
    dev_names, dev_shas = set(), set()
    for p in cfg.dev_labels:
        if not (root / p).is_file():
            why.add("DEV_GUARD", "a configured DEV label set is missing, so DEV frames cannot be ruled out", p)
            continue
        for f in json.loads((root / p).read_text())["frames"]:
            dev_names.add(f["name"])
            dev_shas.add(f.get("image_sha256"))
    split_of = dcfg.preflight.expected()                     # match_id -> roboflow split

    # -- candidates: licence, split, category --------------------------------------------------------------------
    frames: Dict[str, dict] = {}
    unused = {v: 0 for v in (rv.UNSURE, rv.SKIP)}
    for c in sorted(doc["candidates"], key=lambda c: c["id"]):
        cid, mid = c["id"], c["match_id"]
        if c["split"] != rv.TRAIN:
            why.add("DEV_OR_EVAL", f"declared split {c['split']!r}: only TRAIN footage enters the dataset (DEV and "
                    "EVAL frames never do)", cid)
        try:
            authorize(dcfg, mid, "train", third_party=True)
            if mid not in split_of:
                why.add("NOT_ALLOW_LISTED", f"match {mid!r} has no split in preflight.split_matches", cid)
        except PolicyError as e:
            why.add("NOT_ALLOW_LISTED", str(e), cid)
        if c["verdict"] == rv.NOT_BALL and c["category"] not in ctx.categories:
            why.add("CATEGORY", f"{c['category']!r} is not a look-alike category (allowed: {list(ctx.categories)})", cid)
        if c["verdict"] in unused:
            unused[c["verdict"]] += 1
        for f in c["frames"]:
            fr = frames.setdefault(f["name"], {"name": f["name"], "match_id": mid, "entries": [], "left_out": False})
            if fr["match_id"] != mid:
                why.add("FRAME_NAME", "one frame is listed under two matches", f["name"])
            fr["entries"].append((c, f))
            fr["left_out"] |= c["verdict"] not in rv.USED

    # -- frames: names, hashes, pixels, ball rules ---------------------------------------------------------------
    new_frames, left_out = [], []
    for name in sorted(frames):
        fr = frames[name]
        mid = fr["match_id"]
        declared = sorted({f["sha256"] for _, f in fr["entries"] if f.get("sha256")})
        if name in dev_names or dev_shas & set(declared):
            why.add("DEV_OR_EVAL", "the frame is in a frozen DEV label set", name)
        if fr["left_out"]:                                   # shared with an UNSURE / SKIP candidate: never used
            if any(c["verdict"] in rv.USED for c, _ in fr["entries"]):
                left_out.append(name)
            continue
        if not name.startswith(mid + "_"):
            why.add("FRAME_NAME", "a frame's file name must start with its match id", name)
        if name in base_names:
            why.add("FRAME_NAME", "the frame name is already used by the base labels", name)
        if any(not f.get("sha256") for _, f in fr["entries"]):
            why.add("MISSING_HASH", "a used frame has no sha256 in the review", name)
            continue
        if len(declared) != 1:
            why.add("HASH_MISMATCH", "the review gives two different sha256 for one frame", name)
            continue
        path = root / cfg.frames_dir / name
        if not path.is_file():
            why.add("MISSING_FRAME", f"a used frame is not in {cfg.frames_dir}", name)
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != declared[0]:
            why.add("HASH_MISMATCH", "the frame file does not match the sha256 in the review", name)
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            why.add("MISSING_FRAME", "the frame file is not a readable image", name)
            continue
        h, w = img.shape[:2]
        balls, absent, pending = [], False, False
        ok = True
        for c, f in fr["entries"]:
            for k in ("object_box", "match_ball_box"):
                b = f.get(k)
                if b is not None and not (0 <= b[0] and 0 <= b[1] and b[2] <= w and b[3] <= h):
                    why.add("BOX", f"a box lies outside its {w} x {h} frame", name)
                    ok = False
            mb = f.get("match_ball_box")
            if mb is not None and f.get("match_ball_absent", False):
                why.add("BALL_UNCLEAR", "match_ball_box and match_ball_absent are both given", name)
                ok = False
            if mb is not None and not any(_iou(mb, x) >= cfg.same_ball_iou for x in balls):
                balls.append(mb)
            absent |= f.get("match_ball_absent", False)
            if c["verdict"] == rv.MATCH_BALL and mb is None:
                why.add("MATCH_BALL_NO_BOX", "a MATCH_BALL frame has no match_ball_box (the mined ring is a hint, "
                        "not a label)", name)
                ok = False
            if c["verdict"] != rv.MATCH_BALL:
                if f.get("object_box") is None:
                    why.add("BOX", f"a {c['verdict']} frame needs the object_box of that object", name)
                    ok = False
                pending |= mb is None and not f.get("match_ball_absent", False)
        if len(balls) > 1:
            why.add("TWO_BALL_BOXES", "two ball boxes on one frame (there is one match ball)", name)
            ok = False
        if balls and absent:
            why.add("BALL_UNCLEAR", "one candidate boxes the match ball, another says it is not visible", name)
            ok = False
        if not balls and pending:
            why.add("BALL_UNACCOUNTED", "a look-alike / spare-ball frame neither boxes the match ball nor states it "
                    "is not visible (match_ball_absent)", name)
            ok = False
        for c, f in fr["entries"]:
            if c["verdict"] != rv.MATCH_BALL and f.get("object_box") is not None and any(
                    _on(b, f["object_box"], cfg.object_overlap_iou) for b in balls):
                spare = c["verdict"] == rv.SPARE_BALL
                why.add("BOXED_SPARE_BALL" if spare else "BOXED_LOOKALIKE",
                        f"a ball box is on a {'spare ball' if spare else 'look-alike'}: only the match ball is boxed",
                        name)
                ok = False
        if not ok:
            continue
        verdicts = {c["verdict"] for c, _ in fr["entries"]}
        cats = sorted({c["category"] for c, _ in fr["entries"] if c["verdict"] == rv.NOT_BALL})
        split = split_of.get(mid, "")
        tags = frame_tags(mid, split) + [cfg.tags.version, cfg.tags.persons_pending]
        tags += [cfg.tags.small_ball] if rv.MATCH_BALL in verdicts else []
        tags += [cfg.tags.negative_prefix + c for c in cats]
        tags += [cfg.tags.spare_ball] if rv.SPARE_BALL in verdicts else []
        new_frames.append({"name": name, "sha256": declared[0], "match_id": mid, "split": split, "width": w,
                           "height": h, "boxes": [{"class": ball, "box": _box(b)} for b in balls],
                           "tags": sorted(tags), "categories": cats, "verdicts": sorted(verdicts),
                           "candidates": sorted(c["id"] for c, _ in fr["entries"])})
    if not new_frames and not why:
        why.add("EMPTY", "the review leaves no frame to add")

    # -- look-alike categories: enough frames each, or the shortfall -----------------------------------------------
    per_cat = {c: sum(c in f["categories"] for f in new_frames) for c in ctx.categories}
    for c, n in per_cat.items():
        if 0 < n < ctx.min_per_category:
            why.add("CATEGORY_SHORTFALL", f"look-alike category {c!r}: {n} frames, {ctx.min_per_category - n} short of "
                    f"min_per_category {ctx.min_per_category} (nothing is padded)")

    # -- the base labels after the review's verdicts on their ball boxes ---------------------------------------------
    remove: Dict[str, List[List[float]]] = {}
    seen = set()
    for e in doc["existing_labels"]:
        f, box = base_by_id.get(e["image_id"]), [float(x) for x in e["box"]]
        key = (e["image_id"], tuple(box))
        if f is None or box not in [b["box"] for b in f["boxes"] if b["class"] == ball]:
            why.add("EXISTING_LABEL", "no such ball box in the base labels", f"{e['image_id']} {e['box']}")
        elif key in seen:
            why.add("EXISTING_LABEL", "a ball box of the base labels has two verdicts", f"{e['image_id']} {e['box']}")
        elif e["verdict"] == rv.SPARE_BALL:
            remove.setdefault(e["image_id"], []).append(box)
        seen.add(key)
    changes, base_frames = [], []
    for f in sorted(base["frames"], key=lambda f: f["name"]):
        gone = remove.get(f["id"], [])
        boxes = [b for b in f["boxes"] if not (b["class"] == ball and b["box"] in gone)]
        for box in sorted(gone):
            changes.append({"op": "remove_box", "image_id": f["id"], "name": f["name"], "class": ball, "box": box})
        if gone:
            changes.append({"op": "add_tag", "image_id": f["id"], "name": f["name"], "tag": cfg.tags.spare_ball})
        if sum(b["class"] == ball for b in boxes) > 1:
            why.add("TWO_BALL_BOXES", "two ball boxes on one frame (there is one match ball)", f["name"])
        base_frames.append({**f, "boxes": boxes, "spare_ball": bool(gone)})

    # -- frames the scale copies are made from ----------------------------------------------------------------------
    sources = []
    if cfg.scale_copies.enabled:
        for f in base_frames:
            if f["split"] not in set(split_of.values()) or split_of.get(f["match_id"]) != f["split"]:
                continue                                     # platform test frames and anything out of place: no copy
            if cfg.scale_copies.require_ball and not any(b["class"] == ball for b in f["boxes"]):
                continue
            if not (root / dcfg.paths.image_cache_dir / f"{f['id']}.jpg").is_file():
                why.add("MISSING_FRAME", f"a base frame needed for scale copies is not in {dcfg.paths.image_cache_dir}",
                        f["name"])
            sources.append(f)
    if why:
        raise why.refused()
    return {"new_frames": new_frames, "left_out": left_out, "unused_candidates": unused, "changes": changes,
            "copy_sources": sources, "per_category": {c: n for c, n in per_cat.items() if n},
            "base_frames": len(base_frames)}


# ------------------------------------------------------------------------------------------------- scale copies
def _offsets(length: int, tile: int, overlap: float) -> List[int]:
    if length <= tile:
        return [0]
    n = math.ceil((length - tile) / (tile * (1 - overlap)) - 1e-9) + 1
    return sorted({int(round(i * (length - tile) / (n - 1))) for i in range(n)})


def scale_copies(img: np.ndarray, boxes: Sequence[dict], sc: ScaleCopies, ball: str) -> List[dict]:
    """Copies of one labelled frame: [{suffix, kind, image, boxes}]. Deterministic; boxes follow the pixels."""
    import cv2
    h, w = img.shape[:2]
    out = []
    for f in sc.shrink_factors:
        nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
        ox, oy = (w - nw) // 2, (h - nh) // 2
        canvas = np.full_like(img, sc.pad_value)
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        sx, sy = nw / w, nh / h
        out.append({"suffix": f"__shrink{int(round(f * 1000)):04d}", "kind": "shrink", "image": canvas,
                    "boxes": [{"class": b["class"], "box": _box([b["box"][0] * sx + ox, b["box"][1] * sy + oy,
                                                                 b["box"][2] * sx + ox, b["box"][3] * sy + oy])}
                              for b in boxes]})
    t = max(1, min(w, h, int(round(sc.tile_frac_height * h))))
    for y0 in _offsets(h, t, sc.tile_overlap_frac):
        for x0 in _offsets(w, t, sc.tile_overlap_frac):
            kept, cut, has_ball = [], False, False
            for b in boxes:
                x1, y1, x2, y2 = b["box"]
                ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + t), min(y2, y0 + t)
                frac = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / ((x2 - x1) * (y2 - y1))
                if b["class"] == ball:
                    cut |= 0 < frac < 1                       # a ball cut by the tile edge: no such tile
                    has_ball |= frac == 1
                    if frac < 1:
                        continue
                elif frac < sc.min_box_visible_frac:
                    continue
                kept.append((b["class"], [ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0]))
            if cut or (sc.require_ball and not has_ball):
                continue
            crop = img[y0:y0 + t, x0:x0 + t]
            for e in sc.tile_enlarge:
                side = max(1, int(round(t * e)))
                s = side / t
                out.append({"suffix": f"__tile{int(round(e * 1000)):04d}_x{x0:04d}_y{y0:04d}", "kind": "tile",
                            "image": cv2.resize(crop, (side, side),
                                                interpolation=cv2.INTER_CUBIC if e > 1 else cv2.INTER_AREA),
                            "boxes": [{"class": c, "box": _box([v * s for v in bx])} for c, bx in kept]})
    return out


# ---------------------------------------------------------------------------------------------------- assembly
def stage_dir(ctx: Context, review_version: int) -> Path:
    return ctx.root / ctx.cfg.stage_dir / f"v{ctx.cfg.dataset_version}_review{review_version}"


def assemble(plan: dict, frozen: dict, ctx: Context) -> Path:
    """Write the staged upload. Nothing is sent anywhere. Refuses to overwrite a staged version."""
    import cv2
    cfg, dcfg, root = ctx.cfg, ctx.dcfg, ctx.root
    ball, sc = dcfg.prelabel.ball_class, cfg.scale_copies
    dest = stage_dir(ctx, frozen["review_version"])
    if dest.exists():
        raise Refused([("STAGE_EXISTS", f"{dest} already exists: a staged version is never overwritten", [])])
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        staged, copies = [], []
        for f in plan["new_frames"]:
            rel = f"images/{f['split']}/{f['name']}"
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / cfg.frames_dir / f["name"], tmp / rel)
            staged.append({**f, "file": rel, "kind": "mined"})
        for f in plan["copy_sources"]:
            data = (root / dcfg.paths.image_cache_dir / f"{f['id']}.jpg").read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (f["height"], f["width"]):
                raise Refused([("MISSING_FRAME", "a cached base frame is unreadable or not the size its labels were "
                                "made on", [f["name"]])])
            for c in scale_copies(img, f["boxes"], sc, ball):
                ok, enc = cv2.imencode(".jpg", c["image"], [cv2.IMWRITE_JPEG_QUALITY, sc.jpeg_quality])
                if not ok:
                    raise Refused([("ENCODE", "a scale copy could not be encoded", [f["name"]])])
                name = Path(f["name"]).stem + c["suffix"] + ".jpg"
                rel = f"images/{f['split']}/{name}"
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                (tmp / rel).write_bytes(enc.tobytes())
                tags = frame_tags(f["match_id"], f["split"]) + [
                    cfg.tags.version, cfg.tags.scale_copy_prefix + c["kind"], cfg.tags.source_prefix + f["id"]]
                tags += [cfg.tags.spare_ball] if f["spare_ball"] else []
                copies.append({"name": name, "file": rel, "sha256": sha256_bytes(enc.tobytes()), "kind": c["kind"],
                               "match_id": f["match_id"], "split": f["split"], "width": int(c["image"].shape[1]),
                               "height": int(c["image"].shape[0]), "boxes": c["boxes"], "tags": sorted(tags),
                               "source_image_id": f["id"], "source_name": f["name"], "source_sha256": sha256_bytes(data)})
        names = [e["name"] for e in staged + copies]
        if len(set(names)) != len(names):
            raise Refused([("FRAME_NAME", "two staged files have the same name", [])])
        splits = sorted({e["split"] for e in staged + copies})
        manifest = {
            "schema": STAGE_SCHEMA[0], "schema_version": STAGE_SCHEMA[1], "dataset_version": cfg.dataset_version,
            "uploaded": False, "project": dcfg.roboflow.project,
            "review": {"review_version": frozen["review_version"], "record_sha256": frozen["record_sha256"],
                       "reviewer": frozen["reviewer"]},
            "base_labels": {"file": cfg.base_labels, "sha256": cfg.base_labels_sha256, "frames": plan["base_frames"]},
            "config_sha256": ctx.config_sha256,
            "counts": {
                "mined_frames": {s: sum(f["split"] == s for f in staged) for s in splits},
                "match_ball_frames": sum(bool(f["boxes"]) for f in staged),
                "spare_ball_frames": sum(rv.SPARE_BALL in f["verdicts"] for f in staged),
                "lookalike_frames_per_category": plan["per_category"],
                "scale_copies": {k: {s: sum(c["kind"] == k and c["split"] == s for c in copies) for s in splits}
                                 for k in ("shrink", "tile")},
                "frames_left_out_shared_with_unsure_or_skip": len(plan["left_out"]),
                "candidates_never_used": plan["unused_candidates"],
                "project_changes": len(plan["changes"])},
            "project_changes": plan["changes"], "frames_left_out": plan["left_out"],
            "frames": staged, "scale_copies": copies}
        write_json_atomic(tmp / "stage.json", manifest)
        os.rename(tmp, dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return dest


# --------------------------------------------------------------------------------------------------------- CLI
def build(review_path, ctx: Context) -> Path:
    dest, _ = freeze(review_path, ctx)
    doc, frozen = load_frozen(dest)                          # everything after this reads the frozen bytes
    if stage_dir(ctx, frozen["review_version"]).exists():
        raise Refused([("STAGE_EXISTS", f"{stage_dir(ctx, frozen['review_version'])} already exists: a staged "
                        "version is never overwritten", [])])
    return assemble(preflight(doc, ctx), frozen, ctx)


def check(review_path, ctx: Context) -> dict:
    doc = json.loads(Path(review_path).read_bytes())
    bad = rv.validate_review(doc)
    if bad:
        raise Refused([("SCHEMA", b, []) for b in bad])
    return preflight(doc, ctx)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "build"):
        q = sub.add_parser(name)
        q.add_argument("--review", required=True)
        q.add_argument("--config", default=str(DEFAULT_CONFIG))
        q.add_argument("--dataset-config", default=str(REPO / "config" / "roboflow.yaml"))
        q.add_argument("--mine-config", default=str(REPO / "config" / "detector_mine.yaml"))
        q.add_argument("--diag-config", default=str(REPO / "config" / "ball_fp_diag.yaml"))
        q.add_argument("--root", default=str(REPO), help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    ctx = load_context(a.config, a.dataset_config, a.mine_config, a.diag_config, Path(a.root))
    try:
        if a.cmd == "check":
            plan = check(a.review, ctx)
            print(f"preflight passed: {len(plan['new_frames'])} frames to add, {len(plan['copy_sources'])} base frames "
                  f"to copy at scale, {len(plan['changes'])} project changes; look-alike frames per category "
                  f"{plan['per_category']}. Nothing written.")
            return 0
        dest = build(a.review, ctx)
    except Refused as e:
        print(f"REFUSED ({len(e.reasons)} reasons):", *e.lines(), sep="\n  ", file=sys.stderr)
        return 3
    m = json.loads((dest / "stage.json").read_text())
    print(f"staged -> {dest}\n{json.dumps(m['counts'], indent=1)}\nNOT uploaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
