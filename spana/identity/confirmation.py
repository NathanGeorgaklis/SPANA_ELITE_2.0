"""Player confirmations of target identity episodes (spana.target 2.10.0; CODE TASK identity/player-confirmation).

The player's own statement about a track, as two versioned user-action records (docs/INTERFACES.md):

  player_confirmation 1.0.0 {kind, version, confirmation_id, track_id, frame, frame_end, verdict, source}
      verdict IS_ME | NOT_ME, source "player". The track must be live on every frame frame..frame_end, observed
      (ACTIVE) on `frame`, and in ONE uncertainty episode (the same tracks `uncertain_episodes` count) throughout:
      the confirmation covers that episode of that track, nothing else.
  revoke_confirmation 1.0.0 {kind, version, confirmation_id, source}
      undo: the named confirmation is ignored (its episode resolution and gallery crops disappear and every output
      is recomputed as if it had never been made); both records stay in the document and the log.

Episode = (track id, uncertain_episodes count). An IS_ME resolves the target's answer for that episode only (the
tracks document, rivals and every other identity are untouched); a later episode of the same track is uncertain
again unless re-ID against the human-confirmed gallery clears it on its own (target.gallery_reid_resolve).

Conflict (fail closed: human evidence never silently overrides certain evidence). An IS_ME is not applied, logged
CONFLICT and surfaced for re-confirmation when:
  OTHER_IDENTITY_CERTAIN    another followed identity's CERTAIN answer (claims) is on that track in the range;
  TARGET_CERTAIN_ELSEWHERE  the target is answered CERTAIN on another track at that frame;
  PLAYER_CONTRADICTION      the player said NOT_ME for the same episode earlier;
  REID_CONTRARY / COLOUR_CONTRARY  every clean crop of the track in the range is nearer another identity's reference
                            than the target's own by target.confirm_contradict_reid_margin / _colour_margin.
A NOT_ME is not applied (CONFLICT) when the target is answered CERTAIN on that very track at that frame, or the
player said IS_ME for the same episode; the target then abstains there (CONFIRMATION_CONFLICT) until re-confirmed.

Gallery: the clean crops (CropFeatures: observed, uncontested, overlapping no other detection above
reid.clean_max_iou, at least reid.min_crop_height_px tall) of an applied IS_ME's frames form the human-confirmed
partition of the target gallery, each tagged with its confirmation id. Unclean frames add nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..appearance import similarity as colour_similarity
from ..appearance import torso_descriptor
from ..config import SpanaConfig
from ..geometry import iou_matrix
from ..records import TrackState

PLAYER_CONFIRMATION = ("player_confirmation", "1.0.0")
REVOKE_CONFIRMATION = ("revoke_confirmation", "1.0.0")
CONFIRMATION_KEYS = ("kind", "version", "confirmation_id", "track_id", "frame", "frame_end", "verdict", "source")
REVOKE_KEYS = ("kind", "version", "confirmation_id", "source")
VERDICTS = ("IS_ME", "NOT_ME")
SOURCES = ("player",)
CONFLICT_REASONS = ("OTHER_IDENTITY_CERTAIN", "TARGET_CERTAIN_ELSEWHERE", "TARGET_CERTAIN", "PLAYER_CONTRADICTION",
                    "REID_CONTRARY", "COLOUR_CONTRARY")
CROP_FEATURES_SCHEMA = ("spana.crop_features", "1.0.0")


def player_confirmation(confirmation_id: str, track_id: str, frame: int, verdict: str,
                        frame_end: Optional[int] = None, source: str = "player") -> dict:
    return {"kind": PLAYER_CONFIRMATION[0], "version": PLAYER_CONFIRMATION[1], "confirmation_id": confirmation_id,
            "track_id": track_id, "frame": frame, "frame_end": frame if frame_end is None else frame_end,
            "verdict": verdict, "source": source}


def revoke_confirmation(confirmation_id: str, source: str = "player") -> dict:
    return {"kind": REVOKE_CONFIRMATION[0], "version": REVOKE_CONFIRMATION[1], "confirmation_id": confirmation_id,
            "source": source}


def is_record_ok(a: dict) -> bool:
    """Structural check of a confirmation / revocation record (values, not applicability)."""
    key = (a.get("kind"), a.get("version"))
    if key == PLAYER_CONFIRMATION:
        return set(a) == set(CONFIRMATION_KEYS) and isinstance(a["confirmation_id"], str) and \
            bool(a["confirmation_id"]) and isinstance(a["track_id"], str) and _int(a["frame"]) and \
            _int(a["frame_end"]) and a["frame_end"] >= a["frame"] and a["verdict"] in VERDICTS and \
            a["source"] in SOURCES
    if key == REVOKE_CONFIRMATION:
        return set(a) == set(REVOKE_KEYS) and isinstance(a["confirmation_id"], str) and a["source"] in SOURCES
    return False


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ------------------------------------------------------------------------------------------------ clean crop features
@dataclass(frozen=True)
class Crop:
    """Appearance of one clean crop of an observed track box: torso colour histogram and/or re-ID embedding."""
    colour: Optional[np.ndarray]
    reid: Optional[np.ndarray]


class CropFeatures:
    """Clean crops of observed track boxes, keyed (frame, track id); a box that is not clean has no entry.
    Clean = ACTIVE, uncontested, its own detection not multi-body, no other detection of the frame overlapping it
    above reid.clean_max_iou, and at least reid.min_crop_height_px tall (the re-ID training crops' floor)."""

    def __init__(self, crops: Dict[Tuple[int, str], Crop], meta: dict):
        self.crops, self.meta = crops, meta

    def get(self, frame: int, track_id: str) -> Optional[Crop]:
        return self.crops.get((frame, track_id))

    @staticmethod
    def clean_boxes(tracks_frame: dict, det_rows: Sequence[Sequence[float]], cfg: SpanaConfig
                    ) -> List[Tuple[str, List[float]]]:
        rc = cfg.reid
        rows = np.asarray([r[:4] for r in det_rows], dtype=float).reshape(-1, 4)
        out = []
        for t in tracks_frame["tracks"]:
            if t["state"] != TrackState.ACTIVE.value or t["contested"]:
                continue
            b = np.asarray(t["box"], dtype=float)
            if b[3] - b[1] < rc.min_crop_height_px or not len(rows):
                continue
            ious = iou_matrix(b[None], rows)[0]
            own = int(np.argmax(ious))
            if ious[own] <= 0 or (len(det_rows[own]) > 6 and int(det_rows[own][6]) == 1):
                continue
            others = np.delete(ious, own)
            if len(others) and float(others.max()) > rc.clean_max_iou:
                continue
            out.append((t["track_id"], [float(v) for v in t["box"]]))
        return out

    @classmethod
    def build(cls, video, tracks_doc: dict, det_doc: dict, cfg: SpanaConfig, embedder=None,
              tracks_sha256: str = "") -> "CropFeatures":
        """Read the video once; colour for every clean crop, re-ID (embedder, spana.identity.reid) when given."""
        from .reid import embed_boxes
        dets = {fr["frame"]: fr["detections"] for fr in det_doc["frames"]}
        need: Dict[int, List[Tuple[str, List[float]]]] = {}
        for fr in tracks_doc["frames"]:
            rows = dets.get(fr["frame"])
            if rows is None:
                continue
            boxes = cls.clean_boxes(fr, rows, cfg)
            if boxes:
                need[fr["frame"]] = boxes
        crops: Dict[Tuple[int, str], Crop] = {}
        if need:
            for i, _, img in video.frames(min(need), max(need) + 1):
                todo = need.get(i)
                if not todo:
                    continue
                embs = embed_boxes(embedder, img, [b for _, b in todo]) if embedder is not None else [None] * len(todo)
                for (tid, b), e in zip(todo, embs):
                    col = torso_descriptor(img, b, cfg.appearance)
                    if col is not None or e is not None:
                        crops[(i, tid)] = Crop(col, None if e is None else np.asarray(e, dtype=np.float32))
        meta = {"schema": CROP_FEATURES_SCHEMA[0], "schema_version": CROP_FEATURES_SCHEMA[1],
                "tracks_sha256": tracks_sha256, "reid_model": None if embedder is None else embedder.model_id(),
                "clean_max_iou": cfg.reid.clean_max_iou, "min_crop_height_px": cfg.reid.min_crop_height_px}
        return cls(crops, meta)

    def save(self, path: str | Path) -> None:
        import json
        keys = sorted(self.crops)
        col = [self.crops[k].colour for k in keys]
        rei = [self.crops[k].reid for k in keys]
        cd = next((c.shape[0] for c in col if c is not None), 0)
        rd = next((r.shape[0] for r in rei if r is not None), 0)
        np.savez_compressed(
            path, meta=np.array(json.dumps(self.meta, sort_keys=True)),
            frames=np.array([k[0] for k in keys], dtype=np.int64), tracks=np.array([k[1] for k in keys]),
            colour=np.stack([c if c is not None else np.full(cd, np.nan) for c in col]) if keys else np.zeros((0, 0)),
            reid=np.stack([r if r is not None else np.full(rd, np.nan, dtype=np.float32) for r in rei])
            if keys and rd else np.zeros((len(keys), 0), dtype=np.float32))

    @classmethod
    def load(cls, path: str | Path, tracks_sha256: Optional[str] = None) -> "CropFeatures":
        import json
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if (meta.get("schema"), meta.get("schema_version")) != CROP_FEATURES_SCHEMA:
            raise ValueError(f"{path}: not a {CROP_FEATURES_SCHEMA} file")
        if tracks_sha256 is not None and meta["tracks_sha256"] != tracks_sha256:
            raise ValueError(f"{path}: crop features of other tracks ({meta['tracks_sha256'][:12]})")
        crops = {}
        col, rei = np.array(z["colour"]), np.array(z["reid"])   # decompress once (NpzFile re-reads per access)
        for k, (f, t) in enumerate(zip(z["frames"].tolist(), z["tracks"].tolist())):
            c = col[k].copy() if col.shape[1] else None          # copies: no view keeps the whole array alive
            r = rei[k].copy() if rei.shape[1] else None
            crops[(int(f), str(t))] = Crop(None if c is None or np.isnan(c).any() else c,
                                           None if r is None or np.isnan(r).any() else r)
        return cls(crops, meta)


def cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


# ------------------------------------------------------------------------------------------------ evidence rules
def appearance_conflicts(crops: Sequence[Crop], own: Dict[str, Optional[np.ndarray]],
                         others: Dict[str, Dict[str, Optional[np.ndarray]]], cfg: SpanaConfig
                         ) -> Tuple[List[dict], dict]:
    """REID_CONTRARY / COLOUR_CONTRARY: every clean crop of the confirmed range is nearer another identity's reference
    than the target's own reference by the configured margin. own = {"reid": emb, "colour": hist} (None = no
    reference: that cue gives no verdict); others = identity -> the same. Returns (conflicts, evidence)."""
    tc = cfg.target
    conflicts, ev = [], {"crops": len(crops), "identities": {}}
    for q in sorted(others):
        ref = others[q]
        row = {}
        for cue, fn, margin, reason in (("reid", cos, tc.confirm_contradict_reid_margin, "REID_CONTRARY"),
                                        ("colour", colour_similarity, tc.confirm_contradict_colour_margin,
                                         "COLOUR_CONTRARY")):
            ms = []
            for c in crops:
                x = getattr(c, cue)
                a, b = fn(x, ref.get(cue)), fn(x, own.get(cue))
                if a is not None and b is not None:
                    ms.append(round(a - b, 4))
            row[cue] = ms
            if ms and len(ms) == len(crops) and min(ms) >= margin:
                conflicts.append({"reason": reason, "identity": q, "margins": ms})
        ev["identities"][q] = row
    return conflicts, ev


def gallery_frame_verdict(crop: Optional[Crop], own: Optional[np.ndarray], rivals: Dict[str, Optional[np.ndarray]],
                          margin: float) -> Tuple[str, dict]:
    """One evidence frame of the gallery re-ID run: CLEAR (the crop is nearer the human-confirmed gallery reference
    than every rival's reference by `margin`), AGAINST (some rival nearer by `margin`), or HOLD with a reason
    (NO_CLEAN_CROP, NO_GALLERY_REFERENCE, NO_RIVAL, NO_RIVAL_REFERENCE, BELOW_MARGIN). Missing data is no evidence."""
    if crop is None or crop.reid is None:
        return "HOLD", {"reason": "NO_CLEAN_CROP"}
    if own is None:
        return "HOLD", {"reason": "NO_GALLERY_REFERENCE"}
    if not rivals:
        return "HOLD", {"reason": "NO_RIVAL"}
    ms: Dict[str, Optional[float]] = {}
    for rid, ref in sorted(rivals.items()):
        ms[rid] = None if ref is None else round(cos(crop.reid, own) - cos(crop.reid, ref), 4)
    if any(m is not None and m <= -margin for m in ms.values()):
        return "AGAINST", {"margins": ms}
    if any(m is None for m in ms.values()):
        return "HOLD", {"reason": "NO_RIVAL_REFERENCE", "margins": ms}
    if all(m >= margin for m in ms.values()):
        return "CLEAR", {"margins": ms}
    return "HOLD", {"reason": "BELOW_MARGIN", "margins": ms}


def claims_from_targets(docs: Iterable[Tuple[str, dict]]) -> Dict[str, Dict[int, str]]:
    """Other identities' CERTAIN answers: identity -> {frame: track id} of their spana.target TARGET frames with reason
    TRACK_OBSERVED (an observed, confirmed, CERTAIN track the user selected or confirmed). Human-confirmed or
    re-ID-cleared answers are not certain evidence and are left out."""
    out: Dict[str, Dict[int, str]] = {}
    for ident, doc in docs:
        out[ident] = {fr["frame"]: fr["track_id"] for fr in doc["frames"]
                      if fr["state"] == "TARGET" and fr["reason"] == "TRACK_OBSERVED"}
    return out
