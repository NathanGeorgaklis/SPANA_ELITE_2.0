"""Re-ID signatures per track (spana.track_embeddings 1.0.0) for re-entry offers (spana.identity.follower).

A pinned SPANA re-ID model (ONNX, `models/spana_reid_v2.json`) embeds clean crops of each confirmed track: the first
and the last `reid.samples` observations that are confirmed, CERTAIN, uncontested, at least `reid.min_crop_height_px`
tall, at least `reid.sample_min_gap_s` apart, and overlap no other observed box by more than `reid.clean_max_iou`
(a crop mixing two players is not evidence about either). A track's `first` / `last` signature is the normalised mean
of those embeddings. A track without such a crop has no signature (`null`): it is never offered, never guessed.

The signatures are evidence for an OFFER only. Similarity never makes an answer by itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..artifacts import EMBEDDINGS_SCHEMA, sha256_doc, validate_track_embeddings
from ..config import SpanaConfig
from ..provenance import build_provenance
from ..geometry import iou_matrix
from ..video import sha256_file

DECIMALS = 5                       # stored embedding precision (size bound; cosine error < 1e-4)


class ReidModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReidModelRecord:
    name: str
    source: str
    version: str
    licence: str
    onnx_path: str                 # relative to the record file
    onnx_sha256: str
    input_size_hw: Tuple[int, int]
    crop_pad: float                # box padding per side, share of width / height (as in training crops)
    color_order: str
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    embedding_dim: int
    validation: str

    @staticmethod
    def load(path: str | Path) -> "ReidModelRecord":
        p = Path(path)
        if not p.is_file():
            raise ReidModelError(f"re-ID model record not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        names = list(ReidModelRecord.__dataclass_fields__)
        if set(data) != set(names):
            raise ReidModelError(f"re-ID model record {p}: keys {sorted(data)} != {sorted(names)}")
        if data["color_order"] not in ("rgb", "bgr") or len(data["onnx_sha256"]) != 64 or \
                not data["licence"].strip() or not data["version"].strip():
            raise ReidModelError("re-ID model record needs rgb|bgr, a licence, an exact version and a sha256")
        rec = ReidModelRecord(**{**data, "input_size_hw": tuple(data["input_size_hw"]), "mean": tuple(data["mean"]),
                                 "std": tuple(data["std"])})
        weights = (p.parent / rec.onnx_path).resolve()
        if not weights.is_file():
            raise ReidModelError(f"re-ID weights not found: {weights}")
        if sha256_file(weights) != rec.onnx_sha256:
            raise ReidModelError(f"re-ID weights sha256 differs from the pinned {rec.onnx_sha256}")
        object.__setattr__(rec, "onnx_path", str(weights))
        return rec


def cut_crop(img: np.ndarray, box: Sequence[float], pad: float, size_hw: Tuple[int, int]) -> np.ndarray:
    """Padded box, clipped to the image, resized to size_hw (BGR in, BGR out); the training crop convention."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    xa, ya = int(max(0, round(x1 - pad * w))), int(max(0, round(y1 - pad * h)))
    xb, yb = int(min(W, round(x2 + pad * w))), int(min(H, round(y2 + pad * h)))
    c = img[ya:yb, xa:xb]
    if c.size == 0:
        raise ValueError(f"empty crop for box {box}")
    return cv2.resize(c, (size_hw[1], size_hw[0]),
                      interpolation=cv2.INTER_AREA if c.shape[0] > size_hw[0] else cv2.INTER_LINEAR)


class ReidEmbedder:
    def __init__(self, record: ReidModelRecord, batch: int = 32):
        import onnxruntime as ort
        self.rec, self.batch = record, batch
        self.session = ort.InferenceSession(record.onnx_path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def model_id(self) -> dict:
        return {"name": self.rec.name, "version": self.rec.version, "sha256": self.rec.onnx_sha256}

    def __call__(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """L2-normalised embeddings (N x D) of model-size BGR crops."""
        if not len(crops_bgr):
            return np.zeros((0, self.rec.embedding_dim))
        mean, std = np.array(self.rec.mean), np.array(self.rec.std)
        x = np.stack([(((c[:, :, ::-1] if self.rec.color_order == "rgb" else c).astype(np.float32) / 255.0 - mean)
                       / std).transpose(2, 0, 1) for c in crops_bgr]).astype(np.float32)
        y = np.concatenate([self.session.run(None, {self.input: x[k:k + self.batch]})[0]
                            for k in range(0, len(x), self.batch)]).reshape(len(x), -1).astype(np.float64)
        if y.shape[1] != self.rec.embedding_dim:
            raise ReidModelError(f"model output dim {y.shape[1]} != record {self.rec.embedding_dim}")
        return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def select_samples(tracks_doc: dict, cfg: SpanaConfig) -> Dict[str, Dict[str, List[Tuple[int, List[float]]]]]:
    """track -> {"first": [(frame, box)], "last": [(frame, box)]}: the clean observations that will be embedded."""
    rc = cfg.reid
    clean: Dict[str, List[Tuple[int, float, List[float]]]] = {}
    for fr in tracks_doc["frames"]:
        obs = [t for t in fr["tracks"] if t["source"] == "OBSERVED"]
        boxes = np.array([t["box"] for t in obs], dtype=float).reshape(-1, 4)
        ov = iou_matrix(boxes, boxes) if len(obs) else np.zeros((0, 0))
        if len(obs):
            np.fill_diagonal(ov, 0.0)
        for k, t in enumerate(obs):
            b = t["box"]
            if not t["confirmed"] or t["identity"] != "CERTAIN" or t["contested"] or \
                    b[3] - b[1] < rc.min_crop_height_px or (len(obs) > 1 and ov[k].max() > rc.clean_max_iou):
                continue
            clean.setdefault(t["track_id"], []).append((fr["frame"], fr["t"], list(b)))

    def spaced(rows):
        out, last_t = [], None
        for f, t, b in rows:
            if last_t is None or abs(t - last_t) >= rc.sample_min_gap_s - 1e-9:
                out.append((f, b))
                last_t = t
            if len(out) == rc.samples:
                break
        return out

    return {tid: {"first": spaced(rows), "last": sorted(spaced(list(reversed(rows))))} for tid, rows in clean.items()}


def crop_quality(height: float, overlap: float, rc) -> float:
    """Quality weight of a crop in (0, 1]: full from reid.min_crop_height_px tall with overlap <= clean_max_iou;
    smaller boxes scale down with height, overlap beyond clean_max_iou falls linearly to 0.1 at set_max_iou."""
    q_h = min(1.0, height / rc.min_crop_height_px)
    if overlap <= rc.clean_max_iou:
        q_o = 1.0
    else:
        q_o = max(0.1, 1.0 - 0.9 * (overlap - rc.clean_max_iou) / max(rc.set_max_iou - rc.clean_max_iou, 1e-9))
    return round(q_h * q_o, 4)


def select_set_samples(tracks_doc: dict, cfg: SpanaConfig) -> Dict[str, Dict[str, List[Tuple[int, List[float], float]]]]:
    """track -> {"first": [(frame, box, quality)], "last": [...]}: crop sets under the looser reid.set_* rules
    (confirmed, CERTAIN, uncontested; >= set_min_height_px; overlap <= set_max_iou), up to set_samples per end,
    set_min_gap_s apart, from each end of the track inwards."""
    rc = cfg.reid
    rows: Dict[str, List[Tuple[int, float, List[float], float]]] = {}
    for fr in tracks_doc["frames"]:
        obs = [t for t in fr["tracks"] if t["source"] == "OBSERVED"]
        boxes = np.array([t["box"] for t in obs], dtype=float).reshape(-1, 4)
        ov = iou_matrix(boxes, boxes) if len(obs) else np.zeros((0, 0))
        if len(obs):
            np.fill_diagonal(ov, 0.0)
        for k, t in enumerate(obs):
            b = t["box"]
            o = float(ov[k].max()) if len(obs) > 1 else 0.0
            if not t["confirmed"] or t["identity"] != "CERTAIN" or t["contested"] or \
                    b[3] - b[1] < rc.set_min_height_px or o > rc.set_max_iou:
                continue
            rows.setdefault(t["track_id"], []).append((fr["frame"], fr["t"], list(b), crop_quality(b[3] - b[1], o, rc)))

    def spaced(seq):
        out, last_t = [], None
        for f, t, b, q in seq:
            if last_t is None or abs(t - last_t) >= rc.set_min_gap_s - 1e-9:
                out.append((f, b, q))
                last_t = t
            if len(out) == rc.set_samples:
                break
        return out

    return {tid: {"first": spaced(r), "last": sorted(spaced(list(reversed(r))))} for tid, r in rows.items()}


def signature(emb: np.ndarray) -> Optional[List[float]]:
    if not len(emb):
        return None
    m = emb.mean(axis=0)
    m = m / max(float(np.linalg.norm(m)), 1e-12)
    return [round(float(v), DECIMALS) for v in m]


def track_embeddings(video, tracks_doc: dict, embedder: ReidEmbedder, cfg: SpanaConfig) -> Dict[str, dict]:
    """Per track {first, last, first_frames, last_frames} (the 1.0.0 averaged signatures, unchanged) and
    {first_crops, last_crops}: the 1.1.0 crop sets [{frame, quality, emb}]. Reads only the frames that hold a crop."""
    samples = select_samples(tracks_doc, cfg)
    sets = select_set_samples(tracks_doc, cfg)
    need: Dict[int, List[Tuple[str, str, List[float]]]] = {}
    for tid, s in samples.items():
        for end in ("first", "last"):
            for f, b in s[end]:
                need.setdefault(f, []).append((tid, end, b))
    for tid, s in sets.items():
        for end in ("first", "last"):
            for f, b, _ in s[end]:
                need.setdefault(f, []).append((tid, f"{end}_set", b))
    crops: Dict[Tuple[str, str], List[np.ndarray]] = {}
    if need:
        first, last = min(need), max(need)
        for i, _, img in video.frames(first, last + 1):
            for tid, end, b in need.get(i, ()):
                crops.setdefault((tid, end), []).append(
                    cut_crop(img, b, embedder.rec.crop_pad, embedder.rec.input_size_hw))
    out = {}
    for tid in sorted(set(samples) | set(sets)):
        row = {}
        for end in ("first", "last"):
            c = crops.get((tid, end), [])
            row[end] = signature(embedder(c)) if c else None
            row[f"{end}_frames"] = [f for f, _ in samples.get(tid, {}).get(end, [])]
            cs = crops.get((tid, f"{end}_set"), [])
            embs = embedder(cs) if cs else np.zeros((0, 0))
            row[f"{end}_crops"] = [{"frame": f, "quality": q, "emb": [round(float(v), DECIMALS) for v in e]}
                                   for (f, _, q), e in zip(sets.get(tid, {}).get(end, []), embs)]
        out[tid] = row
    return out


def set_similarity(a: Sequence[dict], b: Sequence[dict]) -> Optional[float]:
    """Quality-weighted best-pair similarity of two crop sets: each crop's best cosine in the other set, averaged
    with the crops' quality weights, in both directions, then the mean of the two directions. None if either set is
    empty."""
    if not a or not b:
        return None
    A = np.array([c["emb"] for c in a], float)
    B = np.array([c["emb"] for c in b], float)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    C = A @ B.T
    wa = np.array([c["quality"] for c in a], float)
    wb = np.array([c["quality"] for c in b], float)
    return float(0.5 * ((wa * C.max(axis=1)).sum() / wa.sum() + (wb * C.max(axis=0)).sum() / wb.sum()))


def appearance(sig: Dict[str, dict], lost: str, cand: str, tc_target, upto_frame: Optional[int] = None
               ) -> Optional[float]:
    """Appearance similarity of the lost track's last end and the candidate's first end, by
    target.reentry_signature: "mean" = cosine of the averaged signatures (1.0.0); "set" = set_similarity() of the
    crop sets (1.1.0), the candidate's crops limited to frames <= upto_frame and required to number at least
    target.reentry_wait_samples when that is > 0. None = no evidence (never offered, never guessed)."""
    if tc_target.reentry_signature == "mean":
        return cosine(sig.get(lost, {}).get("last"), sig.get(cand, {}).get("first"))
    a = sig.get(lost, {}).get("last_crops") or []
    b = sig.get(cand, {}).get("first_crops") or []
    if tc_target.reentry_wait_samples > 0:
        if upto_frame is not None:
            b = [c for c in b if c["frame"] <= upto_frame]
        if len(b) < tc_target.reentry_wait_samples:
            return None
    return set_similarity(a, b)


def cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    x, y = np.asarray(a, float), np.asarray(b, float)
    return float(x @ y / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-12))


def embeddings_document(video, tracks_doc: dict, embedder: ReidEmbedder, cfg: SpanaConfig,
                        argv: Sequence[str] = ()) -> dict:
    """Validated spana.track_embeddings 1.1.0 document for a validated spana.tracks document and its video."""
    import dataclasses
    import sys
    if video.meta.sha256 != tracks_doc["video"]["sha256"]:
        raise ReidModelError("the video differs from the tracks document's video")
    tsha = sha256_doc(tracks_doc)
    doc = {"schema": EMBEDDINGS_SCHEMA[0], "schema_version": EMBEDDINGS_SCHEMA[1],
           "provenance": build_provenance(config_sha256=cfg.sha256(), argv=list(argv) or sys.argv,
                                          inputs={"video": video.meta.sha256, "tracks": tsha,
                                                  "model": embedder.rec.onnx_sha256}),
           "video": tracks_doc["video"], "tracks_sha256": tsha,
           "model": {**embedder.model_id(), "embedding_dim": embedder.rec.embedding_dim},
           "reid_config": dataclasses.asdict(cfg.reid),
           "tracks": track_embeddings(video, tracks_doc, embedder, cfg)}
    validate_track_embeddings(doc, tracks_doc)
    return doc


# --------------------------------------------------------------------------- re-entry decision (offer rule)
_BREAK = ("CUT", "UNKNOWN")


def reachable(frames: Sequence[dict], lost_box: Sequence[float], from_idx: int, to_idx: int,
              cand_box: Sequence[float], tc, frame_w: float, frame_h: float) -> Optional[bool]:
    """Could the lost player (last seen as lost_box on frames[from_idx]) be cand_box on frames[to_idx]?
    The lost box's centre is carried through the per-frame camera transforms; the candidate's centre must lie within
    tracking.gate_base_h + tracking.max_speed_h_per_s * gap body heights of it (the tracker's own speed bound).
    None = position cannot judge: a CUT / UNKNOWN camera frame in between, or the carried point left the picture
    (the player went out of shot and came back somewhere else)."""
    from ..geometry import affine_scale, apply_affine_points
    if frame_w is None or frame_h is None:
        return None
    x1, y1, x2, y2 = lost_box
    p = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
    h = y2 - y1
    for k in range(from_idx + 1, to_idx + 1):
        cam = frames[k]["camera"]
        if cam["status"] in _BREAK:
            return None
        A = np.asarray(cam["affine"], dtype=float)
        p = apply_affine_points(A, p)
        h *= affine_scale(A)
    if not (0.0 <= p[0, 0] <= frame_w and 0.0 <= p[0, 1] <= frame_h):
        return None
    c = np.array([(cand_box[0] + cand_box[2]) / 2, (cand_box[1] + cand_box[3]) / 2])
    gap = frames[to_idx]["t"] - frames[from_idx]["t"]
    return float(np.hypot(*(c - p[0]))) / max(h, 1e-9) <= tc.gate_base_h + tc.max_speed_h_per_s * gap + 1e-9


@dataclass(frozen=True)
class ReentryContext:
    """What the re-entry rule needs from a tracks document (built once per document)."""
    frames: Sequence[dict]
    birth: Dict[str, int]                         # track -> frame index of its first row
    last_obs: Dict[str, int]                      # track -> frame index of its last OBSERVED row
    box_at: Dict[Tuple[str, int], List[float]]    # (track, frame index) -> box, for birth / last-observation rows
    width: Optional[float]                        # frame size; unknown -> position cannot judge
    height: Optional[float]

    @staticmethod
    def of(tracks_doc: dict) -> "ReentryContext":
        frames = tracks_doc["frames"]
        birth, last_obs, box_at = {}, {}, {}
        for n, fr in enumerate(frames):
            for t in fr["tracks"]:
                if t["track_id"] not in birth:
                    birth[t["track_id"]] = n
                    box_at[(t["track_id"], n)] = t["box"]
                if t["source"] == "OBSERVED":
                    last_obs[t["track_id"]] = n
                    box_at[(t["track_id"], n)] = t["box"]
        v = tracks_doc["video"]
        return ReentryContext(frames, birth, last_obs, box_at, v.get("width"), v.get("height"))

    def last_obs_upto(self, tid: str, n: int) -> Optional[int]:
        """Last observation of tid at or before frame index n (the follower may ask before the track ended)."""
        lo = self.last_obs.get(tid)
        if lo is not None and lo <= n:
            return lo
        for k in range(min(n, len(self.frames) - 1), self.birth.get(tid, n + 1) - 1, -1):
            row = next((t for t in self.frames[k]["tracks"] if t["track_id"] == tid), None)
            if row is not None and row["source"] == "OBSERVED":
                return k
        return None


SILENCING = ("CAMERA_BREAK", "LEFT_FRAME", "OUT_OF_PICTURE", "PAN", "NO_FRAME_SIZE")
POSITION_REASONS = ("INFORMATIVE", "GAP_TOO_LONG", "CAMERA_BREAK", "LEFT_FRAME", "OUT_OF_PICTURE", "PAN",
                    "NO_FRAME_SIZE", "WEIGHT_ZERO")


def position_fit(ctx: "ReentryContext", lost: str, lost_idx: int, cand: str, tc_target) -> Tuple[float, str]:
    """Spatiotemporal fit in [0, 1] of candidate `cand` (at its birth) for the lost track (at its last observation),
    and why position was or was not informative. Same space as reachable(): the lost box centre carried through the
    per-frame camera transforms, distances in the lost box's (carried) height.
      dist_fit = max(0, 1 - d / (reentry_position_base_h + reentry_position_speed_h_per_s * gap))
      time_fit = exp(-gap / reentry_position_time_scale_s)
      fit      = dist_fit * time_fit
    Not informative (fit 0): gap > reentry_position_max_gap_s; a CUT / UNKNOWN camera frame in the gap; the lost track
    ended LEFT_FRAME by then; the carried point left the picture; the carried point moved more than
    reentry_position_max_pan_frac of the frame width (the camera panned); no frame size."""
    from ..geometry import affine_scale, apply_affine_points
    frames = ctx.frames
    b = ctx.birth[cand]
    gap = frames[b]["t"] - frames[lost_idx]["t"]
    if tc_target.reentry_position_weight <= 0:
        return 0.0, "WEIGHT_ZERO"
    if ctx.width is None or ctx.height is None:
        return 0.0, "NO_FRAME_SIZE"
    x1, y1, x2, y2 = ctx.box_at[(lost, lost_idx)]
    p0 = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
    p, h = p0.copy(), y2 - y1
    for k in range(lost_idx + 1, b + 1):
        cam = frames[k]["camera"]
        if cam["status"] in _BREAK:
            return 0.0, "CAMERA_BREAK"
        if any(e[0] == lost and e[1] == "LEFT_FRAME" for e in frames[k]["ended"]):
            return 0.0, "LEFT_FRAME"
        A = np.asarray(cam["affine"], dtype=float)
        p = apply_affine_points(A, p)
        h *= affine_scale(A)
    if not (0.0 <= p[0, 0] <= ctx.width and 0.0 <= p[0, 1] <= ctx.height):
        return 0.0, "OUT_OF_PICTURE"
    if float(np.hypot(*(p[0] - p0[0]))) > tc_target.reentry_position_max_pan_frac * ctx.width:
        return 0.0, "PAN"
    if gap > tc_target.reentry_position_max_gap_s + 1e-9:     # after the camera / exit checks: those silence more
        return 0.0, "GAP_TOO_LONG"
    cb = ctx.box_at[(cand, b)]
    d = float(np.hypot((cb[0] + cb[2]) / 2 - p[0, 0], (cb[1] + cb[3]) / 2 - p[0, 1])) / max(h, 1e-9)
    reach = tc_target.reentry_position_base_h + tc_target.reentry_position_speed_h_per_s * gap
    dist_fit = max(0.0, 1.0 - d / max(reach, 1e-9))
    time_fit = float(np.exp(-gap / tc_target.reentry_position_time_scale_s))
    return dist_fit * time_fit, "INFORMATIVE"


def exit_side(ctx: "ReentryContext", lost: str, lost_idx: int, cand: str) -> dict:
    """Where did the lost player leave the picture, and how far from that edge was `cand` born?
    {"status": "unknown"}     a CUT / UNKNOWN camera frame between the last observation and cand's birth
    {"status": "in_picture"}  the lost track did not end LEFT_FRAME by then and its last box centre, carried through the
                              per-frame camera transforms to cand's birth, is still inside the picture
    {"status": "off_screen", "side", "edge_dist"}  side = the side the carried centre is beyond (largest excess), or,
                              if still inside (ended LEFT_FRAME at the edge), the nearest side of the carried last box;
                              edge_dist = cand's birth box distance from that side as a share of the frame width
                              (left / right) or height (top / bottom), 0 = touching it.
    records/reentry_exit_side_2026-11-19: every one of 17 labelled DEV off-screen returns was born <= 0.011 from the
    exit side; the three wrong DEV offers were 0.62-0.98 away."""
    from ..geometry import apply_affine_points
    frames = ctx.frames
    b = ctx.birth[cand]
    if ctx.width is None or ctx.height is None:
        return {"status": "unknown"}
    W, H = float(ctx.width), float(ctx.height)
    x1, y1, x2, y2 = ctx.box_at[(lost, lost_idx)]
    pts = np.array([[(x1 + x2) / 2, (y1 + y2) / 2], [x1, y1], [x2, y2]], dtype=float)
    left_frame = False
    for k in range(lost_idx + 1, b + 1):
        cam = frames[k]["camera"]
        if cam["status"] in _BREAK:
            return {"status": "unknown"}
        pts = apply_affine_points(np.asarray(cam["affine"], dtype=float), pts)
        left_frame = left_frame or any(e[0] == lost and e[1] == "LEFT_FRAME" for e in frames[k]["ended"])
    cx, cy = pts[0]
    excess = {"left": -cx, "right": cx - W, "top": -cy, "bottom": cy - H}
    outside = max(excess.values()) > 0
    if not (outside or left_frame):
        return {"status": "in_picture"}
    if outside:
        side = max(excess, key=excess.get)
    else:
        gaps = {"left": pts[1][0], "right": W - pts[2][0], "top": pts[1][1], "bottom": H - pts[2][1]}
        side = min(gaps, key=gaps.get)
    cb = ctx.box_at[(cand, b)]
    dist = {"left": cb[0] / W, "right": (W - cb[2]) / W, "top": cb[1] / H, "bottom": (H - cb[3]) / H}[side]
    return {"status": "off_screen", "side": side, "edge_dist": round(float(max(dist, 0.0)), 6)}


def reentry_choice(lost: str, lost_idx: int, now_idx: int, ctx: ReentryContext, sig: Dict[str, dict], tc_target,
                   tc_tracking, allowed=lambda c: True, causal: bool = False) -> dict:
    """The re-entry rule shared by the follower's offer and the DEV evaluation. Always returns a decision:
    {offer, reason, track, appearance, position, position_informative, position_reason, combined, second_combined,
     margin, reverse_second_combined, candidates: {track: {appearance, position, position_reason, combined}}}.

    Candidates: tracks born after the lost track's last observation (frame index lost_idx), at most
    target.reentry_max_gap_s later, born by now_idx, with a first signature, that `allowed` accepts for the offer
    (the follower: usable now and not yet offered). With target.reentry_position_gate, a candidate the lost player
    could not have reached (reachable() is False) is dropped (unchanged).
    Score: combined = appearance + reentry_position_weight * position_fit (position_fit() is 0 when position carries
    no information). A camera / exit reason (cut, pan, LEFT_FRAME, out of picture, no frame size) for ANY candidate
    silences position for the whole decision, so the ranking is exactly appearance-only; a gap beyond
    reentry_position_max_gap_s silences only that candidate. The best combined offerable candidate is offered
    when its APPEARANCE reaches reentry_min_similarity and its combined score beats every other candidate's (offerable
    or not) by reentry_min_margin; with reentry_mutual, the lost track's combined score must also beat, by the same
    margin, every other track that was lost when the candidate appeared (scored the same way, towards the candidate).
    With target.reentry_exit_edge, when the lost player left the picture (exit_side() off_screen), a candidate born
    more than reentry_exit_edge_max_frac of the frame from the side they left by is not a candidate at all (listed in
    `exit_excluded`): a returning player comes back in where they went out. Unknown (camera cut) or in-picture: no
    effect.
    Abstention reasons: NO_SIGNATURE, NO_CANDIDATE, BELOW_MIN_SIMILARITY, MARGIN, NOT_MUTUAL."""
    frames = ctx.frames
    w = tc_target.reentry_position_weight
    out = {"offer": False, "reason": None, "track": None, "appearance": None, "position": None,
           "position_informative": None, "position_reason": None, "combined": None, "second_combined": None,
           "margin": None, "reverse_second_combined": None, "candidates": {}, "exit_excluded": []}
    upto = frames[now_idx]["frame"] if causal else None
    has_last = sig.get(lost, {}).get("last") is not None if tc_target.reentry_signature == "mean" \
        else bool(sig.get(lost, {}).get("last_crops"))
    if not has_last:
        return {**out, "reason": "NO_SIGNATURE"}
    t0 = frames[lost_idx]["t"]
    lost_box = ctx.box_at[(lost, lost_idx)]

    def gate_ok(src_idx: int, src_box, c: str) -> bool:
        if not tc_target.reentry_position_gate:
            return True
        b = ctx.birth[c]
        return reachable(frames, src_box, src_idx, b, ctx.box_at[(c, b)], tc_tracking, ctx.width, ctx.height) \
            is not False

    pool = [c for c, b in ctx.birth.items() if c != lost and lost_idx < b <= now_idx
            and frames[b]["t"] - t0 <= tc_target.reentry_max_gap_s + 1e-9 and gate_ok(lost_idx, lost_box, c)]
    cands, excluded = {}, []
    for c in pool:
        a = appearance(sig, lost, c, tc_target, upto)
        if a is None:
            continue
        if tc_target.reentry_exit_edge:
            ex = exit_side(ctx, lost, lost_idx, c)
            if ex["status"] == "off_screen" and ex["edge_dist"] > tc_target.reentry_exit_edge_max_frac + 1e-9:
                excluded.append(c)
                continue
        fit, why = position_fit(ctx, lost, lost_idx, c, tc_target)
        cands[c] = {"appearance": a, "position": fit, "position_reason": why}
    # The camera or an exit silences position for the whole decision: a pan return must never lose to a candidate
    # that position happens to favour (a long gap only silences that candidate).
    silenced = next((v["position_reason"] for v in cands.values() if v["position_reason"] in SILENCING), None)
    for v in cands.values():
        if silenced is not None:
            v["position"], v["position_reason"] = 0.0, v["position_reason"] if v["position_reason"] in SILENCING \
                else silenced
        v["combined"] = round(v["appearance"] + w * v["position"], 6)
        v["appearance"], v["position"] = round(v["appearance"], 6), round(v["position"], 6)
    out["candidates"] = cands
    out["exit_excluded"] = sorted(excluded)
    offerable = [c for c in cands if allowed(c)]
    if not offerable:
        return {**out, "reason": "NO_CANDIDATE"}
    best = max(offerable, key=lambda c: (cands[c]["combined"], c))
    cb = cands[best]
    others = [v["combined"] for c, v in cands.items() if c != best]
    second = max(others) if others else None
    out.update(track=best, appearance=cb["appearance"], position=cb["position"],
               position_informative=cb["position_reason"] == "INFORMATIVE", position_reason=cb["position_reason"],
               combined=cb["combined"], second_combined=second,
               margin=None if second is None else round(cb["combined"] - second, 6))
    if cb["appearance"] < tc_target.reentry_min_similarity - 1e-9:
        return {**out, "reason": "BELOW_MIN_SIMILARITY"}
    if second is not None and cb["combined"] - second < tc_target.reentry_min_margin - 1e-9:
        return {**out, "reason": "MARGIN"}
    if tc_target.reentry_mutual:
        b = ctx.birth[best]
        rival = []
        for other in ctx.birth:
            if other in (lost, best):
                continue
            lo = ctx.last_obs_upto(other, b - 1)
            if lo is None or frames[b]["t"] - frames[lo]["t"] > tc_target.reentry_max_gap_s + 1e-9 or \
                    any(t["track_id"] == other and t["source"] == "OBSERVED" for t in frames[b]["tracks"]):
                continue                          # not lost when the candidate appeared
            v = appearance(sig, other, best, tc_target, upto)
            if v is None:
                continue
            obox = ctx.box_at.get((other, lo)) or next(t["box"] for t in frames[lo]["tracks"]
                                                        if t["track_id"] == other)
            if not gate_ok(lo, obox, best):
                continue
            if (other, lo) not in ctx.box_at:
                ctx.box_at[(other, lo)] = obox
            # A silenced decision is appearance-only on both sides: a pan / exit return must not lose the two-way
            # check to another lost player that position happens to favour.
            fit = 0.0 if silenced is not None else position_fit(ctx, other, lo, best, tc_target)[0]
            rival.append(v + w * fit)
        rs = max(rival) if rival else None
        out["reverse_second_combined"] = None if rs is None else round(rs, 6)
        if rs is not None and cb["combined"] - rs < tc_target.reentry_min_margin - 1e-9:
            return {**out, "reason": "NOT_MUTUAL"}
    return {**out, "offer": True}
