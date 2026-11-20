"""Offline global tracklet linking v1: a post-processing stage after the tracker, over a whole clip.

In: a validated `spana.tracks` document (2.1.0 / 2.2.0 / 2.3.0) and per-tracklet appearance signatures from an
`Embedder`. Out: the same contract and version, with linked tracklets written as ONE track, plus a link log. The
production tracker is unchanged.

A link A -> B is written exactly as the tracker writes its own recovery after a gap: B's rows take A's track id, the
frames between A's last observation and B's first are A held (LOST / HELD, camera-carried box), `recoveries` goes up
by one and `last_gap_s` is the link gap. No field is added, so the contract version does not change; the link log
(`*.links.jsonl`) says which recoveries are links.

Candidates (all from config/tracking/linking.yaml, one generic setting):
  * same epoch (no CUT / UNKNOWN camera in between), B born after A's last observation (no time overlap of what
    the two observed), gap <= max_gap_s;
  * camera-carried displacement of the foot point <= gate_base_h + max_speed_h_per_s * gap (body heights), box
    height ratio within max_height_ratio.
Cost (lower is better) = weight_gap * gap / max_gap + weight_motion * (distance from A's camera-carried constant
velocity prediction / gate radius, capped at 1) + weight_appearance * (1 - similarity of A's tail and B's head
signatures; 1 when unknown) + weight_team * (0 same / 1 different; only when both team labels are given).

Fail closed. A pair is linked only if ALL hold, otherwise both tracklets are left as they are and the abstention is
logged with its reasons:
  * B is A's cheapest candidate and A is B's cheapest candidate (two-way), each beating the second best by `margin`
    (every gate-passing tracklet competes, confirmed or not);
  * both tracklets confirmed; appearance known and >= min_similarity; no team conflict;
  * not across an uncertain episode (A not IDENTITY_UNCERTAIN from its last observation on, B not uncertain on its
    first row) and the two never named each other in `identity_rivals` (also anywhere along a chain), unless the
    appearance similarity reaches strict_min_similarity AND allow_uncertain_clearance is on (OFF since 2026-11-20:
    torso colour let a same-kit teammate through on DEV, records/tracklet_linking_v1_2026-11-20);
  * the merged rows can still be written truthfully (an uncertain row whose only rivals are its own chain cannot).
Confidence is never raised by a link: B's rows carry the chain's `uncertain_episodes` (so a follower selected
before the link stays tainted after an earlier episode), B's own identity state, and a `confidence` that is the
lower of B's own and the merged track's window value. `confirmed` is carried forward as the contract requires
(a confirmed track never becomes unconfirmed), as for a tracker recovery.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

import numpy as np
import yaml

from ..appearance import describe, similarity as hist_similarity
from ..config import AppearanceConfig, ConfigError, typed_from_dict

LINKING_CONFIG_VERSION = "1.1.0"   # 1.1.0: + allow_uncertain_clearance
LINK_LOG_SCHEMA = ("spana.links", "1.0.0")
_EPS = 1e-9

# abstention reasons
AMBIGUOUS_SUCCESSORS = "AMBIGUOUS_SUCCESSORS"        # A's best and second-best B within margin
AMBIGUOUS_PREDECESSORS = "AMBIGUOUS_PREDECESSORS"    # B's best and second-best A within margin
NOT_MUTUAL_BEST = "NOT_MUTUAL_BEST"                  # B prefers another predecessor
UNCONFIRMED = "UNCONFIRMED"
APPEARANCE_UNKNOWN = "APPEARANCE_UNKNOWN"
APPEARANCE_BELOW_MIN = "APPEARANCE_BELOW_MIN"
TEAM_CONFLICT = "TEAM_CONFLICT"
UNCERTAIN_EPISODE = "UNCERTAIN_EPISODE"
IDENTITY_RIVAL = "IDENTITY_RIVAL"
IDENTITY_RIVAL_IN_CHAIN = "IDENTITY_RIVAL_IN_CHAIN"
UNREPRESENTABLE_RIVALRY = "UNREPRESENTABLE_RIVALRY"


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class LinkingConfig:
    schema_version: str
    max_gap_s: float
    gate_base_h: float
    max_speed_h_per_s: float
    max_height_ratio: float
    velocity_window_s: float
    extrapolate_max_s: float
    signature_samples: int
    min_similarity: float
    strict_min_similarity: float
    allow_uncertain_clearance: bool   # strict_min_similarity may clear an uncertain episode / rival (else never)
    weight_gap: float
    weight_motion: float
    weight_appearance: float
    weight_team: float
    margin: float

    def validate(self) -> None:
        if self.schema_version != LINKING_CONFIG_VERSION:
            raise ConfigError(f"linking schema_version {self.schema_version!r} != {LINKING_CONFIG_VERSION!r}")
        for n in ("max_gap_s", "gate_base_h", "max_speed_h_per_s", "velocity_window_s", "extrapolate_max_s"):
            if not getattr(self, n) > 0:
                raise ConfigError(f"linking.{n} must be > 0")
        if self.max_height_ratio < 1.0:
            raise ConfigError("linking.max_height_ratio must be >= 1")
        if self.signature_samples < 1:
            raise ConfigError("linking.signature_samples must be >= 1")
        for n in ("min_similarity", "strict_min_similarity"):
            if not 0.0 <= getattr(self, n) <= 1.0:
                raise ConfigError(f"linking.{n} must be in [0, 1]")
        if self.strict_min_similarity < self.min_similarity:
            raise ConfigError("linking.strict_min_similarity must be >= min_similarity")
        for n in ("weight_gap", "weight_motion", "weight_appearance", "weight_team", "margin"):
            if getattr(self, n) < 0:
                raise ConfigError(f"linking.{n} must be >= 0")

    def sha256(self) -> str:
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def linking_config_from_dict(data: Mapping) -> LinkingConfig:
    cfg = typed_from_dict(LinkingConfig, data, "linking")
    cfg.validate()
    return cfg


def load_linking_config(path: str | Path) -> LinkingConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"linking config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return linking_config_from_dict(yaml.safe_load(fh))


# ----------------------------------------------------------------------------- appearance
class Embedder(Protocol):
    """Appearance evidence for linking. Any model can stand behind it (torso colour now, re-ID later)."""
    name: str

    def embed(self, frame_bgr: np.ndarray, boxes: Sequence[Sequence[float]]) -> List[Optional[np.ndarray]]:
        """One vector per box; None when the crop is not clean evidence (too small, overlapped)."""

    def aggregate(self, vectors: Sequence[np.ndarray]) -> np.ndarray:
        """One signature from several crops of the same tracklet end."""

    def similarity(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        """In [0, 1], higher = more alike; None when either is unknown."""


class TorsoColourEmbedder:
    """The tracker's current appearance feature (spana/appearance.py): torso hue-saturation histogram, crops
    overlapping another box by more than appearance.clean_max_iou are unknown."""
    name = "torso_colour_hs_hist_v1"

    def __init__(self, cfg: AppearanceConfig):
        self.cfg = cfg

    def embed(self, frame_bgr, boxes):
        return describe(frame_bgr, boxes, self.cfg)

    def aggregate(self, vectors):
        m = np.mean(np.asarray(vectors, dtype=np.float64), axis=0)
        s = float(m.sum())
        return m / s if s > 0 else m

    def similarity(self, a, b):
        return hist_similarity(a, b)


Signatures = Dict[str, Dict[str, Optional[np.ndarray]]]      # track id -> {"head": vec | None, "tail": vec | None}


def compute_signatures(frames: Iterable[Tuple[int, np.ndarray]], doc: dict, embedder: Embedder,
                       samples: int) -> Signatures:
    """Head / tail signatures per track from (frame index, BGR image) pairs covering the tracks' frames. Every
    observed box of the frame is passed to the embedder together, so overlap with any other player is judged."""
    rows_by_frame = {fr["frame"]: [t for t in fr["tracks"] if t["source"] == "OBSERVED"] for fr in doc["frames"]}
    head: Dict[str, List[np.ndarray]] = {}
    tail: Dict[str, Deque[np.ndarray]] = {}
    for f, img in frames:
        obs = rows_by_frame.get(f)
        if not obs:
            continue
        vecs = embedder.embed(img, [t["box"] for t in obs])
        for t, v in zip(obs, vecs):
            if v is None:
                continue
            tid = t["track_id"]
            h = head.setdefault(tid, [])
            if len(h) < samples:
                h.append(v)
            tail.setdefault(tid, deque(maxlen=samples)).append(v)
    ids = {t["track_id"] for fr in doc["frames"] for t in fr["tracks"]}
    return {tid: {"head": embedder.aggregate(head[tid]) if head.get(tid) else None,
                  "tail": embedder.aggregate(list(tail[tid])) if tail.get(tid) else None} for tid in sorted(ids)}


# ----------------------------------------------------------------------------- geometry
def _foot(b: Sequence[float]) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, b[3]], dtype=float)


class _Camera:
    """Frame-to-frame affines of the tracks document (frame n maps frame n-1 coordinates to frame n)."""

    def __init__(self, doc: dict):
        self.A = [np.asarray(fr["camera"]["affine"], dtype=float) for fr in doc["frames"]]
        self.scale = [float(np.sqrt(abs(np.linalg.det(a[:, :2])))) for a in self.A]

    def carry(self, pt: np.ndarray, n_from: int, n_to: int) -> np.ndarray:
        p = np.asarray(pt, dtype=float)
        for k in range(n_from + 1, n_to + 1):
            p = self.A[k][:, :2] @ p + self.A[k][:, 2]
        return p

    def carry_scale(self, n_from: int, n_to: int) -> float:
        s = 1.0
        for k in range(n_from + 1, n_to + 1):
            s *= self.scale[k]
        return s

    def carry_box(self, b: Sequence[float], n_from: int, n_to: int) -> List[float]:
        c = self.carry(np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]), n_from, n_to)
        s = self.carry_scale(n_from, n_to)
        hw, hh = (b[2] - b[0]) * s / 2.0, (b[3] - b[1]) * s / 2.0
        return [c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------------- tracklets
@dataclass
class _Tracklet:
    tid: str
    epoch: int
    confirmed: bool
    rows: List[Tuple[int, dict]]          # (frame position, row)
    obs: List[int]                        # indices into rows of observed rows

    @property
    def first_pos(self) -> int:
        return self.rows[0][0]

    @property
    def last_obs_pos(self) -> int:
        return self.rows[self.obs[-1]][0]

    def rivals(self) -> Set[str]:
        return {r for _, row in self.rows for r in row["identity_rivals"]}


def _tracklets(doc: dict) -> Dict[str, _Tracklet]:
    conf = {r["track_id"]: r["confirmed"] for r in doc["tracklets"]}
    out: Dict[str, _Tracklet] = {}
    for n, fr in enumerate(doc["frames"]):
        for row in fr["tracks"]:
            tl = out.get(row["track_id"])
            if tl is None:
                tl = out[row["track_id"]] = _Tracklet(row["track_id"], row["epoch"], conf[row["track_id"]], [], [])
            if row["source"] == "OBSERVED":
                tl.obs.append(len(tl.rows))
            tl.rows.append((n, row))
    return out


# ----------------------------------------------------------------------------- candidates and decisions
@dataclass
class _Pair:
    a: str
    b: str
    gap_s: float
    displacement_h: float
    gate_h: float
    similarity: Optional[float]
    terms: Dict[str, Optional[float]]
    cost: float

    def as_log(self) -> dict:
        return {"to_track": self.b, "from_track": self.a, "gap_s": round(self.gap_s, 6),
                "displacement_h": round(self.displacement_h, 4), "gate_h": round(self.gate_h, 4),
                "similarity": None if self.similarity is None else round(self.similarity, 4),
                "terms": {k: None if v is None else round(v, 4) for k, v in self.terms.items()},
                "cost": round(self.cost, 4)}


def _candidates(doc: dict, tls: Dict[str, _Tracklet], sig: Signatures, sim: Callable, cfg: LinkingConfig,
                teams: Optional[Mapping[str, str]]) -> List[_Pair]:
    cam = _Camera(doc)
    times = [fr["t"] for fr in doc["frames"]]
    pairs = []
    for a in tls.values():
        last_n, last_row = a.rows[a.obs[-1]]
        t_last = times[last_n]
        foot_last, h_last = _foot(last_row["box"]), last_row["box"][3] - last_row["box"][1]
        v = np.zeros(2)                                        # A's camera-compensated velocity, px / s (frame of last)
        early = [a.rows[k] for k in a.obs[:-1] if t_last - times[a.rows[k][0]] <= cfg.velocity_window_s + _EPS]
        if early:
            n0, r0 = early[0]
            v = (foot_last - cam.carry(_foot(r0["box"]), n0, last_n)) / (t_last - times[n0])
        for b in tls.values():
            if b.tid == a.tid or b.epoch != a.epoch or b.first_pos <= last_n:
                continue
            gap = times[b.first_pos] - t_last
            if gap > cfg.max_gap_s + _EPS:
                continue
            first_row = b.rows[0][1]
            hb = first_row["box"][3] - first_row["box"][1]
            ha = h_last * cam.carry_scale(last_n, b.first_pos)
            if not 1.0 / cfg.max_height_ratio <= hb / ha <= cfg.max_height_ratio:
                continue
            h = (ha + hb) / 2.0
            fb = _foot(first_row["box"])
            disp = float(np.linalg.norm(fb - cam.carry(foot_last, last_n, b.first_pos))) / h
            gate = cfg.gate_base_h + cfg.max_speed_h_per_s * gap
            if disp > gate + _EPS:
                continue
            pred = cam.carry(foot_last + v * min(gap, cfg.extrapolate_max_s), last_n, b.first_pos)
            motion = min(1.0, float(np.linalg.norm(fb - pred)) / h / gate)
            s = sim(sig.get(a.tid, {}).get("tail"), sig.get(b.tid, {}).get("head"))
            team = None
            if teams is not None and teams.get(a.tid) is not None and teams.get(b.tid) is not None:
                team = 0.0 if teams[a.tid] == teams[b.tid] else 1.0
            terms = {"gap": gap / cfg.max_gap_s, "motion": motion, "appearance": None if s is None else 1.0 - s,
                     "team": team}
            cost = cfg.weight_gap * terms["gap"] + cfg.weight_motion * motion + \
                cfg.weight_appearance * (1.0 if s is None else 1.0 - s) + cfg.weight_team * (team or 0.0)
            pairs.append(_Pair(a.tid, b.tid, gap, disp, gate, s, terms, cost))
    return pairs


def _uncertain_across(a: _Tracklet, b: _Tracklet) -> bool:
    tail = [row for k, (_, row) in enumerate(a.rows) if k >= a.obs[-1]]
    return any(r["identity"] != "CERTAIN" for r in tail) or b.rows[0][1]["identity"] != "CERTAIN"


def _pair_reasons(p: _Pair, tls: Dict[str, _Tracklet], cfg: LinkingConfig) -> Tuple[List[str], bool]:
    """Reasons that refuse this pair on its own (not ranking), and whether a strict appearance clearance was used."""
    a, b = tls[p.a], tls[p.b]
    reasons, cleared = [], False
    if not (a.confirmed and b.confirmed):
        reasons.append(UNCONFIRMED)
    if p.similarity is None:
        reasons.append(APPEARANCE_UNKNOWN)
    elif p.similarity < cfg.min_similarity:
        reasons.append(APPEARANCE_BELOW_MIN)
    if p.terms["team"] == 1.0:
        reasons.append(TEAM_CONFLICT)
    strict = cfg.allow_uncertain_clearance and p.similarity is not None and \
        p.similarity >= cfg.strict_min_similarity
    if _uncertain_across(a, b):
        if strict:
            cleared = True
        else:
            reasons.append(UNCERTAIN_EPISODE)
    if p.b in a.rivals() or p.a in b.rivals():
        if strict:
            cleared = True
        else:
            reasons.append(IDENTITY_RIVAL)
    return reasons, cleared


def decide(doc: dict, sig: Signatures, sim: Callable, cfg: LinkingConfig,
           teams: Optional[Mapping[str, str]] = None) -> Tuple[List[_Pair], List[dict]]:
    """The links to make and the full log (one entry per predecessor tracklet that has any candidate)."""
    tls = _tracklets(doc)
    pairs = _candidates(doc, tls, sig, sim, cfg, teams)
    key = lambda p: (p.cost, p.a, p.b)                       # noqa: E731  deterministic ties
    by_a: Dict[str, List[_Pair]] = {}
    by_b: Dict[str, List[_Pair]] = {}
    for p in sorted(pairs, key=key):
        by_a.setdefault(p.a, []).append(p)
        by_b.setdefault(p.b, []).append(p)
    links, log = [], {}
    for a_id in sorted(by_a):
        cand = by_a[a_id]
        best = cand[0]
        reasons, cleared = _pair_reasons(best, tls, cfg)
        m_succ = cand[1].cost - best.cost if len(cand) > 1 else None
        if m_succ is not None and m_succ < cfg.margin - _EPS:
            reasons.insert(0, AMBIGUOUS_SUCCESSORS)
        rev = by_b[best.b]
        m_pred = None
        if rev[0].a != a_id:
            reasons.insert(0, NOT_MUTUAL_BEST)
        elif len(rev) > 1:
            m_pred = rev[1].cost - rev[0].cost
            if m_pred < cfg.margin - _EPS:
                reasons.insert(0, AMBIGUOUS_PREDECESSORS)
        entry = {"kind": "abstain" if reasons else "link", **best.as_log(),
                 "from_last_observed_frame": doc["frames"][tls[a_id].last_obs_pos]["frame"],
                 "to_first_frame": doc["frames"][tls[best.b].first_pos]["frame"],
                 "margin_successors": None if m_succ is None else round(m_succ, 4),
                 "margin_predecessors": None if m_pred is None else round(m_pred, 4),
                 "cleared_by_appearance": cleared, "reasons": reasons,
                 "candidates": [p.as_log() for p in cand[:5]],
                 "predecessor_candidates": [p.as_log() for p in rev[:5]]}
        log[a_id] = entry
        if not reasons:
            links.append(best)
    # chain-level rules: rivalry between any two members of a chain, and rows the merge could not write truthfully
    while True:
        bad = _chain_conflict(links, tls, cfg)
        if bad is None:
            break
        p, reason = bad
        links.remove(p)
        log[p.a].update(kind="abstain", reasons=[reason])
    return links, [log[k] for k in sorted(log)]


def _chains(links: Sequence[_Pair]) -> List[List[str]]:
    succ = {p.a: p.b for p in links}
    preds = {p.b for p in links}
    out = []
    for h in sorted(set(succ) - preds):
        c = [h]
        while c[-1] in succ:
            c.append(succ[c[-1]])
        out.append(c)
    return out


def _kept_rows(tl: _Tracklet, nxt: Optional[_Tracklet]) -> List[dict]:
    return [row for n, row in tl.rows if nxt is None or n < nxt.first_pos]


def _chain_conflict(links: List[_Pair], tls: Dict[str, _Tracklet], cfg: LinkingConfig
                    ) -> Optional[Tuple[_Pair, str]]:
    by_a = {p.a: p for p in links}
    for chain in _chains(links):
        members = set(chain)
        for i, x in enumerate(chain):
            for j in range(i + 1, len(chain)):
                y = chain[j]
                if y in tls[x].rivals() or x in tls[y].rivals():
                    weak = [by_a[chain[k]] for k in range(i, j)
                            if not cfg.allow_uncertain_clearance or by_a[chain[k]].similarity is None
                            or by_a[chain[k]].similarity < cfg.strict_min_similarity]
                    if weak:
                        return weak[0], IDENTITY_RIVAL_IN_CHAIN
        for i, x in enumerate(chain):
            nxt = tls[chain[i + 1]] if i + 1 < len(chain) else None
            for row in _kept_rows(tls[x], nxt):
                if row["identity"] != "CERTAIN" and not row.get("identity_doubt", False) and \
                        row["identity_rivals"] and set(row["identity_rivals"]) <= members:
                    # cut the chain next to x, towards a rival it names
                    k = chain.index(row["identity_rivals"][0])
                    return by_a[chain[i if k > i else i - 1]], UNREPRESENTABLE_RIVALRY
    return None


# ----------------------------------------------------------------------------- writing the linked document
def apply_links(doc: dict, links: Sequence[_Pair], confidence_window_s: float, score_decimals: int,
                box_decimals: int) -> Tuple[List[dict], List[dict], Dict[str, str]]:
    """New frames and tracklets with every chain written as one track (the head's id). Returns (frames, tracklets,
    member id -> head id)."""
    tls = _tracklets(doc)
    cam = _Camera(doc)
    chains = _chains(links)
    gap_of = {p.b: p.gap_s for p in links}
    head_of = {m: c[0] for c in chains for m in c}
    nxt_of = {c[i]: c[i + 1] for c in chains for i in range(len(c) - 1)}
    cut_pos = {m: tls[s].first_pos for m, s in nxt_of.items()}   # rows of m from here on are replaced by s
    first_frame = {c[0]: doc["frames"][tls[c[0]].first_pos]["frame"] for c in chains}

    def ren(x):
        return head_of.get(x, x)

    def ren_list(xs, me, sort):
        out = []
        for x in xs:
            y = ren(x)
            if y != me and y not in out:
                out.append(y)
        return sorted(out) if sort else out

    st: Dict[str, dict] = {}      # head -> merged state: last row, current member, offsets, history
    frames_out, tracklet_tail = [], {}
    for n, fr in enumerate(doc["frames"]):
        f, t = fr["frame"], fr["t"]
        dropped = {tr["track_id"] for tr in fr["tracks"]
                   if tr["track_id"] in cut_pos and n >= cut_pos[tr["track_id"]]}
        rows, recompute = [], []
        for tr in fr["tracks"]:
            tid = tr["track_id"]
            if tid in dropped:
                continue
            row = copy.deepcopy(tr)
            me = ren(tid)
            row["track_id"] = me
            row["possible_predecessors"] = ren_list(tr["possible_predecessors"], me, sort=False)
            row["identity_rivals"] = ren_list(tr["identity_rivals"], me, sort=True)
            if row["occluder"] is not None:
                row["occluder"] = ren(row["occluder"])
                if row["occluder"] == me:      # pragma: no cover - an unobserved member cannot occlude its chain
                    raise ValueError(f"frame {f}: {tid} occluded by its own chain")
            if row["overlap_with"] is not None:
                if row["overlap_with"] in dropped or ren(row["overlap_with"]) == me:
                    recompute.append(row)
                row["overlap_with"] = ren(row["overlap_with"])
            if tid in head_of:
                _merge_row(row, tr, tid, me, f, t, st, gap_of, first_frame[me], confidence_window_s, score_decimals)
            if row["identity"] != "CERTAIN" and not row["identity_rivals"] and not row.get("identity_doubt", False):
                raise ValueError(f"frame {f}: {me} would be uncertain with no rival")   # guarded by _chain_conflict
            rows.append(row)
        ended = []
        for tid, reason in fr["ended"]:
            if tid in nxt_of:
                continue                      # a member with a successor does not end: the chain continues
            if tid in head_of:
                st[head_of[tid]]["done"] = True
            ended.append([ren(tid), reason])
        live = {r["track_id"] for r in rows}
        for h, s in st.items():       # heads in a gap between two members: A held, camera-carried
            if h in live or s["done"] or n <= s["last_pos"]:
                continue
            row = _held_row(s, n, f, t, cam, box_decimals, confidence_window_s, score_decimals)
            rows.append(row)
            recompute.append(row)
        for row in recompute:
            others = [(o["track_id"], _iou(row["box"], o["box"])) for o in rows if o is not row]
            j, o = max(others, key=lambda x: x[1]) if others else (None, 0.0)
            o = round(o, score_decimals)
            row["max_overlap_iou"], row["overlap_with"] = (o, j) if o > 0 else (0.0, None)
        for row in rows:
            if row["track_id"] in st:
                st[row["track_id"]]["last_pos"] = n
                st[row["track_id"]]["last"] = row
        frames_out.append({**{k: v for k, v in fr.items() if k not in ("tracks", "ended")},
                           "tracks": rows, "ended": ended})
    tracklets = []
    by_id = {r["track_id"]: r for r in doc["tracklets"]}
    for r in doc["tracklets"]:
        tid = r["track_id"]
        if tid in head_of and head_of[tid] != tid:
            continue
        row = dict(r, possible_predecessors=ren_list(r["possible_predecessors"], ren(tid), sort=False))
        if tid in head_of:
            chain = next(c for c in chains if c[0] == tid)
            tail = by_id[chain[-1]]
            last = st[tid]["last"]
            row.update(last_frame=tail["last_frame"], last_observed_frame=tail["last_observed_frame"],
                       observed_frames=last["observed_frames"], confirmed=any(by_id[m]["confirmed"] for m in chain),
                       end_reason=tail["end_reason"], recoveries=last["recoveries"])
        tracklets.append(row)
    return frames_out, tracklets, head_of


def _window_conf(hist: Deque[Tuple[float, float]], t: float, window: float) -> float:
    while hist and t - hist[0][0] > window + _EPS:
        hist.popleft()
    return sum(s for _, s in hist) / len(hist)


def _merge_row(row: dict, tr: dict, tid: str, head: str, f: int, t: float, st: Dict[str, dict],
               gap_of: Dict[str, float], head_first_frame: int, window: float, dec: int) -> None:
    s = st.get(head)
    if s is None:                                  # the head itself: rows unchanged except renamed references
        s = st[head] = {"member": tid, "first_frame": head_first_frame, "obs_off": 0, "rec_off": 0, "eps_off": 0, "gap": 0.0, "conf": False,
                        "hist": deque(), "last": None, "last_pos": -1, "done": False, "last_obs_t": None}
    elif s["member"] != tid:                       # first row of the next member: the link
        last = s["last"]
        s.update(member=tid, obs_off=last["observed_frames"], rec_off=last["recoveries"] + 1,
                 eps_off=last["uncertain_episodes"], gap=gap_of[tid], done=False)
    observed = tr["source"] == "OBSERVED"
    s["hist"].append((t, float(tr["detection_score"]) if observed else 0.0))
    merged_conf = round(_window_conf(s["hist"], t, window), dec)
    if s["member"] != head:
        row["age_frames"] = f - head_first_frame
        row["observed_frames"] = tr["observed_frames"] + s["obs_off"]
        row["recoveries"] = tr["recoveries"] + s["rec_off"]
        row["last_gap_s"] = tr["last_gap_s"] if tr["recoveries"] > 0 else round(s["gap"], 6)
        row["uncertain_episodes"] = tr["uncertain_episodes"] + s["eps_off"]
        row["confirmed"] = tr["confirmed"] or s["conf"]
        row["confidence"] = min(tr["confidence"], merged_conf)
    s["conf"] = s["conf"] or row["confirmed"]
    if observed:
        s["last_obs_t"] = t


def _held_row(s: dict, n: int, f: int, t: float, cam: _Camera, box_dec: int, window: float, dec: int) -> dict:
    last = s["last"]
    box = [round(v, box_dec) for v in cam.carry_box(last["box"], s["last_pos"], n)]
    s["hist"].append((t, 0.0))
    row = dict(copy.deepcopy(last), state="LOST", source="HELD", box=box, detection_index=None,
               detection_score=None, confidence=round(_window_conf(s["hist"], t, window), dec), contested=False,
               age_frames=f - s["first_frame"], missed_frames=last["missed_frames"] + 1,
               time_since_observed_s=round(t - s["last_obs_t"], 6), occluder=None)
    return row


# ----------------------------------------------------------------------------- whole document
def link_document(doc: dict, sig: Signatures, embedder_name: str, sim: Callable, cfg: LinkingConfig, *,
                  confidence_window_s: float, score_decimals: int, box_decimals: int, provenance: dict,
                  teams: Optional[Mapping[str, str]] = None) -> Tuple[dict, List[dict], Dict[str, str]]:
    """(linked tracks document, link log rows, member id -> head id). The output keeps the input's schema version
    and is validated by the caller with validate_tracks."""
    links, log = decide(doc, sig, sim, cfg, teams)
    frames, tracklets, head_of = apply_links(doc, links, confidence_window_s, score_decimals, box_decimals)
    head_of = {m: h for m, h in head_of.items() if m != h}
    for e in log:
        e["head"] = head_of.get(e["from_track"], e["from_track"]) if e["kind"] == "link" else None
    out = {"schema": doc["schema"], "schema_version": doc["schema_version"],
           "provenance": {**provenance, "stage": "tracklet_linking_v1", "embedder": embedder_name,
                          "links": sum(e["kind"] == "link" for e in log),
                          "abstentions": sum(e["kind"] == "abstain" for e in log),
                          "source_tracks_provenance": doc["provenance"]},
           "video": doc["video"], "detections_sha256": doc["detections_sha256"], "frames": frames,
           "tracklets": tracklets}
    return out, log, head_of


def rename_decisions(rows: Iterable[dict], head_of: Mapping[str, str]) -> List[dict]:
    """The tracker's decision log with linked member ids replaced by their chain's id (so tools that look a
    decision's track up in the linked document find it)."""
    def ren(v):
        if isinstance(v, str):
            return head_of.get(v, v)
        if isinstance(v, list):
            return [ren(x) for x in v]
        if isinstance(v, dict):
            return {k: ren(x) for k, x in v.items()}
        return v
    return [ren(r) for r in rows]
