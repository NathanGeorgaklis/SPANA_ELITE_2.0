"""How much could re-identification recover, and how far do clues we already have go? (DEV analysis, no new model.)

    python -m tools.reid_headroom --config config/default.yaml --out REPORT.json \
        --clip data/labels/<clip>/v1 TRK.json [--video VIDEO] [--clip ...]

For each labelled player, the tracks attributed to them on labelled frames (NESTED_V2, unambiguous; same matching
as tools/target_score.py) form a chain. Every change A -> B in that chain is a HANDOVER: a relinker would have to
move the target from track A to track B. Per handover:
  kind          LOSS when A was observed before B was born and not again from B's birth up to B's labelled frame
                (a relinker had to act), else OVERLAP (A and B coexisted: parallel tracks on one player or an
                identity switch; not a relink case)
  gap_s         time from A's last observation to B's first frame
  a_end         A's EndReason (or null if A was still live at the end of the clip)
  cut_between   a CUT / UNKNOWN camera frame lies between A's last observation and B's birth
Clues, each ranking B among the CANDIDATES = tracks born after A's last observation, up to and including B's birth
(what a relinker could have considered when B appeared). Rank 1 = B is the best candidate; ties count against B.
  predecessor   B's own tracklet lists A in possible_predecessors (tracker evidence); also how many other
                candidates list A
  position      distance (in A's last box heights) from A's last observed foot point, carried through the camera
                transforms to B's birth frame, to each candidate's first foot point. Not available across a cut.
  appearance    (only with --video) Bhattacharyya similarity between A's torso colour histogram (mean of clean
                descriptors over its last appearance.tau_s observed) and each candidate's (first appearance.tau_s
                observed); spana.appearance, config appearance section. A candidate with no clean crop is unknown.
Rules: simple relink rules from these clues (unique predecessor, best position, best appearance, agreements), each
counted as picks / correct / wrong over LOSS handovers.
Ceiling: per player (a player no track ever matches unambiguously is not selectable: v1 counts 0), the share of
labelled frames (from their first labelled frame) on which SOME track attributed
to them is ACTIVE, confirmed and CERTAIN, i.e. what the v1 follower would score with perfect relinking, next to
v1's own correct count. Reporting only: nothing here is used by the product, nothing is tuned.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from spana.appearance import describe, similarity
from spana.artifacts import read_json, sha256_doc, validate_tracks, write_json_atomic
from spana.config import load_config
from spana.identity.follower import Selection, follow
from tools.benchmark import load_frozen
from tools.target_score import RULE, attributions

BREAKS = ("CUT", "UNKNOWN")


def _foot(b: Sequence[float]) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, b[3]])


def _best(scores: Dict[str, Optional[float]], higher_better: bool) -> Optional[str]:
    """The strictly best candidate (no tie), or None."""
    known = sorted(((v, k) for k, v in scores.items() if v is not None), reverse=higher_better)
    if not known or (len(known) > 1 and known[0][0] == known[1][0]):
        return None
    return known[0][1]


def _rank(scores: Dict[str, Optional[float]], key: str, higher_better: bool) -> Optional[int]:
    """1 + number of other candidates scoring at least as well as `key` (ties count against it); None if unknown."""
    s = scores.get(key)
    if s is None:
        return None
    others = [v for k, v in scores.items() if k != key and v is not None]
    return 1 + sum(1 for v in others if (v >= s if higher_better else v <= s))


class TrackIndex:
    def __init__(self, doc: dict):
        self.frames = doc["frames"]
        self.pos = {fr["frame"]: n for n, fr in enumerate(self.frames)}
        self.obs: Dict[str, Dict[int, List[float]]] = {}
        self.first: Dict[str, int] = {}
        self.live: Dict[int, Dict[str, dict]] = {}
        for fr in self.frames:
            self.live[fr["frame"]] = {t["track_id"]: t for t in fr["tracks"]}
            for t in fr["tracks"]:
                self.first.setdefault(t["track_id"], fr["frame"])
                if t["source"] == "OBSERVED":
                    self.obs.setdefault(t["track_id"], {})[fr["frame"]] = t["box"]
        self.tracklet = {r["track_id"]: r for r in doc["tracklets"]}

    def t(self, frame: int) -> float:
        return self.frames[self.pos[frame]]["t"]

    def carry(self, point: np.ndarray, f_from: int, f_to: int) -> Optional[np.ndarray]:
        """Map an image point from frame f_from to frame f_to through the per-frame camera affines; None across a
        CUT / UNKNOWN frame."""
        p = np.array([point[0], point[1], 1.0])
        for fr in self.frames[self.pos[f_from] + 1: self.pos[f_to] + 1]:
            if fr["camera"]["status"] in BREAKS:
                return None
            p = np.array([*(np.asarray(fr["camera"]["affine"]) @ p), 1.0])
        return p[:2]

    def cut_between(self, f_from: int, f_to: int) -> bool:
        return any(fr["camera"]["status"] in BREAKS for fr in self.frames[self.pos[f_from] + 1: self.pos[f_to] + 1])


def chains(gt, attr) -> Dict[str, List[Tuple[int, str]]]:
    out: Dict[str, List[Tuple[int, str]]] = {}
    for f in gt.frames:
        for tid, (p, amb) in attr[f].items():
            if p is not None and not amb:
                out.setdefault(p, []).append((f, tid))
    return out


def handovers(chain: List[Tuple[int, str]]) -> List[Tuple[int, str, int, str]]:
    """(last labelled frame of A, A, first labelled frame of B, B) for every track change; a track that comes back
    later in the chain (A -> B -> A) counts as a handover each time."""
    return [(fa, a, fb, b) for (fa, a), (fb, b) in zip(chain, chain[1:]) if a != b]


class Appearance:
    """Mean clean torso descriptors per (track, frame set), computed from the video in one pass."""

    def __init__(self, video: Optional[str], ti: TrackIndex, cfg, sha: str):
        self.desc: Dict[Tuple[str, int], Optional[np.ndarray]] = {}
        self.enabled = video is not None
        self.ti, self.cfg, self.video, self.sha = ti, cfg, video, sha

    def compute(self, needed: Dict[int, set]) -> None:
        if not self.enabled or not needed:
            return
        from spana.video import VideoSource
        src = VideoSource(self.video, expected_sha256=self.sha)
        lo, hi = min(needed), max(needed)
        for i, _, img in src.frames(lo, hi + 1):
            if i not in needed:
                continue
            obs = [(tid, t["box"]) for tid, t in self.ti.live[i].items() if t["source"] == "OBSERVED"]
            ds = describe(img, [b for _, b in obs], self.cfg.appearance)
            for (tid, _), d in zip(obs, ds):
                if tid in needed[i]:
                    self.desc[(tid, i)] = d

    def mean(self, tid: str, frames: Sequence[int]) -> Optional[np.ndarray]:
        ds = [self.desc.get((tid, f)) for f in frames]
        ds = [d for d in ds if d is not None]
        if not ds:
            return None
        m = np.mean(ds, axis=0)
        return m / m.sum()


def window(ti: TrackIndex, tid: str, anchor: int, tau: float, after: bool) -> List[int]:
    """Observed frames of tid within tau seconds after (or up to and including) the anchor frame."""
    t0 = ti.t(anchor)
    return [f for f in sorted(ti.obs.get(tid, {}))
            if (0 <= ti.t(f) - t0 <= tau if after else 0 <= t0 - ti.t(f) <= tau)]


def analyse_clip(label_dir: Path, tracks_path: Path, video: Optional[str], cfg) -> dict:
    frozen, gt = load_frozen(label_dir)
    doc = read_json(tracks_path)
    validate_tracks(doc)
    if doc["video"]["sha256"] != frozen["clip"]["sha256"]:
        raise SystemExit("refusing: the tracks were made from a different video than the labels")
    ti = TrackIndex(doc)
    attr = attributions(gt, doc)
    ch = chains(gt, attr)
    tau = cfg.appearance.tau_s

    raw = []
    for pid in sorted(ch, key=int):
        for fa, a, fb, b in handovers(ch[pid]):
            born = ti.first[b]
            # A's last observation before B was born (A may live on after B's birth: OVERLAP)
            a_obs_before = [f for f in ti.obs[a] if f < born]
            a_last = max(a_obs_before) if a_obs_before else min(ti.obs[a])
            # LOSS: A was seen before B was born and not seen again from B's birth up to B's labelled frame, so a
            # relinker had to act. Otherwise A and B coexisted (parallel tracks or an identity switch): OVERLAP.
            parallel = not a_obs_before or any(born <= f <= fb for f in ti.obs[a])
            cands = [tid for tid, f1 in ti.first.items() if a_last < f1 <= born and tid != a]
            raw.append({"player": pid, "a": a, "b": b, "a_last_obs": a_last, "b_born": born, "fa": fa, "fb": fb,
                        "parallel": parallel, "cands": cands})

    needed: Dict[int, set] = {}
    for e in raw:
        for f in window(ti, e["a"], e["a_last_obs"], tau, after=False):
            needed.setdefault(f, set()).add(e["a"])
        for c in e["cands"]:
            for f in window(ti, c, ti.first[c], tau, after=True):
                needed.setdefault(f, set()).add(c)
    app = Appearance(video, ti, cfg, doc["video"]["sha256"])
    app.compute(needed)

    events = []
    for e in raw:
        a, b, a_last, born, cands = e["a"], e["b"], e["a_last_obs"], e["b_born"], e["cands"]
        a_box = ti.obs[a][a_last]
        h = a_box[3] - a_box[1]
        foot = ti.carry(_foot(a_box), a_last, born)
        dist = {}
        for c in cands:
            f1 = ti.first[c]                       # a track is born observed
            pc = None if foot is None else ti.carry(_foot(ti.obs[c][f1]), f1, born)
            dist[c] = None if pc is None else float(np.linalg.norm(pc - foot) / h)
        sims = {}
        if app.enabled:
            da = app.mean(a, window(ti, a, a_last, tau, after=False))
            for c in cands:
                sims[c] = similarity(da, app.mean(c, window(ti, c, ti.first[c], tau, after=True)))
        trk_b = ti.tracklet[b]
        listers = [c for c in cands if a in ti.tracklet[c]["possible_predecessors"]]
        events.append({
            "player": e["player"], "from_track": a, "to_track": b,
            "kind": "OVERLAP" if e["parallel"] else "LOSS",
            "a_last_observed_frame": a_last, "b_first_frame": born,
            "gap_s": round(ti.t(born) - ti.t(a_last), 4),
            "a_end": ti.tracklet[a]["end_reason"], "cut_between": ti.cut_between(a_last, born),
            "candidates": len(cands),
            "predecessor": {"b_lists_a": a in trk_b["possible_predecessors"], "candidates_listing_a": len(listers),
                            "pick": listers[0] if len(listers) == 1 else None},
            "position": {"b_distance_h": None if dist.get(b) is None else round(dist[b], 3),
                         "rank": _rank(dist, b, higher_better=False), "pick": _best(dist, higher_better=False)},
            "appearance": None if not app.enabled else {
                "b_similarity": None if sims.get(b) is None else round(sims[b], 4),
                "best_other": max((v for k, v in sims.items() if k != b and v is not None), default=None),
                "rank": _rank(sims, b, higher_better=True), "pick": _best(sims, higher_better=True)},
            "labelled_gap": [e["fa"], e["fb"]],
        })

    ceiling = ceiling_rows(gt, attr, ti, doc, cfg)
    return {"clip": frozen["clip"]["name"], "tracks_sha256": sha256_doc(doc), "video_used": video is not None,
            "handovers": events, "summary": summarise(events, app.enabled), "ceiling": ceiling}


def ceiling_rows(gt, attr, ti: TrackIndex, doc: dict, cfg) -> dict:
    rows = []
    for pid in sorted({p for f in gt.frames for p in gt.players.get(f, {})}, key=int):
        vis = [f for f in gt.frames if pid in gt.players.get(f, {})]
        if not vis:
            continue
        ok = any_attr = 0
        for f in vis:
            mine = [tid for tid, (p, amb) in attr[f].items() if p == pid and not amb]
            any_attr += bool(mine)
            ok += any(ti.live[f][tid]["state"] == "ACTIVE" and ti.live[f][tid]["confirmed"]
                      and ti.live[f][tid]["identity"] == "CERTAIN" for tid in mine)
        v1 = v1_correct(pid, vis, attr, doc, cfg)
        rows.append({"player": pid, "labelled_frames": len(vis), "selectable": v1 is not None,
                     "v1_correct": v1 or 0, "ceiling_correct": ok, "attributed_any_state": any_attr})
    n = sum(r["labelled_frames"] for r in rows)
    pooled = {k: sum(r[k] for r in rows) for k in ("v1_correct", "ceiling_correct", "attributed_any_state")}
    return {"per_player": rows, "labelled_frames": n, **pooled,
            "v1_correct_time": round(pooled["v1_correct"] / n, 4) if n else None,
            "ceiling_correct_time": round(pooled["ceiling_correct"] / n, 4) if n else None}


def v1_correct(pid: str, vis: List[int], attr, doc: dict, cfg) -> Optional[int]:
    """v1 follower correct frames, selected as tools/target_score.py does (first unambiguous match)."""
    for f in vis:
        hit = [tid for tid, (p, amb) in attr[f].items() if p == pid and not amb]
        if hit:
            ans = {fr["frame"]: fr for fr in follow(doc, Selection.track_at_frame(hit[0], f), cfg)[0]}
            return sum(1 for g in vis if g >= f and ans[g]["state"] == "TARGET"
                       and attr[g].get(hit[0], (None, True)) == (pid, False))
    return None


def summarise(events: List[dict], with_app: bool) -> dict:
    loss = [e for e in events if e["kind"] == "LOSS"]
    r1 = lambda k: sum(1 for e in loss if e[k]["rank"] == 1)                     # noqa: E731
    known = lambda k: sum(1 for e in loss if e[k]["rank"] is not None)           # noqa: E731
    out = {"handovers": len(events), "overlap": len(events) - len(loss), "loss": len(loss),
           "loss_cut_between": sum(e["cut_between"] for e in loss),
           "loss_gap_s_median": float(np.median([e["gap_s"] for e in loss])) if loss else None,
           "loss_candidates_median": float(np.median([e["candidates"] for e in loss])) if loss else None,
           "loss_b_lists_a": sum(e["predecessor"]["b_lists_a"] for e in loss),
           "loss_b_lists_a_uniquely": sum(e["predecessor"]["b_lists_a"] and e["predecessor"]["candidates_listing_a"] == 1
                                          for e in loss),
           "loss_position_known": known("position"), "loss_position_rank1": r1("position")}
    out["rules"] = rules(loss, with_app)
    if with_app:
        out.update({"loss_appearance_known": known("appearance"), "loss_appearance_rank1": r1("appearance"),
                    "loss_position_and_appearance_rank1": sum(1 for e in loss if e["position"]["rank"] == 1
                                                              and e["appearance"]["rank"] == 1)})
    return out


def rules(loss: List[dict], with_app: bool) -> dict:
    """Simple relink rules on LOSS handovers: each picks a candidate (or abstains); a pick that is not the player's
    next track is WRONG (pessimistic: that candidate may be an unlabelled fragment of the same player)."""
    def pick(e, names):
        ps = [e[n]["pick"] for n in names]
        return ps[0] if all(p is not None for p in ps) and len(set(ps)) == 1 else None
    names = {"predecessor_unique": ["predecessor"], "position_best": ["position"]}
    if with_app:
        names.update({"appearance_best": ["appearance"], "position_and_appearance_agree": ["position", "appearance"],
                      "all_three_agree": ["predecessor", "position", "appearance"]})
    out = {}
    for rule, ns in names.items():
        picks = [(pick(e, ns), e["to_track"]) for e in loss]
        out[rule] = {"picks": sum(p is not None for p, _ in picks), "correct": sum(p == b for p, b in picks),
                     "wrong": sum(p is not None and p != b for p, b in picks)}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--clip", nargs=2, action="append", required=True, metavar=("LABELS_DIR", "TRACKS_JSON"))
    p.add_argument("--video", action="append", default=None,
                   help="video per --clip, in the same order (enables the appearance clue)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.video and len(a.video) != len(a.clip):
        raise SystemExit("give one --video per --clip, or none")
    cfg = load_config(a.config)
    clips = [analyse_clip(Path(l), Path(t), a.video[n] if a.video else None, cfg) for n, (l, t) in enumerate(a.clip)]
    report = {"tool": "tools.reid_headroom", "match_rule": RULE.name, "config_sha256": cfg.sha256(), "clips": clips}
    write_json_atomic(a.out, report)
    for c in clips:
        print(c["clip"], c["summary"], {k: c["ceiling"][k] for k in ("v1_correct_time", "ceiling_correct_time")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
