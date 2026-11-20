"""Tracker bench scoring against frozen DEV labels (labelled frames only; labels every `frame_step` video frames).

Overall: HOTA / DetA / AssA, IDF1, ID switches, MOTA from TrackEval (`trackeval` 1.3.0) on the labelled frames, IoU
similarity. Track boxes inside a human ignore region (IoA >= tools.benchmark.IGNORE_IOA) are dropped first, as in
tools/benchmark.py. ID switches are therefore counted between consecutive labelled frames.

Target player (every labelled player in turn, as tools/target_score.py does). Who a box is follows the repo's
NESTED_V2 rule; an ambiguous attribution counts as nobody. The player is "tapped" on their first labelled frame with
an unambiguous track, and the tapped track is then followed as it is: no relink, no user help.
- covered: labelled frames of the player, from the tap on, on which the tapped track is that player;
- coverage = covered / the player's labelled frames from the tap on;
- longest segment: longest run of consecutive covered labelled frames, in seconds (frames x frame_step / fps);
- wrong frames: labelled frames from the tap on where the tapped track is attributed to ANOTHER labelled player;
  `unflagged` = those on which the tracker called the identity certain (a silent wrong answer);
- ID switches: changes of the player's track between consecutive frames where the player is attributed to a track;
- fragments: the player's labelled frames split into runs of one track (a frame without a track also breaks a run);
  fragments = runs - 1.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from spana.evaluation import NESTED_V2, RULES, _match       # the repo's attribution rule (private helper, reused)
from spana.geometry import iou_matrix
from spana.tracking.protocol import TrackOutput
from tools.benchmark import IGNORE_IOA, _ioa

TRACKEVAL_VERSION = "1.3.0"

Run = List[Tuple[int, List[TrackOutput]]]


@dataclass
class TargetStats:
    players: int = 0
    untapped: int = 0
    frames: int = 0
    covered: int = 0
    wrong: int = 0
    wrong_unflagged: int = 0
    id_switches: int = 0
    fragments: int = 0
    longest_s: List[float] = field(default_factory=list)

    def add(self, o: "TargetStats") -> None:
        for k in ("players", "untapped", "frames", "covered", "wrong", "wrong_unflagged", "id_switches", "fragments"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        self.longest_s += o.longest_s

    def row(self) -> dict:
        return {"target_players": self.players, "target_untapped": self.untapped,
                "target_idsw": self.id_switches, "target_fragments": self.fragments,
                "target_coverage": round(self.covered / self.frames, 4) if self.frames else None,
                "target_longest_seg_s_median": round(statistics.median(self.longest_s), 2) if self.longest_s else None,
                "target_wrong_frames": self.wrong, "target_wrong_frames_unflagged": self.wrong_unflagged}


def _kept(gt, f: int, outs: Sequence[TrackOutput]) -> List[TrackOutput]:
    ign = gt.ignore.get(f, [])
    return [o for o in outs if not any(_ioa(o.box, g) >= IGNORE_IOA for g in ign)]


def trackeval_data(gt, run: Run) -> dict:
    """TrackEval's per-sequence input, built in memory over the labelled frames."""
    outs = dict(run)
    frames = gt.frames
    gid_of: Dict[str, int] = {}
    tid_of: Dict[str, int] = {}
    gt_ids, tr_ids, sims = [], [], []
    for f in frames:
        if f not in outs:
            raise ValueError(f"labelled frame {f} was not processed by the run")
        players = gt.players.get(f, {})
        g = sorted(players)
        kept = _kept(gt, f, outs[f])
        gt_ids.append(np.array([gid_of.setdefault(p, len(gid_of)) for p in g], int))
        tr_ids.append(np.array([tid_of.setdefault(o.track_id, len(tid_of)) for o in kept], int))
        G = np.array([players[p] for p in g], float).reshape(-1, 4)
        T = np.array([o.box for o in kept], float).reshape(-1, 4)
        sims.append(iou_matrix(G, T) if len(G) and len(T) else np.zeros((len(G), len(T))))
    return {"num_timesteps": len(frames), "num_gt_ids": len(gid_of), "num_tracker_ids": len(tid_of),
            "num_gt_dets": int(sum(len(x) for x in gt_ids)), "num_tracker_dets": int(sum(len(x) for x in tr_ids)),
            "gt_ids": gt_ids, "tracker_ids": tr_ids, "similarity_scores": sims}


def _metrics():
    from importlib.metadata import version
    if version("trackeval") != TRACKEVAL_VERSION:
        raise ValueError(f"trackeval {version('trackeval')} installed, bench pins {TRACKEVAL_VERSION}")
    from trackeval.metrics import CLEAR, HOTA, Identity
    return HOTA(), Identity({"PRINT_CONFIG": False}), CLEAR({"PRINT_CONFIG": False})


def trackeval_sequence(data: dict) -> dict:
    return {type(m).__name__: m.eval_sequence(data) for m in _metrics()}


def trackeval_combine(per_seq: Sequence[dict]) -> dict:
    ms = _metrics()
    return {type(m).__name__: m.combine_sequences({str(k): r[type(m).__name__] for k, r in enumerate(per_seq)})
            for m in ms}


def trackeval_row(res: dict) -> dict:
    h, i, c = res["HOTA"], res["Identity"], res["CLEAR"]
    return {"hota": round(float(np.mean(h["HOTA"])), 4), "deta": round(float(np.mean(h["DetA"])), 4),
            "assa": round(float(np.mean(h["AssA"])), 4), "idf1": round(float(i["IDF1"]), 4),
            "idsw": int(c["IDSW"]), "mota": round(float(c["MOTA"]), 4)}


def target_stats(gt, run: Run, frame_step: int, fps: float, rule_name: str = NESTED_V2.name) -> TargetStats:
    rule = RULES[rule_name]
    outs = dict(run)
    # per labelled frame: player -> (track, certain) and track -> attributed player, unambiguous only
    who: Dict[int, Dict[str, Tuple[str, bool]]] = {}
    shown: Dict[int, Dict[str, Tuple[Optional[str], bool]]] = {}
    for f in gt.frames:
        players = gt.players.get(f, {})
        kept = _kept(gt, f, outs[f])
        assign, _, ambiguous = _match(sorted(players), players, [o.box for o in kept], rule,
                                      frozenset(gt.occluded.get(f, ())))
        who[f] = {p: (kept[b].track_id, kept[b].identity_certain) for b, p in assign.items() if b not in ambiguous}
        shown[f] = {o.track_id: (assign.get(b) if b not in ambiguous else None, o.identity_certain)
                    for b, o in enumerate(kept)}
    st = TargetStats()
    for p in sorted({p for f in gt.frames for p in gt.players.get(f, {})}):
        fp = [f for f in gt.frames if p in gt.players.get(f, {})]
        st.players += 1
        seq = [who[f].get(p, (None, True))[0] for f in fp]
        runs = 0
        prev = None
        for tid in seq:
            if tid is not None and tid != prev:
                runs += 1
            prev = tid
        st.fragments += max(0, runs - 1)
        matched = [t for t in seq if t is not None]
        st.id_switches += sum(1 for a, b in zip(matched, matched[1:]) if a != b)
        tap = next((k for k, t in enumerate(seq) if t is not None), None)
        if tap is None:
            st.untapped += 1
            continue
        track, tap_frame = seq[tap], fp[tap]
        after = [f for f in fp if f >= tap_frame]
        cov = [who[f].get(p, (None, True))[0] == track for f in after]
        st.frames += len(after)
        st.covered += sum(cov)
        best = cur = 0
        for c in cov:
            cur = cur + 1 if c else 0
            best = max(best, cur)
        st.longest_s.append(best * frame_step / fps)
        for f in gt.frames:
            if f < tap_frame or track not in shown[f]:
                continue
            q, certain = shown[f][track]
            if q is not None and q != p:
                st.wrong += 1
                st.wrong_unflagged += int(certain)
    return st
