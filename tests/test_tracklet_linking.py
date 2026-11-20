"""Offline global tracklet linking v1 (spana/tracking/linking.py) on synthetic tracks documents.

Each scenario is a hand-built, validated spana.tracks 2.3.0 document with generic ids; appearance signatures are given
directly (no video), except in the CLI plumbing test. Covers gap linking, overlap / gate rejection, ambiguity
abstention, uncertain-episode and rival blocking, and confidence inheritance (incl. the follower's view).
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import yaml

from spana import cli
from spana.artifacts import read_json, validate_tracks, write_json_atomic
from spana.config import ConfigError, load_config
from spana.identity.follower import Selection, follow
from spana.tracking.linking import (AMBIGUOUS_PREDECESSORS, AMBIGUOUS_SUCCESSORS, APPEARANCE_BELOW_MIN,
                                    APPEARANCE_UNKNOWN, IDENTITY_RIVAL, NOT_MUTUAL_BEST, UNCERTAIN_EPISODE,
                                    UNCONFIRMED, UNREPRESENTABLE_RIVALRY, link_document, linking_config_from_dict, load_linking_config,
                                    rename_decisions)
from spana.appearance import similarity as hist_similarity
from spana.video import VideoSource

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")
LCFG_PATH = ROOT / "config" / "tracking" / "linking.yaml"
LCFG = load_linking_config(LCFG_PATH)
LCFG_CLEAR = dataclasses.replace(LCFG, allow_uncertain_clearance=True)   # the clearance path (off in config)
FPS = 10.0
H = 100.0                      # body height in px of every synthetic player

# per-frame spec of a track: (state "A" observed / "L" lost, foot x, foot y, rivals or None)
Row = Tuple[str, float, float, Optional[List[str]]]


def tracks_doc(tracks: Dict[str, Dict[int, Row]], n_frames: int, unconfirmed: Sequence[str] = (),
               video: Optional[dict] = None) -> dict:
    """A valid spana.tracks 2.3.0 document. A track's rows must be contiguous; it ends (LOST_EXPIRED) on the frame
    after its last row unless that is past the clip. Tracks in `unconfirmed` never confirm."""
    frames, st = [], {}
    for f in range(n_frames):
        rows, det = [], 0
        for tid in sorted(tracks):
            spec = tracks[tid].get(f)
            if spec is None:
                continue
            state, x, y, rivals = spec
            s = st.setdefault(tid, {"first": f, "obs": 0, "miss": 0, "eps": 0, "unc": False, "last_obs_t": None})
            observed = state == "A"
            s["obs"] += observed
            s["miss"] = 0 if observed else s["miss"] + 1
            if observed:
                s["last_obs_t"] = f / FPS
            unc = bool(rivals)
            if unc and not s["unc"]:
                s["eps"] += 1
            s["unc"] = unc
            rows.append({
                "track_id": tid, "state": "ACTIVE" if observed else "LOST",
                "confirmed": tid not in unconfirmed and f > s["first"],
                "source": "OBSERVED" if observed else "PREDICTED", "box": [x - 20.0, y - H, x + 20.0, y],
                "detection_index": det if observed else None, "detection_score": 0.9 if observed else None,
                "confidence": 0.9, "contested": False, "age_frames": f - s["first"], "missed_frames": s["miss"],
                "time_since_observed_s": 0 if observed else round(f / FPS - s["last_obs_t"], 6),
                "observed_frames": s["obs"], "epoch": 0, "occluder": None, "max_overlap_iou": 0.0,
                "overlap_with": None, "recoveries": 0, "last_gap_s": 0, "possible_predecessors": [],
                "identity": "IDENTITY_UNCERTAIN" if unc else "CERTAIN", "identity_rivals": sorted(rivals or []),
                "uncertain_episodes": s["eps"], "identity_doubt": False, "resolved_by": None})
            det += observed
        ended = [[tid, "LOST_EXPIRED"] for tid in sorted(tracks) if max(tracks[tid]) == f - 1]
        frames.append({"frame": f, "t": f / FPS,
                       "camera": {"status": "FIRST" if f == 0 else "OK",
                                  "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "inliers": 0, "hist_distance": None},
                       "tracks": rows, "ended": ended})
    tracklets = []
    for tid in sorted(tracks):
        fs = sorted(tracks[tid])
        obs = [f for f in fs if tracks[tid][f][0] == "A"]
        tracklets.append({"track_id": tid, "epoch": 0, "first_frame": fs[0], "last_frame": fs[-1],
                          "last_observed_frame": obs[-1], "observed_frames": len(obs),
                          "confirmed": tid not in unconfirmed and len(fs) > 1,
                          "end_reason": None if fs[-1] == n_frames - 1 else "LOST_EXPIRED", "recoveries": 0,
                          "possible_predecessors": []})
    doc = {"schema": "spana.tracks", "schema_version": "2.3.0", "provenance": {},
           "video": video or {"sha256": "v" * 64}, "detections_sha256": "d" * 64, "frames": frames,
           "tracklets": tracklets}
    validate_tracks(doc)
    return doc


def run(frames: Dict[int, Row], start: int, stop: int, x0: float, vx: float, y: float = 500.0) -> Dict[int, Row]:
    """Observed rows from start to stop-1, moving vx px per frame."""
    for f in range(start, stop):
        frames[f] = ("A", x0 + vx * (f - start), y, None)
    return frames


def lost(frames: Dict[int, Row], start: int, stop: int) -> Dict[int, Row]:
    last = frames[start - 1]
    for f in range(start, stop):
        frames[f] = ("L", last[1], last[2], None)
    return frames


def vec(k: int, mix: float = 0.0) -> np.ndarray:
    """A normalised 8-bin histogram concentrated on bin k, with `mix` of its mass spread on bin k+1."""
    v = np.zeros(8)
    v[k] = 1.0 - mix
    v[(k + 1) % 8] += mix
    return v


def sigs(**ends: Tuple[Optional[np.ndarray], Optional[np.ndarray]]) -> dict:
    return {tid: {"head": h, "tail": t} for tid, (h, t) in ends.items()}


def link(doc: dict, sig: dict, cfg=LCFG_CLEAR):
    out, log, head_of = link_document(doc, sig, "test", hist_similarity, cfg,
                                      confidence_window_s=CFG.tracking.confidence_window_s,
                                      score_decimals=CFG.output.score_decimals, box_decimals=CFG.output.box_decimals,
                                      provenance={})
    validate_tracks(out)
    return out, log, head_of


def entry(log: List[dict], a: str) -> dict:
    return next(e for e in log if e["from_track"] == a)


def ids_on(doc: dict, f: int) -> Set[str]:
    return {t["track_id"] for t in doc["frames"][f]["tracks"]}


def row(doc: dict, f: int, tid: str) -> dict:
    return next(t for t in doc["frames"][f]["tracks"] if t["track_id"] == tid)


SAME = vec(0)


class GapLinkingTest(unittest.TestCase):
    def test_gap_after_end_is_linked_and_written_as_a_recovery(self):
        a = run({}, 0, 10, 100.0, 5.0)
        lost(a, 10, 12)                                       # A lost 10-11, ended on frame 12
        b = run({}, 14, 25, 100.0 + 5.0 * 14, 5.0)            # B where A would be, 0.5 s after A's last observation
        doc = tracks_doc({"A": a, "B": b}, 25)
        out, log, head_of = link(doc, sigs(A=(SAME, SAME), B=(SAME, SAME)))
        e = entry(log, "A")
        self.assertEqual(e["kind"], "link", e)
        self.assertEqual(e["reasons"], [])
        self.assertEqual(head_of, {"B": "A"})
        self.assertEqual([t["track_id"] for t in out["tracklets"]], ["A"])
        for f in range(25):
            self.assertEqual(ids_on(out, f), {"A"}, f)            # one track throughout, gap held
        for f in (12, 13):
            r = row(out, f, "A")
            self.assertEqual((r["state"], r["source"]), ("LOST", "HELD"))
        r = row(out, 14, "A")
        self.assertEqual((r["source"], r["recoveries"], r["last_gap_s"]), ("OBSERVED", 1, 0.5))
        self.assertEqual(r["observed_frames"], 11)
        self.assertEqual(r["age_frames"], 14)
        self.assertEqual(out["tracklets"][0]["last_frame"], 24)
        self.assertIsNone(out["tracklets"][0]["end_reason"])
        self.assertFalse(any(e2 for fr in out["frames"] for e2 in fr["ended"]))
        for k in ("gap", "motion", "appearance", "team"):
            self.assertIn(k, e["terms"])
        self.assertIsNone(e["terms"]["team"])

    def test_successor_born_while_predecessor_still_lost(self):
        a = run({}, 0, 10, 100.0, 5.0)
        lost(a, 10, 18)                                       # A LOST until 17; B born at 13
        b = run({}, 13, 25, 100.0 + 5.0 * 13, 5.0)
        doc = tracks_doc({"A": a, "B": b}, 25)
        out, log, _ = link(doc, sigs(A=(SAME, SAME), B=(SAME, SAME)))
        self.assertEqual(entry(log, "A")["kind"], "link")
        self.assertEqual(ids_on(out, 13), {"A"})
        self.assertEqual(row(out, 13, "A")["source"], "OBSERVED")  # A's lost rows from B's birth are replaced

    def test_chain_of_three(self):
        a = run({}, 0, 8, 100.0, 5.0)
        b = run({}, 10, 18, 150.0, 5.0)
        c = run({}, 20, 30, 250.0, 5.0)
        doc = tracks_doc({"A": a, "B": b, "C": c}, 30)
        out, log, head_of = link(doc, sigs(A=(SAME, SAME), B=(SAME, SAME), C=(SAME, SAME)))
        self.assertEqual(head_of, {"B": "A", "C": "A"})
        self.assertEqual(row(out, 25, "A")["recoveries"], 2)
        self.assertEqual(len(out["tracklets"]), 1)

    def test_decisions_renamed(self):
        rows = [{"kind": "recovery", "track_id": "B", "frame": 3, "rivals": ["B", "X"]}]
        self.assertEqual(rename_decisions(rows, {"B": "A"}),
                         [{"kind": "recovery", "track_id": "A", "frame": 3, "rivals": ["A", "X"]}])


class RejectionTest(unittest.TestCase):
    def test_time_overlap_is_never_a_candidate(self):
        a = run({}, 0, 15, 100.0, 5.0)
        b = run({}, 10, 25, 150.0, 5.0)                       # B observed while A still observed
        doc = tracks_doc({"A": a, "B": b}, 25)
        out, log, head_of = link(doc, sigs(A=(SAME, SAME), B=(SAME, SAME)))
        self.assertEqual((log, head_of), ([], {}))
        self.assertEqual(len(out["tracklets"]), 2)

    def test_gap_over_max_is_not_a_candidate(self):
        a = run({}, 0, 10, 100.0, 0.0)
        gap_frames = int(round((LCFG.max_gap_s + 0.3) * FPS))
        b = run({}, 9 + gap_frames, 9 + gap_frames + 5, 100.0, 0.0)
        doc = tracks_doc({"A": a, "B": b}, 9 + gap_frames + 5)
        _, log, head_of = link(doc, sigs(A=(SAME, SAME), B=(SAME, SAME)))
        self.assertEqual((log, head_of), ([], {}))

    def test_displacement_over_max_speed_is_not_a_candidate(self):
        a = run({}, 0, 10, 100.0, 0.0)
        gap = 0.5
        reach_px = (LCFG.gate_base_h + LCFG.max_speed_h_per_s * gap) * H
        b = run({}, 9 + int(gap * FPS), 20, 100.0 + reach_px * 1.2, 0.0)
        doc = tracks_doc({"A": a, "B": b}, 20)
        _, log, head_of = link(doc, sigs(A=(SAME, SAME), B=(SAME, SAME)))
        self.assertEqual((log, head_of), ([], {}))

    def test_unconfirmed_or_unknown_or_different_appearance_abstains(self):
        a = run({}, 0, 10, 100.0, 0.0)
        b = run({}, 12, 20, 100.0, 0.0)
        for kw, sig, reason in (({"unconfirmed": ["B"]}, sigs(A=(SAME, SAME), B=(SAME, SAME)), UNCONFIRMED),
                                ({}, sigs(A=(SAME, SAME), B=(None, None)), APPEARANCE_UNKNOWN),
                                ({}, sigs(A=(SAME, SAME), B=(vec(4), vec(4))), APPEARANCE_BELOW_MIN)):
            doc = tracks_doc({"A": dict(a), "B": dict(b)}, 20, **kw)
            out, log, head_of = link(doc, sig)
            self.assertEqual(entry(log, "A")["kind"], "abstain")
            self.assertIn(reason, entry(log, "A")["reasons"])
            self.assertEqual(head_of, {})
            self.assertEqual(len(out["tracklets"]), 2)


class AmbiguityTest(unittest.TestCase):
    def test_two_equally_good_successors_abstain(self):
        a = run({}, 0, 10, 500.0, 0.0)
        b1 = run({}, 12, 20, 520.0, 0.0)                      # mirror positions, same appearance
        b2 = run({}, 12, 20, 480.0, 0.0, y=500.0)
        doc = tracks_doc({"A": a, "B1": b1, "B2": b2}, 20)
        out, log, head_of = link(doc, sigs(A=(SAME, SAME), B1=(SAME, SAME), B2=(SAME, SAME)))
        e = entry(log, "A")
        self.assertEqual(e["kind"], "abstain")
        self.assertIn(AMBIGUOUS_SUCCESSORS, e["reasons"])
        self.assertLess(e["margin_successors"], LCFG.margin)
        self.assertEqual(head_of, {})
        self.assertEqual(len(out["tracklets"]), 3)

    def test_two_equally_good_predecessors_abstain(self):
        a1 = run({}, 0, 10, 480.0, 0.0)
        a2 = run({}, 0, 10, 520.0, 0.0)
        b = run({}, 12, 20, 500.0, 0.0)
        doc = tracks_doc({"A1": a1, "A2": a2, "B": b}, 20)
        _, log, head_of = link(doc, sigs(A1=(SAME, SAME), A2=(SAME, SAME), B=(SAME, SAME)))
        reasons = {r for e in log for r in e["reasons"]}
        self.assertTrue({AMBIGUOUS_PREDECESSORS, NOT_MUTUAL_BEST} & reasons, log)
        self.assertEqual(head_of, {})

    def test_clear_winner_by_appearance_is_linked(self):
        a = run({}, 0, 10, 500.0, 0.0)
        b1 = run({}, 12, 20, 520.0, 0.0)
        b2 = run({}, 12, 20, 480.0, 0.0)
        doc = tracks_doc({"A": a, "B1": b1, "B2": b2}, 20)
        _, log, head_of = link(doc, sigs(A=(SAME, SAME), B1=(SAME, SAME), B2=(vec(3), vec(3))))
        self.assertEqual(head_of, {"B1": "A"})
        self.assertGreaterEqual(entry(log, "A")["margin_successors"], LCFG.margin)


class FailClosedIdentityTest(unittest.TestCase):
    def _uncertain_end(self):
        a = run({}, 0, 10, 100.0, 0.0)
        a[9] = ("A", 100.0, 500.0, ["X"])                      # A IDENTITY_UNCERTAIN at its last observation
        x = run({}, 0, 20, 900.0, 0.0)                        # the rival, far away and alive throughout
        b = run({}, 12, 20, 100.0, 0.0)
        return {"A": a, "X": x, "B": b}

    def test_uncertain_episode_blocks_unless_strict_appearance(self):
        # similarity between min and strict: Bhattacharyya of vec(0) and vec(0, mix) = sqrt(1 - mix)
        s = 0.5 * (LCFG.min_similarity + LCFG.strict_min_similarity)
        mid = vec(0, mix=1 - s ** 2)
        doc = tracks_doc(self._uncertain_end(), 20)
        sig = sigs(A=(SAME, SAME), X=(vec(5), vec(5)), B=(mid, mid))
        _, log, head_of = link(doc, sig)
        self.assertIn(UNCERTAIN_EPISODE, entry(log, "A")["reasons"])
        self.assertEqual(head_of, {})
        strong = vec(0, mix=1 - ((1 + LCFG.strict_min_similarity) / 2) ** 2)
        _, log, head_of = link(doc, sigs(A=(SAME, SAME), X=(vec(5), vec(5)), B=(strong, strong)))
        self.assertEqual(head_of, {"B": "A"})
        self.assertTrue(entry(log, "A")["cleared_by_appearance"])

    def _rival_doc(self, rivals_of_b: List[str]) -> dict:
        """A observed 0-9 then LOST to 14; B born at 12 where A was; on frame 13 B is uncertain naming `rivals_of_b`
        (A among them: the tracker could not tell B from the lost A). X is far away throughout."""
        a = lost(run({}, 0, 10, 100.0, 0.0), 10, 15)
        x = run({}, 0, 20, 900.0, 0.0)
        b = run({}, 12, 20, 100.0, 0.0)
        b[13] = ("A", 100.0, 500.0, rivals_of_b)
        if "X" in rivals_of_b:
            x[13] = ("A", 900.0, 500.0, ["B"])
        return tracks_doc({"A": a, "X": x, "B": b}, 20)

    def test_rival_blocks_unless_strict_appearance(self):
        s = 0.5 * (LCFG.min_similarity + LCFG.strict_min_similarity)
        mid = vec(0, mix=1 - s ** 2)
        strong = vec(0, mix=1 - ((1 + LCFG.strict_min_similarity) / 2) ** 2)
        doc = self._rival_doc(["A", "X"])
        _, log, head_of = link(doc, sigs(A=(SAME, SAME), X=(vec(5), vec(5)), B=(mid, mid)))
        self.assertIn(IDENTITY_RIVAL, entry(log, "A")["reasons"])
        self.assertEqual(head_of, {})
        out, log, head_of = link(doc, sigs(A=(SAME, SAME), X=(vec(5), vec(5)), B=(strong, strong)))
        self.assertEqual(head_of, {"B": "A"})
        self.assertTrue(entry(log, "A")["cleared_by_appearance"])
        self.assertEqual(row(out, 13, "A")["identity_rivals"], ["X"])     # still uncertain, against X only

    def test_rivalry_with_the_partner_alone_is_never_linked(self):
        strong = vec(0, mix=1 - ((1 + LCFG.strict_min_similarity) / 2) ** 2)
        doc = self._rival_doc(["A"])     # merged, that row would be uncertain with no rival: cannot be written
        _, log, head_of = link(doc, sigs(A=(SAME, SAME), X=(vec(5), vec(5)), B=(strong, strong)))
        self.assertEqual(head_of, {})
        self.assertEqual(entry(log, "A")["reasons"], [UNREPRESENTABLE_RIVALRY])


    def test_config_flag_off_blocks_the_strict_clearance(self):
        self.assertFalse(LCFG.allow_uncertain_clearance)              # shipped setting
        strong = vec(0, mix=1 - ((1 + LCFG.strict_min_similarity) / 2) ** 2)
        doc = tracks_doc(self._uncertain_end(), 20)
        _, log, head_of = link(doc, sigs(A=(SAME, SAME), X=(vec(5), vec(5)), B=(strong, strong)), cfg=LCFG)
        self.assertEqual(head_of, {})
        self.assertIn(UNCERTAIN_EPISODE, entry(log, "A")["reasons"])
        self.assertFalse(entry(log, "A")["cleared_by_appearance"])
        doc = self._rival_doc(["A", "X"])
        _, log, head_of = link(doc, sigs(A=(SAME, SAME), X=(vec(5), vec(5)), B=(strong, strong)), cfg=LCFG)
        self.assertEqual(head_of, {})
        self.assertIn(IDENTITY_RIVAL, entry(log, "A")["reasons"])
        # a clean link is unaffected by the flag
        doc = tracks_doc({"A": run({}, 0, 10, 100.0, 0.0), "B": run({}, 12, 20, 100.0, 0.0)}, 20)
        _, _, head_of = link(doc, sigs(A=(SAME, SAME), B=(SAME, SAME)), cfg=LCFG)
        self.assertEqual(head_of, {"B": "A"})


class ConfidenceInheritanceTest(unittest.TestCase):
    def _doc(self):
        a = run({}, 0, 10, 100.0, 0.0)
        x = run({}, 0, 20, 900.0, 0.0)
        a[3] = ("A", 100.0, 500.0, ["X"])                      # an episode early in A, resolved by frame 4
        x[3] = ("A", 900.0, 500.0, ["A"])
        b = run({}, 12, 20, 100.0, 0.0)
        return tracks_doc({"A": a, "X": x, "B": b}, 20)

    def test_linked_rows_inherit_the_lower_identity_confidence(self):
        doc = self._doc()
        out, _, head_of = link(doc, sigs(A=(SAME, SAME), X=(vec(5), vec(5)), B=(SAME, SAME)))
        self.assertEqual(head_of, {"B": "A"})
        for f in range(12, 20):
            merged, own = row(out, f, "A"), row(doc, f, "B")
            self.assertEqual(merged["uncertain_episodes"], own["uncertain_episodes"] + 1)   # A's episode carried
            self.assertEqual(merged["identity"], own["identity"])
            self.assertLessEqual(merged["confidence"], own["confidence"])
        self.assertLess(row(out, 12, "A")["confidence"], row(doc, 12, "B")["confidence"])  # gap zeros count

    def test_follower_tainted_before_the_link_stays_tainted_after(self):
        doc = self._doc()
        out, _, _ = link(doc, sigs(A=(SAME, SAME), X=(vec(5), vec(5)), B=(SAME, SAME)))
        cfg = dataclasses.replace(CFG, target=dataclasses.replace(CFG.target, resume_after_uncertain=False))
        frames, *_ = follow(out, Selection.track_at_frame("A", 1), cfg)
        after = [fr for fr in frames if fr["frame"] >= 12]
        self.assertTrue(after)
        self.assertTrue(all(fr["state"] != "TARGET" for fr in after), after)
        # selected after the episode: the link carries the clean identity on
        frames, *_ = follow(out, Selection.track_at_frame("A", 6), cfg)
        self.assertTrue(all(fr["state"] == "TARGET" for fr in frames if fr["frame"] >= 13))


class ConfigTest(unittest.TestCase):
    def test_config_fails_closed(self):
        raw = yaml.safe_load(LCFG_PATH.read_text())
        bad = dict(raw)
        del bad["margin"]
        with self.assertRaises(ConfigError):
            linking_config_from_dict(bad)
        with self.assertRaises(ConfigError):
            linking_config_from_dict({**raw, "extra": 1})
        with self.assertRaises(ConfigError):
            linking_config_from_dict({**raw, "strict_min_similarity": raw["min_similarity"] - 0.1})
        with self.assertRaises(ConfigError):
            linking_config_from_dict({**raw, "schema_version": "1.0.0"})      # 1.0.0 lacks the flag
        with self.assertRaises(ConfigError):
            linking_config_from_dict({**raw, "allow_uncertain_clearance": "no"})


class CliTest(unittest.TestCase):
    def test_link_command_writes_valid_outputs(self):
        td = Path(tempfile.mkdtemp())
        video = td / "clip.avi"
        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (640, 360))
        for f in range(20):
            img = np.full((360, 640, 3), (40, 140, 40), np.uint8)
            if f < 10 or f >= 12:
                cv2.rectangle(img, (80, 220), (120, 320), (20, 20, 200), -1)     # one red player, briefly missing
            vw.write(img)
        vw.release()
        meta = VideoSource(video).meta.to_dict()
        doc = tracks_doc({"A": run({}, 0, 10, 100.0, 0.0, y=320.0), "B": run({}, 12, 20, 100.0, 0.0, y=320.0)},
                         20, video=meta)
        write_json_atomic(td / "in.trk.json", doc)
        (td / "in.trk.decisions.jsonl").write_text(json.dumps({"kind": "spawn", "track_id": "B", "frame": 12}) + "\n")
        out = td / "out.trk.json"
        self.assertEqual(cli.main(["link", "--video", str(video), "--config", str(ROOT / "config" / "default.yaml"),
                                   "--linking-config", str(LCFG_PATH), "--tracks", str(td / "in.trk.json"),
                                   "--decisions", str(td / "in.trk.decisions.jsonl"), "--out", str(out)]), 0)
        validate_tracks(read_json(out))
        links = [json.loads(l) for l in out.with_suffix(".links.jsonl").read_text().splitlines()]
        self.assertEqual(links[0]["kind"], "header")
        self.assertEqual([e["kind"] for e in links[1:]], ["link"])
        dec = [json.loads(l) for l in out.with_suffix(".decisions.jsonl").read_text().splitlines()]
        self.assertEqual(dec[0]["track_id"], "A")
        self.assertEqual(dec[-1]["kind"], "link")
        with self.assertRaises(Exception):                   # never overwrites
            cli.main(["link", "--video", str(video), "--config", str(ROOT / "config" / "default.yaml"),
                      "--linking-config", str(LCFG_PATH), "--tracks", str(td / "in.trk.json"),
                      "--decisions", str(td / "in.trk.decisions.jsonl"), "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
