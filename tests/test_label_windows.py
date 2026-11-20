"""DEV labelling windows (tools/dev/label_windows.py) and range label packs (tools/label_pack.py 1.1.0): events are read
from track logs generically, windows are scored with caps and chosen deterministically without overlapping labelled
footage, wrong answers get dense windows, and a range pack keeps the source video's hash and frame numbering."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_cli_e2e import write_video
from tools.dev import label_windows as LW
from tools.freeze_labels import freeze
from tools.label_pack import make_pack
from spana.labels import LabelError

CFG = LW.load_cfg("config/label_windows.yaml")


def trk(track_id, state="ACTIVE", source="OBSERVED", box=(100.0, 100.0, 140.0, 200.0)):
    return {"track_id": track_id, "state": state, "source": source, "box": list(box)}


def doc(frames, confirmed=("A", "B", "C"), w=1000, h=500):
    return {"video": {"width": w, "height": h, "fps": 10.0},
            "tracklets": [{"track_id": t, "confirmed": t in confirmed} for t in ("A", "B", "C", "U")],
            "frames": [{"frame": f, "tracks": ts} for f, ts in frames]}


class Events(unittest.TestCase):
    def test_each_event_type_from_the_logs(self):
        frames = [(0, [trk("A"), trk("B", box=(0.0, 100.0, 30.0, 200.0)), trk("U", box=(0.0, 0.0, 9.0, 9.0))]),
                  (1, [trk("A", state="OCCLUDED", source="PREDICTED"), trk("B")]),
                  (2, [trk("A", state="OCCLUDED", source="PREDICTED"), trk("B")]),     # still the same occlusion
                  (3, [trk("A"), trk("B")])]
        dec = [{"frame": 3, "kind": "end", "track_id": "B", "reason": "LEFT_FRAME"},
               {"frame": 3, "kind": "end", "track_id": "A", "reason": "LOST_EXPIRED"},
               {"frame": 2, "kind": "identity_uncertain", "track_id": "A", "rival": "B", "rival_similarity": 0.9},
               {"frame": 2, "kind": "identity_uncertain", "track_id": "B", "rival": "A", "rival_similarity": 0.2},
               {"frame": 2, "kind": "identity_uncertain", "track_id": "U", "rival": "A", "rival_similarity": 0.9},
               {"frame": 3, "kind": "identity_uncertain", "track_id": "A", "rival": "B", "rival_similarity": 0.9}]
        ev = LW.events(doc(frames), dec, CFG, min_similarity=0.7)
        self.assertEqual(ev["occlusion"], [1])
        self.assertEqual(ev["entry"], [0])               # B born at the left border; U unconfirmed: ignored
        self.assertEqual(ev["exit"], [3])                # LOST_EXPIRED is not leaving the picture
        self.assertEqual(ev["flagged"], [2, 2])          # A-B on frames 2 and 3 is one episode
        self.assertEqual(ev["similar_rival"], [2])       # only the rival that looked alike


class Choose(unittest.TestCase):
    def ev(self, **kw):
        return {t: sorted(kw.get(t, [])) for t in LW.TYPES}

    def test_score_caps_and_needs_several_types(self):
        cfg = dict(CFG, window_s=1.0, stride_s=1.0, cap=2, min_event_types=2)
        ev = self.ev(flagged=[1, 2, 3, 4, 5], occlusion=[5], exit=[25])
        c = LW.candidates("S", ev, 0, 39, 10.0, cfg, [])
        self.assertEqual([(x["start_frame"], x["score"]) for x in c], [(0, 3.0)])   # 2 (capped) + 1; 1 type -> out

    def test_exclusion_per_source_limit_and_determinism(self):
        cfg = dict(CFG, window_s=1.0, stride_s=1.0, count=3, max_per_source=1, min_event_types=1)
        evs = {"S1": self.ev(flagged=[1], occlusion=[2], exit=[3], entry=[15]), "S2": self.ev(flagged=[5]),
               "S3": self.ev(flagged=[5])}
        cands = []
        for n in ("S1", "S2", "S3"):
            cands += LW.candidates(n, evs[n], 0, 29, 10.0, cfg, [(0, 10)] if n == "S3" else [])
        a = LW.choose(cands, ["S1", "S2", "S3"], cfg)
        self.assertEqual([(x["source"], x["start_frame"]) for x in a], [("S1", 0), ("S2", 0)])  # S3's event excluded
        self.assertEqual(a, LW.choose(list(reversed(cands)), ["S1", "S2", "S3"], cfg))
        cfg2 = dict(cfg, max_per_source=2)
        b = LW.choose(cands, ["S1", "S2", "S3"], cfg2)
        self.assertEqual([(x["source"], x["start_frame"]) for x in b], [("S1", 0), ("S1", 10), ("S2", 0)])


class Dense(unittest.TestCase):
    def test_every_wrong_answer_gets_a_window(self):
        td = Path(tempfile.mkdtemp())
        try:
            lab = td / "labels"
            lab.mkdir()
            (lab / "FROZEN.json").write_text(json.dumps({"split": "DEV", "clip": {"fps": 30.0, "sha256": "ab" * 32}}))
            (lab / "labels_frozen.json").write_text(json.dumps({"frames_labelled": [0, 6, 300], "frame_step": 6}))
            rep = {"clips": [{"clip": "CLIP", "labels": {"dir": str(lab)}, "per_player": [      # target_score layout
                {"player": "9", "events": [{"kind": "wrong", "frame": 276, "attributed_to": "10", "track_id": "T45"},
                                           {"kind": "correct", "frame": 270}]},
                {"player": "4", "events": [{"kind": "wrong", "frame": 10, "attributed_to": "8"}]}]}]}
            w = LW.dense_windows(rep, td, CFG)
            self.assertEqual([(x["wrong_frame"], x["start_frame"], x["end_frame"]) for x in w],
                             [(276, 246, 306), (10, 0, 41)])            # +-1 s at 30 fps, clipped at frame 0
            self.assertEqual(w[0]["step_frames"], 1)
            self.assertEqual(w[0]["name"], "CLIP_dense_f246_306")
            self.assertEqual([x["player"] for x in w], ["9", "4"])                     # the player ids, not positions
        finally:
            shutil.rmtree(td, ignore_errors=True)


class RangePack(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.video = self.td / "MATCH.avi"
        write_video(self.video, 40, 25.0)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_range_pack_keeps_source_numbering_and_hash(self):
        m = make_pack(str(self.video), "DEV", None, str(self.td / "p"), name="MATCH_f12_30", start_frame=12,
                      end_frame=30, step_frames=5)
        self.assertEqual(m["schema_version"], "1.1.0")
        self.assertEqual([f["index"] for f in m["frames"]], [12, 17, 22, 27])
        self.assertEqual(m["clip"]["name"], "MATCH_f12_30")
        self.assertEqual(m["clip"]["sha256"], hashlib.sha256(self.video.read_bytes()).hexdigest())
        self.assertEqual(m["source"], {"video": "MATCH.avi", "start_frame": 12, "end_frame": 30})
        whole = make_pack(str(self.video), "DEV", 0.2, str(self.td / "w"))          # 1.0.0 unchanged
        self.assertEqual((whole["schema_version"], whole["clip"]["name"]), ("1.0.0", "MATCH"))
        self.assertNotIn("source", whole)

    def test_range_pack_refusals(self):
        bad = [dict(name="X", start_frame=0, end_frame=None, step_frames=1),           # partial range
               dict(name=None, start_frame=0, end_frame=10, step_frames=1),
               dict(name="X", start_frame=10, end_frame=10, step_frames=1),            # empty
               dict(name="a/b", start_frame=0, end_frame=10, step_frames=1),           # not a folder name
               dict(name="X", start_frame=30, end_frame=60, step_frames=1)]            # past the end of the video
        for n, kw in enumerate(bad):
            with self.subTest(n=n), self.assertRaises(SystemExit):
                make_pack(str(self.video), "DEV", None, str(self.td / f"b{n}"), **kw)
        with self.assertRaises(SystemExit):
            make_pack(str(self.video), "EVAL", None, str(self.td / "e"), name="X", start_frame=0, end_frame=10,
                      step_frames=1)
        with self.assertRaises(SystemExit):                                            # both steps
            make_pack(str(self.video), "DEV", 0.2, str(self.td / "s"), name="X", start_frame=0, end_frame=10,
                      step_frames=1)

    def test_freeze_names_the_label_set_and_records_the_range(self):
        pack = self.td / "p"
        m = make_pack(str(self.video), "DEV", None, str(pack), name="MATCH_f10_20", start_frame=10, end_frame=20,
                      step_frames=5)
        gt = "11,1,10,10,20,40,1,1,1\n16,1,12,10,20,40,1,1,1\n"
        g, r = self.td / "gt.txt", self.td / "rec.json"
        g.write_text(gt)
        rec = {"schema": "spana.labels_frozen", "split": "DEV", "clip": m["clip"], "frame_step": 5, "label_version": 1,
               "annotator": "QA", "frozen_utc": "2026-11-20T00:00:00Z", "frames_labelled": [10, 15],
               "gt_sha256": hashlib.sha256(gt.encode()).hexdigest()}
        r.write_text(json.dumps(rec))
        dest = freeze(str(pack), str(g), str(r), str(self.video), root=self.td / "repo")
        self.assertEqual(dest.parent.name, "MATCH_f10_20")
        fz = json.loads((dest / "FROZEN.json").read_text())
        self.assertEqual(fz["source"], {"video": "MATCH.avi", "start_frame": 10, "end_frame": 20})
        rec["clip"] = dict(m["clip"], name="OTHER")                                    # a record of another pack
        r.write_text(json.dumps(rec))
        with self.assertRaises(LabelError):
            freeze(str(pack), str(g), str(r), None, root=self.td / "repo2")


def detections_doc(video: Path, frames, boxes):
    """A valid spana.detections document of `video` with the same boxes (box, score) on each listed frame."""
    from spana.artifacts import DETECTIONS_SCHEMA, detection_frame
    from spana.config import load_config
    from spana.records import Detection
    cfg = load_config("config/default.yaml")
    return {"schema": DETECTIONS_SCHEMA[0], "schema_version": DETECTIONS_SCHEMA[1], "provenance": {},
            "video": {"sha256": hashlib.sha256(video.read_bytes()).hexdigest()}, "model": {"name": "test-detector"},
            "class_names": ["person"], "detection_config": {}, "frame_stride": 1,
            "frames": [detection_frame(f, f / 25.0, [Detection(b, sc, 0) for b, sc in boxes], cfg.output)
                       for f in frames]}


class AssistedPack(unittest.TestCase):
    """Detector-assisted packs (1.2.0): suggestions per frame above the threshold, refused when the detections are
    of another video or do not cover the pack; the frozen labels carry `assisted` and must agree with the pack."""
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.video = self.td / "MATCH.avi"
        write_video(self.video, 40, 25.0)
        self.dets = self.td / "MATCH.det.json"
        self.dets.write_text(json.dumps(detections_doc(self.video, range(0, 40),
                                                       [((10.0, 10.0, 30.0, 60.0), 0.9), ((50.0, 10.0, 70.0, 60.0), 0.3)])))

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def pack(self, out, dets=None, **kw):
        args = dict(name="MATCH_f10_20", start_frame=10, end_frame=20, step_frames=5,
                    suggestions=str(dets or self.dets), suggest_min_score=0.5, cover_iou=0.5)
        args.update(kw)
        return make_pack(str(self.video), "DEV", None, str(self.td / out), **args)

    def test_suggestions_above_the_threshold_and_the_assisted_note(self):
        m = self.pack("a")
        self.assertEqual(m["schema_version"], "1.2.0")
        self.assertEqual([f["suggestions"] for f in m["frames"]], [[[10.0, 10.0, 30.0, 60.0, 0.9]]] * 2)
        self.assertEqual(m["assisted"]["detections_sha256"], hashlib.sha256(self.dets.read_bytes()).hexdigest())
        self.assertEqual((m["assisted"]["min_score"], m["assisted"]["cover_iou"]), (0.5, 0.5))
        plain = make_pack(str(self.video), "DEV", None, str(self.td / "p"), name="X", start_frame=10, end_frame=20,
                          step_frames=5)
        self.assertNotIn("assisted", plain)
        self.assertNotIn("suggestions", plain["frames"][0])

    def test_refusals(self):
        other = self.td / "OTHER.avi"
        write_video(other, 40, 25.0, seed=3)
        od = self.td / "other.det.json"
        od.write_text(json.dumps(detections_doc(other, range(40), [((1.0, 1.0, 9.0, 9.0), 0.9)])))
        short = self.td / "short.det.json"
        short.write_text(json.dumps(detections_doc(self.video, range(0, 12), [((1.0, 1.0, 9.0, 9.0), 0.9)])))
        for n, kw in enumerate([dict(dets=od), dict(dets=short), dict(suggest_min_score=None), dict(cover_iou=0.0)]):
            with self.subTest(n=n), self.assertRaises(SystemExit):
                self.pack(f"r{n}", **kw)
        with self.assertRaises(SystemExit):                                            # whole-clip packs: never
            make_pack(str(self.video), "DEV", 0.2, str(self.td / "w"), suggestions=str(self.dets),
                      suggest_min_score=0.5, cover_iou=0.5)

    def test_freeze_carries_assistance_and_refuses_a_mismatch(self):
        pack = self.td / "a"
        m = self.pack("a")
        gt = "11,1,10,10,20,50,1,1,1\n"
        g, r = self.td / "gt.txt", self.td / "rec.json"
        g.write_text(gt)
        rec = {"schema": "spana.labels_frozen", "split": "DEV", "clip": m["clip"], "frame_step": 5, "label_version": 1,
               "annotator": "QA", "frozen_utc": "2026-11-20T00:00:00Z", "frames_labelled": [10],
               "gt_sha256": hashlib.sha256(gt.encode()).hexdigest(), "assisted": m["assisted"]}
        r.write_text(json.dumps(rec))
        dest = freeze(str(pack), str(g), str(r), str(self.video), root=self.td / "repo")
        self.assertEqual(json.loads((dest / "FROZEN.json").read_text())["assisted"], m["assisted"])
        rec.pop("assisted")                                                            # export lost the note
        r.write_text(json.dumps(rec))
        with self.assertRaises(LabelError):
            freeze(str(pack), str(g), str(r), None, root=self.td / "repo2")


class Page(unittest.TestCase):
    def test_range_packs_keep_their_own_saved_work(self):
        html = Path("tools/labeller.html").read_text()
        self.assertIn('PACK.clip.sha256 + (PACK.source ? ":" + PACK.clip.name : "")', html)
        self.assertEqual(html.count("samePack(s)"), 2)                                # used on load and on import


if __name__ == "__main__":
    unittest.main()
