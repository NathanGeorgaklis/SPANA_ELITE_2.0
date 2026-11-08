"""Re-ID tracklet labelling workflow: match manifest, tracklet proposals, pack, freeze, split guard, labeller page."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from spana.artifacts import tracklet_rows, tracks_frame, validate_tracks
from spana.config import load_config
from spana.records import CameraObservation, CameraStatus, Detection
from spana.reid_labels import ReidLabelError, load_frozen, manifest_template, split_guard, validate_manifest
from spana.tracking.tracker import MultiPlayerTracker
from spana.video import VideoSource, sha256_file
from tests.test_cli_e2e import write_video
from tools.freeze_reid_labels import freeze
from tools.reid_pack import make_pack, propose

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "default.yaml")


def tracks_doc(video: Path, gap_frames=()):
    """Two players walking apart, one briefly unobserved (gap_frames) -> a valid spana.tracks artifact."""
    src = VideoSource(str(video))
    trk = MultiPlayerTracker(CFG, src.meta.width, src.meta.height)
    frames = []
    for i, t, _ in src.frames():
        dets = [Detection((40.0 + i, 40.0, 70.0 + i, 120.0), 0.9, 0)]
        if i not in gap_frames:
            dets.append(Detection((200.0 - i, 40.0, 230.0 - i, 120.0), 0.9, 0))
        cam = CameraObservation.identity(CameraStatus.FIRST if i == 0 else CameraStatus.OK, inliers=100)
        frames.append(tracks_frame(trk.update(i, t, dets, cam), CFG.output))
    doc = {"schema": "spana.tracks", "schema_version": "2.1.0", "provenance": {}, "video": src.meta.to_dict(),
           "detections_sha256": "0" * 64, "frames": frames, "tracklets": tracklet_rows(trk.summaries())}
    validate_tracks(doc)
    return doc


class Workflow(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.video = self.td / "match.avi"
        write_video(self.video, 40, 25.0)
        self.tracks = self.td / "match.trk.json"
        self.tracks.write_text(json.dumps(tracks_doc(self.video, gap_frames=range(20, 35))))
        self.manifest = self.td / "match.json"
        m = manifest_template("M_TEST_001", "TRAIN", sha256_file(str(self.video)))
        m.update(footage_licence="SPANA-owned test footage", rights_holder="SPANA", consent_basis="synthetic",
                 training_permitted=True, agreement_reference="synthetic test",
                 permissions={k: True for k in m["permissions"]})
        self.manifest.write_text(json.dumps(m))
        self.repo = self.td / "repo"

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _pack(self, name="pack", **kw):
        return make_pack(str(self.video), str(self.tracks), str(self.manifest), str(self.td / name), **kw)

    def _export(self, pack, players=None, parts=None, **kw):
        tl = pack["tracklets"]
        players = players or {"A07": {"side": "A", "shirt_number": "7"}, "B04": {"side": "B", "shirt_number": None}}
        if parts is None:
            parts = []
            for t, p in zip(tl, ["A07", "B04", "B04"]):
                parts.append({"tracklet": t["id"], "first_frame": t["first_frame"], "last_frame": t["last_frame"],
                              "decision": "player", "player": p})
        doc = {"schema": "spana.reid_label_export", "schema_version": "1.0.0", "match_id": "M_TEST_001",
               "split": "TRAIN", "tracklets_sha256": pack["tracklets_sha256"], "annotator": "QA", "label_version": 1,
               "reason": "", "exported_utc": "2026-11-08T00:00:00Z", "players": players, "parts": parts, **kw}
        path = self.td / f"export_{len(list(self.td.glob('export_*')))}.json"
        path.write_text(json.dumps(doc))
        return path

    def test_manifest_requires_split_rights_and_consent(self):
        m = json.loads(self.manifest.read_text())
        for k, v in (("split", "TEST"), ("footage_licence", " "), ("consent_basis", ""), ("training_permitted", False),
                     ("agreement_reference", "")):
            bad = {**m, k: v}
            with self.assertRaises(ReidLabelError, msg=k):
                validate_manifest(bad)
        for perm in ("copy_and_process", "ml_development", "commercial_use", "player_consent", "minors_covered",
                     "retain_derived_data"):
            with self.assertRaises(ReidLabelError, msg=perm):
                validate_manifest({**m, "permissions": {**m["permissions"], perm: False}})
        validate_manifest({**m, "permissions": {**m["permissions"], "model_survives_footage_removal": False}})
        validate_manifest({**m, "split": "EVAL", "training_permitted": False,
                           "permissions": {k: False for k in m["permissions"]}})   # EVAL may be labelled

    def test_proposals_cut_at_gaps(self):
        doc = json.loads(self.tracks.read_text())
        tl = propose(doc, 25.0, max_gap_s=0.5, min_observed=3)
        self.assertEqual(len(tl), 3)                                     # one player + the other split by the gap
        for t in tl:
            self.assertEqual(len(t["frames"]), len(t["boxes"]))
            self.assertTrue(all((b - a) / 25.0 <= 0.5 for a, b in zip(t["frames"], t["frames"][1:])))
        # a gap inside one track (no new track) is cut too
        doc2 = json.loads(json.dumps(doc))
        for fr in doc2["frames"]:
            if 10 <= fr["frame"] < 30:
                for t in fr["tracks"]:
                    t["source"] = "PREDICTED"
        cut = propose(doc2, 25.0, max_gap_s=0.5, min_observed=3)
        self.assertTrue(any(t["after_gap"] for t in cut))
        # entering IDENTITY_UNCERTAIN (a possible swap point) starts a new tracklet
        doc3 = json.loads(json.dumps(doc))
        first = doc3["frames"][0]["tracks"][0]["track_id"]
        for fr in doc3["frames"]:
            for t in fr["tracks"]:
                if t["track_id"] == first and fr["frame"] >= 15:
                    t["uncertain_episodes"], t["identity"] = 1, "IDENTITY_UNCERTAIN"
        cut = [t for t in propose(doc3, 25.0, 0.5, 3) if t["track"] == first]
        self.assertEqual([(t["first_frame"], t["starts_uncertain"]) for t in cut], [(0, False), (15, True)])

    def test_pack_contents_and_refusals(self):
        pack = self._pack()
        d = self.td / "pack"
        for f in ("index.html", "pack.js", "pack.json", "tracklets.json"):
            self.assertTrue((d / f).is_file(), f)
        for t in pack["tracklets"]:
            for th in t["thumbs"]:
                self.assertTrue((d / th["file"]).is_file())
        with self.assertRaises(ReidLabelError):
            self._pack()                                                   # never overwrite
        m = json.loads(self.manifest.read_text())
        self.manifest.write_text(json.dumps({**m, "rights_holder": ""}))
        with self.assertRaises(ReidLabelError):
            self._pack("pack2")

    def test_freeze_happy_path_and_hashes(self):
        pack = self._pack()
        dest = freeze(str(self.td / "pack"), str(self._export(pack)), str(self.video), root=self.repo)
        self.assertEqual(dest, self.repo / "data/reid/TRAIN/M_TEST_001/v1")
        f = json.loads((dest / "FROZEN.json").read_text())
        self.assertEqual(f["counts"]["players"], 2)
        rows = [l.split(",") for l in (dest / "boxes.mot.txt").read_text().split()]
        self.assertTrue(rows and all(len(r) == 9 for r in rows))       # MOT 1.1 boxes from the tracker
        lab = load_frozen(dest, "train")
        self.assertEqual(len(lab["identities"]), 3)
        with self.assertRaises(ReidLabelError):
            freeze(str(self.td / "pack"), str(self._export(pack)), root=self.repo)       # never overwrite
        (dest / "players.json").write_text("{}")
        with self.assertRaises(ReidLabelError):
            load_frozen(dest, "train")                                     # edited after freezing

    def test_split_and_conflict_rules(self):
        pack = self._pack()
        t0 = json.loads((self.td / "pack" / "tracklets.json").read_text())[0]
        mid = t0["frames"][len(t0["frames"]) // 2]
        before = t0["frames"][t0["frames"].index(mid) - 1]
        parts = [{"tracklet": t0["id"], "first_frame": t0["first_frame"], "last_frame": before, "decision": "player",
                  "player": "A07"},
                 {"tracklet": t0["id"], "first_frame": mid, "last_frame": t0["last_frame"], "decision": "reject",
                  "player": None}]
        rest = [{"tracklet": t["id"], "first_frame": t["first_frame"], "last_frame": t["last_frame"],
                 "decision": "player", "player": "B04"} for t in pack["tracklets"][1:]]
        dest = freeze(str(self.td / "pack"), str(self._export(pack, parts=parts + rest)), root=self.repo)
        self.assertEqual(json.loads((dest / "FROZEN.json").read_text())["counts"]["rejected_parts"], 1)
        # one player on two tracklets at the same time is refused
        clash = [{**p, "player": "A07"} for p in rest]
        with self.assertRaises(ReidLabelError):
            freeze(str(self.td / "pack"), str(self._export(pack, parts=parts + clash, label_version=2,
                                                            reason="test")), root=self.repo)
        # gaps between parts, undecided tracklets, unknown players are refused
        for bad in (parts[1:] + rest, rest, [{**parts[0], "player": "X99"}, parts[1]] + rest):
            with self.assertRaises(ReidLabelError):
                freeze(str(self.td / "pack"), str(self._export(pack, parts=bad, label_version=3, reason="t")),
                       root=self.repo)

    def test_new_version_needs_reason(self):
        pack = self._pack()
        with self.assertRaises(ReidLabelError):
            freeze(str(self.td / "pack"), str(self._export(pack, label_version=2)), root=self.repo)


class SplitGuard(unittest.TestCase):
    def test_eval_never_for_training_or_tuning(self):
        split_guard("TRAIN", "train")
        split_guard("DEV", "tune")
        for split, purpose in (("EVAL", "train"), ("EVAL", "tune"), ("DEV", "train"), ("TRAIN", "tune"),
                               ("TRAIN", "eval")):
            with self.assertRaises(ReidLabelError, msg=(split, purpose)):
                split_guard(split, purpose, "approved")
        with self.assertRaises(ReidLabelError):
            split_guard("EVAL", "eval")                                    # needs a recorded approval
        split_guard("EVAL", "eval", "Nathan 2026-11-08")


def _playwright():
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError:
        return None


@unittest.skipIf(_playwright() is None or shutil.which("google-chrome") is None, "Playwright/Chrome not available")
class LabellerPage(unittest.TestCase):
    def test_label_split_and_export_session(self):
        w = Workflow("test_freeze_happy_path_and_hashes")
        w.setUp()
        try:
            pack = w._pack()
            with _playwright()() as p:
                b = p.chromium.launch(executable_path=shutil.which("google-chrome"), headless=True)
                page = b.new_page(accept_downloads=True)
                page.goto((w.td / "pack" / "index.html").as_uri())
                page.fill("#annotator", "QA")
                for key, side in (("A07", "A"), ("B04", "B")):
                    page.fill("#pkey", key)
                    page.select_option("#pside", side)
                    page.click("#padd")
                cards = page.locator(".card")
                self.assertEqual(cards.count(), 3)
                cards.nth(0).locator(".cut").first.click()               # split the first tracklet
                self.assertEqual(cards.nth(0).locator(".part").count(), 2)
                page.locator(".player", has_text="A07").click()
                cards.nth(0).locator(".part").nth(0).locator("button", has_text="Assign").click()
                cards.nth(0).locator(".part").nth(1).locator("button", has_text="Reject").click()
                page.locator(".player", has_text="B04").click()
                for k in (1, 2):
                    cards.nth(k).locator("button", has_text="Assign").click()
                with page.expect_download() as dl:
                    page.click("#export")
                out = w.td / "export_browser.json"
                dl.value.save_as(out)
                b.close()
            dest = freeze(str(w.td / "pack"), str(out), root=w.repo)
            f = json.loads((dest / "FROZEN.json").read_text())
            self.assertEqual((f["counts"]["player_parts"], f["counts"]["rejected_parts"]), (3, 1))
        finally:
            w.tearDown()


if __name__ == "__main__":
    unittest.main()
