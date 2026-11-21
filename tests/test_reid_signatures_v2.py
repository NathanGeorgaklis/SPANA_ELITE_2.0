"""Option 2, better re-ID signatures (records/reid_signatures_v2_2026-11-19).

spana.track_embeddings 1.1.0 keeps per track end a crop set [{frame, quality, emb}] chosen by looser rules
(reid.set_*), each crop weighted by its quality. target.reentry_signature "set" compares two tracks by
quality-weighted best-pair matching of their crop sets instead of one averaged signature; reentry_wait_samples
judges a returning track only from its crops seen so far, once it has that many; reentry_withdraw drops an offer the
evidence so far no longer supports (spana.target 2.6.0, dropped_reason WITHDRAWN)."""
from __future__ import annotations

import copy
import dataclasses
import unittest

import numpy as np

from spana.artifacts import ArtifactError, sha256_doc, validate_target, validate_track_embeddings
from spana.config import ConfigError
from spana.identity.follower import follow, target_document
from spana.identity.reid import crop_quality, set_similarity, signature
from tests.target_versions import older
from tests.test_reentry_offer import ME, ME_AGAIN, SEL, SOMEONE, cfg, returning
from tests.test_target_follow import CFG


def crops(*rows):
    """rows: (frame, vector, quality)."""
    return [{"frame": f, "quality": q, "emb": list(v)} for f, v, q in rows]


def emb11(tracks: dict, sets: dict) -> dict:
    """sets: track -> (first_crops, last_crops); the 1.0.0 averaged signatures are derived from them."""
    doc = {"schema": "spana.track_embeddings", "schema_version": "1.1.0", "provenance": {}, "video": tracks["video"],
           "tracks_sha256": sha256_doc(tracks), "model": {"name": "test", "version": "t", "sha256": "0" * 64,
                                                          "embedding_dim": 3},
           "reid_config": {}, "tracks": {}}
    for tid, (first, last) in sets.items():
        row = {}
        for end, cs in (("first", first), ("last", last)):
            row[end] = signature(np.array([c["emb"] for c in cs])) if cs else None
            row[f"{end}_frames"] = [c["frame"] for c in cs]
            row[f"{end}_crops"] = cs
        doc["tracks"][tid] = row
    validate_track_embeddings(doc, tracks)
    return doc


def scfg(**kw):
    base = dict(reentry_signature="set", reentry_wait_samples=1, reentry_withdraw=True, reentry_min_margin=0.0,
                reentry_min_similarity=0.7, reentry_position_weight=0.0)
    return cfg(**{**base, **kw})


LOST = crops((3, ME, 1.0))


class Similarity(unittest.TestCase):
    def test_quality_weight(self):
        rc = CFG.reid
        self.assertEqual(crop_quality(rc.min_crop_height_px, 0.0, rc), 1.0)
        self.assertAlmostEqual(crop_quality(rc.min_crop_height_px / 2, 0.0, rc), 0.5, places=3)
        self.assertAlmostEqual(crop_quality(rc.min_crop_height_px, rc.set_max_iou, rc), 0.1, places=3)

    def test_one_occluded_crop_does_not_sink_the_match(self):
        """The averaged signature is dragged off by one bad crop; best-pair matching is not."""
        mine = crops((1, ME_AGAIN, 1.0), (2, ME_AGAIN, 1.0), (3, SOMEONE, 0.2))
        mean = float(np.dot(signature(np.array([ME])), signature(np.array([c["emb"] for c in mine]))))
        self.assertGreater(set_similarity(LOST, mine), mean)

    def test_low_quality_crops_count_less(self):
        hi = crops((1, ME_AGAIN, 1.0), (2, SOMEONE, 0.1))
        lo = crops((1, ME_AGAIN, 0.1), (2, SOMEONE, 1.0))
        self.assertGreater(set_similarity(LOST, hi), set_similarity(LOST, lo))

    def test_empty_set_is_no_evidence(self):
        self.assertIsNone(set_similarity(LOST, []))


class WaitAndWithdraw(unittest.TestCase):
    def setUp(self):
        self.tr = returning()               # t1 lost after frame 3; t2 born at frame 8, observed 9..13

    def offers(self, sets, **kw):
        emb = emb11(self.tr, sets)
        frames, info = follow(self.tr, SEL, scfg(**kw), (), emb)
        return frames, info, emb

    def test_waits_for_enough_crops_seen_so_far(self):
        sets = {"t1": ([], LOST), "t2": (crops((9, ME_AGAIN, 1.0), (10, ME_AGAIN, 1.0), (11, ME_AGAIN, 1.0)), [])}
        first = {w: self.offers(sets, reentry_wait_samples=w)[1]["relinks"][0]["proposed_frame"] for w in (1, 3)}
        self.assertEqual(first, {1: 9, 3: 11})
        self.assertFalse(self.offers(sets, reentry_wait_samples=4)[1]["relinks"])      # never enough: never offered

    def test_offer_is_withdrawn_when_later_crops_disagree(self):
        sets = {"t1": ([], LOST), "t2": (crops((9, ME_AGAIN, 1.0), (10, SOMEONE, 1.0), (11, SOMEONE, 1.0),
                                               (12, SOMEONE, 1.0)), [])}
        frames, info, emb = self.offers(sets)
        r = info["relinks"][0]
        self.assertEqual((r["proposed_frame"], r["outcome"], r["dropped_reason"]), (9, "DROPPED", "WITHDRAWN"))
        cand = [f["frame"] for f in frames if f["state"] == "CANDIDATE"]
        self.assertEqual(cand, [9, 10])                          # withdrawn on frame 11 (similarity 0.67 < 0.70)
        self.assertIn("BELOW_MIN_SIMILARITY", [x["reason"] for x in info["reentry_log"]])
        doc = target_document(self.tr, SEL, scfg(), ["test"], (), emb)
        validate_target(doc, self.tr, emb)
        self.assertEqual(doc["schema_version"], "2.9.0")
        _, info2, _ = self.offers(sets, reentry_withdraw=False)   # policy off: the offer stays
        self.assertEqual(info2["relinks"][0]["outcome"], "PENDING")

    def test_mean_mode_ignores_crop_sets(self):
        sets = {"t1": ([], LOST), "t2": (crops((9, ME_AGAIN, 1.0)), [])}
        _, info, _ = self.offers(sets, reentry_signature="mean", reentry_withdraw=False)
        self.assertAlmostEqual(info["relinks"][0]["evidence"]["similarity"], 0.95, places=2)


class Validation(unittest.TestCase):
    def setUp(self):
        self.tr = returning()
        self.sets = {"t1": ([], LOST), "t2": (crops((9, ME_AGAIN, 1.0), (10, SOMEONE, 1.0), (11, SOMEONE, 1.0)), [])}
        self.emb = emb11(self.tr, self.sets)
        self.doc = target_document(self.tr, SEL, scfg(), ["test"], (), self.emb)

    def _refused(self, doc, emb=None):
        with self.assertRaises(ArtifactError):
            validate_target(doc, self.tr, emb or self.emb)

    def test_tampered_set_similarity_is_refused(self):
        d = copy.deepcopy(self.doc)
        d["relinks"][0]["evidence"]["similarity"] += 0.05
        self._refused(d)

    def test_withdrawn_needs_2_6_0(self):
        self.assertEqual(self.doc["relinks"][0]["dropped_reason"], "WITHDRAWN")
        self._refused(older(self.doc, "2.5.0"))

    def test_set_mode_needs_crop_sets(self):
        e = copy.deepcopy(self.emb)
        e["schema_version"] = "1.0.0"
        for row in e["tracks"].values():
            row.pop("first_crops"), row.pop("last_crops")
        d = copy.deepcopy(self.doc)
        d["embeddings_sha256"] = sha256_doc(e)
        self._refused(d, e)

    def test_crop_on_an_unconfirmed_frame_is_refused(self):
        e = copy.deepcopy(self.emb)
        e["tracks"]["t2"]["first_crops"].insert(0, {"frame": 8, "quality": 1.0, "emb": list(ME)})   # t2 fresh at 8
        with self.assertRaises(ArtifactError):
            validate_track_embeddings(e, self.tr)

    def test_bad_quality_is_refused(self):
        e = copy.deepcopy(self.emb)
        e["tracks"]["t2"]["first_crops"][0]["quality"] = 0.0
        with self.assertRaises(ArtifactError):
            validate_track_embeddings(e, self.tr)


class Config(unittest.TestCase):
    def test_set_rules_must_be_looser(self):
        with self.assertRaises(ConfigError):
            dataclasses.replace(CFG.reid, set_min_height_px=CFG.reid.min_crop_height_px + 1).validate()
        with self.assertRaises(ConfigError):
            dataclasses.replace(CFG.target, reentry_signature="best").validate()


if __name__ == "__main__":
    unittest.main()
