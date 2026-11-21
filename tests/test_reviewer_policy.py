"""Simulated reviewer policy (decision 2026-11-22, config/reviewer_policy.yaml): a confirmation stretches over
labelled frames where its track is not observed, never over a two-player box; short stretches get no action."""
import unittest
from pathlib import Path

import tools.target_score as TS

REPO = Path(__file__).resolve().parents[1]


def doc(n, tid="T1", eps=1, missing=()):
    return {"frames": [{"frame": f, "tracks": [] if f in missing else [{"track_id": tid, "uncertain_episodes": eps}]}
                       for f in range(n)]}


GAP = {"end_frame": 100}


class StretchEnd(unittest.TestCase):
    def test_earlier_reviewer_stops_at_an_unobserved_labelled_frame(self):
        attr = {0: {"T1": ("7", False)}, 6: {}, 12: {"T1": ("7", False)}}
        self.assertEqual(TS.confirmation_end(doc(20), None, attr, "7", "T1", 0, GAP), 0)

    def test_stretch_passes_over_unobserved_frames(self):
        attr = {0: {"T1": ("7", False)}, 6: {}, 12: {"T1": ("7", False)}}
        self.assertEqual(TS.confirmation_end(doc(20), None, attr, "7", "T1", 0, GAP, True), 12)

    def test_never_over_a_two_player_box(self):
        attr = {0: {"T1": ("7", False)}, 6: {"T1": ("7", True)}, 12: {"T1": ("7", False)}}
        self.assertEqual(TS.confirmation_end(doc(20), None, attr, "7", "T1", 0, GAP, True), 0)

    def test_stops_at_another_player_or_the_player_elsewhere(self):
        other = {0: {"T1": ("7", False)}, 6: {"T1": ("8", False)}, 12: {"T1": ("7", False)}}
        self.assertEqual(TS.confirmation_end(doc(20), None, other, "7", "T1", 0, GAP, True), 0)
        moved = {0: {"T1": ("7", False)}, 6: {"T2": ("7", False)}, 12: {"T1": ("7", False)}}
        self.assertEqual(TS.confirmation_end(doc(20), None, moved, "7", "T1", 0, GAP, True), 0)

    def test_stops_at_track_end_and_gap_end(self):
        attr = {0: {"T1": ("7", False)}, 6: {}, 12: {"T1": ("7", False)}}
        self.assertEqual(TS.confirmation_end(doc(20, missing=(9,)), None, attr, "7", "T1", 0, GAP, True), 0)
        self.assertEqual(TS.confirmation_end(doc(20), None, attr, "7", "T1", 0, {"end_frame": 10}, True), 0)


class PolicyConfig(unittest.TestCase):
    def test_repo_policy_is_the_decision(self):
        p = TS.load_reviewer_policy(REPO / "config/reviewer_policy.yaml")
        self.assertEqual((p.stretch_over_unobserved, p.min_stretch_labelled_frames), (True, 2))

    def test_bad_policy_refused(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write('schema_version: "1.0.0"\nstretch_over_unobserved: true\nmin_stretch_labelled_frames: 0\n')
        with self.assertRaises(SystemExit):
            TS.load_reviewer_policy(f.name)


if __name__ == "__main__":
    unittest.main()
