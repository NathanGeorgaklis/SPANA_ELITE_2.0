"""DEV resolution check (tools/dev/resolution_check.py): episodes from tracks rows, right / wrong / unknown."""
import unittest

from tools.dev.resolution_check import episodes, judge, majority


def doc(rows):
    return {"frames": [{"frame": f, "t": f / 10.0, "tracks": [{"track_id": "T1", "identity": i, "resolved_by": rb}]}
                       for f, (i, rb) in enumerate(rows)]}


class Check(unittest.TestCase):
    def test_episode_and_verdicts(self):
        rows = [("CERTAIN", None)] * 5 + [("IDENTITY_UNCERTAIN", None)] * 3 + [("CERTAIN", "REID")] * 5
        eps = episodes(doc(rows))
        self.assertEqual(len(eps), 1)
        ep = eps[0]
        self.assertEqual((ep["start_t"], ep["clear_t"], ep["resolved_by"]), (0.5, 0.8, "REID"))
        t_of = {f: f / 10.0 for f in range(13)}
        right = {f: "7" for f in range(13)}
        self.assertEqual(judge(right, ep, t_of, 0.5)["verdict"], "right")
        swapped = {**{f: "7" for f in range(6)}, **{f: "9" for f in range(8, 13)}}
        self.assertEqual(judge(swapped, ep, t_of, 0.5)["verdict"], "wrong")
        self.assertEqual(judge({f: "7" for f in range(6)}, ep, t_of, 0.5)["verdict"], "unknown")

    def test_majority_ties_are_unknown_and_2_1_0_is_appearance(self):
        self.assertIsNone(majority({0: "a", 1: "b"}, 0, 1, {0: 0.0, 1: 1.0}))
        rows = [("CERTAIN", None), ("IDENTITY_UNCERTAIN", None), ("CERTAIN", None)]
        d = doc(rows)
        for fr in d["frames"]:
            fr["tracks"][0].pop("resolved_by")
        self.assertEqual(episodes(d)[0]["resolved_by"], "APPEARANCE")


if __name__ == "__main__":
    unittest.main()
