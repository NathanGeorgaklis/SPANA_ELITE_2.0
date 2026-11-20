"""Transfer flagging in spana.evaluation (tools/benchmark.py): a transfer is flagged only if its track was
IDENTITY_UNCERTAIN within the configured tolerance of the transfer frame; recoveries and links never flag it.
Recovery- and link-caused transfers are reported in their own columns. Synthetic, generic ids."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spana.config import ConfigError
from spana.evaluation import IOU_050, EvalFrame, TrackBox, TransferFlags, evaluate
from tools.benchmark import DEFAULT_BENCH_CONFIG, load_bench_config

FPS = 10.0
P, Q = [100.0, 100.0, 140.0, 200.0], [600.0, 100.0, 640.0, 200.0]


def scenario(switch: int, n: int = 12, uncertain=(), recovery_at=None, link_at=None, tol=0.5):
    """One track T1 on player P until `switch`, on player Q from then on. Returns the report."""
    frames, timeline = [], []
    for f in range(n):
        box = P if f < switch else Q
        frames.append(EvalFrame(f, {"P": P, "Q": Q}, [P, Q], [TrackBox("T1", box, True, 0, f in uncertain)], []))
        rec = 1 if (recovery_at is not None and f >= recovery_at) or (link_at is not None and f >= link_at) else 0
        timeline.append((f, f / FPS, [("T1", f in uncertain, rec)]))
    flags = TransferFlags(timeline, {"T1": [link_at]} if link_at is not None else {}, tol)
    return evaluate(frames, rule=IOU_050, flags=flags)


class TransferFlaggingTest(unittest.TestCase):
    def test_a_inside_an_uncertain_episode_is_flagged(self):
        r = scenario(5, uncertain={4, 5, 6})
        self.assertEqual((r.identity_transfers, r.unflagged_transfers), (1, 0))
        self.assertTrue(r.transfer_events[0]["flagged"])

    def test_b_recovered_track_outside_any_episode_is_silent(self):
        r = scenario(5, recovery_at=5)
        self.assertEqual((r.identity_transfers, r.unflagged_transfers), (1, 1))
        self.assertEqual((r.recovery_caused_transfers, r.recovery_caused_silent), (1, 1))
        self.assertEqual((r.link_caused_transfers, r.transfers_on_linked_tracks), (0, 0))
        e = r.transfer_events[0]
        self.assertEqual((e["flagged"], e["recovery_caused"], e["link_caused"], e["previous_frame"]),
                         (False, True, False, 4))

    def test_c_linked_track_outside_any_episode_is_silent(self):
        r = scenario(5, link_at=5)
        self.assertEqual((r.identity_transfers, r.unflagged_transfers), (1, 1))
        self.assertEqual((r.link_caused_transfers, r.link_caused_silent, r.transfers_on_linked_tracks), (1, 1, 1))
        self.assertEqual(r.recovery_caused_transfers, 0)      # the link's own recoveries step is not a recovery

    def test_c_recovery_before_the_previous_attribution_is_not_the_cause(self):
        r = scenario(5, recovery_at=2)
        self.assertEqual(r.recovery_caused_transfers, 0)
        self.assertEqual(r.unflagged_transfers, 1)

    def test_d_edge_of_the_tolerance_window(self):
        tol = 0.5
        edge = scenario(5, uncertain={0}, tol=tol)            # uncertain exactly tol before the transfer frame
        self.assertEqual(edge.unflagged_transfers, 0)
        past = scenario(6, uncertain={0}, tol=tol)            # one frame further: outside
        self.assertEqual(past.unflagged_transfers, 1)
        after = scenario(5, uncertain={10}, tol=tol)          # uncertain exactly tol after the transfer frame
        self.assertEqual(after.unflagged_transfers, 0)
        self.assertEqual(scenario(5, uncertain={11}, tol=tol).unflagged_transfers, 1)
        self.assertEqual(scenario(5, uncertain={4}, tol=0.0).unflagged_transfers, 1)   # zero tolerance: frame only
        self.assertEqual(scenario(5, uncertain={5}, tol=0.0).unflagged_transfers, 0)

    def test_without_flags_only_the_frame_itself_counts(self):
        frames = [EvalFrame(f, {"P": P, "Q": Q}, [P, Q], [TrackBox("T1", P if f < 3 else Q, True, 0, f == 2)], [])
                  for f in range(5)]
        r = evaluate(frames, rule=IOU_050)
        self.assertEqual((r.identity_transfers, r.unflagged_transfers), (1, 1))
        self.assertIsNone(r.recovery_caused_transfers)


class BenchConfigTest(unittest.TestCase):
    def test_config_loads_and_fails_closed(self):
        self.assertGreaterEqual(load_bench_config(DEFAULT_BENCH_CONFIG).transfer_flag_tolerance_s, 0)
        td = Path(tempfile.mkdtemp())
        for text in ('schema_version: "1.0.0"\n',
                     'schema_version: "0.9.0"\ntransfer_flag_tolerance_s: 0.5\n',
                     'schema_version: "1.0.0"\ntransfer_flag_tolerance_s: -1\n',
                     'schema_version: "1.0.0"\ntransfer_flag_tolerance_s: 0.5\nextra: 1\n'):
            (td / "b.yaml").write_text(text)
            with self.assertRaises(ConfigError, msg=text):
                load_bench_config(td / "b.yaml")


if __name__ == "__main__":
    unittest.main()
