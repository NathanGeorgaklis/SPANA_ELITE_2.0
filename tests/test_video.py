"""Video input timing on real container timestamps (regression for the fresh DEV clips' trailing PTS gap)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spana.video import VideoSource


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not available")
class ContainerTimes(unittest.TestCase):
    def _make(self, path: Path, drop_frame: int | None) -> None:
        vf = "format=yuv420p"
        if drop_frame is not None:
            vf = f"select='not(eq(n\\,{drop_frame}))',{vf}"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=25:duration=1",
                        "-vf", vf, "-fps_mode", "passthrough", "-c:v", "libx264", "-y", str(path)], check=True)

    def test_constant_rate_matches_index_over_fps(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cfr.mp4"
            self._make(p, None)
            times = [t for _, t, _ in VideoSource(p).frames()]
            np.testing.assert_allclose(times, np.arange(len(times)) / 25.0, atol=1e-3)

    def test_dropped_frame_is_a_longer_step_not_a_wrong_time(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gap.mp4"
            self._make(p, 10)
            times = [t for _, t, _ in VideoSource(p).frames()]
            steps = np.diff(times)
            self.assertAlmostEqual(float(steps.max()), 2 / 25.0, delta=1e-3)   # the gap is kept
            self.assertAlmostEqual(times[-1], 24 / 25.0, delta=1e-3)            # later times stay true
            self.assertTrue(np.all(steps > 0))


if __name__ == "__main__":
    unittest.main()
