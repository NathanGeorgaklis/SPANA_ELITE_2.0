"""tools/roboflow_ball_sheet.py: offline contact sheets of ball pre-label boxes."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools import roboflow_ball_sheet as bs


class BallSheetTests(unittest.TestCase):
    def test_one_cell_per_ball_box_and_pages_are_padded(self):
        with tempfile.TemporaryDirectory() as d:
            cv2.imwrite(str(Path(d) / "id1.jpg"), np.full((360, 640, 3), 90, np.uint8))
            doc = {"entries": [
                {"id": "id1", "name": "M_f0000010.jpg", "match_id": "M", "boxes": [
                    {"class": "ball", "box": [1.0, 1.0, 9.0, 9.0], "score": 0.6},          # at the frame corner
                    {"class": "ball", "box": [630.0, 350.0, 640.0, 360.0], "score": 0.7},
                    {"class": "player", "box": [100.0, 100.0, 130.0, 180.0], "score": 0.9}]},
                {"id": "missing", "name": "M_f0000020.jpg", "match_id": "M", "boxes": []}]}  # no ball: never read
            cells = list(bs.ball_cells(doc, "ball", Path(d)))
        self.assertEqual([m for m, _ in cells], ["M", "M"])
        self.assertTrue(all(c.shape == (bs.CELL_PX, bs.CELL_PX, 3) for _, c in cells))
        pages = list(bs.sheets([c for _, c in cells] * 3, cols=4, rows=1))
        self.assertEqual([p.shape for p in pages], [(bs.CELL_PX, 4 * bs.CELL_PX, 3)] * 2)
        self.assertEqual(int(pages[1][:, 2 * bs.CELL_PX:].max()), 0)                       # padding is black


if __name__ == "__main__":
    unittest.main()
