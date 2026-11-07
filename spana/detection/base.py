"""Player-detector interface. Backends return SPANA Detection records only; no vendor objects leak out."""
from __future__ import annotations

from typing import List, Protocol

import numpy as np

from ..records import Detection


class PlayerDetector(Protocol):
    @property
    def model_id(self) -> dict:
        """Pinned identity of the model (name, version, sha256, licence...)."""

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Person detections for one BGR frame, in source pixels, after NMS and score/size filtering."""
