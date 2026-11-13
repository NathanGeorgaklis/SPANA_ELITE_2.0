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


class Detector(Protocol):
    """Generic multi-class detector: boxes, class and score only. No football (or any domain) logic.

    `Detection.class_id` indexes `class_names` (the configured class list)."""

    @property
    def model_id(self) -> dict:
        """Identity of the model (name / id, version, source...)."""

    @property
    def class_names(self) -> tuple:
        """Class names; Detection.class_id is an index into this tuple."""

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detections for one BGR frame, in source pixels."""
