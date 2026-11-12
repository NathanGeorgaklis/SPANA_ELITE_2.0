"""Perceptual hash (64-bit DCT pHash) and deterministic near-duplicate grouping. numpy + OpenCV only."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def phash(image_bgr: np.ndarray) -> int:
    """Grey 32x32 (area resize) -> 2-D DCT -> top-left 8x8 without the DC term's influence -> bits above the median."""
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    med = np.median(low[1:])
    bits = low > med
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def phash_hex(h: int) -> str:
    return f"{h:016x}"


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_groups(items: Sequence[Tuple[str, Tuple, int]], max_distance: int) -> List[Dict]:
    """items: (id, sort_key, hash). Pairs within `max_distance` bits are linked; each connected group keeps its member
    with the smallest sort_key and lists the rest as `exclude`. Deterministic for any input order."""
    order = sorted(items, key=lambda x: (x[1], x[0]))
    parent = list(range(len(order)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            d = hamming(order[i][2], order[j][2])
            if d <= max_distance:
                pairs.append((order[i][0], order[j][0], d))
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: Dict[int, List[int]] = {}
    for i in range(len(order)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for root in sorted(groups):
        members = groups[root]
        if len(members) < 2:
            continue
        out.append({"keep": order[members[0]][0], "exclude": [order[m][0] for m in members[1:]],
                    "pairs": [p for p in pairs if p[0] in {order[m][0] for m in members}]})
    return out
