"""HTTP 429 handling shared by the Roboflow clients: wait (Retry-After if given, else base x attempt) and retry."""
from __future__ import annotations

import time
from typing import Callable


def retry_after_s(response, attempt: int, base_wait_s: float) -> float:
    try:
        v = float(response.headers.get("Retry-After"))
        if v >= 0:
            return min(v, 300.0)
    except (TypeError, ValueError, AttributeError):
        pass
    return base_wait_s * attempt


def call_with_429_retry(send: Callable, retries: int, base_wait_s: float, sleep=time.sleep):
    """`send()` -> response. Re-sends while the status is 429, at most `retries` times; returns the last response."""
    r = send()
    for attempt in range(1, retries + 1):
        if getattr(r, "status_code", None) != 429:
            return r
        sleep(retry_after_s(r, attempt, base_wait_s))
        r = send()
    return r
