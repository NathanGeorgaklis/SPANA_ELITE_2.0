"""HTTP 429 handling shared by the Roboflow clients: wait (Retry-After if given, else base x attempt) and retry."""
from __future__ import annotations

import sys
import time
from typing import Callable


def retry_after_s(response, attempt: int, base_wait_s: float, max_wait_s: float = 300.0) -> float:
    """Seconds to wait before re-sending: the server's Retry-After (capped at max_wait_s), else base x attempt.
    Re-sending before Retry-After has passed is refused again, so max_wait_s must cover the server's window."""
    try:
        v = float(response.headers.get("Retry-After"))
        if v >= 0:
            return min(v, max_wait_s)
    except (TypeError, ValueError, AttributeError):
        pass
    return base_wait_s * attempt


def call_with_429_retry(send: Callable, retries: int, base_wait_s: float, sleep=time.sleep,
                        max_wait_s: float = 300.0):
    """`send()` -> response. Re-sends while the status is 429, at most `retries` times; returns the last response."""
    r = send()
    for attempt in range(1, retries + 1):
        if getattr(r, "status_code", None) != 429:
            return r
        wait = retry_after_s(r, attempt, base_wait_s, max_wait_s)
        print(f"HTTP 429: waiting {wait:.0f}s (retry {attempt}/{retries}, Retry-After="
              f"{getattr(r, 'headers', {}).get('Retry-After')!r})", file=sys.stderr, flush=True)
        sleep(wait)
        r = send()
    return r
