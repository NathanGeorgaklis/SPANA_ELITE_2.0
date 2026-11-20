"""Per-stage wall-clock timing of the pipeline passes (config `profiling`, off by default).

Measurement only: timing never changes what a stage computes. When `profiling.enabled` is false every call is a
no-op (a shared null context), so the default pipeline pays nothing. When it is true, `detect` / `track` / `target`
write `<out>.profile.json` (schema `spana.profile` 1.0.0) next to their artifact: per stage total seconds, calls,
ms per processed frame and share of the pass total, plus whatever the caller records about devices and batch sizes.

GPU work is asynchronous: a stage that runs a model must end with a host copy of its outputs (the runners do), so the
time lands in the stage that launched it.
"""
from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, Iterator, Optional

PROFILE_SCHEMA = ("spana.profile", "1.0.0")
_NULL = nullcontext()


class StageTimer:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._tot: Dict[str, float] = {}
        self._calls: Dict[str, int] = {}
        self._order: list = []
        self.frames = 0
        self.info: Dict[str, Any] = {}
        self._t0 = time.perf_counter()

    def add(self, name: str, seconds: float) -> None:
        if name not in self._tot:
            self._tot[name], self._calls[name] = 0.0, 0
            self._order.append(name)
        self._tot[name] += seconds
        self._calls[name] += 1

    def stage(self, name: str):
        return self._stage(name) if self.enabled else _NULL

    @contextmanager
    def _stage(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t)

    def timed_iter(self, it: Iterable, name: str, first_name: Optional[str] = None) -> Iterator:
        """Yield from `it`, timing each step under `name`. The first step can be timed apart (`first_name`): a
        sequential decoder reads every frame before the window's start there."""
        if not self.enabled:
            yield from it
            return
        it = iter(it)
        k = 0
        while True:
            t = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                self.add(name, time.perf_counter() - t)
                return
            self.add(first_name if (k == 0 and first_name) else name, time.perf_counter() - t)
            k += 1
            yield item

    def report(self, pass_name: str, realtime_fps: Optional[float] = None, **extra: Any) -> dict:
        wall = time.perf_counter() - self._t0
        timed = sum(self._tot.values())
        n = max(1, self.frames)
        stages = [{"stage": s, "seconds": round(self._tot[s], 4), "calls": self._calls[s],
                   "ms_per_frame": round(1000.0 * self._tot[s] / n, 3),
                   "share": round(self._tot[s] / timed, 4) if timed > 0 else 0.0} for s in self._order]
        stages.sort(key=lambda r: -r["seconds"])
        out = {"schema": PROFILE_SCHEMA[0], "schema_version": PROFILE_SCHEMA[1], "pass": pass_name,
               "frames": self.frames, "wall_s": round(wall, 4), "timed_s": round(timed, 4),
               "ms_per_frame": round(1000.0 * wall / n, 3), "stages": stages, "info": self.info, **extra}
        if realtime_fps:
            out["realtime_factor"] = round((wall / n) * realtime_fps, 4)   # processing time / footage time
        return out
