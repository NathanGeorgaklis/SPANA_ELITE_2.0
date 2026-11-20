"""Pinned third-party tracker code, fetched outside Git (bench/external/ is ignored).

    python -m bench.external fetch            # clone each pinned repo at its commit
    python -m bench.external check            # verify the checkouts

Code without a commercial licence (Deep-EIoU) is never vendored into this repository; it is only read from here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import Candidate, load_bench


def checkout_dir(external_dir: Path, cand: Candidate) -> Path:
    return Path(external_dir) / cand.name


def head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def verified_path(external_dir: Path, cand: Candidate) -> Path:
    """The pinned checkout (plus its subdir) or an error; adapters import third-party code only from here."""
    d = checkout_dir(external_dir, cand)
    if not (d / ".git").exists():
        raise SystemExit(f"{cand.name}: no checkout at {d}; run `python -m bench.external fetch`")
    got, want = head(d), cand.implementation["commit"]
    if got != want:
        raise SystemExit(f"{cand.name}: checkout at {got}, config pins {want}")
    return d / cand.implementation.get("subdir", "")


def fetch(external_dir: Path, cand: Candidate) -> Path:
    d = checkout_dir(external_dir, cand)
    if not (d / ".git").exists():
        d.mkdir(parents=True, exist_ok=True)
        run = lambda *a: subprocess.run(["git", "-C", str(d), *a], check=True)   # noqa: E731
        run("init", "-q")
        run("remote", "add", "origin", cand.implementation["repo"])
        run("fetch", "-q", "--depth", "1", "origin", cand.implementation["commit"])
        run("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD")
    return verified_path(external_dir, cand)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fetch", "check"])
    p.add_argument("--bench", default=None)
    a = p.parse_args(argv)
    bench = load_bench(a.bench) if a.bench else load_bench()
    for c in bench.candidates:
        if "repo" in c.implementation:
            path = (fetch if a.cmd == "fetch" else verified_path)(bench.external_dir, c)
            print(f"{c.name}: {path} @ {c.implementation['commit']} (licence {c.licence['spdx']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
