"""Run provenance: everything needed to reproduce an output artifact."""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional, Sequence

_PACKAGES = ("numpy", "opencv-python", "opencv-python-headless", "scipy", "onnxruntime", "PyYAML", "roboflow")


def git_state(repo_dir: Path) -> Dict[str, Optional[str]]:
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else str(bool(status)).lower()}


def package_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_provenance(*, config_sha256: str, inputs: Dict[str, str], argv: Sequence[str]) -> dict:
    repo = Path(__file__).resolve().parent.parent
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(repo),
        "config_sha256": config_sha256,
        "inputs": dict(inputs),
        "argv": list(argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }
