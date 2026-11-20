"""Strict loaders for config/tracking/*.yaml (unknown or missing keys are refused, as in spana.config)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

REPO = Path(__file__).resolve().parents[1]
TRACKING_CONFIG_DIR = REPO / "config" / "tracking"

_CAND_KEYS = {"schema", "schema_version", "candidate", "name", "kind", "implementation", "licence", "needs_image",
              "needs_embeddings", "params"}
_CAND_OPTIONAL = {"applies_to"}
_BENCH_KEYS = {"schema", "schema_version", "clips", "candidates", "embedder", "scoring", "external_dir", "output_csv",
               "ranking"}


class BenchConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    path: Path
    candidate: str
    name: str
    kind: str                      # online | offline
    implementation: dict
    licence: dict
    needs_image: bool
    needs_embeddings: bool
    params: dict
    applies_to: tuple = ()

    @property
    def commercial_ok(self) -> bool:
        return bool(self.licence["commercial_ok"])


@dataclass(frozen=True)
class Clip:
    name: str
    labels: Path


@dataclass(frozen=True)
class Bench:
    path: Path
    clips: List[Clip]
    candidates: List[Candidate]
    embedder: dict
    scoring: dict
    external_dir: Path
    output_csv: Path
    ranking: tuple


def _keys(doc: dict, required: set, optional: set, where: Path) -> None:
    missing, unknown = required - set(doc), set(doc) - required - optional
    if missing or unknown:
        raise BenchConfigError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def load_candidate(path: Path) -> Candidate:
    doc = yaml.safe_load(Path(path).read_text())
    _keys(doc, _CAND_KEYS, _CAND_OPTIONAL, path)
    if doc["schema"] != "spana.tracker_candidate" or doc["schema_version"] != "1.0.0":
        raise BenchConfigError(f"{path}: not a spana.tracker_candidate 1.0.0")
    if doc["kind"] not in ("online", "offline"):
        raise BenchConfigError(f"{path}: kind must be online or offline")
    if (doc["kind"] == "offline") != bool(doc.get("applies_to")):
        raise BenchConfigError(f"{path}: an offline candidate (and only one) names the online ones it applies to")
    if not isinstance(doc["licence"].get("commercial_ok"), bool) or not doc["licence"].get("spdx"):
        raise BenchConfigError(f"{path}: licence needs spdx and a boolean commercial_ok")
    return Candidate(Path(path), str(doc["candidate"]), doc["name"], doc["kind"], doc["implementation"],
                     doc["licence"], bool(doc["needs_image"]), bool(doc["needs_embeddings"]), dict(doc["params"]),
                     tuple(doc.get("applies_to") or ()))


def load_bench(path: Path = TRACKING_CONFIG_DIR / "bench_v1.yaml") -> Bench:
    path = Path(path)
    doc = yaml.safe_load(path.read_text())
    _keys(doc, _BENCH_KEYS, set(), path)
    if doc["schema"] != "spana.tracker_bench" or doc["schema_version"] != "1.0.0":
        raise BenchConfigError(f"{path}: not a spana.tracker_bench 1.0.0")
    cands = [load_candidate(path.parent / c) for c in doc["candidates"]]
    names = [c.name for c in cands]
    if len(set(names)) != len(names):
        raise BenchConfigError(f"{path}: candidate names must be unique")
    for c in cands:
        if c.kind == "offline" and not set(c.applies_to) <= set(names):
            raise BenchConfigError(f"{c.path}: applies_to names unknown candidates")
    clips = [Clip(c["name"], REPO / c["labels"]) for c in doc["clips"]]
    ranking = []
    for r in doc["ranking"]:
        key, order = r.split()
        if order not in ("asc", "desc"):
            raise BenchConfigError(f"{path}: ranking order must be asc or desc")
        ranking.append((key, order))
    return Bench(path, clips, cands, dict(doc["embedder"]), dict(doc["scoring"]), REPO / doc["external_dir"],
                 REPO / doc["output_csv"], tuple(ranking))
