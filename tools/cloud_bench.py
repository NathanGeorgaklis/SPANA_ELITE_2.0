"""Cloud speed / identity test on Modal GPUs (DEV only; measurement, not product code). Config: config/cloud_bench.yaml.

    python -m tools.cloud_bench upload --file VIDEO [--file VIDEO ...]
    python -m tools.cloud_bench run --suite dev_standard --tag TAG [--gpu L4] [--gpus N]
    python -m tools.cloud_bench score --suite dev_standard --tag TAG          # re-score a finished run locally

upload  Puts footage and the detector weights on the private Modal volume. A video is uploaded only if its sha256 is
        listed under `footage` for a match in `modal.confirmed_matches` that is also on the licensed allow-list of
        config/roboflow.yaml as split dev with third_party_processing_confirmed (fail closed: anything else, EVAL
        included, is refused before any byte leaves the machine). Local paths are remembered in
        runs/cloud_bench/footage_local.json for the decode check.
run     Runs each unit of the suite (detect + track of one footage range, the committed code at HEAD, the product
        config with profiling on, `--device cuda`) on its own GPU, in parallel (at most --gpus at once), and writes
        the outputs to runs/cloud_bench/<tag>/. Then `score`.
score   Locally, no GPU: decode check (the first `decode_check_frames` frames of each unit hashed in the cloud and on
        this Mac must be identical, otherwise the run is not comparable), speed (tools/identity_speed.py profile on
        the window unit) with a full-match projection for 1..max_gpus GPUs, window identity (identity_speed compare
        vs the Mac baseline), labelled identity (tools/benchmark.py + tools/target_score.py, judged with the
        performance.option_* limits vs the Mac outputs of the same clips). Writes summary.json
        (spana.cloud_bench 1.0.0). `comparable_to_mac` is true only if every check passes.
The full-match projection is arithmetic on the window's per-frame stage times (detection split over N GPUs,
tracking sequential), not a measurement.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"
VOL = "/vol"                     # volume mount point in the container
REMOTE_REPO = "/root/repo"
CODE_PATHS = ("spana", "config", "tools", "models/rfdetr_v1.json")   # what the container gets, from git HEAD
OUT_ROOT = REPO / "runs" / "cloud_bench"


class BenchConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Footage:
    name: str
    match_id: str
    file: str
    sha256: str


@dataclass(frozen=True)
class Unit:
    unit: str
    footage: str
    start: int
    stop: Optional[int]
    labels: Optional[str]


@dataclass(frozen=True)
class BenchConfig:
    app_name: str
    volume_name: str
    python_version: str
    gpu: str
    max_gpus: int
    timeout_s: int
    confirmed_matches: Tuple[str, ...]
    footage: Tuple[Footage, ...]
    suites: Dict[str, Tuple[Unit, ...]]
    window_identity: str
    window_tracks: str
    labelled_dir: str
    labelled_tag: str
    decode_check_frames: int

    def footage_by_name(self, name: str) -> Footage:
        for f in self.footage:
            if f.name == name:
                return f
        raise BenchConfigError(f"unknown footage {name!r}")


def _keys(d, where, keys):
    if not isinstance(d, dict):
        raise BenchConfigError(f"{where}: expected a mapping")
    missing, unknown = sorted(set(keys) - set(d)), sorted(set(d) - set(keys))
    if missing or unknown:
        raise BenchConfigError(f"{where}: missing {missing}, unknown {unknown}")
    return d


def _pos_int(v, where):
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise BenchConfigError(f"{where} must be a positive integer")
    return v


def _str(v, where):
    if not isinstance(v, str) or not v.strip():
        raise BenchConfigError(f"{where} must be a non-empty string")
    return v


def parse_config(raw: dict, licensed) -> BenchConfig:
    """raw: config/cloud_bench.yaml; licensed: spana.dataset.config LicensedMatch list (config/roboflow.yaml)."""
    _keys(raw, "cloud_bench", ("schema_version", "modal", "footage", "suites", "baseline", "decode_check_frames"))
    if raw["schema_version"] != SCHEMA_VERSION:
        raise BenchConfigError(f"schema_version {raw['schema_version']!r} != {SCHEMA_VERSION!r}")
    m = _keys(raw["modal"], "modal", ("app_name", "volume_name", "python_version", "gpu", "max_gpus", "timeout_s",
                                      "confirmed_matches"))
    lic = {x.match_id: x for x in licensed}
    confirmed = tuple(m["confirmed_matches"] or ())
    for mid in confirmed:
        x = lic.get(mid)
        if x is None or x.split != "dev" or not x.third_party_processing_confirmed:
            raise BenchConfigError(f"modal.confirmed_matches: {mid} is not a licensed DEV match with third-party "
                                   f"processing confirmed (config/roboflow.yaml)")
    footage = []
    for i, f in enumerate(raw["footage"] or ()):
        f = _keys(f, f"footage[{i}]", ("name", "match_id", "file", "sha256"))
        if f["match_id"] not in confirmed:
            raise BenchConfigError(f"footage {f['name']}: match {f['match_id']} is not in modal.confirmed_matches")
        if not isinstance(f["sha256"], str) or len(f["sha256"]) != 64:
            raise BenchConfigError(f"footage {f['name']}: sha256 must be 64 hex characters")
        if Path(_str(f["file"], "footage.file")).name != f["file"]:
            raise BenchConfigError(f"footage {f['name']}: file must be a bare file name")
        footage.append(Footage(_str(f["name"], "footage.name"), f["match_id"], f["file"], f["sha256"]))
    if len({f.name for f in footage}) != len(footage) or len({f.sha256 for f in footage}) != len(footage):
        raise BenchConfigError("footage: names and sha256 must be unique")
    names = {f.name for f in footage}
    suites = {}
    for s, units in (raw["suites"] or {}).items():
        us = []
        for i, u in enumerate(units or ()):
            u = _keys(u, f"suites.{s}[{i}]", ("unit", "footage", "start", "stop", "labels"))
            if u["footage"] not in names:
                raise BenchConfigError(f"suites.{s}: unit {u['unit']} uses unlisted footage {u['footage']!r}")
            if not isinstance(u["start"], int) or u["start"] < 0:
                raise BenchConfigError(f"suites.{s}: unit {u['unit']} start must be >= 0")
            if u["stop"] is not None and (not isinstance(u["stop"], int) or u["stop"] <= u["start"]):
                raise BenchConfigError(f"suites.{s}: unit {u['unit']} stop must be > start or null")
            us.append(Unit(_str(u["unit"], "unit"), u["footage"], u["start"], u["stop"], u["labels"]))
        if not us or len({u.unit for u in us}) != len(us):
            raise BenchConfigError(f"suites.{s}: needs units with unique names")
        if sum(u.labels is None for u in us) > 1:
            raise BenchConfigError(f"suites.{s}: at most one window (unlabelled) unit")
        suites[s] = tuple(us)
    b = _keys(raw["baseline"], "baseline", ("window_identity", "window_tracks", "labelled_dir", "labelled_tag"))
    return BenchConfig(
        app_name=_str(m["app_name"], "modal.app_name"), volume_name=_str(m["volume_name"], "modal.volume_name"),
        python_version=_str(m["python_version"], "modal.python_version"), gpu=_str(m["gpu"], "modal.gpu"),
        max_gpus=_pos_int(m["max_gpus"], "modal.max_gpus"), timeout_s=_pos_int(m["timeout_s"], "modal.timeout_s"),
        confirmed_matches=confirmed, footage=tuple(footage), suites=suites,
        **{k: _str(b[k], f"baseline.{k}") for k in b},
        decode_check_frames=_pos_int(raw["decode_check_frames"], "decode_check_frames"))


def load_bench_config(path, dataset_config) -> BenchConfig:
    from spana.dataset.config import load_dataset_config
    return parse_config(yaml.safe_load(Path(path).read_text()), load_dataset_config(dataset_config).licensed_matches)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def footage_for_upload(cfg: BenchConfig, path: Path, digest: str) -> Footage:
    """The listed footage this file is, by content. Refuses anything not listed (fail closed)."""
    for f in cfg.footage:
        if f.sha256 == digest:
            if path.name != f.file:
                raise BenchConfigError(f"{path}: content is footage {f.name} but the name differs from {f.file}")
            return f
    raise BenchConfigError(f"{path.name}: sha256 {digest[:12]}… is not listed in config/cloud_bench.yaml footage; "
                           f"only confirmed DEV footage may be uploaded")


def frame_hashes(video: str, start: int, n: int) -> List[str]:
    """sha256 of the decoded BGR pixels of frames [start, start + n), read with spana.video.VideoSource."""
    from spana.video import VideoSource
    src = VideoSource(video)
    return [hashlib.sha256(f.tobytes()).hexdigest() for _, _, f in src.frames(start, start + n)]


def projection(det_ms: float, trk_ms: float, fps: float, footage_s: float, max_gpus: int) -> Dict[str, float]:
    """Full-match minutes if detection is split over N GPUs and tracking stays sequential (arithmetic)."""
    frames = fps * footage_s
    return {str(n): round((det_ms / n + trk_ms) * frames / 1000.0 / 60.0, 1) for n in range(1, max_gpus + 1)}


def tracked_code_dirty(status_porcelain: str) -> List[str]:
    """Modified / staged tracked files under CODE_PATHS (untracked files are not shipped, see stage_code)."""
    return [ln[3:] for ln in status_porcelain.splitlines() if ln[3:].startswith(CODE_PATHS)]


# ----------------------------------------------------------------------------------------------- remote side
def remote_unit(job: dict) -> dict:
    """Runs in the Modal container: detect + track of one unit on the GPU; returns every output file."""
    import shutil
    os.chdir(REMOTE_REPO)
    w = Path(REMOTE_REPO, "models", "rfdetr_v1", "weights.pt")
    w.parent.mkdir(parents=True, exist_ok=True)
    if not w.exists():
        w.symlink_to(Path(VOL, "models", "rfdetr_v1", "weights.pt"))
    video = str(Path(VOL, "footage", job["footage"], job["file"]))
    got = sha256_file(video)
    if got != job["sha256"]:
        raise RuntimeError(f"volume footage {job['footage']} sha256 {got[:12]}… != {job['sha256'][:12]}…")
    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    cfg["profiling"]["enabled"] = True
    out = Path("/tmp/out")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    (out / "profile.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stem = out / f"{job['unit']}_{job['tag']}"
    rng = ["--start", str(job["start"])] + (["--stop", str(job["stop"])] if job["stop"] is not None else [])
    cmds = {
        "detect": [sys.executable, "-m", "spana.cli", "detect", "--backend", "local", "--model-record",
                   "models/rfdetr_v1.json", "--dataset-config", "config/roboflow.yaml", "--runtime", "torch",
                   "--device", "cuda", "--video", video, "--video-sha256", job["sha256"], "--config",
                   str(out / "profile.yaml"), "--out", f"{stem}.det.json", *rng],
        "track": [sys.executable, "-m", "spana.cli", "track", "--video", video, "--video-sha256", job["sha256"],
                  "--config", str(out / "profile.yaml"), "--detections", f"{stem}.det.json",
                  "--out", f"{stem}.trk.json"],
    }
    wall = {}
    t0 = time.perf_counter()
    hashes = frame_hashes(video, job["start"], job["decode_check_frames"])
    wall["decode_check_s"] = round(time.perf_counter() - t0, 2)
    for name, cmd in cmds.items():
        t = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / f"{stem.name}.{name[:3]}.log").write_text(r.stdout + r.stderr)
        wall[f"{name}_s"] = round(time.perf_counter() - t, 2)
        if r.returncode != 0:
            break
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    import torch
    return {"unit": job["unit"], "ok": r.returncode == 0, "wall": wall, "frame_hashes": hashes,
            "host": {"gpu": smi, "torch": torch.__version__, "cuda": torch.version.cuda},
            "files": {p.name: p.read_bytes() for p in sorted(out.iterdir()) if p.name != "profile.yaml"}}


# ----------------------------------------------------------------------------------------------- local side
def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def stage_code(dest: Path) -> Tuple[str, Dict[str, str]]:
    """Committed code at HEAD (CODE_PATHS only) into dest; returns the commit and the git object id of each code
    path (they identify the code even if the commit is later amended). Refuses modified tracked code."""
    dirty = tracked_code_dirty(_git("status", "--porcelain", "-uno"))
    if dirty:
        raise SystemExit(f"commit first: modified tracked code would not match the run record: {dirty}")
    commit = _git("rev-parse", "HEAD").strip()
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD", *CODE_PATHS], cwd=REPO, capture_output=True,
                         check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar, check=True)
    return commit, {p: _git("rev-parse", f"HEAD:{p}").strip() for p in CODE_PATHS}


def _image(cfg: BenchConfig, code_dir: Path):
    import modal
    pins = [ln.strip() for f in ("requirements.lock.txt", "requirements-local-detector.txt")
            for ln in (REPO / f).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return (modal.Image.debian_slim(python_version=cfg.python_version)
            .apt_install("libglib2.0-0")
            .pip_install(*pins)
            .add_local_dir(str(code_dir), REMOTE_REPO))


def cmd_upload(a, cfg: BenchConfig) -> int:
    import modal
    from spana.detection.rfdetr_local import LocalModelRecord
    plan = []
    for p in map(Path, a.file):
        f = footage_for_upload(cfg, p, sha256_file(p))          # refuses before anything is sent
        plan.append((p, f"footage/{f.name}/{f.file}", f))
    weights = LocalModelRecord.load(REPO / "models" / "rfdetr_v1.json").file("weights")   # sha-checked
    plan.append((weights, "models/rfdetr_v1/weights.pt", None))
    vol = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)
    have = {e.path.lstrip("/"): e.size for e in vol.listdir("/", recursive=True)}
    todo = [(p, r) for p, r, _ in plan if have.get(r) != p.stat().st_size]
    if todo:
        with vol.batch_upload(force=True) as batch:
            for p, r in todo:
                print(f"upload {p.name} -> {cfg.volume_name}:/{r} ({p.stat().st_size / 1e6:.0f} MB)")
                batch.put_file(str(p), r)
    print(f"{len(plan) - len(todo)} already on the volume, {len(todo)} uploaded")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    idx_path = OUT_ROOT / "footage_local.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx.update({f.name: {"path": str(p.resolve()), "sha256": f.sha256} for p, _, f in plan if f})
    idx_path.write_text(json.dumps(idx, indent=1, sort_keys=True))
    return 0


def cmd_run(a, cfg: BenchConfig) -> int:
    import modal
    units = cfg.suites.get(a.suite) or sys.exit(f"unknown suite {a.suite!r}")
    gpus = a.gpus or min(len(units), cfg.max_gpus)
    if not 1 <= gpus <= cfg.max_gpus:
        raise SystemExit(f"--gpus must be 1..{cfg.max_gpus} (modal.max_gpus)")
    out = OUT_ROOT / a.tag
    if out.exists():
        raise SystemExit(f"{out} exists: pick another --tag (runs are never overwritten)")
    gpu = a.gpu or cfg.gpu
    with tempfile.TemporaryDirectory() as td:
        commit, code_ids = stage_code(Path(td))
        jobs = [{**dataclasses.asdict(u), "tag": a.tag, "file": cfg.footage_by_name(u.footage).file,
                 "sha256": cfg.footage_by_name(u.footage).sha256, "decode_check_frames": cfg.decode_check_frames}
                for u in units]
        app = modal.App(cfg.app_name)
        vol = modal.Volume.from_name(cfg.volume_name)

        def run_unit(job: dict) -> dict:
            import sys as _s
            _s.path.insert(0, "/root/repo")
            from tools.cloud_bench import remote_unit as _r
            return _r(job)

        fn = app.function(image=_image(cfg, Path(td)), gpu=gpu, volumes={VOL: vol}, timeout=cfg.timeout_s,
                          serialized=True, max_containers=gpus)(run_unit)
        t0 = time.perf_counter()
        with modal.enable_output(), app.run():
            results = list(fn.map(jobs, return_exceptions=True))
        elapsed = time.perf_counter() - t0
    out.mkdir(parents=True)
    meta = {"schema": "spana.cloud_bench_run", "schema_version": SCHEMA_VERSION, "argv": sys.argv, "commit": commit,
            "code_object_ids": code_ids, "suite": a.suite, "tag": a.tag, "gpu_requested": gpu, "gpus": gpus, "elapsed_s": round(elapsed, 1),
            "units": []}
    for job, res in zip(jobs, results):
        if isinstance(res, BaseException):
            meta["units"].append({"unit": job["unit"], "ok": False, "error": repr(res)})
            continue
        for name, data in res.pop("files").items():
            (out / name).write_bytes(data)
        meta["units"].append(res)
    (out / "run.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    print(out / "run.json")
    if not all(u["ok"] for u in meta["units"]):
        print("some units failed (see run.json and the *.log files); not scored")
        return 1
    return cmd_score(a, cfg)


def _py(*args) -> None:
    r = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(args)}\n{r.stdout}{r.stderr}")


def cmd_score(a, cfg: BenchConfig) -> int:
    from spana.config import load_config
    from tools.detector_options import judge
    perf = load_config(a.config).performance
    units = cfg.suites.get(a.suite) or sys.exit(f"unknown suite {a.suite!r}")
    out = OUT_ROOT / a.tag
    run = json.loads((out / "run.json").read_text())
    for pat in ("*.profile_summary.json", "*.compare.json", "*.bench.json", "target_*.json", "judge.json",
                "summary.json"):            # derived files only; the run's own outputs are never touched
        for f in out.glob(pat):
            f.unlink()
    by_unit = {u["unit"]: u for u in run["units"]}
    idx = json.loads((OUT_ROOT / "footage_local.json").read_text())
    summary = {"schema": "spana.cloud_bench", "schema_version": SCHEMA_VERSION, "argv": sys.argv,
               "commit": run["commit"], "code_object_ids": run["code_object_ids"], "tag": a.tag, "suite": a.suite, "gpus": run["gpus"],
               "gpu": sorted({u["host"]["gpu"] for u in run["units"]}), "elapsed_s": run["elapsed_s"],
               "gpu_seconds": round(sum(sum(u["wall"].values()) for u in run["units"]), 1), "checks": {}}
    # decode check: same pixels in the cloud as on this Mac
    dec = {}
    for u in units:
        f = cfg.footage_by_name(u.footage)
        loc = idx.get(f.name) or sys.exit(f"no local path for {f.name}: run `upload --file` for it on this Mac")
        if loc["sha256"] != f.sha256 or sha256_file(loc["path"]) != f.sha256:
            raise SystemExit(f"local {f.name} differs from the listed sha256")
        dec[u.unit] = frame_hashes(loc["path"], u.start, cfg.decode_check_frames) == by_unit[u.unit]["frame_hashes"]
    summary["checks"]["decode_identical"] = {"ok": all(dec.values()), "per_unit": dec}
    window = [u for u in units if u.labels is None]
    if window:
        w = out / f"{window[0].unit}_{a.tag}"
        _py("-m", "tools.identity_speed", "profile", "--config", a.config, "--profile", f"{w}.det.profile.json",
            "--profile", f"{w}.trk.profile.json", "--out", f"{w}.profile_summary.json")
        ps = json.loads(Path(f"{w}.profile_summary.json").read_text())
        det = next(p for p in ps["passes"] if p["pass"] == "detect")["ms_per_frame"]
        trk = next(p for p in ps["passes"] if p["pass"] == "track")["ms_per_frame"]
        summary["speed"] = {"ms_per_frame": ps["ms_per_frame"], "detect_ms": det, "track_ms": trk,
                            "realtime_factor_1gpu": ps["realtime_factor"],
                            "full_match_minutes_by_gpus (projection)": projection(
                                det, trk, ps["fps"], ps["projection_footage_s"], cfg.max_gpus)}
        _py("-m", "tools.identity_speed", "compare", "--config", a.config, "--baseline", cfg.window_identity,
            "--base-tracks", cfg.window_tracks, "--tracks", f"{w}.trk.json", "--out", f"{w}.compare.json")
        cmp_ = json.loads(Path(f"{w}.compare.json").read_text())
        summary["checks"]["window_identity"] = {"ok": bool(cmp_["pass"]), "detail": cmp_["checks"]}
    lab = [u for u in units if u.labels]
    if lab:
        base_dir, bt = REPO / cfg.labelled_dir, cfg.labelled_tag
        base, cand = {}, {}
        for u in lab:
            s = out / f"{u.unit}_{a.tag}"
            _py("-m", "tools.benchmark", "--labels", u.labels, "--detections", f"{s}.det.json", "--tracks",
                f"{s}.trk.json", "--decisions", f"{s}.trk.decisions.jsonl", "--out", f"{s}.bench.json")
            cand[u.unit] = json.loads(Path(f"{s}.bench.json").read_text())
            base[u.unit] = json.loads((base_dir / f"{u.unit}_{bt}.bench.json").read_text())
        targets = {}
        for who, d, tag in (("base", base_dir, bt), ("cand", out, a.tag)):
            clips = [x for u in lab for x in ("--clip", u.labels, str(d / f"{u.unit}_{tag}.trk.json"))]
            p = out / f"target_{who}_reviewer.json"
            _py("-m", "tools.target_score", "--config", a.config, "--out", str(p), "--simulate-reviewer", *clips)
            targets[who] = json.loads(p.read_text())
        j = judge(base, cand, targets["base"], targets["cand"], perf)
        (out / "judge.json").write_text(json.dumps(j, indent=1, sort_keys=True))
        summary["checks"]["labelled_identity"] = {"ok": j["viable"], "detail": j["checks"],
                                                  "info": j["info_base_vs_candidate"]}
    summary["comparable_to_mac"] = all(c["ok"] for c in summary["checks"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: summary[k] for k in ("gpu", "gpus", "elapsed_s", "gpu_seconds", "comparable_to_mac")}
                     | {"speed": summary.get("speed"), "checks": {k: v["ok"] for k, v in summary["checks"].items()}},
                     indent=1))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bench-config", default=str(REPO / "config" / "cloud_bench.yaml"))
    p.add_argument("--dataset-config", default=str(REPO / "config" / "roboflow.yaml"))
    p.add_argument("--config", default=str(REPO / "config" / "default.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload")
    u.add_argument("--file", action="append", required=True)
    for name in ("run", "score"):
        s = sub.add_parser(name)
        s.add_argument("--suite", required=True)
        s.add_argument("--tag", required=True)
    sub.choices["run"].add_argument("--gpu", help="Modal GPU type (default: modal.gpu)")
    sub.choices["run"].add_argument("--gpus", type=int, help="max GPUs at once (default: one per unit)")
    a = p.parse_args(argv)
    cfg = load_bench_config(a.bench_config, a.dataset_config)
    return {"upload": cmd_upload, "run": cmd_run, "score": cmd_score}[a.cmd](a, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
