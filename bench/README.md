# Tracker bench v1 (DEV only)

Which tracker to build on? Every candidate runs on the **same frozen detections** of the 4 labelled DEV clips and is
scored by the same code. Configs: `config/tracking/*.yaml` (published defaults, no per-clip tuning).
EVAL is never read; no full-match runs.

| | candidate | code | licence | product use |
|---|---|---|---|---|
| REF | SPANA tracker (`config/default.yaml`) | `spana/tracking` | SPANA | current |
| A | ByteTrack, Roboflow `trackers` 2.7.0 | pip | Apache-2.0 | allowed |
| B | BoT-SORT + camera motion compensation, `trackers` 2.7.0 | pip | Apache-2.0 | allowed |
| C | Deep-EIoU @ `2fc7d27` | `bench/external/` (not in Git) | **none** | **EVAL-ONLY** (benchmark use only) |
| D | GTA-Link @ `e4d5cc4` on B and on C | `bench/external/` (not in Git) | MIT (code) | allowed on B; not on C |

C and D use SPANA re-ID v2 embeddings (computed once into the cache) instead of their OSNet sports weights, whose
training data and download are not licence-cleared.

## Run (Nathan's Mac, bench venv)
```bash
cd ~/SPANA_ELITE_2.0
python3 -m venv .venv-bench && source .venv-bench/bin/activate
pip install -r bench/requirements-bench.txt
python -m bench.external fetch
# once per clip: detector output (skip `detect` if the local-detector artifact already exists) + embeddings
python -m spana.cli detect --backend local --model-record models/rfdetr_v1.json --dataset-config config/roboflow.yaml \
  --runtime torch --device mps --video <CLIP>.mp4 --config config/default.yaml --out runs/det/<CLIP>.det.json
python -m bench.run cache --clip FRESH_02 --detections runs/det/<CLIP>.det.json --video <CLIP>.mp4 --embed
# ... FRESH_05 / FRESH_07 / FRESH_08
python -m bench.run track --video FRESH_02=<clip>.mp4 --video FRESH_05=... --video FRESH_07=... --video FRESH_08=...
python -m bench.run score          # -> bench/results/tracker_bench_v1.csv + ranked table
```

## Metrics (`bench/metrics.py`)
- Overall, TrackEval (`trackeval` 1.3.0, MIT): HOTA / DetA / AssA, IDF1, ID switches, MOTA on the labelled frames.
- Target player (each labelled player, tapped on their first unambiguous NESTED_V2 frame, tapped track followed as
  is): ID switches, fragments, % of GT frames covered, longest continuous segment (s), frames wrongly on another
  player (and how many of those the tracker called certain = silent).
- Runtime: frames / seconds inside `Tracker.update` (D: online + offline time). Decoding and embedding excluded.

Recommendation rule (fixed in `bench_v1.yaml` before any result): commercial-compatible only; fewest silent
wrong-player frames, then target coverage, then IDF1.
