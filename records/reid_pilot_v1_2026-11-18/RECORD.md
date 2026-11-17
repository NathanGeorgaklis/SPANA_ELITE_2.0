# SPANA re-ID model, pilot v1 (2026-11-18): one TRAIN match, compared with shirt colour on DEV

Scope:
- First training run of the re-ID model in `records/design/REID_MODEL_DESIGN.md`.
- Trained on TRAIN_M001 only: frozen labels v1, 25 identities (20 players, 2 goalkeepers, 3 referees).
- Evaluated on the DEV appearance test (`tools/dev/appearance_cue_eval.py`), FRESH_02 / 05 / 07 / 08.
- Product code unchanged: nothing in `spana/` reads the model. EVAL not used.
- Everything ran on Nathan's Mac (Apple GPU); crops, weights and the model stay there (`HASHES.txt`).

## Steps
1. **Backbone pinned:** `facebook/dinov2-small`, revision `ed25f3a3…`, Apache-2.0, weights sha in `models/dinov2_small.json`.
   The training script refuses other weights.
2. **Crops** (`tools/reid_crops.py`, 22 s).
   - Labels loaded through `load_frozen(..., "train")` (hashes + split guard).
   - The video must equal the manifest's sha256. The AV1 window clip was refused, as intended; the H.264 transcode
     (`…_h264.mp4`, sha `2475f26c…`) was used.
   - Per player one crop every 0.25 s, height ≥ 40 px, 224×112: 10,139 crops (39 to 1,148 per identity).
   - The last 20 % of the labelled frames are holdout: 7,950 train / 2,189 holdout crops.
   - A contact sheet matched shirt numbers to labels. Visible label noise: two grey-kit #5 crops under A5, and one
     blurred other figure under B10. The labels are frozen, so they stay.
3. **Training** (`tools/reid_train.py`, `config/reid_train.yaml`).
   - DINOv2-S/14 with the first 8 of 12 blocks frozen.
   - CLS + 3 part pools → 256-d embedding.
   - ArcFace + batch-hard triplet; PK batches of 16 × 4 with at least half from one side.
   - Augmentations: flip, down-scaling, blur, JPEG, small hue, erasing.
   - 12 epochs × 150 batches, 16.6 min on the M3.
   - Holdout rival check (closed-set, same players, later frames): 0.867 untrained → 0.970 (`train_log.jsonl`).
4. **Export:** ONNX; max difference from PyTorch 7.5e-7.
5. **DEV appearance test** (`cue_eval_summary.json`).
   - The tracks are the local-detector runs of the target records.
   - Both cues are scored on the same crops, pairs and rival triplets.

## Results (DEV)

| clip | rival tests | rival accuracy hist | rival accuracy spana | TPR @ 1 % FAR on hard pairs, hist | same, spana |
|---|---|---|---|---|---|
| FRESH_05 | 53 | **1.000** | 0.962 | **1.000** | 0.657 |
| FRESH_07 | 1,496 | **0.914** | 0.909 | 0.106 | **0.229** |
| FRESH_02 | 1,329 | **0.889** | 0.867 | 0.055 | **0.221** |
| FRESH_08 | 1,610 | **0.899** | 0.883 | 0.058 | **0.152** |
| pooled | 4,488 | **0.902** | 0.888 | | |

- **Rival test:** the pilot is level with shirt colour or slightly worse (pooled 0.888 vs 0.902). It is better on some
  pairs colour finds hard:
  - FRESH_07 5|6: 0.844 vs 0.697;
  - FRESH_08 16|17: 0.861 vs 0.762;
  - FRESH_02 12|16: 0.812 vs 0.750.
  It is worse on others (FRESH_07 4|5: 0.885 vs 0.975).
- **Strict acceptance:** at a threshold allowing 1 % wrong accepts on hard pairs, the pilot accepts 2–4× more true
  same-player pairs than colour on FRESH_07 / 02 / 08. This is the regime the product needs: a margin calibrated for
  ≤ 0.5 % false resolutions.
- **Occluded queries:** worse than colour on FRESH_07 / 08 (0.78 vs 0.84 / 0.86).
- **Frozen DINOv2 baseline:** 0.85 / 0.79 / 0.80 on FRESH_05 / 07 / 02 (`records/appearance_cues_2026-11-08`).
  Those runs used other tracks, so the comparison is only indicative: the trained model is 0.06–0.12 higher.

## Findings
1. **Training works end to end on the Mac:** labels → crops → model → ONNX in under 20 min per match. The data path
   (frozen labels, video sha, split guard, pinned weights) fails closed.
2. **One match is not enough to beat colour on the rival test.** The holdout score (0.970, same 25 people) is far
   above DEV (0.888, unseen people): the model learns these players, not what separates players in general. The
   design's pilot was 3 TRAIN matches. M002 and M003 have video and tracks but no frozen identity labels yet.
3. **The two cues disagree on different pairs.** The pilot is better where it matters most for fail-closed use (the
   strict threshold), and colour is better on easy pairs. A combination may beat both. This is not measured yet.
4. **Not adopted.** The tracker and the follower are unchanged. No threshold was calibrated.

## Next (suggested)
1. Label M002 (and M003) in the tracklet-linking labeller, freeze them, cut crops, and retrain on 2–3 matches. Same
   config, so the effect of more identities is clear.
2. Add a combined cue (colour + re-ID) to the DEV test, and report TPR at 1 % and 0.5 % FAR on hard pairs as the
   primary number next to rival accuracy.
3. Only then calibrate the margin on DEV and try the model as extra evidence for IDENTITY_UNCERTAIN in the tracker,
   behind a config flag, off by default.

## Files
- `cue_eval_summary.json`: compact DEV results.
- `train_log.jsonl`: per-epoch losses and holdout check.
- `HASHES.txt`: inputs and outputs on Nathan's Mac.
