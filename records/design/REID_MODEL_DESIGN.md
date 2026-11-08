# SPANA-owned football re-ID model: design (2026-11-08, proposal, nothing trained)

## Purpose and scope
- The model only distinguishes players within one match, mainly same-kit players. It is used only as extra evidence
  to resolve IDENTITY_UNCERTAIN.
- It never forces an association.
- It returns UNKNOWN when the evidence is weak.

## Architecture
- **Backbone:** DINOv2 ViT-S/14 (Apache-2.0, about 22 M parameters). The first 8 of 12 blocks are frozen; the last 4
  are fine-tuned.
- **Input:** a 224×112 person crop (RGB) from the detector box.
- **Head:** CLS token concatenated with 3 horizontal part pools (upper torso / shorts / legs) → linear → 256-d,
  L2-normalised embedding.
- **Quality output:** computed in code, not learned. It is crop height, a clean-crop flag, an occluded flag and a
  blur score, and gates every comparison.
- **Losses:** ArcFace over the training identities, plus batch-hard triplet loss with same-team hard negatives.
- **Use (fail closed):** the pair is cleared only when all of these hold:
  - both crops pass the quality gates;
  - sim(q, A_ref) − sim(q, B_ref) ≥ m, where m is calibrated on DEV for a false-resolution rate ≤ 0.5 %;
  - the condition is sustained for `identity_resolve_s`;
  - no contradicting evidence. Otherwise the result is UNKNOWN.
- **Export:** ONNX on onnxruntime CPU. Run on demand, only for uncertain tracks and their rivals.

## Data and splits
- **Unit:** an identity is (match_id, player_key), so identities are match-local.
- **Split by match only, assigned and frozen before any training.** TRAIN / DEV / EVAL use disjoint matches.
  EVAL should also have disjoint venues or competitions where possible.
- **The 3 source matches of the FRESH_* DEV clips stay DEV.** They are never TRAIN.
- **Rights:** each match manifest records the footage licence and the consent basis that covers ML training.
  A match without both is not used.
- **Pilot:** 3 TRAIN matches plus the existing DEV.
- **Target:** about 40–60 TRAIN matches, about 1,000–1,500 identities, 50–100 crops per identity sampled across the
  match (≥ 2 s apart, different halves), contact episodes oversampled. About 100 k crops.

## Labels (frozen, never edited; corrections are a new version)
- `data/reid/<split>/<match_id>/v<N>/manifest.json`: match_id, split, video sha256, licence, consent basis,
  annotator, frozen hashes.
- `identities.jsonl`, one row per tracklet:
  `{"tracklet": "T000123", "player": "A07", "side": "A|B|REF|GK_A|GK_B", "first_frame": 1200,
  "last_frame": 1460, "verified": true, "shirt_number": "7"|null}`
- Boxes come from the existing MOT 1.1 tracklet export; crops are cut at extraction time from the hashed video.
- **Labelling** extends the existing labeller with a tracklet-linking mode. The tracker proposes tracklets; the human
  links them to a player, splits wrong tracklets and rejects bad ones. Nobody draws boxes.

## Training pipeline
1. Freeze labels and the match split (`tools/freeze_reid_labels.py`).
2. Extract crops plus metadata (height, occluded, clean) with hashes.
3. Sample PK batches within a single match: 16 identities × 4 crops, at least half from the same team.
4. Augment:
   - down-scaling to small player heights;
   - random erasing (occlusion);
   - blur and compression;
   - horizontal flip;
   - only small hue jitter (kit colour carries identity).
5. Train with AdamW, about 30 epochs, early stopping on DEV rival-test accuracy.
6. Calibrate on DEV: the margin m and the quality gates, at false resolution ≤ 0.5 %.
7. Export ONNX, write a model card (data manifest hashes, licence, metrics), pin in `models/`.
8. EVAL once, only with Nathan's approval.

## Evaluation (DEV for development; EVAL once)
- **Primary:**
  - same-team rival-test accuracy (`tools/dev/appearance_cue_eval.py`);
  - false-resolution rate at m;
  - share of uncertain cases resolved at that rate.
- **Verification:** TPR at FAR 1 % and 0.1 % on same-team pairs; within-match rank-1 / mAP.
- **Slices:** player height, occlusion, time gap (> 1 min, other half), replays / camera cuts, unseen venue (EVAL).
- **End-to-end on the DEV clips:**
  - silent swaps stay 0;
  - wrong resolutions are 0;
  - flagged swaps do not rise;
  - uncertain track-frames go down.
- **Baselines:** the colour histogram and frozen DINOv2. Youtu is used as a DEV-only research reference and is never
  shipped.

## Compute
- **Training:** about 1–2 GPU-hours per run on a single T4/A10-class GPU; about 20–40 GPU-hours including sweeps.
  Not feasible on the 2-CPU workspace.
- **Inference:** about 25–40 ms per crop on 2 CPUs (estimate, to be measured), on demand only.
- **Labelling (the dominant cost):** about 3–6 human-hours per full match with tracklet proposals, so the pilot is
  about 10–20 h.
