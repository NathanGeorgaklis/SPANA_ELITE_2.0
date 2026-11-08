# Re-ID model licence check (2026-11-08)

- Desk review of public sources. This is not legal advice; confirm with counsel before any commercial release.
- No engine change, no training, EVAL not used.

## Youtu ReID (`opencv/person_reid_youtureid`, `person_reid_youtu_2021nov.onnx`, sha256 05796833…580d)

| question | finding | source |
|---|---|---|
| Model file licence | Apache-2.0: an Apache LICENSE was added to `models/person_reid_youtureid` in the OpenCV Zoo commit "Add YoutuReID for person ReID (#24)" | huggingface.co/opencv/opencv_zoo/commit/0f201989… |
| Origin | Tencent Youtu Lab baseline from github.com/ReID-Team/ReID_extra_testdata | OpenCV Zoo README |
| Training data | The authors list **Market-1501, DukeMTMC, CUHK03 and MSMT17** as used for the baseline training | ReID_extra_testdata README |
| DukeMTMC | Released under **CC BY-NC-SA 4.0** (non-commercial). **Taken down in June 2019** after privacy / IRB criticism. The original licence prohibits distributing the dataset or derivatives | Princeton CITP blog; Peng et al. (NSF PAR 10346631) |
| CUHK03 | CUHK's data repository terms put datasets generally under CC BY-NC (non-commercial, research and instructional use) | libguides.lib.cuhk.edu.hk |
| MSMT17 | Distributed by request (pkuvmc.com); the access terms restrict use to non-commercial research. Exact agreement text not retrieved | paperswithcode MSMT17 entry |
| Market-1501 | No licence declared in the public distributions (empty licence field) | academictorrents Market-1501 |
| Attribution | Apache-2.0: keep the licence and NOTICE; credit Tencent Youtu Lab. Dataset licences add their own conditions (CC BY-NC-SA: attribution, non-commercial, share-alike) | |

**Verdict: not clearly licence-safe.** The Apache-2.0 licence covers the model file, but the weights were trained on
three datasets with non-commercial terms, one of them withdrawn for privacy reasons. Whether those terms bind a
trained model is legally unsettled. Researchers have found production use of models trained on non-commercial
datasets to be common, but that is not a safe basis for a commercial product.

**The same problem applies to OSNet MSMT17** (`anriha/osnet_x0_25_msmt17`).

## Alternatives

| candidate | licence of weights | training data | commercial | our DEV result |
|---|---|---|---|---|
| DINOv2-small (Meta) | Apache-2.0 (relicensed from CC-BY-NC) | LVD-142M curated web images, not released | yes, standard Apache-2.0 | rival accuracy 0.85 / 0.79 / 0.80: below the colour histogram |
| NVIDIA TAO ReIdentificationNet (ResNet50; Swin) | NVIDIA model EULA; card states "ready for commercial use" | Market-1501 (751 IDs) + 2023 AI City MTMC (156 IDs, 148 synthetic); the Swin version adds an NVIDIA proprietary set | yes under NVIDIA's terms, **but restricted to NVIDIA hardware / TAO / DeepStream / TensorRT**; ONNX deployable available | not tested |
| Intel Open Model Zoo person-reidentification-retail-0277/0286/0287/0288 | Open Model Zoo is Apache-2.0; Intel human-rights use condition | **not disclosed** (only Market-1501 metrics published); trained with Intel's deep-object-reid (Torchreid fork) | Apache-2.0, but data provenance unknown | not tested |
| Our own re-ID (fine-tuned on SPANA footage we hold rights to, from an Apache-2.0 backbone such as DINOv2) | ours | ours | yes, if the footage rights and consent cover it | not available: training not approved |
