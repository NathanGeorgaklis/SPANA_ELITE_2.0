# DEV matches: permission as stated by Nathan (2026-11-14)

Recorded so the licence allow-list in `config/roboflow.yaml` has a written source. Nathan gave these statements
in the session of 2026-11-14 and asked for both matches to be added to the allow-list. The written permissions
themselves are held by Nathan, not in this repository.

| match_id | Match | Local DEV clips | Rights holder | Permission |
|---|---|---|---|---|
| DEV_G1_SYDNEY_UNITED_APIA | Sydney United 58 v APIA Leichhardt (Game 1) | FRESH_01, FRESH_02 | Sydney United 58 / APIA Leichhardt | written permission dated 31 October 2026 |
| DEV_G2_NWS_SPIRIT_SD_RAIDERS | NWS Spirit v SD Raiders (Game 2) | FRESH_03 to FRESH_06 | NWS Spirit / SD Raiders | written permission dated 31 October 2026 |

For both, as stated by Nathan:
- Consent basis for players filmed: authorised football match participation and written permission covering use of
  the footage and player imagery / data for SPANA's AI/ML development.
- SPANA Sports Pty Ltd has permission to use the match footage and associated player imagery / data for commercial
  AI and machine-learning development, including player detection, tracking, re-identification, model training,
  validation and testing. ML training is expressly permitted.
- The permission covers third-party processing (frames sent to Roboflow's hosted inference): confirmed.

Not covered by this record: Game 3 (Sydney FC v St George City; FRESH_07 to FRESH_10). It is not on the allow-list,
so no tool here will send its frames anywhere.

## Full match footage (confirmed by Nathan, 2026-11-15)
Asked before the full match videos were read, Nathan confirmed in the session of 2026-11-15 that the written
permissions dated 31 October 2026 cover the full match footage of both DEV matches, not only the FRESH clips. That
includes third-party processing by Roboflow (upload and hosted inference). Videos used for the DEV benchmark
(`tools/dev_bench.py`): DEV_G1_SYDNEY_UNITED_APIA `Sydney_United_APIA.mp4`, DEV_G2_NWS_SPIRIT_SD_RAIDERS
`NWS_Spirit_SD_Raiders.mp4` (one file per match; sha256 in the `records/dev_bench/` manifests).
