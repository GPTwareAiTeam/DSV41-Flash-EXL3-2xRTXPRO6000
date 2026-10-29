# Diffbot drafter head — attribution and licence

**Diffbot — [DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)**
(MIT). Their published `drafter-code-k7` DSpark pack is the alternative drafter head this runtime can load
(`DSPARK_HEAD=diffbot`, see [env_vars.md](../../doc/env_vars.md) and `qmesh/tp2/SHARD_SPEC.md`). No diffbot code and
no diffbot weights are redistributed here: users download the pack from the Hub under its own MIT licence and
convert it locally with this tree's converter, `qmesh/tp2/drafter_overlay.py`, which reads only the pack's `mtp.*`
drafter tensors. The target model stays our own EXL3 quantisation of DeepSeek-V4.1-Flash-1017, so their quantised
target weights are never used. The pack is a fine-tune of DeepSeek's released multi-token-prediction head and
carries the upstream MIT notice (`Copyright (c) 2023 DeepSeek`), whose text is kept verbatim at
[diffbot-drafter-k7.LICENSE.txt](diffbot-drafter-k7.LICENSE.txt) so the "include the notice" condition is met by a
file in this repository rather than by a URL. Diffbot's published TP=2 throughput figures on the same GPU class are
the external comparison point quoted in our results notes; the default drafter head remains ours.
