# Ah — Questions for the human (agent asks)

<!-- The agent writes questions; the human answers in-line. Group into sections (one per
     questioning session). Prefix each question subsection Q1., Q2., ... Put dates in headings. -->

## 1. Fine-tuning recipe decisions (2026-09-14)

### Q1.

Should the OpenVLA-OFT fine-tune use `--use_proprio True` (feed the 16-D proprioceptive state to
the model) or `--use_proprio False` (matching the sibling `openvla` project's existing behavior,
where the schema declares state fields but the model never sees them)? Enabling it would be a new
capability, not parity with the existing checkpoint — recommend leaving it `False` for a
like-for-like comparison unless there's a specific reason to add it now.

> leave it false for now.

### Q2.

`NUM_ACTIONS_CHUNK=8` was picked as a starting point (~1.3 s of open-loop motion per query at
this dataset's ~6 Hz effective step rate — see `05a_codemap.md`). Any preference for a different
starting chunk size before the first fine-tune, or is 8 fine to try first and adjust later based
on measured inference latency?

> no, 8 is fine.
