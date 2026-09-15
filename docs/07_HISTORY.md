# 07 — History

<!-- Chronological summary: one line per significant event, newest at the bottom of each
     section or clearly dated. Lossless dated snapshots of 06_current live in 07_archive/
     as 06_YYYY-MM-DD_<tag>.md. -->

## 1. Timeline

- (2026-09-14) Project kicked off: the sibling `openvla` project's second rollout attempt moved
  the arms but control frequency was too slow to be usable (one ~280 ms forward pass per
  timestep, no action chunking). Investigated OpenVLA-OFT as a fix
  (`~/dev/openvla/docs/04s_openvla_oft_feasibility.md`) — feasible, decided to try it.
- (2026-09-14) Forked `moojink/openvla-oft` to `DevolRobots/openvla-oft`, cloned to
  `~/dev/openvla-oft`, branch `ting/dev` (matching the `openvla`/`openpi` fork convention).
- (2026-09-14) Ported the Flexiv dual-arm dataset registration from the sibling `openvla` fork
  (`configs.py`, `transforms.py`, `materialize.py`) and added a `FLEXIV_CONSTANTS` platform
  profile to `constants.py`. Not yet committed.
- (2026-09-14) Project docs scaffolded, populated with background from the sibling project and
  this session's research, for a fresh agent to pick up from here.
- (2026-09-15) Correction: the dataset-registration port and the docs scaffold were both actually
  folded into `"Initial commit"` (`e9b7e09`) already — `0A_agent.md`/`06_current.md` had been
  saying step 1 was still uncommitted; fixed to match `git log`.
- (2026-09-15) Both open fine-tuning decisions answered (`Ah_for_human.md`§1): `--use_proprio
  False`, `NUM_ACTIONS_CHUNK=8` confirmed. Both already matched the recommended defaults, so no
  plan/command changes — just closes out the open items.
