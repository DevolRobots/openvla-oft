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
