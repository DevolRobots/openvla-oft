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
- (2026-09-15) Task scope widened: `h0_inputs.md`§2 added three source batches/tasks (dual-arm
  box-stacking, left-arm-only box-stacking, DIN-rail wire-connector), replacing the single
  Ethernet-insertion dataset as the active target.
- (2026-09-15) Dedicated venv built and verified (real import, not just clean exit) on both
  `devolremote` and `gpu245` — same `/cpfs01` checkout, no separate clone needed on the second
  machine.
- (2026-09-15) Pre-fine-tuning review (`08r_gpt_review.md`) run before any submission: found the
  environment couldn't actually import `tensorflow_datasets` (protobuf/`tensorflow-metadata`
  mismatch), the task list didn't match the code, the inherited no-op-filtered RLDS conversion is
  unsafe for action chunks, the stride/cadence assumption inherited from vanilla OpenVLA doesn't
  hold for OFT, and no queue-safe submitter existed. Verdict: do not submit yet.
- (2026-09-15) Fixed the code/config-level blocking findings from that review: pinned
  `tensorflow-metadata==1.16.1` + `protobuf>=4.21.6,<4.26` (reinstalled + verified remotely),
  pinned the `transformers`/`dlimp` git-fork revisions, ported `scripts/submit_finetune.py` +
  `openvla_train.sh` (queue-safe, `merge_lora_during_training=False` by default), and registered
  `devol_flexiv_leftarm_stackboxes` / `devol_flexiv_dualarm_dinrail` for the two new tasks. Task
  scope, action-execution cadence, and the no-op-filter re-conversion remain open —
  `Ah_for_human.md`§2 Q3–Q5.
- (2026-09-15) `Ah_for_human.md`§2 Q3–Q5 answered: all three tasks, run as separate fine-tunes
  serially (one GPU each, not concurrent); action-execution cadence is native 30 Hz, not the
  inherited stride-5-at-6Hz (matches the sibling `openpi` project's `action_horizon=30` on the
  same data) — `constants.py`'s `NUM_ACTIONS_CHUNK` updated `8`→`30`; dataset names renamed to
  match `openpi`'s convention with an `openvla_oft_` prefix
  (`openvla_oft_flexiv_dualarm_stackboxes` / `openvla_oft_flexiv_leftarm_stackboxes` /
  `openvla_oft_flexiv_dualarm_dinrail` — the first of these was `devol_flexiv_dualarm_stackboxes`
  since step 1, renamed now since no RLDS build existed for it yet). All blocking findings from
  `08r_gpt_review.md` that were fixable in this repo's code/config are now applied; the remaining
  blocker (no RLDS build exists for any of the 3 tasks) is sibling-repo (`~/dev/openvla`)
  data-conversion work.
