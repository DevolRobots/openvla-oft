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
- (2026-09-15) Created/renamed the 3 TFDS builder packages in the sibling `~/dev/openvla` repo's
  `scripts/data_conversion/` to match the renamed dataset names: renamed
  `devol_flexiv_dualarm_stackboxes` → `openvla_oft_flexiv_dualarm_stackboxes` (no RLDS build had
  ever been produced under the old name, and this repo's own active docs had no competing plan
  for it — verified before renaming), and created two new packages from scratch,
  `openvla_oft_flexiv_leftarm_stackboxes` and `openvla_oft_flexiv_dualarm_dinrail`. **Uncommitted**
  in both the local `~/dev/openvla` checkout and the remote `/cpfs01/wutingsh/openvla` one — that
  repo also has unrelated in-progress work of its own (`deployment/flexiv_dualarm/schema.py`/
  `serve.py` modified, a new `test_decode_jpeg_color.py`) which was left untouched.
- (2026-09-15) Ran the LeRobot→RLDS conversion for all three tasks via the sibling repo's
  `convert.py`, at **native 30 Hz** (`--stride 1`) with **no-op filtering off**
  (`--no-noop-filter`), `--image-size 224`: `openvla_oft_flexiv_dualarm_stackboxes` (200 eps, 16
  shards, ~1.6 GiB, ~17 min), `openvla_oft_flexiv_leftarm_stackboxes` (200 eps, 32 shards, ~25
  min — its `right_gripper` action dim is fully degenerate/zero-span, confirmed harmless: the
  normalizer's `zeros_mask` already maps `min==max` dims to a constant 0), `openvla_oft_flexiv_dualarm_dinrail`
  (395 eps, 32 shards, ~35 min). All three verified importable and present in
  `OXE_DATASET_CONFIGS`/`OXE_STANDARDIZATION_TRANSFORMS` on both `devolremote` and `gpu245`
  (shared `/cpfs01`, no separate build needed per machine). `measure_noops.py` run on all three
  source batches first — no assumption violations.
- (2026-09-15) Copied the sibling `openvla` project's gitignored `.env` (W&B key) to
  `/cpfs01/wutingsh/openvla-oft/.env` — a same-machine `cp`, contents never read by the agent, at
  the human's explicit instruction. Confirmed working: job 411 (below) picked it up and logged
  into W&B as the same entity/project the sibling project uses.
- (2026-09-15) Extended `scripts/submit_finetune.py` with `--depends-on` (passthrough to
  `gbatch --depends-on`) and stdout job-ID capture (`JOB_ID=<n>`), so the three fine-tunes could
  be queued serially (one GPU each, per the human's instruction) without polling. **Uncommitted**
  (synced to the remote checkout via `rsync` for testing, matching local, but never `git commit`).
- (2026-09-15/16) Checked GPU availability: `devolremote` fully saturated (one 8-GPU job running,
  8 more queued behind it); `gpu245` had a genuinely free GPU (node 7, confirmed via both
  `gqueue`'s Running list and `nvidia-smi`). Submitted all three fine-tunes to `gpu245`, chained
  with `--depends-on`: job 411 (`openvla_oft_flexiv_dualarm_stackboxes`, no dependency) → job 412
  (`..._leftarm_stackboxes`, depends on 411) → job 413 (`..._dualarm_dinrail`, depends on 412),
  each `--gpus 1 --time 24:00:00` (the script's then-default). Job 411 started immediately on
  GPU 7 and was confirmed healthy (wandb login OK, training steps advancing).
- (2026-09-16) **Job 411 hit its 24h `--time` limit and was killed (`State=Timeout`) at step
  138,462/200,000 (69.2%)**, having measured ~1.58 it/s in practice — i.e. a full 200,000-step run
  actually needs **~35 hours**, not the ~24h the docs had estimated (that estimate came from the
  sibling project's stride-5/`NUM_ACTIONS_CHUNK=8` numbers; native-rate data plus
  `NUM_ACTIONS_CHUNK=30` is a real per-step cost increase, not a miscalculation in this run). Per
  gbatch's default auto-cancel-on-dependency-failure, **job 412 was cancelled
  (`DependencyFailed:411`) and job 413 cascaded (`DependencyFailed:412`)** — neither ever ran.
  13 checkpoints survive from job 411 (steps 10k–130k, LoRA adapter + action head only since
  `merge_lora_during_training=False`, ~977 MiB each, ~13 GiB total) — enough to resume rather than
  restart (`--resume True --resume_step 130000 --vla_path <...--130000_chkpt dir>`, see
  `09_commands.md`§3).
- (2026-09-17) Agent noticed the timeout mid-session (before the human returned) and asked
  permission to cancel + resubmit with a longer `--time`; the cancel itself was blocked by the
  permission system (destructive action needing explicit approval), and no answer arrived before
  the human stepped away. **Nothing was resubmitted.** Human returned, said other work now has
  priority — training resubmission is paused, not abandoned; see `Ah_for_human.md`§3 for the open
  decision (new `--time` value; resume-from-130k vs. restart). Docs refreshed
  (`06_current.md` archived to `07_archive/`) to reflect this pause.
