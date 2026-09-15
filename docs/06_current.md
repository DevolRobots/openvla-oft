# 06 — Current task

<!-- Coherent description of the CURRENT active task. Not a dumping ground. If idle, write
     "between tasks — last completed X". Snapshot into 07_archive/06_YYYY-MM-DD_<tag>.md
     before a major pivot. -->

## 1. Active task (2026-09-15)

**Get an OpenVLA-OFT checkpoint fine-tuned and served on the Flexiv dual-arm task family, to test
whether action chunking fixes the control-frequency problem found with vanilla OpenVLA.**
Background: [`01_overview.md`](01_overview.md), full plan: [`04_plan.md`](04_plan.md). Task scope
widened 2026-09-14/15 from one dataset (Ethernet insertion) to three (`h0_inputs.md`§2:
dual-arm box-stacking, left-arm-only box-stacking, DIN-rail wire-connector) — see §3 below.

**Step 1 (dataset registration) and step 2 (venv) are both done.** Step 1: committed in
`"Initial commit"` (`e9b7e09`). Step 2: venv built and verified (real import) on both
`devolremote` and `gpu245` (shared `/cpfs01` checkout, `ting/dev` @ `68c3462`).

**Current status: pre-fine-tuning review found blockers; all decisions now made, only data
conversion is left.** [`08r_gpt_review.md`](08r_gpt_review.md) (2026-09-15) is the authoritative
preflight status — verdict "do not submit yet." Fixed/decided this session (uncommitted): the
protobuf/`tensorflow-metadata` import failure (pinned + reinstalled + verified remotely), a
queue-safe `scripts/submit_finetune.py`/`openvla_train.sh` (ported from the sibling project,
`merge_lora_during_training` off by default), pinned `transformers`/`dlimp` git-fork revisions,
and — per `Ah_for_human.md`§2 Q3–Q5 — task scope (**all 3 tasks, three separate fine-tunes, run
serially, one GPU each**), cadence (**native 30 Hz**, `NUM_ACTIONS_CHUNK` `8`→`30`), and dataset
names (`openvla_oft_flexiv_dualarm_stackboxes` / `openvla_oft_flexiv_leftarm_stackboxes` /
`openvla_oft_flexiv_dualarm_dinrail`, matching the sibling `openpi` project's convention).

## 2. Next steps

In order:

1. Re-run the sibling repo's LeRobot→RLDS conversion for all three tasks, at **native 30 Hz** (not
   the old stride-5 default) with the interior no-op deletion disabled/redesigned for OFT
   (`08r_gpt_review.md`§2.3 — the existing converter's `DEVOL_NOOP_FILTER=0` env var, not yet
   verified sufficient alone for chunk continuity). The box-stacking task also needs the sibling
   repo's `DevolFlexivDualarmStackboxes` builder class renamed to match the new dataset name
   first (`05a_codemap.md`).
2. Submit a LoRA fine-tune per task via `python3 scripts/submit_finetune.py --gpus 1
   --dataset-name <name>` (`09_commands.md`§3), **one at a time** — wait for each to
   finish/checkpoint before starting the next (Q3: serial, not concurrent). GPU allocation must be
   re-confirmed fresh (`gqueue -u all -s Running` + `nvidia-smi` on both machines) before each.
3. Write the new `serve.py` analog (`05a_codemap.md`§3).
4. Re-run the pre-flight ladder, then the rollout — with client-side timing instrumentation this
   time (`04_plan.md`§2 step 5).

## 3. Open / blocked

- ~~**`--use_proprio` is an open decision**~~ **Decided 2026-09-15: `False`** (`Ah_for_human.md`§1
  Q1). ~~Chunk size 8 confirmed (Q2)~~ **superseded — now `30`, native-rate cadence (Q4)**.
- ~~**Task scope, action-execution cadence, dataset naming — open**~~ **All decided 2026-09-15**
  (`Ah_for_human.md`§2 Q3–Q5) — see §1 above.
- **No RLDS build exists for any of the 3 tasks yet** — only `devol_flexiv_dualarm` (the original
  Ethernet-insertion dataset, now out of the active task list) is built on `/cpfs01`. This is the
  one remaining blocker before step 2 above.
- **This is a scope-limited experiment** (`01_overview.md`§2): if the fine-tune or the new
  serving code turns out substantially harder than expected, or the measured control frequency
  still isn't usable, the plan is to drop this rather than keep pushing — flag that decision
  point rather than grinding through it unilaterally if it comes up.
