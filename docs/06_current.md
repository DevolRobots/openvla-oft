# 06 — Current task

<!-- Coherent description of the CURRENT active task. Not a dumping ground. If idle, write
     "between tasks — last completed X". Snapshot into 07_archive/06_YYYY-MM-DD_<tag>.md
     before a major pivot. -->

## 1. Active task (2026-09-14)

**Get an OpenVLA-OFT checkpoint fine-tuned and served on the Flexiv dual-arm task, to test
whether action chunking fixes the control-frequency problem found with vanilla OpenVLA.**
Background: [`01_overview.md`](01_overview.md), full plan: [`04_plan.md`](04_plan.md).

**Step 1 is done (2026-09-14):** forked `moojink/openvla-oft` to `DevolRobots/openvla-oft`
(`~/dev/openvla-oft`, branch `ting/dev`), ported the dataset registration
(`devol_flexiv_dualarm`/`devol_flexiv_dualarm_stackboxes`) and added the `FLEXIV_CONSTANTS`
platform profile. Full detail: [`05a_codemap.md`](05a_codemap.md). **Not yet committed** in this
repo — the working tree has the 4-file diff, uncommitted, as of this doc scaffold.

## 2. Next steps

In order, per [`04_plan.md`](04_plan.md)§2:

1. Decide whether to commit the current uncommitted dataset-registration changes (probably yes —
   check with the user if there's a reason not to; the sibling project's convention has been to
   confirm before committing).
2. Build the dedicated venv (`05b_remote.md`§3) — **cannot** reuse `~/dev/openvla/.venv`.
3. Push this repo to a remote checkout on `/cpfs01` (neither `devolremote` nor `gpu245` has one
   yet) and submit a LoRA fine-tune (`--use_l1_regression True --use_film False
   --num_images_in_input 1 --lora_rank 32`, `--use_proprio` undecided — see `Ah_for_human.md`).
   Needs its own `submit_finetune.py`-equivalent (`05b_remote.md`§4) and GPU allocation
   re-confirmed at submit time (both machines were fully occupied by other jobs as of
   2026-09-14 — check fresh with `gqueue -u all -s Running` + `nvidia-smi` on both before
   assuming either is free).
4. Write the new `serve.py` analog (`05a_codemap.md`§3).
5. Re-run the pre-flight ladder, then the rollout — with client-side timing instrumentation this
   time (`04_plan.md`§2 step 5).

## 3. Open / blocked

- **Nothing blocks step 1's continuation** (the venv build), but steps 3 onward need GPU
  allocation, which must be re-checked fresh, not assumed from this doc.
- **`--use_proprio` is an open decision**, not yet made — see `Ah_for_human.md`§1 Q1.
- **This is a scope-limited experiment** (`01_overview.md`§2): if the fine-tune or the new
  serving code turns out substantially harder than expected, or the measured control frequency
  still isn't usable, the plan is to drop this rather than keep pushing — flag that decision
  point rather than grinding through it unilaterally if it comes up.
