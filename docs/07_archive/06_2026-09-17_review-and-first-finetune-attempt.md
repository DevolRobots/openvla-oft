# 06 — Current task

<!-- Coherent description of the CURRENT active task. Not a dumping ground. If idle, write
     "between tasks — last completed X". Snapshot into 07_archive/06_YYYY-MM-DD_<tag>.md
     before a major pivot. -->

## 1. Active task (2026-09-17, snapshot before archive)

**Get an OpenVLA-OFT checkpoint fine-tuned and served on the Flexiv dual-arm task family, to test
whether action chunking fixes the control-frequency problem found with vanilla OpenVLA.**
Background: [`01_overview.md`](01_overview.md), full plan: [`04_plan.md`](04_plan.md). Task scope
widened 2026-09-14/15 from one dataset (Ethernet insertion) to three (`h0_inputs.md`§2:
dual-arm box-stacking, left-arm-only box-stacking, DIN-rail wire-connector).

**Steps 1–2 done** (dataset registration port, committed `e9b7e09`; venv built+verified on both
`devolremote`/`gpu245`). **Step 3 (submit fine-tune) is in progress but currently paused**, not
by any open decision — everything was decided and executed — but by an operational setback plus
a shift in priorities.

## 2. What happened, in order

1. A pre-fine-tuning review (`08r_gpt_review.md`, 2026-09-15) found: the environment couldn't
   actually import `tensorflow_datasets` (protobuf/`tensorflow-metadata` mismatch); the task list
   had grown to 3 datasets with no matching code; the inherited no-op-filtered RLDS conversion is
   unsafe for action chunks; the stride/cadence assumption inherited from vanilla OpenVLA doesn't
   hold for OFT; no queue-safe submitter existed. Verdict: do not submit yet.
2. Fixed the code/config-level findings: pinned `tensorflow-metadata`/`protobuf`
   (reinstalled+verified remotely), pinned `transformers`/`dlimp` git-fork revisions, ported
   `scripts/submit_finetune.py` + `openvla_train.sh` (queue-safe `gbatch` launcher,
   `merge_lora_during_training=False` by default). Committed+pushed as `23c8ab2`.
3. `Ah_for_human.md`§2 Q3–Q5 answered: all 3 tasks, run serially/1 GPU each; cadence is native
   30 Hz (`NUM_ACTIONS_CHUNK` `8`→`30`); dataset names renamed to
   `openvla_oft_flexiv_dualarm_stackboxes` / `_leftarm_stackboxes` / `_dualarm_dinrail`, matching
   the sibling `openpi` project's convention. Also committed+pushed as part of `23c8ab2`.
4. Built all three RLDS datasets in the sibling `~/dev/openvla` repo, at native 30 Hz with no-op
   filtering off: renamed its `devol_flexiv_dualarm_stackboxes` builder to match, created two new
   builder packages from scratch. **Uncommitted** in that repo (both local and remote checkouts).
5. Copied that repo's `.env` (W&B key) to this repo, un-read, at the human's instruction — same
   W&B project/entity as the sibling project, confirmed working.
6. Extended `submit_finetune.py` with `--depends-on` (chains `gbatch` jobs serially without
   polling) — **uncommitted** in this repo.
7. Submitted all three fine-tunes to `gpu245` (the only machine with a free GPU at the time),
   chained: job 411 (`dualarm_stackboxes`) → 412 (`leftarm_stackboxes`, depends on 411) → 413
   (`dualarm_dinrail`, depends on 412). Each `--gpus 1 --time 24:00:00`.
8. **Job 411 hit that 24h limit and was killed (`Timeout`) at step 138,462/200,000 (69.2%).**
   Real measured throughput was ~1.58 it/s — a full 200k-step run needs ~35h, not the ~24h
   estimated from the sibling project's stride-5/8-chunk numbers. Jobs 412/413 auto-cancelled via
   gbatch's dependency-failure cascade and never ran. 13 checkpoints (steps 10k–130k) survive.
9. Noticed the timeout mid-run, asked for permission to cancel+resubmit with a longer `--time`
   (permission-blocked, needs explicit approval), got no answer before the human stepped away.
   **Nothing currently queued.** Human returned, said other priorities came first — pausing here,
   not abandoning.

## 3. Next steps, when priorities allow

1. Answer `Ah_for_human.md`§3 Q6 (new `--time`; resume job 411 from its 130k checkpoint, or
   restart clean) and Q7 (commit the uncommitted code in both repos, or hold for review).
2. Resubmit per `09_commands.md`§3 — re-check GPU allocation fresh first on both machines.
3. Write the new `serve.py` analog (`05a_codemap.md`§3) once all three fine-tunes are done.
4. Re-run the pre-flight ladder, then the rollout, with client-side timing instrumentation
   (`04_plan.md`§2 step 5).

## 4. Open / blocked

- `Ah_for_human.md`§3 Q6/Q7 — see above.
- GPU allocation must be re-checked fresh, not assumed from this snapshot's date.
- **This is a scope-limited experiment** (`01_overview.md`§2): if the fine-tune or the new
  serving code turns out substantially harder than expected, or the measured control frequency
  still isn't usable, the plan is to drop this rather than keep pushing — flag that decision
  point rather than grinding through it unilaterally if it comes up.
