# 0A — Agent working notes

<!-- The agent entry point. Active working notes, current status, handoff context. For the
     file index, see 00_README.md (do not duplicate the table here). For shared cross-workstream
     codebase truths, see AA_intra_project_context.md if present. -->

## 1. Status & next task (2026-09-17)

**Start here, then read in this order:** `01_overview.md` (what/why) → `04_plan.md` (the 5-step
plan, step 1–2 done) → `05a_codemap.md` (exactly what code changed) → `05b_remote.md` (hosts,
paths, venv gotchas) → `09_commands.md`§3 (fine-tuning commands + current blocker) →
`Ah_for_human.md`§3 (the open decision blocking the next step) → `06_current.md` (today's task).

**Steps 1–2 done** (dataset registration port, committed `e9b7e09`; venv, built+verified on both
`devolremote`/`gpu245`). **Step 3 (fine-tune) is IN PROGRESS, PAUSED — not blocked on decisions
anymore, blocked on priorities + one operational decision.**

All recipe decisions from the 2026-09-15 review are made and applied: task scope (all 3 tasks,
serial, 1 GPU each), cadence (native 30 Hz, `NUM_ACTIONS_CHUNK=30`), dataset naming
(`openvla_oft_flexiv_dualarm_stackboxes` / `_leftarm_stackboxes` / `_dualarm_dinrail`). All three
RLDS builds exist on `/cpfs01/wutingsh/rlds224` (2026-09-15). Three fine-tune jobs were submitted
to `gpu245` chained with `gbatch --depends-on` — **job 411 hit its 24h `--time` limit at step
138,462/200,000 (real throughput is ~1.58 it/s ⇒ ~35h needed, not ~24h) and was killed; jobs
412/413 auto-cancelled via the dependency-failure cascade. Nothing is currently queued.** 13
checkpoints (10k–130k) survive from job 411 — resumable, not a restart from scratch.

**Immediate next action, when priorities allow:** answer `Ah_for_human.md`§3 Q6 (new `--time`,
resume-vs-restart) and Q7 (commit the uncommitted code — see below), then resubmit per
`09_commands.md`§3.

**Uncommitted right now** (deliberately — a lot happened in one unattended stretch, holding for
review per the usual convention):
- This repo: `scripts/submit_finetune.py`'s `--depends-on` + job-ID-capture addition.
- The sibling `~/dev/openvla` repo: the renamed `openvla_oft_flexiv_dualarm_stackboxes` builder
  package and the two new ones (`_leftarm_stackboxes`, `_dualarm_dinrail`) — needed for the RLDS
  builds above to exist at all. That repo also has unrelated in-progress work of its own
  (`deployment/flexiv_dualarm/`) untouched by this project.

**Read the sibling project's own investigation before doing anything else**, if not already
familiar: `~/dev/openvla/docs/04s_openvla_oft_feasibility.md`. It's the source of nearly every
non-obvious decision baked into this repo's docs (chunk size, normalization type, why FiLM is
off, why proprio is undecided, why a new serve.py is needed rather than reusing upstream's
`deploy.py`).

## 2. Orientation

- File index: see `00_README.md`.
- Current task detail: `06_current.md`.
- Open questions: `Ah_for_human.md` (needs a human answer) / `hA_for_agent.md` (none yet).
- The sibling project this one depends on for data, motivation, and the serving contract to
  port: `~/dev/openvla` (its own `docs/` has the full history — dataset conversion, the
  discretization/gripper-hardware caveats that apply here too, the `DevolInference` client
  contract).

## 3. Open items

- ~~**`--use_proprio`, chunk size, task scope, cadence, dataset naming**~~ **all decided**
  (`Ah_for_human.md`§1–2) — see §1 above for the current values.
- **`Ah_for_human.md`§3 Q6 — new `--time` + resume-vs-restart for the killed job 411.** Blocks
  resubmission. Proposed: `--time 48:00:00`, resume from the 130k checkpoint.
- **`Ah_for_human.md`§3 Q7 — OK to commit the uncommitted code** (this repo's
  `submit_finetune.py` change; the sibling repo's new/renamed builder packages)?
- **GPU allocation must be re-checked fresh before resubmitting** — `gpu245` had a free GPU as of
  2026-09-15/16 (`devolremote` did not), but that was multiple days ago by the time this is picked
  back up; don't assume either machine's state from this doc.
- **This is a scope-limited experiment, not a committed direction** (`01_overview.md`§2,
  `04_plan.md`§3) — if step 3 (fine-tune) or step 4 (new serving code) turns out substantially
  harder than expected, or the measured control frequency still isn't usable once tested, flag
  that as a real decision point for the user rather than continuing to push through it.
