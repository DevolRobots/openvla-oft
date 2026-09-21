# 0A — Agent working notes

<!-- The agent entry point. Active working notes, current status, handoff context. For the
     file index, see 00_README.md (do not duplicate the table here). For shared cross-workstream
     codebase truths, see AA_intra_project_context.md if present. -->

## 1. Status & next task (2026-09-21)

**Start here, then read in this order:** `01_overview.md` (what/why) → `04_plan.md` (the 5-step
plan, step 1–2 done) → `05a_codemap.md` (exactly what code changed) → `05b_remote.md` (hosts,
paths, venv gotchas) → `09_commands.md`§3 (fine-tuning commands, on hold) →
`Ah_for_human.md`§3–4 (decisions + the current resubmission hold) → `06_current.md` (today's task).

**Steps 1–2 done** (dataset registration port, committed `e9b7e09`; venv, built+verified on both
`devolremote`/`gpu245`). **Step 3 (fine-tune) is ON HOLD by explicit user instruction (2026-09-21)
— do not resubmit jobs.** More data/details are coming out of the `openpi` runs that may change
the recipe here; the user is fetching that data and will give the go-ahead.

All recipe decisions from the 2026-09-15 review are made and applied: task scope (all 3 tasks,
serial, 1 GPU each), cadence (native 30 Hz, `NUM_ACTIONS_CHUNK=30`), dataset naming
(`openvla_oft_flexiv_dualarm_stackboxes` / `_leftarm_stackboxes` / `_dualarm_dinrail`). All three
RLDS builds exist on `/cpfs01/wutingsh/rlds224` (2026-09-15). Three fine-tune jobs were submitted
to `gpu245` chained with `gbatch --depends-on` — **job 411 hit its 24h `--time` limit at step
138,462/200,000 (real throughput is \~1.58 it/s ⇒ \~35h needed, not \~24h) and was killed; jobs
412/413 auto-cancelled via the dependency-failure cascade. Nothing is currently queued.** 13
checkpoints (10k–130k) survive from job 411 — resumable, not a restart from scratch.

**Resubmission plan when the hold lifts** (`Ah_for_human.md`§3 Q6/Q7, answered 2026-09-21):
`--time 48:00:00` for all three; resume job 411 from its 130,000-step checkpoint rather than
restarting. Do **not** act on this until the user explicitly says the `openpi`-derived data/details
are in and resubmission is cleared — check `Ah_for_human.md`§4 and this file's date first.

**Committed (2026-09-21):** `scripts/submit_finetune.py`'s `--depends-on` + job-ID-capture addition
was already committed here as `791c47b`. The sibling `~/dev/openvla` repo's builder packages
(`openvla_oft_flexiv_dualarm_stackboxes` rename, plus new `_leftarm_stackboxes` and
`_dualarm_dinrail`) are now committed there as `def4ed3`. **Neither has been pushed to remote
yet** — ask before pushing. That sibling repo also has unrelated in-progress work of its own
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
- ~~**`Ah_for_human.md`§3 Q6 — new `--time` + resume-vs-restart**~~ **answered 2026-09-21:**
  `--time 48:00:00`, resume job 411 from the 130k checkpoint.
- ~~**`Ah_for_human.md`§3 Q7 — OK to commit the uncommitted code?**~~ **done 2026-09-21:** committed
  in both repos, not pushed.
- **`Ah_for_human.md`§4 — resubmission is on hold** pending data/details from the `openpi` runs
  that the user is fetching. This is the actual current blocker, not the two above.
- **GPU allocation must be re-checked fresh before resubmitting**, whenever the hold lifts —
  `gpu245` had a free GPU as of 2026-09-15/16 (`devolremote` did not), but that's stale; don't
  assume either machine's state from this doc.
- **This is a scope-limited experiment, not a committed direction** (`01_overview.md`§2,
  `04_plan.md`§3) — if step 3 (fine-tune) or step 4 (new serving code) turns out substantially
  harder than expected, or the measured control frequency still isn't usable once tested, flag
  that as a real decision point for the user rather than continuing to push through it.
