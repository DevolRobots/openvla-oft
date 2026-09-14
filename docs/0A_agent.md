# 0A — Agent working notes

<!-- The agent entry point. Active working notes, current status, handoff context. For the
     file index, see 00_README.md (do not duplicate the table here). For shared cross-workstream
     codebase truths, see AA_intra_project_context.md if present. -->

## 1. Status & next task (2026-09-14)

**Start here, then read in this order:** `01_overview.md` (what/why) → `04_plan.md` (the 5-step
plan, step 1 done) → `05a_codemap.md` (exactly what code changed so far) → `05b_remote.md`
(hosts, paths, the venv gotchas) → `09_commands.md` (actual commands to run).

**Immediate next action: `04_plan.md` step 2 — build the venv.** Step 1 (dataset registration
port) is done but **uncommitted** in this repo's working tree (`git status` in
`~/dev/openvla-oft` to see the 4-file diff: `prismatic/vla/constants.py`,
`prismatic/vla/datasets/rlds/oxe/{configs,transforms,materialize}.py`). Confirm with the user
before committing — the working convention on the sibling `openvla` project has been to review
diffs before committing, not commit unprompted.

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

- **`--use_proprio` for the fine-tune** — not decided (`Ah_for_human.md`§1 Q1).
- **GPU allocation** — neither `devolremote` nor `gpu245` had a free GPU as of 2026-09-14 when
  last checked (in the sibling project's session); re-check fresh with `gqueue -u all -s
  Running` + `nvidia-smi` on both, don't assume either is free from anything in these docs.
- **No remote checkout of this repo exists yet** on `/cpfs01` — needed before venv build or
  fine-tuning (`05b_remote.md`§1).
- **This is a scope-limited experiment, not a committed direction** (`01_overview.md`§2,
  `04_plan.md`§3) — if step 3 (fine-tune) or step 4 (new serving code) turns out substantially
  harder than expected, or the measured control frequency still isn't usable once tested, flag
  that as a real decision point for the user rather than continuing to push through it.
