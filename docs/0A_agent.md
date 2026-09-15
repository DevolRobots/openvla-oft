# 0A — Agent working notes

<!-- The agent entry point. Active working notes, current status, handoff context. For the
     file index, see 00_README.md (do not duplicate the table here). For shared cross-workstream
     codebase truths, see AA_intra_project_context.md if present. -->

## 1. Status & next task (2026-09-15)

**Start here, then read in this order:** `01_overview.md` (what/why) → `04_plan.md` (the 5-step
plan, step 1 done) → `05a_codemap.md` (exactly what code changed so far) → `05b_remote.md`
(hosts, paths, the venv gotchas) → `09_commands.md` (actual commands to run) → **`08r_gpt_review.md`
(current no-go verdict — read before touching step 3)**.

Step 1 (dataset registration port) is done and committed (`e9b7e09`). **Step 2 (venv) is also
done** — built and verified (real import, not just clean exit) on both `devolremote` and `gpu245`
(same `/cpfs01` checkout, `ting/dev` @ `68c3462`).

**Immediate next action: NOT step 3 (submit fine-tune) yet — but all decisions are made, only
data prep is left.** A pre-fine-tuning review (`08r_gpt_review.md`, 2026-09-15) found the
environment couldn't import `tensorflow_datasets`, the task list changed (`h0_inputs.md`§2, added
while the venv build was in progress — 3 tasks now, not 1), the inherited no-op-filtered RLDS
conversion is unsafe for action chunks, and no queue-safe submitter existed. All fixed or decided
(uncommitted, this session):
- `pyproject.toml`: pinned `tensorflow-metadata==1.16.1` + `protobuf>=4.21.6,<4.26` (fixes the
  import failure — reinstalled + verified remotely with a real `import tensorflow_datasets,
  dlimp, wandb`, not just `torch`/`transformers`) and pinned the `transformers`/`dlimp` git-fork
  revisions instead of moving branch heads.
- `scripts/submit_finetune.py` + `openvla_train.sh`: ported the sibling project's queue-safe
  `gbatch` launcher, adapted for this repo's OFT flags; `merge_lora_during_training` defaults to
  `False` (disk safety, `08r_gpt_review.md`#3.1). Verified with `--dry-run`.
- **Task scope decided (`Ah_for_human.md`§2 Q3): all three tasks, as three separate fine-tunes,
  run serially, one GPU each** — not concurrently.
- **Cadence decided (Q4): native 30 Hz**, not the inherited stride-5-at-6Hz — matches the sibling
  `openpi` project's `action_horizon=30` on the same data. `constants.py`'s `NUM_ACTIONS_CHUNK`
  updated `8` → `30` accordingly.
- **Dataset naming decided (Q5):** renamed to match `openpi`'s convention with an `openvla_oft_`
  prefix — `openvla_oft_flexiv_dualarm_stackboxes` / `openvla_oft_flexiv_leftarm_stackboxes` /
  `openvla_oft_flexiv_dualarm_dinrail` (registered in `configs.py`/`transforms.py`, verified
  remotely). **Note:** the sibling `openvla` repo's existing `DevolFlexivDualarmStackboxes`
  builder class still has the OLD name — needs a matching rename there before
  `openvla_oft_flexiv_dualarm_stackboxes` resolves via TFDS (not done — out of this repo's scope,
  see `05a_codemap.md`).

**Still blocking a real submission, regardless of the decisions above:** no RLDS build exists yet
for any of the 3 tasks — the inherited converter's interior no-op deletion needs
disabling/redesigning first for OFT (`08r_gpt_review.md`§2.3 — env var `DEVOL_NOOP_FILTER=0` in
the sibling repo's converter, not yet verified sufficient on its own for chunk continuity), and
conversion needs to target native 30 Hz per the cadence decision above (not the old stride-5
default). This is sibling-repo (`~/dev/openvla`) data-conversion work, not something this repo's
code changes.

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

- ~~**`--use_proprio` for the fine-tune** — not decided~~ **Decided 2026-09-15: `False`**
  (`Ah_for_human.md`§1 Q1). ~~`NUM_ACTIONS_CHUNK=8`~~ **now `30`** (Q2 confirmed 8 as a starting
  point on 2026-09-14; superseded by the native-30Hz cadence decision on 2026-09-15, Q4).
- ~~**Task scope, cadence, dataset naming — open**~~ **All decided 2026-09-15**
  (`Ah_for_human.md`§2 Q3–Q5): all 3 tasks, serially, 1 GPU each; native 30 Hz; `openvla_oft_
  flexiv_*` names.
- **GPU allocation** — neither `devolremote` nor `gpu245` had a free GPU as of 2026-09-14 when
  last checked (in the sibling project's session); re-check fresh with `gqueue -u all -s
  Running` + `nvidia-smi` on both, don't assume either is free from anything in these docs.
- **No RLDS build exists yet for any of the 3 tasks in `h0_inputs.md`§2** on `/cpfs01` — code is
  ready (all three registered under their final names) but none have been converted from LeRobot
  yet, conversion needs to target native 30 Hz (not the old stride-5 default), and the existing
  converter's interior no-op deletion needs disabling/redesigning first for OFT
  (`08r_gpt_review.md`§2.3). This is the one remaining blocker before step 3.
- **This is a scope-limited experiment, not a committed direction** (`01_overview.md`§2,
  `04_plan.md`§3) — if step 3 (fine-tune) or step 4 (new serving code) turns out substantially
  harder than expected, or the measured control frequency still isn't usable once tested, flag
  that as a real decision point for the user rather than continuing to push through it.
