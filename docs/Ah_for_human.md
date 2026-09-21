# Ah — Questions for the human (agent asks)

<!-- The agent writes questions; the human answers in-line. Group into sections (one per
     questioning session). Prefix each question subsection Q1., Q2., ... Put dates in headings. -->

## 1. Fine-tuning recipe decisions (2026-09-14)

### Q1.

Should the OpenVLA-OFT fine-tune use `--use_proprio True` (feed the 16-D proprioceptive state to
the model) or `--use_proprio False` (matching the sibling `openvla` project's existing behavior,
where the schema declares state fields but the model never sees them)? Enabling it would be a new
capability, not parity with the existing checkpoint — recommend leaving it `False` for a
like-for-like comparison unless there's a specific reason to add it now.

> leave it false for now.

### Q2.

`NUM_ACTIONS_CHUNK=8` was picked as a starting point (\~1.3 s of open-loop motion per query at
this dataset's \~6 Hz effective step rate — see `05a_codemap.md`). Any preference for a different
starting chunk size before the first fine-tune, or is 8 fine to try first and adjust later based
on measured inference latency?

> no, 8 is fine.

## 2. Post-review decisions (2026-09-15)

A pre-fine-tuning review (`08r_gpt_review.md`) checked the environment, the new task list in
`h0_inputs.md`§2, and the inherited data-conversion assumptions before anything gets submitted.
Verdict: **do not submit yet.** Two of its blocking findings are code/config fixes and are
already applied (protobuf/`tensorflow-metadata` pin, verified with a real remote import; a
queue-safe `submit_finetune.py`/`openvla_train.sh`; pinned git revisions for the `transformers`
fork and `dlimp`; new dataset registrations for the two not-yet-built tasks below). Three need a
call only you can make:

### Q3.

**Task scope.** `h0_inputs.md`§2 lists three source batches/tasks (dual-arm box-stacking,
left-arm-only box-stacking, DIN-rail wire-connector). Only `devol_flexiv_dualarm_stackboxes`
(dual-arm box-stacking) has code + an existing sibling-repo builder ready to point at data — no
RLDS build exists yet for any of the three on `/cpfs01`, and the other two only got dataset
registrations added in this repo just now (`devol_flexiv_leftarm_stackboxes`,
`devol_flexiv_dualarm_dinrail` — names are my choice, not yet confirmed by you, see Q5). Is the
intended deliverable **all three as separate fine-tunes**, or **just the dual-arm box-stacking
run** first (matching the "Stack the boxes" framing at the top of `h0_inputs.md`§2), with the
other two deferred?

> All three as seperate fine-tunes, but serially; use 1 GPU at a time.

### Q4.

**Action execution cadence.** Stride 5 (of the source 30 Hz data) was inherited from vanilla
OpenVLA, where it was chosen because one model query = one action at \~6 Hz. OFT breaks that
assumption: one query now returns a `NUM_ACTIONS_CHUNK`-length chunk, so query rate and action
*execution* rate are no longer the same number — executing a stride-5 action stream at the
robot's native 30 Hz would apply \~5x the intended motion per second, vs. 6 Hz which preserves the
demonstrated timescale (`08r_gpt_review.md`#2.4). This needs to be settled **before** re-running
data conversion for the new tasks, since stride and `NUM_ACTIONS_CHUNK` should be chosen together.
Two shapes to choose between: (a) keep stride-5 data, stream actions client-side at 6 Hz (closest
to today's default); or (b) convert at native 30 Hz with a larger chunk length. Which one, or do
you want to see numbers first (e.g. a quick client-timing check) before deciding?

> option (b) as recommended.

### Q5.

**Naming for the two new dataset registrations.** I picked `devol_flexiv_leftarm_stackboxes`
(source: `batch_20260910_170153_..._3cam`, "Stack realsense boxes with only left arm") and
`devol_flexiv_dualarm_dinrail` (source: `batch_20260904_102647_..._3cam`, "mount the wire
connector onto the DIN rail"), following the existing `devol_flexiv_dualarm[_stackboxes]`
convention — same schema/transform as the other two, just a distinct self-describing name per
`08r_gpt_review.md`#2.2. Fine as-is, or rename before anyone builds RLDS data or starts a run
under these names (renaming after that means re-registering or migrating a build)?

> do whatever is the most similar to the existing convention used in `openpi`. A prefix of `openvla_oft_` would be nice too.

## 3. Training resubmission (2026-09-16/17) — paused, needs a decision when priorities allow

`07_HISTORY.md`'s 2026-09-16/17 entries have the full story. Short version: job 411 (dual-arm
box-stacking) hit its `--time 24:00:00` limit at step 138,462/200,000 (measured \~1.58 it/s, so a
full run actually needs \~35h, not the \~24h originally estimated) and was killed; jobs 412/413
auto-cancelled via dependency-failure cascade without ever running. Nothing is currently queued.
13 checkpoints (steps 10k–130k) survive from job 411.

### Q6.

**New `--time` for resubmission, and resume vs. restart for job 411.** Proposed: `--time
48:00:00` (margin over the measured \~35h need) for all three, and resume job 411 from its
130,000-step checkpoint (`--resume True --resume_step 130000 --vla_path
<run_dir>--130000_chkpt`, `09_commands.md`§3) rather than restarting from scratch — saves \~22h of
already-completed compute. Confirm both, or adjust (e.g. a different `--time`, or restart 411
clean instead of resuming, or reduce `--max-steps` given the review's own note that the
optimization budget was never really decided per-task, `08r_gpt_review.md`#3.2).

> 48h confirmed.

### Q7.

**`scripts/submit_finetune.py`'s `--depends-on` addition and the sibling `~/dev/openvla` repo's
new/renamed builder packages are both still uncommitted** (in both local and remote checkouts of
each repo) — deliberately, per the "confirm before committing" convention, since a lot happened
in one unattended stretch. OK to commit+push both now, or hold for review first?

> Yes — committed (`def4ed3` in `~/dev/openvla`; the `submit_finetune.py` change was already
> committed as `791c47b` in this repo). Not pushed to remote yet.

## 4. Resubmission hold (2026-09-21)

> Hold off on resubmitting the three fine-tunes for now — more data and details are coming out
> of the `openpi` runs that may affect this. I'll fetch the data and details myself; don't submit
> jobs until then.
