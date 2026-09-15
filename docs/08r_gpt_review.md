# 08r — GPT pre-fine-tuning review

<!-- One-off review performed before any OpenVLA-OFT fine-tuning was submitted. Findings are
     ordered by severity. This records the state observed on 2026-09-15; re-check mutable remote
     state before acting on it. -->

## 1. Verdict

**Do not submit a fine-tuning run yet.** The custom Flexiv registration is internally consistent,
but the remote environment cannot currently import the RLDS stack, the requested task scope does
not match the configured dataset, and the inherited dataset conversion makes assumptions that are
invalid for action chunks. The launch and checkpoint configuration also need to be made safe for
the shared cluster before consuming GPU time.

## 2. Blocking findings

### 2.1. The remote environment cannot import `tensorflow_datasets`

The dedicated venv exists at `/cpfs01/wutingsh/openvla-oft/.venv`, but a direct import check on
`devolremote` failed before reaching training:

```text
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

Installed versions at the time of the check were:

```text
protobuf==4.25.9
tensorflow-metadata==1.21.0
tensorflow-datasets==4.9.3
tensorflow==2.15.0
wandb==0.28.0
torch==2.2.0
transformers==4.40.1
```

`tensorflow-metadata==1.21.0` generated code expects an API absent from the installed protobuf.
This is the same dependency failure already investigated in the sibling `openvla` project. Its
verified compatible pins are:

```toml
"tensorflow-metadata==1.16.1",
"protobuf>=4.21.6,<4.26",
```

This repo currently specifies only an unconstrained `protobuf` in `pyproject.toml` and does not
pin `tensorflow-metadata`. Apply the verified constraints, reinstall, and require a successful
real import of `prismatic`, `tensorflow_datasets`, `dlimp`, and `wandb` before continuing.

### 2.2. The requested tasks do not match the configured run

The new task list in [`h0_inputs.md`§2](h0_inputs.md#2-task-list-2026-09-14) contains three raw
LeRobot datasets:

| Source batch | Canonical task string | Shape |
|---|---|---|
| `batch_20260909_203200_flexiv_action_superset_3cam` | `Stack realsense boxes with both arm` | dual arm |
| `batch_20260910_170153_flexiv_action_superset_3cam` | `Stack realsense boxes with only left arm` | dual-arm schema; right arm idle |
| `batch_20260904_102647_flexiv_action_superset_3cam` | `mount the wire connector onto the DIN rail` | dual arm |

All three raw paths exist on `/cpfs01`. However, the sample command in
[`09_commands.md`§3](09_commands.md#3-fine-tuning-once-the-venv-exists--not-yet-run) still selects
`devol_flexiv_dualarm`, which is the old Ethernet-insertion dataset. Submitting it unchanged would
train the wrong task.

Remote path checks found only the old converted dataset:

```text
/cpfs01/wutingsh/rlds224/devol_flexiv_dualarm/1.0.0     exists
/cpfs01/wutingsh/rlds224/devol_flexiv_dualarm_stackboxes absent
```

The code registers `devol_flexiv_dualarm_stackboxes` for the first box-stacking task, but no RLDS
build exists yet. There are no distinct builder or registration names for the left-arm-only and
DIN-rail tasks. Each task needs its own self-describing RLDS dataset name and run output name; do
not distinguish tasks only by changing `data_root_dir` under one reused name.

The first prompt in `h0_inputs.md` is written as `Stack the boxes`, but the source metadata used by
training says `Stack realsense boxes with both arm`. Serving requires an exact task-string match,
so the source metadata string above is canonical.

### 2.3. Interior no-op deletion breaks fixed-rate action chunks

OpenVLA-OFT constructs each target as the current action plus
`NUM_ACTIONS_CHUNK - 1` future actions in `prismatic/vla/datasets/datasets.py`. With the Flexiv
profile, that is an eight-action sequence.

The reused converter deletes interior no-op timesteps. Its own comment at
`../openvla/scripts/data_conversion/devol_flexiv_dualarm/devol_flexiv_dualarm_dataset_builder.py`
states that this is safe **precisely because vanilla OpenVLA is a single-step policy**, with no
chunk or history for a gap to break. That premise is false for this project: after deletion, the
OFT loader concatenates retained actions across gaps, so a target chunk no longer necessarily
represents eight consecutive fixed-rate controls.

The existing Ethernet build removed 759 of approximately 28,200 candidate transitions. Each
interior deletion can affect as many as seven preceding chunk windows, so the exposure is not
limited to 2.7% of training targets. For OFT datasets, preserve interior timesteps (the existing
converter supports `--no-noop-filter`) or redesign filtering so chunks never cross a removed
timestep. Do not reuse the existing filtered Ethernet RLDS build as-is.

### 2.4. The stride must be chosen from the execution cadence, not inherited from vanilla OpenVLA

Stride 5 was derived for vanilla OpenVLA because one model query produced one action and the
expected query rate was approximately 6 Hz against 30 Hz demonstrations. The sibling analysis
explicitly confirmed stride 5 on the condition that the client queried every step with
`action_chunk_size: 1`.

OFT changes that condition: one query returns several actions, so inference frequency and action
execution frequency are no longer the same quantity. A stride-5 action represents approximately
167 ms of demonstrated motion. Executing those actions at 30 Hz would apply approximately five
times the intended motion per second; executing them at 6 Hz would preserve the demonstrated
timescale.

Settle the client-side action-stream cadence before conversion and training. Then choose stride
and chunk length together. In particular, re-evaluate whether the intended recipe is native 30 Hz
data with a larger chunk, or stride-5 data explicitly streamed at 6 Hz. The new `openpi` task
configs use native-rate 30-action horizons, but that does not by itself prove which cadence the
new OFT serving path should use.

### 2.5. No queue-safe submission path exists yet

The remote checkout exists and is clean on `ting/dev`, but the following launch prerequisites are
absent:

```text
/cpfs01/wutingsh/openvla-oft/.env
/cpfs01/wutingsh/openvla-oft/scripts/submit_finetune.py
/cpfs01/wutingsh/openvla-oft/openvla_train.sh
```

The sample `torchrun` command in [`09_commands.md`](09_commands.md) must not be run directly on a
shared GPU. Port the sibling project's queue-safe launcher before submission. It needs to retain
the established `gbatch`, `runuser -u wutingsh`, environment re-export, secret-file, unique-run-ID,
`tee`, and `pipefail` protections.

## 3. High-risk configuration findings

### 3.1. The documented checkpoint combination can consume more than 300 GB

The sample command sets `save_latest_checkpoint_only=False`. The trainer defaults to
`max_steps=200000`, `save_freq=10000`, and `merge_lora_during_training=True`. Together these
produce 20 checkpoint directories, each containing a merged 7B model in addition to the LoRA
adapter and continuous action head. At approximately 15 GB per merged model, the merged copies
alone can exceed 300 GB.

At review time, `/cpfs01` was 97% full with 652 GB available. Consuming roughly half the remaining
shared capacity for intermediate merged copies is unsafe even if the run technically fits at the
start.

Save the LoRA adapter and action head at the approved 10,000-step interval with
`merge_lora_during_training=False`, then merge only selected checkpoints offline. If preserving
multiple checkpoints, ensure both the adapter and separate action-head state are retained; the
adapter alone is not a complete OFT checkpoint.

### 3.2. The optimization budget is still implicit

The command leaves GPU count and per-device batch size as placeholders and silently inherits:

- `batch_size=8` per device;
- `max_steps=200000`;
- learning-rate decay at step 100,000;
- no validation set;
- no warmup.

Those values are consequential and should not be accepted merely because they are defaults. The
effective batch is `batch_size * number of GPUs * grad_accumulation_steps`, and each new dataset
has a different number of transitions. Derive and log the effective batch, approximate epochs,
maximum steps, LR-decay step, wall-time limit, and stopping criterion separately for every task.

The RLDS builders currently expose no validation split, so training L1 loss is not evidence of
generalization. If no holdout will be created, record on-robot evaluation as the actual selection
criterion and avoid allowing a nominal 200,000-step default to determine the stopping point.

### 3.3. The environment is not reproducibly locked

`pyproject.toml` installs the patched `transformers` and `dlimp` packages from moving Git branch
heads, and the repo has no lockfile. The remote venv currently contains these revisions:

```text
transformers-openvla-oft  bc339d9ad707454c0c115970db43c260067c61ab
dlimp_openvla             040105d256bd28866cc6620621a3d5f7b6b91b46
```

Those matched the repositories' current heads on 2026-09-15, but a future rebuild can silently
resolve different code. Pin tested revisions or create a committed lock before the real run.

`flash_attn` is also absent even though upstream `SETUP.md` asks for it for training. The trainer
does not explicitly request FlashAttention, so absence is not established as a correctness bug;
verify the resolved attention implementation and benchmark a short synthetic step before deciding
whether installing it is worthwhile.

## 4. Documentation drift

The active docs say that no remote checkout or venv exists, but both now exist. They also describe
the Ethernet-insertion experiment as the immediate run while the latest human input lists three
new tasks. The generated index still calls the already-answered proprio and chunk-size questions
open.

Update [`0A_agent.md`](0A_agent.md), [`04_plan.md`](04_plan.md),
[`05b_remote.md`](05b_remote.md), [`06_current.md`](06_current.md), and
[`09_commands.md`](09_commands.md) after the task scope and data cadence are decided. Until then,
this review is the authoritative preflight status, not the older "next: fine-tune" wording.

## 5. Checks that passed

- The custom 14-D action layout matches the sibling converter: per arm,
  `[delta_xyz(3), delta_rotvec(3), gripper(1)]`.
- The absolute-action and normalization masks correctly repeat six relative dimensions plus one
  unnormalized absolute gripper dimension for each arm.
- Flexiv platform detection selected `NUM_ACTIONS_CHUNK=8`, `ACTION_DIM=14`, `PROPRIO_DIM=16`, and
  `BOUNDS_Q99` in a direct smoke test.
- The custom Python sources passed `python3 -m compileall`.
- Local `main`, `origin/main`, and freshly fetched `upstream/main` all resolve to `e4287e9`; no
  newer upstream training changes were missing.
- Local and remote `ting/dev` resolve to `68c3462`; the remote working tree was clean.
- Critical non-RLDS imports succeeded remotely: PyTorch `2.2.0+cu121`, patched Transformers
  `4.40.1`, W&B `0.28.0`, and CUDA availability all loaded successfully.
- All three requested raw LeRobot source directories exist.

The local worktree was not changed by the review except for the human's pre-existing edits to
`h0_inputs.md` and this review document/index addition.

## 6. Required decisions and recommended order

Before implementation, confirm whether the intended deliverable is three separate fine-tunes or
only the dual-arm box-stacking run. Then proceed in this order:

1. Decide the action execution cadence and derive stride plus chunk length together.
2. Add distinct RLDS builders/registrations and build the requested datasets without interior
   no-op deletion.
3. Run the full source-to-RLDS verification for each build, extended to validate chunk continuity.
4. Pin the compatible environment and require all critical imports to pass.
5. Add the queue-safe launcher and explicit per-task optimization recipes.
6. Disable in-training merges, estimate artifact size, and re-check `/cpfs01` capacity.
7. Run a short queued smoke test before submitting any full fine-tune.
