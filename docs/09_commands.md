# 09 — Commands

<!-- Lazy quick-commands reference: actual commands to type, not paraphrases. -->

## 1. Repo basics

```bash
# Already done (2026-09-14) -- reference only
gh repo fork moojink/openvla-oft --org DevolRobots
git clone git@github-devol:DevolRobots/openvla-oft.git ~/dev/openvla-oft
cd ~/dev/openvla-oft
git remote add upstream https://github.com/moojink/openvla-oft.git
git checkout -b ting/dev && git push -u origin ting/dev

# See what's been ported so far vs. what the sibling openvla project has, if re-deriving anything
cd ~/dev/openvla && git diff upstream/main -- prismatic/vla/datasets/rlds/oxe/{configs,transforms,materialize}.py

# Pull upstream OFT fixes into main, then rebase ting/dev (same pattern as openvla/openpi)
git fetch upstream && git checkout main && git merge upstream/main
git checkout ting/dev && git rebase main
```

## 2. Build the venv (see `05b_remote.md`§3 for the full gotcha list first)

```bash
export UV_CACHE_DIR=/cpfs01/wutingsh/.cache/uv
cd /cpfs01/wutingsh/openvla-oft   # after cloning the repo there
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
# Verify with a real import, not just a clean exit code (and check tensorflow_datasets/dlimp too,
# not just torch/transformers -- a protobuf/tensorflow-metadata mismatch broke that import once,
# docs/08r_gpt_review.md#2.1; pyproject.toml now pins compatible versions):
.venv/bin/python -c "
import torch, transformers, tensorflow_datasets, dlimp, wandb
print(torch.__version__, transformers.__version__, torch.cuda.is_available())
"
```

## 3. Fine-tuning — **RLDS builds done; resubmission paused, see `Ah_for_human.md`§3 Q6/Q7**

All recipe decisions are settled (`Ah_for_human.md`§1–2, `08r_gpt_review.md`): all three tasks in
`h0_inputs.md`§2, run as three separate fine-tunes, serially, one GPU each; native 30 Hz
(`NUM_ACTIONS_CHUNK=30`); dataset names `openvla_oft_flexiv_dualarm_stackboxes` /
`openvla_oft_flexiv_leftarm_stackboxes` / `openvla_oft_flexiv_dualarm_dinrail`. **All three RLDS
builds exist** on `/cpfs01/wutingsh/rlds224` (2026-09-15, `07_HISTORY.md`).

**Real measured throughput: ~1.58 it/s on an H200, i.e. a full 200,000-step run takes ~35h, not
the ~24h originally estimated** (that estimate came from the sibling project's
stride-5/`NUM_ACTIONS_CHUNK=8` numbers; native-rate data + `NUM_ACTIONS_CHUNK=30` costs more per
step for real). The first attempt (job 411, `dualarm_stackboxes`) used the old `--time 24:00:00`
default, got killed by the wall-time limit at step 138,462/200,000, and cascaded (gbatch's
default auto-cancel-on-dependency-failure) into cancelling the other two queued jobs, which never
ran. **Nothing is currently queued.** 13 checkpoints (steps 10k–130k) survive from job 411.

`scripts/submit_finetune.py` + `openvla_train.sh` (ported from the sibling project's queue-safe
launcher, `08r_gpt_review.md`#2.5) exist and work — verified with `--dry-run` and with a real
submission. `submit_finetune.py` also has `--depends-on <job_id>` (chains a submission after
another via `gbatch`'s own dependency mechanism — auto-cancels if the dependency fails) and
prints a `JOB_ID=<n>` line for scripting the next `--depends-on` off of.

```bash
# Dry-run only, to see the exact recipe/env/gbatch command without submitting anything:
python3 scripts/submit_finetune.py --gpus 1 --dataset-name openvla_oft_flexiv_dualarm_stackboxes --dry-run

# Chaining three jobs serially (adjust --time per Ah_for_human.md§3 Q6 before really doing this):
python3 scripts/submit_finetune.py --gpus 1 --time 48:00:00 --dataset-name openvla_oft_flexiv_dualarm_stackboxes --tag stackboxes
# -> parse JOB_ID from the output, then:
python3 scripts/submit_finetune.py --gpus 1 --time 48:00:00 --dataset-name openvla_oft_flexiv_leftarm_stackboxes --tag leftarm_stackboxes --depends-on <job1_id>
python3 scripts/submit_finetune.py --gpus 1 --time 48:00:00 --dataset-name openvla_oft_flexiv_dualarm_dinrail --tag dinrail --depends-on <job2_id>

# To RESUME job 411 from its 130k checkpoint instead of restarting from scratch (saves ~22h of
# already-completed compute) -- vla-scripts/finetune.py derives the run ID from --vla_path when
# --resume is set, so this continues writing into the SAME run directory:
#   --vla_path "/cpfs01/wutingsh/openvla_oft_runs/openvla-7b+openvla_oft_flexiv_dualarm_stackboxes+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--openvla_oft_ft_20260915_180339_3fe7a0_stackboxes--130000_chkpt" \
#   --resume True --resume_step 130000
# submit_finetune.py doesn't expose --resume/--resume_step/a custom --vla_path yet -- either add
# them, or invoke openvla_train.sh's underlying torchrun command directly under gbatch by hand.

# GPU allocation must be re-confirmed fresh before submitting anything (gqueue -u all -s Running +
# nvidia-smi on BOTH devolremote and gpu245 -- gpu245 had a genuinely free GPU last time,
# devolremote did not; ginfo's idle count isn't trustworthy on either machine).
```

`--use_proprio False` and `NUM_ACTIONS_CHUNK=30` are baked into `openvla_train.sh`'s/
`constants.py`'s defaults already, nothing to pass explicitly. `MERGE_LORA_DURING_TRAINING`
defaults to `False` in `openvla_train.sh` (`08r_gpt_review.md`#3.1 — merging at every 10k-step
save can consume 300GB+ of shared `/cpfs01` for one run); pass `--merge-lora-during-training` to
`submit_finetune.py` only if you specifically want that.

## 4. Serving (not yet written)

No `serve.py` exists in this repo yet (`04_plan.md` step 4). Once written, it should mirror the
sibling project's own invocation shape:

```bash
python -m deployment.flexiv_dualarm.serve \
  --ckpt-dir <MERGED OR ADAPTER+BASE CHECKPOINT DIR> \
  --dataset-dir /cpfs01/data/devol/lerobot_vla_jepa/batch_20260902_145405_flexiv_action_superset_3cam \
  --task "Insert the Ethernet connector" \
  --host 127.0.0.1 --port 9000 --debug
```
