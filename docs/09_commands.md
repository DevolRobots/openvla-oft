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

## 3. Fine-tuning — **decisions made, still blocked on RLDS builds**

All open recipe decisions are now settled (`Ah_for_human.md`§1–2, `08r_gpt_review.md`): task scope
is **all three tasks in `h0_inputs.md`§2, run as three separate fine-tunes, serially, one GPU each**
(not concurrently); action-execution cadence is **native 30 Hz** (`NUM_ACTIONS_CHUNK=30` in
`constants.py`, not the old stride-5-at-6Hz); dataset names are `openvla_oft_flexiv_dualarm_stackboxes`
/ `openvla_oft_flexiv_leftarm_stackboxes` / `openvla_oft_flexiv_dualarm_dinrail`
(`05a_codemap.md`). **Still blocking a real submission:** no RLDS build exists yet for any of the
three at native rate — the sibling `openvla` repo's converter needs to run with its no-op filter
disabled (`08r_gpt_review.md`§2.3, env var `DEVOL_NOOP_FILTER=0`) and at native (not stride-5)
sampling, and (for the box-stacking task) its `DevolFlexivDualarmStackboxes` builder class
needs the matching rename noted in `05a_codemap.md`.

`scripts/submit_finetune.py` + `openvla_train.sh` (ported from the sibling project's queue-safe
launcher, `08r_gpt_review.md`#2.5) exist and work — verified with `--dry-run` below.

```bash
# Dry-run only, to see the exact recipe/env/gbatch command without submitting anything:
python3 scripts/submit_finetune.py --gpus 1 --dataset-name openvla_oft_flexiv_dualarm_stackboxes --dry-run

# Once each task's RLDS build exists, drop --dry-run and submit the three ONE AT A TIME (Q3:
# serially, 1 GPU each — wait for one to finish/checkpoint before starting the next, don't queue
# all three concurrently). GPU allocation must be re-confirmed fresh (gqueue -u all -s Running +
# nvidia-smi on both machines) before each submission.
python3 scripts/submit_finetune.py --gpus 1 --dataset-name openvla_oft_flexiv_dualarm_stackboxes
python3 scripts/submit_finetune.py --gpus 1 --dataset-name openvla_oft_flexiv_leftarm_stackboxes
python3 scripts/submit_finetune.py --gpus 1 --dataset-name openvla_oft_flexiv_dualarm_dinrail
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
