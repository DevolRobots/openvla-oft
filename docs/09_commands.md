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
# Verify with a real import, not just a clean exit code:
.venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

## 3. Fine-tuning (once the venv exists — not yet run)

Sample command, adapted from upstream `ALOHA.md`'s recipe shape and this project's own
`FLEXIV_CONSTANTS` (`05a_codemap.md`). **Do not run directly on `devolremote`/`gpu245` — GPU jobs
must go through `gbatch`**; this needs its own submitter script first
(`05b_remote.md`§4), analogous to the sibling project's `scripts/submit_finetune.py`.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /cpfs01/wutingsh/rlds224 \
  --dataset_name devol_flexiv_dualarm \
  --run_root_dir /cpfs01/wutingsh/openvla_oft_runs \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --lora_rank 32 \
  --batch_size <PER-DEVICE, see finetune.py default> \
  --learning_rate 5e-4 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --wandb_entity "<SAME AS SIBLING PROJECT'S openvla_train.sh, IF REUSING>" \
  --wandb_project "<SAME>" \
  --run_id_note flexiv_dualarm--oft--l1_regression--8_acts_chunk--no_film
```

`--use_proprio False` above is confirmed (`Ah_for_human.md`§1 Q1, decided 2026-09-15) — matches
the sibling project's existing behavior (proprio declared in the schema but never fed to the
model).

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
