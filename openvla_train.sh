#!/usr/bin/env bash
# openvla_train.sh -- OpenVLA-OFT LoRA fine-tuning command for the Flexiv dual-arm task family.
#
# This file is the single source of truth for the training FLAGS; it submits nothing. GPU jobs
# on devolremote/gpu245 must go through the gbatch queue, so launch via
#
#     python scripts/submit_finetune.py --gpus 2 --dataset-name <name>
#
# which wraps this script in a gbatch submission (runuser so output is not root-owned, env
# re-export, log tee). Running this script directly takes GPUs outside the queue -- do that only
# for a deliberate debug run on a machine you own.
#
# Ported from the sibling `openvla` project's openvla_train.sh (docs/05b_remote.md), adapted for
# this repo's OFT-specific vla-scripts/finetune.py flags (use_l1_regression, use_film,
# num_images_in_input, use_proprio, save_freq not save_steps, merge_lora_during_training, no
# adapter_tmp_dir -- OFT's finetune.py has no such flag). See docs/09_commands.md and
# docs/08r_gpt_review.md.
#
# Every value below is overridable by environment variable; the defaults are NOT yet a decided
# first run -- docs/Ah_for_human.md has open questions (dataset/task scope, action cadence) that
# must be answered before submitting for real. See docs/08r_gpt_review.md.
set -euo pipefail

GPUS="${GPUS:-1}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/cpfs01/wutingsh/rlds224}"
DATASET_NAME="${DATASET_NAME:-openvla_oft_flexiv_dualarm_stackboxes}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/cpfs01/wutingsh/openvla_oft_runs}"

# OFT recipe (docs/04_plan.md#2 step 3, decided 2026-09-15 in Ah_for_human.md#1):
USE_L1_REGRESSION="${USE_L1_REGRESSION:-True}"
USE_DIFFUSION="${USE_DIFFUSION:-False}"
USE_FILM="${USE_FILM:-False}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-1}"
USE_PROPRIO="${USE_PROPRIO:-False}"

LORA_RANK="${LORA_RANK:-32}"          # paper: rank has negligible effect, do not sweep
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
IMAGE_AUG="${IMAGE_AUG:-True}"

# --batch_size is PER DEVICE (each process builds its own DataLoader with it), so the effective
# global batch is BATCH_SIZE x GPUS x GRAD_ACCUM. finetune.py's own default (8/device) is the
# paper's LoRA recipe; scale down per-GPU only if you deliberately want a larger effective batch
# from extra GPUs rather than just more throughput.
BATCH_SIZE="${BATCH_SIZE:-8}"

# Checkpoint policy on this cluster: at most every 10000 steps (disk conservation). Merging is
# OFF by default -- docs/08r_gpt_review.md#3.1: merge_lora_during_training=True writes a full
# merged ~15GB 7B checkpoint at every save, which at the default save_freq/max_steps can exceed
# 300GB of /cpfs01 for one run. Merge selected checkpoints offline instead
# (merge_lora_weights_and_save.py).
SAVE_FREQ="${SAVE_FREQ:-10000}"
MAX_STEPS="${MAX_STEPS:-200000}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-False}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-False}"

# Unique per submission (submit_finetune.py sets this to the job's timestamped run name) so that
# two identical recipes -- including jobs launched independently on both GPU machines against
# shared /cpfs01 -- don't write to the same run_dir/adapter_dir. Empty by default for a bare
# manual invocation of this script; pass your own to stay unique.
RUN_ID_NOTE="${RUN_ID_NOTE:-}"

# The dataset holds on the order of tens of thousands of transitions (varies per task/batch), so
# finetune.py's 100k-frame default shuffle buffer costs several GB of RAM for no benefit once it
# exceeds the dataset size. Lower it per-dataset via SHUFFLE_BUFFER_SIZE if needed.
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-30000}"

# Same W&B target as the sibling openvla project's runs (matching entity keeps the same shared
# API key working -- the entity is NOT a preference, wandb.init fails if it doesn't match the key).
WANDB_PROJECT="${WANDB_PROJECT:-wutingsh-devol-robots}"
WANDB_ENTITY="${WANDB_ENTITY:-ryan-chong-devol-robots}"
export WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE}" == "online" ]]; then
  : "${WANDB_API_KEY:?WANDB_API_KEY must be set, or pass WANDB_MODE=offline}"
fi

# Echo the effective recipe into the log. Cheap, and it means a run's own log answers "what was
# this trained with" without cross-referencing the submitter or this file's git history.
cat <<EOF
=== openvla_train.sh effective config ===
  vla_path                     ${VLA_PATH}
  data_root_dir                ${DATA_ROOT_DIR}
  dataset_name                 ${DATASET_NAME}
  run_root_dir                 ${RUN_ROOT_DIR}
  gpus (nproc)                 ${GPUS}
  use_l1_regression/diffusion  ${USE_L1_REGRESSION} / ${USE_DIFFUSION}
  use_film                     ${USE_FILM}
  num_images_in_input          ${NUM_IMAGES_IN_INPUT}
  use_proprio                  ${USE_PROPRIO}
  lora_rank                    ${LORA_RANK}
  batch_size/device            ${BATCH_SIZE}   -> effective batch ${BATCH_SIZE} x ${GPUS} x ${GRAD_ACCUM} = $(( BATCH_SIZE * GPUS * GRAD_ACCUM ))
  learning_rate                ${LEARNING_RATE}
  image_aug                    ${IMAGE_AUG}
  max_steps                    ${MAX_STEPS}
  save_freq                    ${SAVE_FREQ}
  merge_lora_during_training   ${MERGE_LORA_DURING_TRAINING}
  save_latest_checkpoint_only  ${SAVE_LATEST_CHECKPOINT_ONLY}
  shuffle_buffer                ${SHUFFLE_BUFFER_SIZE}
  wandb                         ${WANDB_MODE}  ${WANDB_ENTITY}/${WANDB_PROJECT}
=========================================
EOF

# No CUDA_VISIBLE_DEVICES here on purpose: under gbatch the scheduler decides which GPUs the job
# sees, and hardcoding it fights that.
RUN_ID_NOTE_ARGS=()
if [[ -n "${RUN_ID_NOTE}" ]]; then
  RUN_ID_NOTE_ARGS=(--run_id_note "${RUN_ID_NOTE}")
fi

exec torchrun --standalone --nnodes 1 --nproc-per-node "${GPUS}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --use_l1_regression "${USE_L1_REGRESSION}" \
  --use_diffusion "${USE_DIFFUSION}" \
  --use_film "${USE_FILM}" \
  --num_images_in_input "${NUM_IMAGES_IN_INPUT}" \
  --use_proprio "${USE_PROPRIO}" \
  --lora_rank "${LORA_RANK}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LEARNING_RATE}" \
  --image_aug "${IMAGE_AUG}" \
  --max_steps "${MAX_STEPS}" \
  --save_freq "${SAVE_FREQ}" \
  --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING}" \
  --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}" \
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  "${RUN_ID_NOTE_ARGS[@]}" \
  "$@"
