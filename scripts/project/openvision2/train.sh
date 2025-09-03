#!/usr/bin/env bash
# OpenVision 2 TPU-VM training template
# Pretrain on DataComp then finetune on 224.
# This script runs commands remotely on a TPU-VM via gcloud.

set -euo pipefail

############################################
# 1. TPU VM and platform settings
############################################
export ZONE="your-gce-zone"
export PROJECT_ID="your-gcp-project-id"
export TPU_NAME="your-tpu-name"
export REPO_DIR="simple_clip_jax"            # repo directory on the TPU VM
export VENV_ACTIVATE="~/${REPO_DIR}/maxtext_env/bin/activate"  # python venv activate path

############################################
# 2. Data locations on GCS
############################################
export IN1K_DATA_DIR="gs://your-bucket/imagenet2012"
export COCO_DATA_DIR="gs://your-bucket/coco"
export Flickr_DATA_DIR="gs://your-bucket/flickr30k"
export DATACOMP_PATH="gs://your-bucket/datacomp1b/shards/shards"   # or your TFRecords root

############################################
# 3. Experiment bookkeeping
############################################
export EXP_PATH="gs://your-bucket/openvision2/runs/${TPU_NAME}"    # where to write checkpoints
export ABLATION_NAME="ov2_baseline_datacomp_v2_keep35"
export WANDB_PROJECT="your-wandb-project"
export WANDB_ENTITY="your-wandb-entity"
# Best practice: pass the API key via env when launching this script. Do not hardcode.
#   export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY}"

############################################
# 4. Model and training hyperparameters
############################################
# Model family
export MODEL="L/14"          # S/16, B/16, L/16, L/14, H/14, g/14
export DECODER="L"           # S, B, L, H, g

# Parallelism for a v4-512 pod. Adjust to your topology.
export data_parallelism=256
export fsdp_parallelism=1
export tensor_parallelism=1

# Optim and schedule
export PRE_LR=8e-6
export FT_LR=4e-7
export PRE_TRAIN_EPOCH=10000
export PRE_WARMUP_EPOCH=40
export FT_TRAIN_EPOCH=800
export FT_WARMUP_EPOCH=20

# Resolution and keep ratio
export PRE_RES=84
export FT_RES=224
export PRE_KEEP_RATIO=0.35
export FT_KEEP_RATIO=0.35

# Global batch controls
export BATCH_FACTOR=2       # pretraining 32k global batch means 32 * BATCH_FACTOR on your infra
export FT_BATCH_FACTOR=1    # finetuning batch factor

# Config file inside the repo
export TRAIN_CONFIG=src/configs/openvision2.py

############################################
# 5. Derived names and paths
############################################
export PRE_WANDB_NAME="${MODEL}_bz_${BATCH_FACTOR}_${ABLATION_NAME}_fp32_pre_${PRE_TRAIN_EPOCH}_res_${PRE_RES}"
export FT_WANDB_NAME="${MODEL}_bz_${FT_BATCH_FACTOR}_${ABLATION_NAME}_fp32_ft_${FT_TRAIN_EPOCH}_res_${FT_RES}"

export PRE_WORK_DIR="${EXP_PATH}/${ABLATION_NAME}/${PRE_WANDB_NAME}"
export FT_WORK_DIR="${PRE_WORK_DIR}/ft/${FT_WANDB_NAME}"

# You can point FT_MODEL_INIT to any checkpoint you want to resume from
export FT_MODEL_INIT="${PRE_WORK_DIR}/checkpoint.npz"

############################################
# 6. Basic required checks
############################################
: "${ZONE:?Set ZONE}"
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${TPU_NAME:?Set TPU_NAME}"
: "${DATACOMP_PATH:?Set DATACOMP_PATH}"
: "${EXP_PATH:?Set EXP_PATH}"

############################################
# 7. Helpers
############################################
run_on_tpu() {
  local CMD="$1"
  gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" --worker=all \
    --command "${CMD}"
}

############################################
# 8. Pretraining on DataComp
############################################
echo "[OV2] Launch pretraining on ${TPU_NAME}"
run_on_tpu "cd ${REPO_DIR} && \
  if [ -f preflight.sh ]; then bash preflight.sh PLATFORM=GCE; fi && \
  . ${VENV_ACTIVATE} && \
  if [ -n \"\${WANDB_API_KEY:-}\" ]; then wandb login \${WANDB_API_KEY}; else echo 'W&B API key not set, skipping login'; fi && \
  python3 -m src.main_openvision2 \
  --config=${TRAIN_CONFIG}:\
res=${PRE_RES},img=${MODEL},txt_decoder_name=${DECODER},base_lr=${PRE_LR},keep_ratio=${PRE_KEEP_RATIO},\
batch_factor=${BATCH_FACTOR},\
data_parallelism=${data_parallelism},fsdp_parallelism=${fsdp_parallelism},tensor_parallelism=${tensor_parallelism},\
imagenet_epoch=${PRE_TRAIN_EPOCH},\
vitual_warmup_epoch=${PRE_WARMUP_EPOCH} \
  --workdir=${PRE_WORK_DIR} \
  --config.eval_only=False \
  --config.wandb.log_wandb=True \
  --config.wandb.experiment=${PRE_WANDB_NAME} \
  --config.wandb.project=${WANDB_PROJECT} \
  --config.wandb.entity=${WANDB_ENTITY} \
  --config.model.image.use_flash_attn=False \
  --config.model.image.dtype=float32 \
  --config.model.image.param_dtype=float32 \
  --config.input.data.data_dir=${DATACOMP_PATH} \
  --config.evals.disclf.data_dir=${IN1K_DATA_DIR} \
  --config.evals.retrieval.data_dir=${COCO_DATA_DIR} \
  --config.evals.retrieval_flikr.data_dir=${Flickr_DATA_DIR}
"

############################################
# 9. Optional: clean stray processes
############################################
echo "[OV2] Cleaning potential stray python processes"
run_on_tpu "sudo pkill -f python3 || true; sudo pkill -f python || true"

############################################
# 10. Finetuning at 224
############################################
echo "[OV2] Launch finetune 224 on ${TPU_NAME}"
run_on_tpu "cd ${REPO_DIR} && \
  . ${VENV_ACTIVATE} && \
  if [ -n \"\${WANDB_API_KEY:-}\" ]; then wandb login \${WANDB_API_KEY}; else echo 'W&B API key not set, skipping login'; fi && \
  python3 -m src.main_openvision2 \
  --config=${TRAIN_CONFIG}:\
res=${FT_RES},img=${MODEL},txt_decoder_name=${DECODER},base_lr=${FT_LR},keep_ratio=${FT_KEEP_RATIO},\
data_parallelism=${data_parallelism},fsdp_parallelism=${fsdp_parallelism},tensor_parallelism=${tensor_parallelism},\
batch_factor=${FT_BATCH_FACTOR},\
imagenet_epoch=${FT_TRAIN_EPOCH},\
vitual_warmup_epoch=${FT_WARMUP_EPOCH} \
  --workdir=${FT_WORK_DIR} \
  --config.wandb.log_wandb=True \
  --config.ft_from=${FT_MODEL_INIT} \
  --config.wandb.experiment=${FT_WANDB_NAME} \
  --config.wandb.project=${WANDB_PROJECT} \
  --config.wandb.entity=${WANDB_ENTITY} \
  --config.model.image.dtype=float32 \
  --config.model.image.param_dtype=float32 \
  --config.input.data.data_dir=${DATACOMP_PATH} \
  --config.evals.disclf.data_dir=${IN1K_DATA_DIR} \
  --config.evals.retrieval.data_dir=${COCO_DATA_DIR} \
  --config.evals.retrieval_flikr.data_dir=${Flickr_DATA_DIR}
"

echo "[OV2] All done."
