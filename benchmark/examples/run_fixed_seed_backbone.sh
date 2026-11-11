#!/usr/bin/env bash
# Train one backbone with a fixed seed, using the trainers in ../../Fixed_Seed_Runs/backbones.
#
# Before running:
#   1. Activate a Python environment with PyTorch and a GPU (see ../../Fixed_Seed_Runs/README.md).
#   2. Edit the paths in the block below.
#
# Usage: bash examples/run_fixed_seed_backbone.sh modelA_CNN | modelB_ResNet | modelC_GAN15 | hicarn_robust
#   modelA_CNN = HiCNN, modelB_ResNet = SRHiC, modelC_GAN15 = DeepHiC, hicarn_robust = HiCARN-1

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BB="${REPO}/Fixed_Seed_Runs/backbones"

# ---- Edit these ---------------------------------------------------------------
# Training tiles from ../../preprocessing (train_lr, train_hr, ...).
TRAIN_NPZ=/path/to/GM12878_test/tensors/data/GM12878/40x40/train_data_raw_ratio16.npz
# Each run gets its own folder under here: <RUNS>/<backbone>_s<SEED>.
RUNS=/path/to/backbone_runs
SEED=17
# The GPU to use.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# -------------------------------------------------------------------------------

NAME=${1:?usage: $0 modelA_CNN|modelB_ResNet|modelC_GAN15|hicarn_robust}
RUN="${RUNS}/${NAME}_s${SEED}"
if [[ -e "$RUN" ]]; then echo "${RUN} already exists; remove it or pick another seed" >&2; exit 1; fi
mkdir -p "$RUN"
cd "$RUN"   # HiCARN writes ./checkpoints_robust in the working directory

# The HiCARN trainer imports Models/ and Utils/ from ../../HiCARN.
export PYTHONPATH="${REPO}/HiCARN:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

case $NAME in
  modelA_CNN)    CMD=(python "${BB}/v2TK_train_modelA_CNN.py" --train_npz "$TRAIN_NPZ" --output_dir "$RUN"
                      --epochs 100 --batch_size 64 --seed "$SEED") ;;
  modelB_ResNet) CMD=(python "${BB}/v2TK_train_modelB_ResNet.py" --data_path "$TRAIN_NPZ" --save_dir "$RUN"
                      --device cuda:0 --epochs 100 --batch_size 32 --seed "$SEED") ;;
  modelC_GAN15)  CMD=(python "${BB}/v2TK_train_modelC_GAN15.py" --train_npz "$TRAIN_NPZ" --output_dir "$RUN"
                      --epochs 100 --batch_size 64 --seed "$SEED") ;;
  hicarn_robust) CMD=(python "${BB}/v2TK_train_hicarn_robust.py" --train_npz "$TRAIN_NPZ"
                      --seed "$SEED") ;;   # 100 epochs and batch size 64 are fixed in the script
  *) echo "unknown backbone: $NAME" >&2; exit 2 ;;
esac

"${CMD[@]}" 2>&1 | tee "${RUN}/train.log"
echo "Done: ${RUN} (per-epoch log: epoch_log.jsonl)"
