#!/usr/bin/env bash
# Refine one trained backbone with the fixed-seed ChromDiffusion scripts, once per training seed,
# and score every result with stages 1-3 of the benchmark.
#   1. Backbone predictions on the training and test tiles  (Fixed_Seed_Runs/tools/v2TK_predict.py)
#   2. Normalize them for the refiner                         (Fixed_Seed_Runs/tools/backbone_to_refiner.py)
#   3. Score the backbone alone, for comparison
#   4. For each seed: train the refiner (V17TK_train.py --seed <seed>), refine chr18-22
#      (V16TK_test.py --seed 17 for every run, so runs differ only in training), and score.
#
# Before running:
#   1. Train the backbone (examples/run_fixed_seed_backbone.sh).
#   2. Make the stage 0 reference calls for the cell line (examples/run_example.sh does this).
#   3. Activate an environment with PyTorch and a GPU plus the benchmark tools in ../environment.yml.
#   4. Edit the paths in the block below.
#
# Usage: bash examples/run_fixed_seed_refiner.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
FSR="${REPO}/Fixed_Seed_Runs"
SCRIPTS="${REPO}/benchmark/scripts"

# ---- Edit these ---------------------------------------------------------------
# Cell-line folder, as in run_example.sh.
CELL_TEST_DIR=/path/to/GM12878_test
CELL=GM12878
RATIO=16
# Tile folder from ../../preprocessing: train_/test_data_raw_ratio<RATIO>.npz and the
# per-chromosome lr_test_chr{N}_ratio<RATIO>.npy files.
TILES="${CELL_TEST_DIR}/tensors/data/${CELL}/40x40"
# The refiner's normalization is fitted on the GM12878 /16 training tiles, for every cell line.
STATS_FROM=/path/to/GM12878_test/tensors/data/GM12878/40x40/train_data_raw_ratio16.npz
# modelA_CNN | modelB_ResNet | modelC_GAN15 | hicarn_robust, and its run folder.
BACKBONE=modelA_CNN
BACKBONE_RUN=/path/to/backbone_runs/modelA_CNN_s17
# Everything this script writes goes under here.
OUT=/path/to/refiner_runs/${BACKBONE}
SEEDS="17 117 217 317 417"
CHROM_SIZES=/path/to/hg19.chrom.sizes
export JUICER_JAR=/path/to/juicer_tools_1.22.01.jar
# HiCCUPS needs a CUDA installation on PATH.
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# The GPU to use.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# -------------------------------------------------------------------------------

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}/HiCARN:${PYTHONPATH:-}"   # HiCARN's Models/, for hicarn_robust
export PYTHONUNBUFFERED=1
GT_DIR="${CELL_TEST_DIR}/tensors/predictions/ground_truth"
CHROMS="chr18 chr19 chr20 chr21 chr22"

score() {  # stages 1-3 of the benchmark for a folder of preds_lr_test_chr{N}_ratio<RATIO>.npy
    python "${SCRIPTS}/prediction_convert_combined_TK.py" \
        --input-dir "$1" --input-format npy \
        --juicer-root "${CELL_TEST_DIR}/juicer_ready" --hr-cell "$CELL" \
        --chrom-sizes "$CHROM_SIZES" --ratio "$RATIO"
    bash "${SCRIPTS}/feature_calling_combined_TK.sh" --mode all --pred-dir "$1" --ratio "$RATIO"
    python "${SCRIPTS}/benchmark_combined_TK.py" --mode all \
        --pred-dir "$1" --gt-dir "$GT_DIR" --ratio "$RATIO" \
        --output "$1/benchmark_results.tsv"
}

mkdir -p "$OUT"

# 1-2. Backbone predictions, put into the refiner's normalized space.
python "${FSR}/tools/v2TK_predict.py" --backbone "$BACKBONE" --run "$BACKBONE_RUN" \
    --tiles "$TILES" --ratio "$RATIO" --out "${OUT}/bb_pred"
python "${FSR}/tools/backbone_to_refiner.py" --name "$BACKBONE" --tiles "$TILES" --ratio "$RATIO" \
    --pred "${OUT}/bb_pred" --out "$OUT" --stats_from "$STATS_FROM"

# 3. The backbone alone.
score "${OUT}/${BACKBONE}_raw"

# 4. One refiner per training seed.
for S in $SEEDS; do
    R="${OUT}/refiner_s${S}"
    # The per-epoch loss log is appended to, so every run needs a fresh folder.
    if [[ -e "$R" ]]; then echo "${R} already exists; remove it first" >&2; exit 1; fi
    # Validation results are printed during training, so keep the log.
    python "${FSR}/refiner/V17TK_train.py" \
        --train_hicarn "${OUT}/bb_train/norm/predictions.npy" --train_gt "${OUT}/bb_train/norm/ground_truth.npy" \
        --output_dir "$R" --device cuda:0 --epochs 100 --batch_size 64 --lr 1e-4 \
        --base_channels 64 --num_timesteps 1000 --seed "$S" 2>&1 | tee "${OUT}/refiner_s${S}_train.log"

    python "${FSR}/refiner/V16TK_test.py" --checkpoint "${R}/best_model_iou.pt" \
        --preprocess_file "${OUT}/norm_stats.pt" --chromosomes $CHROMS \
        --hicarn_dir "${OUT}/bb_test" --gt_dir "$TILES" --output_dir "${OUT}/test_s${S}" \
        --device cuda:0 --ratio "$RATIO" --num_steps 20 --seed 17

    # The benchmark reads preds_lr_test_chr{N}_ratio<RATIO>.npy.
    P="${OUT}/refined_s${S}"
    mkdir -p "$P"
    for c in $CHROMS; do
        ln -sf "${OUT}/test_s${S}/raw/refined_${c}.npy" "${P}/preds_lr_test_${c}_ratio${RATIO}.npy"
    done
    score "$P"
done

echo "Results: ${OUT}/${BACKBONE}_raw/benchmark_results.tsv and ${OUT}/refined_s<seed>/benchmark_results.tsv"
