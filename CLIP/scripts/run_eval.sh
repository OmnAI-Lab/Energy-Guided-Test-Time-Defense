#!/bin/bash

set -e

SECONDS=0

export PYTHONPATH="./:${PYTHONPATH}"

GPU=3

DATASET="wds/imagenet1k"
DATASET_SAFE="wds_imagenet1k"

DATASET_ROOT="https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main"
WDS_CACHE_DIR= "" #path to cache the downloaded wds dataset" 

MODEL="ViT-B-32" #ViT-L-14,
PRETRAINED="" #path to the pretrained clip image encoder

OUT_DIR="./results/debug_clip"
ADV_DATA_DIR="./results/adv_datasets"

mkdir -p "${OUT_DIR}"
mkdir -p "${ADV_DATA_DIR}"
mkdir -p "${WDS_CACHE_DIR}"

N_SAMPLES=1000
BATCH_SIZE=50

EPS=4
ITERATIONS=100

TRANSFORM_ALPHA=2.5
TRANSFORM_EPS=5
TRANSFORM_STEPS=2

TRANSFORM_CLASSIFIER="dataset"
# To use ImageNet21k labels, set:
# TRANSFORM_CLASSIFIER="imagenet21k"
# IMAGENET21K_LABELS="" #"/path/to/imagenet21k_wordnet_lemmas.txt"

COMMON_ARGS=(
  --gpu ${GPU}
  --dataset ${DATASET}
  --dataset_root "${DATASET_ROOT}"
  --wds_cache_dir "${WDS_CACHE_DIR}"
  --model "${MODEL}"
  --pretrained "${PRETRAINED}"
  --task zeroshot_classification
  --n_samples ${N_SAMPLES}
  --batch_size ${BATCH_SIZE}
)

TRANSFORM_ARGS=(
  --transform
  --transform_alpha ${TRANSFORM_ALPHA}
  --transform_eps ${TRANSFORM_EPS}
  --transform_max_iters ${TRANSFORM_STEPS}
  --transform_classifier ${TRANSFORM_CLASSIFIER}
)

if [ "${TRANSFORM_CLASSIFIER}" = "imagenet21k" ]; then
  TRANSFORM_ARGS+=(--imagenet21k_labels "${IMAGENET21K_LABELS}")
fi


echo "=============================="
echo "1. Clean without transform"
echo "=============================="

python -m clip_benchmark.cli eval \
  "${COMMON_ARGS[@]}" \
  --attack none \
  --output "${OUT_DIR}/clean_no_transform.json"


echo "=============================="
echo "2. Clean with transform"
echo "=============================="

python -m clip_benchmark.cli eval \
  "${COMMON_ARGS[@]}" \
  --attack none \
  "${TRANSFORM_ARGS[@]}" \
  --output "${OUT_DIR}/clean_with_transform.json"


echo "=============================="
echo "3. Attack without transform"
echo "=============================="

python -m clip_benchmark.cli eval \
  "${COMMON_ARGS[@]}" \
  --attack aa \
  --norm Linf \
  --eps ${EPS} \
  --iterations_adv ${ITERATIONS} \
  --save_adv \
  --save_adv_path "${ADV_DATA_DIR}/adv_${MODEL}_${DATASET_SAFE}_${N_SAMPLES}_aa_${EPS}_${ITERATIONS}.pth" \
  --output "${OUT_DIR}/attack_no_transform.json"


echo "=============================="
echo "4. Attack with transform"
echo "=============================="

python -m clip_benchmark.cli eval \
  "${COMMON_ARGS[@]}" \
  --attack aa \
  --norm Linf \
  --eps ${EPS} \
  --iterations_adv ${ITERATIONS} \
  "${TRANSFORM_ARGS[@]}" \
  --save_adv \
  --save_adv_path "${ADV_DATA_DIR}/adv_${MODEL}_${DATASET_SAFE}_${N_SAMPLES}_aa_${EPS}_${ITERATIONS}.pth" \
  --output "${OUT_DIR}/attack_with_transform.json"


echo "=============================="
echo "Done"
echo "=============================="

echo "Compare:"
echo "  ${OUT_DIR}/attack_no_transform.json"
echo "  ${OUT_DIR}/attack_with_transform.json"

hours=$((SECONDS / 3600))
minutes=$(((SECONDS % 3600) / 60))
echo "[Runtime] $hours h $minutes min"